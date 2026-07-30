from __future__ import annotations

import argparse
import base64
import importlib
import json
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Any

from common import (
    EXPERIMENT_ROOT,
    append_jsonl,
    endpoint_identity,
    load_json,
    load_jsonl,
    load_yaml,
    sha256_file,
    utc_now,
    write_json,
)
from parse_response import CANDIDATE_IDS_JSON, RESPONSE_PARSERS, parse_response

RETRYABLE_EXCEPTION_NAMES = {"APIConnectionError", "APITimeoutError", "TimeoutException"}
GENERIC_PROMPT_MODE = "candidate_ids_json"
TABLE_V_LEGACY_PROMPT_MODE = "table_v_legacy"
PROMPT_MODES = frozenset({GENERIC_PROMPT_MODE, TABLE_V_LEGACY_PROMPT_MODE})
IMAGE_DETAILS = frozenset({"auto", "low", "high"})

RESERVED_PROVIDER_EXTRA_BODY_FIELDS = frozenset({"model", "messages", "temperature", "max_tokens"})


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Configuration field {name} must be a mapping.")
    return value


def _config_limit(
    config: dict[str, Any], requested_limit: int | None
) -> tuple[dict[str, Any], int]:
    experiment = _mapping(config.get("experiment"), "experiment")
    if experiment.get("api_enabled") is not True:
        raise RuntimeError(
            "API execution is disabled. Keep Phase 1 offline until human review is complete."
        )
    configured_limit = experiment.get("max_calls")
    if type(configured_limit) is not int or configured_limit < 0:
        raise ValueError("experiment.max_calls must be a non-negative integer.")
    limit = configured_limit if requested_limit is None else requested_limit
    if type(limit) is not int or limit <= 0:
        raise ValueError("The requested API-call limit must be a positive integer.")
    if limit > configured_limit:
        raise ValueError("--max-calls cannot exceed experiment.max_calls.")
    if limit > 10:
        approval_path = experiment.get("approval_file")
        if not isinstance(approval_path, str) or not approval_path:
            raise RuntimeError("More than ten API calls require experiment.approval_file.")
        approval = load_json(Path(approval_path))
        if approval.get("approved") is not True or approval.get("max_calls", 0) < limit:
            raise RuntimeError("Approval file does not authorize the requested API-call limit.")
    return experiment, limit


def _load_environment(model_config: dict[str, Any]) -> tuple[str, str, str]:
    try:
        dotenv = importlib.import_module("dotenv")
    except ImportError as error:
        raise RuntimeError(
            "Install project dependencies with `uv sync` before API execution."
        ) from error
    dotenv.load_dotenv(EXPERIMENT_ROOT / ".env")
    api_key_env = model_config.get("api_key_env")
    base_url_env = model_config.get("base_url_env")
    model_env = model_config.get("model_env")
    if (
        not isinstance(api_key_env, str)
        or not api_key_env
        or not isinstance(base_url_env, str)
        or not base_url_env
        or not isinstance(model_env, str)
        or not model_env
    ):
        raise ValueError(
            "model.api_key_env, model.base_url_env, and model.model_env must be non-empty strings."
        )
    api_key = os.environ.get(api_key_env)
    base_url = os.environ.get(base_url_env)
    model = os.environ.get(model_env)
    if not api_key or not base_url or not model:
        missing = [
            label
            for label, value in (("api key", api_key), ("base URL", base_url), ("model", model))
            if not value
        ]
        raise RuntimeError(f"Missing environment value(s): {', '.join(missing)}")
    return api_key, base_url, model


def _prompt_path(config: dict[str, Any]) -> tuple[str, Path, str]:
    prompt = _mapping(config.get("prompt"), "prompt")
    version, location = prompt.get("version"), prompt.get("path")
    mode = prompt.get("mode", GENERIC_PROMPT_MODE)
    if not isinstance(version, str) or not isinstance(location, str):
        raise ValueError("prompt.version and prompt.path must be strings.")
    if mode not in PROMPT_MODES:
        raise ValueError(f"unsupported prompt mode: {mode}")
    path = Path(location)
    if not path.is_absolute():
        path = EXPERIMENT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file does not exist: {path}")
    return version, path, mode


def _image_detail(request_config: dict[str, Any]) -> str:
    image_detail = request_config.get("image_detail", "auto")
    if image_detail not in IMAGE_DETAILS:
        choices = ", ".join(sorted(IMAGE_DETAILS))
        raise ValueError(f"request.image_detail must be one of: {choices}.")
    return image_detail


def _provider_extra_body(request_config: dict[str, Any]) -> dict[str, Any] | None:
    extra_body = request_config.get("provider_extra_body")
    if extra_body is None:
        return None
    if not isinstance(extra_body, dict):
        raise ValueError("request.provider_extra_body must be a mapping when set.")
    reserved_fields = RESERVED_PROVIDER_EXTRA_BODY_FIELDS.intersection(extra_body)
    if reserved_fields:
        names = ", ".join(sorted(reserved_fields))
        raise ValueError(f"request.provider_extra_body cannot override: {names}.")
    try:
        json.dumps(extra_body)
    except (TypeError, ValueError) as error:
        raise ValueError("request.provider_extra_body must be JSON serializable.") from error
    return extra_body


def _mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type not in {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}:
        raise ValueError(f"Unsupported or unknown image type for {path}")
    return mime_type


def _image_content(path_text: str, image_detail: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest image does not exist: {path}")
    mime_type = _mime_type(path)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    descriptor = {
        "basename": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "mime_type": mime_type,
    }
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}", "detail": image_detail},
    }, descriptor


def _candidate_actions(sample: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = sample.get("candidate_actions")
    if (
        not isinstance(candidates, list)
        or not candidates
        or not all(isinstance(item, dict) for item in candidates)
    ):
        raise ValueError(f"{sample.get('sample_id')} has no candidate actions")
    candidate_ids = [candidate.get("id") for candidate in candidates]
    if candidate_ids != list(range(len(candidates))):
        raise ValueError(f"{sample.get('sample_id')} candidate IDs must be consecutive and ordered")
    if not all(isinstance(candidate.get("text"), str) for candidate in candidates):
        raise ValueError(f"{sample.get('sample_id')} candidate action text must be strings")
    return candidates


def _image_content_groups(
    sample: dict[str, Any], labelled: bool, image_detail: str
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    content: list[dict[str, Any]] = []
    descriptors: dict[str, list[dict[str, Any]]] = {"start": [], "goal": []}
    for label, key in (("START", "start_images"), ("GOAL", "end_images")):
        images = sample.get(key)
        if not isinstance(images, list) or not all(isinstance(image, str) for image in images):
            raise ValueError(f"{sample.get('sample_id')} has invalid {key}")
        for index, image in enumerate(images, start=1):
            image_content, descriptor = _image_content(image, image_detail)
            if labelled:
                content.append({"type": "text", "text": f"{label} image {index}"})
            content.append(image_content)
            descriptors["start" if label == "START" else "goal"].append(descriptor)
    return content, descriptors


def _legacy_prompt(
    template: str, candidate_text: str, horizon: int
) -> tuple[list[str], str, str, str]:
    try:
        prompt = json.loads(template)
    except json.JSONDecodeError as error:
        raise ValueError("table_v_legacy prompt must be valid JSON") from error
    system_messages = prompt.get("system_messages")
    user_start = prompt.get("user_start")
    user_end = prompt.get("user_end")
    user_output = prompt.get("user_output", "")
    if (
        not isinstance(system_messages, list)
        or not all(isinstance(message, str) for message in system_messages)
        or not isinstance(user_start, str)
        or not isinstance(user_end, str)
        or not isinstance(user_output, str)
    ):
        raise ValueError(
            "table_v_legacy prompt needs string system_messages, user_start, user_end, "
            "and user_output"
        )
    variables = {"T": horizon, "candidate_action_list": candidate_text}
    return (
        [message.format(**variables) for message in system_messages],
        user_start.format(**variables),
        user_end.format(**variables),
        user_output.format(**variables),
    )


def _prepare_request(
    sample: dict[str, Any],
    prompt_template: str,
    prompt_mode: str,
    response_parser: str,
    image_detail: str = "auto",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = _candidate_actions(sample)
    system_message_count = 0
    if prompt_mode == GENERIC_PROMPT_MODE:
        candidate_text = "\n".join(
            f"{candidate['id']}: {candidate['text']}" for candidate in candidates
        )
        text = prompt_template.format(T=sample["T"], candidate_action_list=candidate_text)
        content, descriptors = _image_content_groups(
            sample, labelled=True, image_detail=image_detail
        )
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
    else:
        candidate_text = ",".join(candidate["text"] for candidate in candidates)
        system_messages, user_start, user_end, user_output = _legacy_prompt(
            prompt_template, candidate_text, sample["T"]
        )
        system_message_count = 1
        start_content, descriptors = _image_content_groups(
            {**sample, "end_images": []}, labelled=False, image_detail=image_detail
        )
        end_content, end_descriptors = _image_content_groups(
            {**sample, "start_images": []}, labelled=False, image_detail=image_detail
        )
        descriptors["goal"] = end_descriptors["goal"]
        content = [{"type": "text", "text": user_start}, *start_content]
        content.extend([{"type": "text", "text": user_end}, *end_content])
        if user_output:
            content.append({"type": "text", "text": user_output})
        messages = [
            {"role": "system", "content": "\n\n".join(system_messages)},
            {"role": "user", "content": content},
        ]
    metadata = {
        "sample_id": sample["sample_id"],
        "image_setting": sample["image_setting"],
        "image_transport": "data_url",
        "image_detail": image_detail,
        "system_message_count": system_message_count,
        "start_images": descriptors["start"],
        "goal_images": descriptors["goal"],
        "candidate_count": len(candidates),
        "candidate_order_seed": sample["candidate_order_seed"],
        "pool_type": sample["pool_type"],
        "prompt_mode": prompt_mode,
        "response_parser": response_parser,
        "prompt_variables": {"T": sample["T"]},
    }
    return messages, metadata


def _is_retryable(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    return (
        status_code == 429
        or isinstance(status_code, int)
        and 500 <= status_code <= 599
        or error.__class__.__name__ in RETRYABLE_EXCEPTION_NAMES
    )


def _serialized(value: Any) -> Any:
    if value is None:
        return None
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return value


def _response_parser(config: dict[str, Any]) -> str:
    response_parser = config.get("response_parser", CANDIDATE_IDS_JSON)
    if response_parser not in RESPONSE_PARSERS:
        raise ValueError(f"unsupported response parser: {response_parser}")
    return response_parser


def _safe_config_snapshot(config: dict[str, Any], model: str, base_url: str) -> dict[str, Any]:
    experiment = _mapping(config.get("experiment"), "experiment")
    request = _mapping(config.get("request"), "request")
    prompt_version, prompt_path, prompt_mode = _prompt_path(config)
    return {
        "captured_at": utc_now(),
        "model": model,
        "endpoint": endpoint_identity(base_url),
        "request": request,
        "experiment": {"expected_split": experiment.get("expected_split")},
        "response_parser": _response_parser(config),
        "prompt": {"version": prompt_version, "path": str(prompt_path), "mode": prompt_mode},
    }


def _validate_manifest(manifest: list[dict[str, Any]], expected_split: str | None = None) -> None:
    sample_ids = [sample.get("sample_id") for sample in manifest]
    if not all(isinstance(sample_id, str) and sample_id for sample_id in sample_ids):
        raise ValueError("Every manifest record needs a non-empty sample_id.")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Manifest sample_id values must be unique.")
    if expected_split is None:
        return
    if expected_split not in {"base", "novel"}:
        raise ValueError("experiment.expected_split must be 'base' or 'novel'.")
    if any(sample.get("split") != expected_split for sample in manifest):
        raise ValueError(f"Manifest does not contain only the expected {expected_split} split.")


def run(
    config_path: Path, manifest_path: Path, run_name: str, requested_limit: int | None
) -> dict[str, int]:
    if Path(run_name).name != run_name or run_name in {"", ".", ".."}:
        raise ValueError("--run-name must be a single directory name.")
    config = load_yaml(config_path)
    experiment, call_limit = _config_limit(config, requested_limit)
    model_config = _mapping(config.get("model"), "model")
    request_config = _mapping(config.get("request"), "request")
    if request_config.get("concurrency") != 1:
        raise ValueError("This runner is intentionally serialized; request.concurrency must be 1.")
    max_retries = request_config.get("max_retries")
    backoff = request_config.get("backoff_seconds")
    if not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("request.max_retries must be a non-negative integer.")
    if not isinstance(backoff, (int, float)) or backoff <= 0:
        raise ValueError("request.backoff_seconds must be a positive number.")
    backoff_seconds = float(backoff)
    image_detail = _image_detail(request_config)
    provider_extra_body = _provider_extra_body(request_config)
    api_key, base_url, model = _load_environment(model_config)
    prompt_version, prompt_path, prompt_mode = _prompt_path(config)
    response_parser = _response_parser(config)
    prompt_template = prompt_path.read_text(encoding="utf-8")
    manifest = load_jsonl(manifest_path)
    _validate_manifest(manifest, experiment.get("expected_split"))
    run_dir = EXPERIMENT_ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", _safe_config_snapshot(config, model, base_url))
    requests_path = run_dir / "requests.jsonl"
    responses_path = run_dir / "responses.jsonl"
    predictions_path = run_dir / "predictions.jsonl"
    failures_path = run_dir / "failures.jsonl"
    ledger_path = EXPERIMENT_ROOT / "runs" / "api_call_ledger.jsonl"
    completed_ids = {
        record["sample_id"]
        for record in load_jsonl(responses_path)
        if record.get("status") == "success"
    }
    saved_request_ids = {record["sample_id"] for record in load_jsonl(requests_path)}
    call_count = sum(1 for record in load_jsonl(ledger_path) if record.get("run_name") == run_name)
    attempted_samples = 0
    completed_samples = 0

    try:
        openai = importlib.import_module("openai")
    except ImportError as error:
        raise RuntimeError(
            "Install project dependencies with `uv sync` before API execution."
        ) from error
    client = openai.OpenAI(
        api_key=api_key, base_url=base_url, timeout=request_config["timeout_seconds"]
    )

    for sample in manifest:
        sample_id = sample["sample_id"]
        if sample_id in completed_ids:
            continue
        if call_count >= call_limit:
            break
        try:
            messages, request_metadata = _prepare_request(
                sample, prompt_template, prompt_mode, response_parser, image_detail
            )
        except Exception as error:
            failure = {
                "timestamp": utc_now(),
                "sample_id": sample_id,
                "status": "local_validation_failure",
                "error_type": error.__class__.__name__,
                "error": str(error),
            }
            append_jsonl(failures_path, failure)
            continue
        if sample_id not in saved_request_ids:
            append_jsonl(
                requests_path,
                {
                    **request_metadata,
                    "timestamp": utc_now(),
                    "vid": sample["vid"],
                    "event": sample["event"],
                    "split": sample["split"],
                    "T": sample["T"],
                    "prompt_version": prompt_version,
                },
            )
        response_record: dict[str, Any] | None = None
        for retry_count in range(max_retries + 1):
            if call_count >= call_limit:
                break
            append_jsonl(
                ledger_path,
                {
                    "timestamp": utc_now(),
                    "run_name": run_name,
                    "sample_id": sample_id,
                    "attempt": retry_count + 1,
                    "endpoint": endpoint_identity(base_url),
                    "model": model,
                },
            )
            call_count += 1
            attempted_samples += 1
            started_at = time.monotonic()
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=request_config["temperature"],
                    max_tokens=request_config["max_tokens"],
                    **({"extra_body": provider_extra_body} if provider_extra_body else {}),
                )
                latency_seconds = time.monotonic() - started_at
                raw_content = completion.choices[0].message.content or ""
                parsed = parse_response(
                    raw_content, sample["T"], sample["candidate_actions"], response_parser
                )
                response_record = {
                    "timestamp": utc_now(),
                    "sample_id": sample_id,
                    "status": "success",
                    "model": model,
                    "endpoint": endpoint_identity(base_url),
                    "prompt_version": prompt_version,
                    "prompt_mode": prompt_mode,
                    "response_parser": response_parser,
                    "usage": _serialized(completion.usage),
                    "response_id": completion.id,
                    "latency_seconds": latency_seconds,
                    "retry_count": retry_count,
                    "raw_response": raw_content,
                    "raw_api_response": _serialized(completion),
                    "parse_status": parsed.status,
                    "parse_error": parsed.error,
                    "parsed_action_texts": parsed.action_texts,
                }
                append_jsonl(responses_path, response_record)
                append_jsonl(
                    predictions_path,
                    {
                        "sample_id": sample_id,
                        "status": "success",
                        "parse_status": parsed.status,
                        "action_ids": parsed.actions,
                        "action_texts": parsed.action_texts,
                        "parse_error": parsed.error,
                    },
                )
                completed_samples += 1
                break
            except Exception as error:
                latency_seconds = time.monotonic() - started_at
                transient = _is_retryable(error)
                failure = {
                    "timestamp": utc_now(),
                    "sample_id": sample_id,
                    "status": "api_failure",
                    "model": model,
                    "endpoint": endpoint_identity(base_url),
                    "retry_count": retry_count,
                    "latency_seconds": latency_seconds,
                    "error": str(error),
                    "retryable": transient,
                }
                append_jsonl(failures_path, failure)
                if not transient or retry_count == max_retries:
                    append_jsonl(responses_path, failure)
                    break
                time.sleep(backoff_seconds * (2**retry_count))
        if response_record is None and call_count >= call_limit:
            break
    return {
        "api_attempts": attempted_samples,
        "completed_samples": completed_samples,
        "ledger_total": call_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run resumable, ledger-capped Qwen3-VL OEPP API evaluation."
    )
    parser.add_argument("--config", type=Path, default=EXPERIMENT_ROOT / "configs" / "qwen3vl.yaml")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-calls", type=int)
    arguments = parser.parse_args()
    try:
        result = run(arguments.config, arguments.manifest, arguments.run_name, arguments.max_calls)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
