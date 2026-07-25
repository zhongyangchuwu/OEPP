from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import load_jsonl, write_jsonl


@dataclass(frozen=True)
class ParseResult:
    status: str
    actions: list[int] | None
    error: str | None


def _json_candidates(content: str) -> Iterable[Any]:
    try:
        yield json.loads(content)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, character in enumerate(content):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        yield value


def parse_actions(content: str, horizon: int, candidate_ids: set[int]) -> ParseResult:
    if not isinstance(content, str) or not content.strip():
        return ParseResult("parse_failure", None, "response content is empty")
    for payload in _json_candidates(content):
        if not isinstance(payload, dict):
            continue
        actions = payload.get("actions")
        if not isinstance(actions, list):
            continue
        if any(type(action) is not int for action in actions):
            return ParseResult("parse_failure", None, "actions must be an integer list")
        if len(actions) != horizon:
            return ParseResult(
                "parse_failure", None, f"expected {horizon} actions, received {len(actions)}"
            )
        unknown = [action for action in actions if action not in candidate_ids]
        if unknown:
            return ParseResult("parse_failure", None, f"candidate IDs are out of range: {unknown}")
        return ParseResult("ok", actions, None)
    return ParseResult("parse_failure", None, "no JSON object with an actions field found")


LEGACY_NUMBERED_ACTION_NAMES = "legacy_numbered_action_names"
CANDIDATE_IDS_JSON = "candidate_ids_json"
RESPONSE_PARSERS = frozenset({CANDIDATE_IDS_JSON, LEGACY_NUMBERED_ACTION_NAMES})


def _legacy_action_key(action: str) -> str:
    return "".join(action.casefold().split())


def _candidate_name_ids(candidates: list[dict[str, Any]]) -> dict[str, int]:
    identifiers = [candidate.get("id") for candidate in candidates if isinstance(candidate, dict)]
    if identifiers != list(range(len(candidates))):
        raise ValueError("candidate IDs must be consecutive and ordered")
    names: dict[str, int] = {}
    for candidate in candidates:
        action = candidate.get("text")
        identifier = candidate["id"]
        if not isinstance(action, str):
            raise ValueError("candidate action text must be a string")
        key = _legacy_action_key(action)
        if not key or key in names:
            raise ValueError("candidate actions must have unique normalized names")
        names[key] = identifier
    return names


def parse_numbered_action_names(
    content: str, horizon: int, candidates: list[dict[str, Any]]
) -> ParseResult:
    if not isinstance(content, str) or not content.strip():
        return ParseResult("parse_failure", None, "response content is empty")
    action_ids = _candidate_name_ids(candidates)
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) != horizon:
        return ParseResult(
            "parse_failure", None, f"expected {horizon} numbered actions, received {len(lines)}"
        )
    actions: list[int] = []
    for expected_index, line in enumerate(lines, start=1):
        match = re.fullmatch(r"(?P<index>[1-9]\d*)\.\s+(?P<action>.+?)\s*", line)
        if match is None:
            return ParseResult(
                "parse_failure", None, f"invalid numbered action at position {expected_index}"
            )
        if int(match.group("index")) != expected_index:
            return ParseResult("parse_failure", None, f"expected action number {expected_index}")
        action_id = action_ids.get(_legacy_action_key(match.group("action")))
        if action_id is None:
            return ParseResult(
                "parse_failure",
                None,
                f"action is not in the candidate pool: {match.group('action')}",
            )
        actions.append(action_id)
    return ParseResult("ok", actions, None)


def parse_response(
    content: str, horizon: int, candidates: list[dict[str, Any]], response_parser: str
) -> ParseResult:
    if response_parser == CANDIDATE_IDS_JSON:
        return parse_actions(content, horizon, set(_candidate_name_ids(candidates).values()))
    if response_parser == LEGACY_NUMBERED_ACTION_NAMES:
        return parse_numbered_action_names(content, horizon, candidates)
    raise ValueError(f"unsupported response parser: {response_parser}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly parse OEPP API responses without repairing predictions."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--response-parser", choices=sorted(RESPONSE_PARSERS), default=CANDIDATE_IDS_JSON
    )
    arguments = parser.parse_args()
    manifest = {record["sample_id"]: record for record in load_jsonl(arguments.manifest)}
    predictions: list[dict[str, Any]] = []
    for response in load_jsonl(arguments.responses):
        sample_id = response.get("sample_id")
        if sample_id not in manifest:
            raise ValueError(f"Response references unknown sample_id: {sample_id}")
        sample = manifest[sample_id]
        result = parse_response(
            str(response.get("raw_response", "")),
            int(sample["T"]),
            sample.get("candidate_actions", []),
            arguments.response_parser,
        )
        predictions.append(
            {
                "sample_id": sample_id,
                "status": response.get("status"),
                "parse_status": result.status,
                "action_ids": result.actions,
                "parse_error": result.error,
            }
        )
    write_jsonl(arguments.output, predictions)
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
