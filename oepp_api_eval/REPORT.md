# API 基线任务进展报告

**更新日期：** 2026-07-26  
**范围：** OEPP Table V 的 $T=4$、3+3 图像设置下的新 MLLM API 基线。  
**代码边界：** 仅修改本 fork 的 `OEPP/`；`OEPP_server/` 仅用于核对历史协议、路径和结果，未修改。

## 1. 当前结论

Qwen3-VL-32B 的真实 API 全量评测与双评分已完成：Base 855 / 855、Novel 1,297 / 1,297 observation 均获得 response，API failure 为 0。论文兼容主指标为 Base `1.52/27.22/31.35`、Novel `3.93/29.74/51.77`（SR/Acc/mIoU）；严格候选池/格式审计另行报告。Base 的 33 个缺失源视频窗口与历史 Table V 结果一致地不在此可评测分母内。

| 阶段 | 状态 | 说明 |
|---|---|---|
| Table V 协议冻结 | 完成 | $T=4$；起始 `start_f+[0,1,2]`；终止 `end_f+[-2,-1,0]`；Base→Base pool，Novel→Novel pool。 |
| 服务器提帧工具 | 完成 | 提取、JPEG 校验、relative-path observation index、缺失视频记录。 |
| 本地 manifest 构建 | 完成 | 校验六张图片、唯一映射本地 annotation、生成候选动作 ID 与 GT。 |
| 历史动作名称解析 | 完成 | 严格接收 `1. action name` 格式，不做语义修复。 |
| API 执行与费用控制 | 完成 | 默认禁用、私有配置启用、调用账本、重试、恢复、usage/latency/raw response 保存。 |
| Base 全量提帧 | 完成 | 855 条可用 observation；33 条缺失源视频。 |
| Novel 全量提帧 | 完成 | 1,297 条可用 observation；与历史 Novel Table V 样本集合一致。 |
| 本地 API pilot 与全量 | 完成 | 历史对齐后的 prompt/pool/config 已完成 2,152 条串行全量调用、response 审计与双模式评分。 |

## 2. 已实现内容

### 2.1 Table V 图像提取

新增 `src/extract_tablev_frames.py`：

- 输入历史 `T=4_base.json` 或 `T=4_novel.json`；
- 从每条记录的 `video_path` 读取原始 MP4；
- 每个可用窗口导出六张 JPEG，最长边统一为 512；
- 写出 observation JSONL，其中图片路径相对 `--frame-root`，可整体同步到本地；
- 写出 unavailable JSONL，保留每个失败 observation 的原因；
- 不会将缺失视频静默跳过或伪造图片。

### 2.2 本地 manifest 与 API 执行

新增 `src/build_tablev_manifest.py`：

- 将服务器 observation 与 `OEPP/data/` 的本地 annotation 交叉验证；
- 对历史 Base 888 条与 Novel 1,297 条 $T=4$ 序列均已验证唯一匹配；
- 检查每个 manifest 样本的六张图片、GT action、split pool 与候选 ID；
- 输出可直接交给 `src/run_api.py` 的 manifest。

扩展 `src/run_api.py` 与 `src/parse_response.py`：

- 支持历史 Table V 风格的 system/user prompt 布局；
- 使用 `legacy_numbered_action_names` parser；
- 响应必须包含编号连续、长度正确、且严格属于候选池的动作名称；
- 保存 prompt mode、parser、raw response、usage、latency、retry 和非秘密请求元数据；
- 保留原有 JSON action-ID protocol，避免破坏通用 API 实验流程。

### 2.3 安全与配置

新增：

- `configs/qwen3vl_tablev_t4_3x3.yaml`
- `configs/prompts/tablev_t4_3x3_legacy.json`

配置默认：

```yaml
experiment:
  api_enabled: false
  max_calls: 0
```

API key、实际模型 ID 与启用后的配置仅能放在 `.env` 和 gitignore 的 `private/` 目录中。`manifests/`、`sampled_frames/`、`runs/` 也不会被提交。

## 3. Base 提帧结果与问题记录

服务器执行结果：

```text
Extracted 855 observation(s); recorded 33 unavailable observation(s).
```

### 3.1 缺失视频

33 条 unavailable 均为：

```text
FileNotFoundError
source video does not exist: /data1/wuyilu/data/crosstask/OEPP/<task_id_old>/<vid>.mp4
```

它们是缺失的 CrossTask 原始视频，不是 API 失败，也不是模型预测失败。33 是滑窗数，不代表 33 个独立视频；同一个缺失视频会贡献多个 $T=4$ 滑窗。

评测约定：

| 类别 | 是否进入 API manifest / 指标分母 | 处理 |
|---|---|---|
| 原始视频缺失，无法取得 observation | 否 | 在 unavailable JSONL 中报告；单列可用覆盖率。 |
| 有效图片已生成，但 API 请求失败 | 是 | 记入分母，零分。 |
| API 返回动作数错误、未知动作或格式错误 | 是 | 记入分母，零分。 |

Base 结果必须报告为：

```text
历史 T=4 Base 窗口：888
不可获得视觉 observation：33
有效 Base API manifest：855
```

不得用空白图、其他视频、预提取 feature 或人工补全替代缺失视频。

### 3.2 H.264 解码警告

服务器提帧中出现：

```text
[h264] mmco: unref short failure
```

这是 OpenCV 底层 FFmpeg 在部分 H.264 文件 seek / 解码时发出的警告。当前 unavailable JSONL 的 33 条全部是 `FileNotFoundError`，因此这些 warning 不是 33 条缺失 observation 的原因。只要对应样本的六张 JPEG 已成功生成，warning 不影响该样本进入 manifest。

若后续 unavailable JSONL 出现 `cannot open source video` 或 `cannot decode a frame`，应单独记录并对该视频进行 ffmpeg 诊断；不要把解码错误伪装为 API 失败。

## 4. 数据位置

历史服务器代码表明：

```text
Table V Base 序列： /data1/wuyilu/OEPP/LLM/T=4_base.json
Table V Novel 序列：/data1/wuyilu/OEPP/LLM/T=4_novel.json

COIN 视频：         /data1/zhuchenhui/pvg_data/coin/<task_id_old>/<vid>.mp4
CrossTask 视频：    /data1/wuyilu/data/crosstask/OEPP/<task_id_old>/<vid>.mp4
历史 annotation：   /data1/wuyilu/OEPP/data_prepare/
```

提帧器优先使用历史序列 JSON 中每条记录的 `video_path`，因此不需要手工拼接视频路径。

## 5. 已验证证据

- 历史 T=4 sequence 与本地 OEPP annotation 的唯一匹配：Base 888 / 888；Novel 1,297 / 1,297；
- `uv sync` 成功，`opencv-python-headless` 已安装；
- 修改文件的 Ruff format/check 通过；
- 14 个单元测试通过；
- `extract_tablev_frames.py --help` 与 `build_tablev_manifest.py --help` 可运行；
- 默认 Qwen Table V 配置被 offline gate 拒绝，未发送 API 请求。

## 6. 下一步

1. 在服务器执行 Novel 的全量提帧；
2. 保留 Base / Novel 的 `*_unavailable.jsonl`；
3. 将 `sampled_frames/tablev_t4_3x3/` 与 observation JSONL 同步到本地，保持相对路径结构；
4. 本地分别构建 Base 与 Novel manifest；
5. 复制安全配置到 `private/`，填入 `.env` 的真实 Qwen endpoint 与精确模型 ID；
6. 在批准调用额度后先运行小规模 pilot；
7. 检查图片顺序、prompt、raw response、usage、解析失败和费用后，再运行全量；
8. Base / Novel 分别评测，最后仅在模型 ID、prompt、候选池与候选顺序一致时汇总。

## 7. 相关文件

| 文件 | 用途 |
|---|---|
| `README.md` | 操作命令与服务器/本地工作流。 |
| `configs/qwen3vl_tablev_t4_3x3.yaml` | 默认禁用的 Table V Qwen 配置。 |
| `src/extract_tablev_frames.py` | 服务器原视频提帧。 |
| `src/build_tablev_manifest.py` | 本地验证与 manifest 构建。 |
| `src/run_api.py` | 带额度、重试、账本和恢复的 API 执行器。 |
| `src/parse_response.py` | JSON ID 与历史编号动作名称 parser。 |
| `tests/test_tablev_protocol.py` | Table V 便携 manifest、缺视频和图片顺序回归测试。 |

## 8. 当前任务要求与执行规划

当前工作分为两个最高优先级任务。两项任务共享原则：代码只维护在本 fork；服务器只执行训练或原视频提帧；所有原始结果、配置、模型 ID、checkpoint 和失败记录必须可追溯。

### 8.1 任务 A：MLLM API 基线

**目标：** 先完成一个 SiliconFlow 托管的 Qwen 视觉模型，在 Table V 的 $T=4$、3+3、split-specific action-pool 协议下得到 Base 和 Novel 的可复现实验结果。

**当前模型调研：**

- SiliconFlow Chat Completions endpoint 是 `https://api.siliconflow.cn/v1/chat/completions`，使用 `Authorization: Bearer <API key>`；
- 官方多模态文档确认 Qwen 系列通过 `messages[].content` 的多个 `image_url` 条目支持多图输入，`image_url.url` 可使用 URL 或 base64 data；现有 `run_api.py` 已输出该兼容格式；
- **已确认的模型约束：** 首个 MLLM 仅使用 SiliconFlow 的 **Qwen3-VL 系列**；本轮不使用 Qwen3.5、Qwen3.6 或其他 Qwen 系列模型；
- SiliconFlow 官方 2026-04-15 release note 曾声明 `Qwen/Qwen3-VL-235B-A22B-Instruct` 和 `Qwen/Qwen3-VL-235B-A22B-Thinking` 下线，因此不能假定历史 235B ID 当前仍可调用；
- 正式运行前必须通过带认证的 `GET https://api.siliconflow.cn/v1/models` 查询当前可用的 **Qwen3-VL chat/VLM** 精确 ID，记录查询时间、模型页价格和额度。若返回中没有可用的 Qwen3-VL chat/VLM，本任务应停在 provider-availability blocker，不能自动切换到 Qwen3.5/3.6。

**代码审查结论：**

| 项目 | 状态 | 处理 |
|---|---|---|
| OpenAI-compatible base URL | 已支持 | `run_api.py` 从环境变量读取 `base_url_env`，无需改 runner。 |
| 多图 data URL | 已支持 | 现有 runner 为每张 JPEG 构造 `image_url` data URL；与 SiliconFlow 文档兼容。 |
| 六图 Table V prompt | 已支持 | `table_v_legacy` 按 start 三图、goal 三图发送。 |
| SiliconFlow 环境变量 | 已私有配置 | `.env` 仅保存 endpoint/model；key 只由继承的 `SILICONFLOW_API_KEY` 读取。 |
| 精确可用模型 ID | 已账户验证 | `/models` 返回 `Qwen/Qwen3-VL-32B-Instruct`；本次 pilot 固定使用该 ID。 |
| 图像 `detail` 参数 | 已固定 | 每张图显式记录 `auto`；SiliconFlow 当前文档说明它采用 low-resolution 预处理。 |
| 成本汇总 | token 投影已完成 | runner 保存实际 usage；公开模型页未暴露 unit price，正式运行前必须保存账户可见的 input/output 价格。 |

**已执行：**

1. 已同步并验证 Base 855 条有效 observation（另有 33 条缺视频）和 Novel 1,297 条有效 observation；
2. 已为两个 split 重新构建本地 manifest；Base 122 项和 Novel 55 项 source pool 的原始文本、顺序与历史脚本一致，包含 Base 的 whitespace-only alias；
3. 已用账户 `/models` 确认 `Qwen/Qwen3-VL-32B-Instruct` 可用；
4. 已完成 8 次有账本的受限 API 尝试；2+2 pilot 只用于 transport/format 验证，发生在本次 Table V 对齐修正之前，不能作为最终 prompt/scoring 配置的质量结论；
5. 已将 Table V 历史 prompt 和评分对齐修正落实到 Base/Novel 独立配置、manifest、parser 和 evaluator。全量运行尚未开始。

**全量运行前置：** 保存账户 Model Square 的 input/output 单价与查询日期；用户批准基于下列 token/时间投影的预算；单独创建超过 10 calls 的 approval file。

**环境变量说明：** API key 已在执行 runner 的 shell 中存在，未打印、未写入 `.env`、config snapshot、JSONL 或 Git；`.env` 仅保存 `SILICONFLOW_BASE_URL` 和 `SILICONFLOW_QWEN_MODEL`。

**官方资料：**

- SiliconFlow Chat Completions API: <https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions/api-reference/chat-completions/chat-completions>
- SiliconFlow 多模态输入: <https://api-docs.siliconflow.cn/docs/userguide/capabilities/multimodal-vision>
- SiliconFlow 模型列表 API: <https://docs.siliconflow.cn/en/api-reference/models/get-model-list>
- SiliconFlow release notes: <https://api-docs.siliconflow.cn/docs/release-notes/overview>
- SiliconFlow rate limits: <https://api-docs.siliconflow.cn/docs/userguide/faqs/rate-limit-and-upgradation>

### 8.1.1 本地 Qwen3-VL-32B-Instruct pilot（已完成）

- **pre-alignment pilot 调用协议：** `Qwen/Qwen3-VL-32B-Instruct`、`https://api.siliconflow.cn/v1`、Table V $T=4$ 3+3、`image_detail=auto`、temperature 0、max tokens 256、串行、无重试；历史对齐后的公开 Base/Novel 配置将 `max_tokens` 恢复为历史值 4096，但尚未发起新调用；
- **接入与候选池修复：** 当前 shell 使用 SOCKS proxy，而 OpenAI/httpx 环境缺少 `socksio`；在 `pyproject.toml` 显式加入 `httpx[socks]` 后，client 才能初始化。pre-alignment pilot 曾折叠 Base 的 whitespace-only alias；当前 manifest 已保留历史 122 项原始 pool text，parser/evaluator 仅在匹配时 casefold/去空白；
- **输出约束修复：** 原始历史 prompt 在该模型上会解释或虚构动作。保留原始 `legacy_v1` 文件，新增 Base/Novel 独立 `legacy_*_v2`：保留对应历史 split 的 user instruction、合并相同的 system 内容，并在第六张图后追加严格四行输出指令。最终 pilot 配置的 5/5 响应严格解析成功；
- **pilot 指标：** Base 2 条：SR 0.0%、Acc 50.0%、mIoU 58.33%；Novel 2 条：SR 0.0%、Acc 25.0%、mIoU 58.33%。样本太少，只验证运输、格式和评测链路，不能作为模型质量结论；
- **最终配置观测：** Base 3 条平均 1,888 prompt tokens、29 completion tokens、2.801 s；Novel 2 条平均 1,449 prompt tokens、28 completion tokens、2.797 s。证据位于 `runs/siliconflow-qwen3vl32-pilot-20260726/pilot_report.json`；
- **全量投影：** 2,152 条有效视觉 observation（Base 855 + Novel 1,297）约为 3,493,593 input tokens + 61,111 output tokens。按当前串行均值为 6,022.8 秒（100.4 分钟）；服务波动、429 和保守恢复空间后预留 2 小时；
- **费用投影：** 若账户 Model Square 的输入/输出单价分别为 $P_{in}$、$P_{out}$ CNY / 1M tokens，预计费用为 $3.493593P_{in}+0.061111P_{out}$ CNY。SiliconFlow 公开文档确认视觉 token 计入模型 context 并按 token 计费，但公开 reader 与未登录模型页未显示该模型单价；不得杜撰金额。

**当前决策：** GPT-compatible Base/Novel prompt、原始候选池文本、paper-compatible 主评分与 strict 审计评分均已实现并通过历史回放验证；等待价格快照和全量预算批准，期间不再发送 API 请求。

### 8.2 任务 B：Embedding 质量分析

**目标：** 重新训练选定的论文 baseline，保存最终 checkpoint，并在冻结 checkpoint 上导出 Base / Novel 每个窗口、每个预测步骤的连续预测 embedding、GT text embedding、MSE、cosine similarity、候选 margin 和最终动作是否正确。

**代码审查结论：**

- `TransformerEncoder.forward()` 和 `MLP.forward()` 都返回每个 step 的连续 embedding 列表；
- `Seq_action.__getitem__()` 已返回 GT `action_tensor`，可直接作为 GT text embedding；
- 当前 `eval.py` 只保存动作文本与 SR/Acc/mIoU，不保存连续 embedding；必须新增独立 exporter，不能靠现有 `base_output.json` / `novel_output.json` 反推；
- 当前本地仓库没有有效 `.pth/.pt/.ckpt`；`attention_config.yaml` 指向服务器输出目录；
- `pdpp_train.py --evaluate` 仍在 epoch 循环中调用 `model.train(...)`，不是纯评测入口。PDPP 的 embedding 分析必须新增 `eval-only` 加载路径，使用冻结的 EMA checkpoint，不能在 test 阶段继续训练。
- `train.py` 仅在 validation SR 严格高于初始值 0 时调用 `torch.save(model, ...)`；若所有 validation SR 都为 0，`model_dir` 不会被定义。重训前应改为无条件保存最后 checkpoint，并以明确的 validation 规则保存 best checkpoint；
- PDPP 虽定义了 checkpoint 保存函数，但实际最佳 checkpoint 调用已被注释；仅重跑当前 `pdpp_train.py` 不保证留下可用于 embedding 导出的 checkpoint；
- `Seq_action` 在返回值中没有 event、dataset 和滑窗起点。exporter 必须按相同的滑窗/padding 规则重建 metadata，并断言其动作序列与 dataloader 输出一致，否则无法可靠按事件和 step 分组。

**执行顺序：**

1. 先实现参数化 embedding exporter、稳定 sample metadata、逐 step 数据格式、统计和绘图脚本；
2. 先修正 Transformer 的 checkpoint 保存策略：每个 run 保存最后 checkpoint，best checkpoint 只由 validation 选择，并保存 config 与 hash；
3. 在服务器以固定的 split、feature、seed、loss 和 validation 规则重新训练 Transformer baseline；
4. 在 `model.eval()` 与 `torch.inference_mode()` 下，对 Base / Novel 各执行一次 exporter；
5. 保存 float32 raw embedding 分片、逐 step CSV、汇总 CSV 和图：cosine by step、MSE by step、Base-vs-Novel、correct-vs-wrong、cosine margin；
6. 之后取消 PDPP best-checkpoint 保存的缺口，并新增纯 eval/export 路径；不直接使用现有 `pdpp_train.py --evaluate`。

**最小交付：**

```text
embedding_results/
├── run_metadata.json
├── raw_embeddings/
├── metrics_per_sample.csv
├── summary_metrics.csv
├── cosine_by_step.png
├── mse_by_step.png
├── base_vs_novel.png
├── correct_vs_wrong.png
└── cosine_margin_by_step.png
```

## 9. 当前阻塞项

1. SiliconFlow 账户可见 input/output 单价尚未以 provider 记录固化；当前仅有用户提供的 2 CNY / 1,000 requests 估算，不能写作实际结算价；
2. 托管 API response 仅返回 `Qwen/Qwen3-VL-32B-Instruct` 字符串，未提供权重 revision/hash；同时 `image_detail=auto` 的实际预处理分辨率和未显式设定的 provider sampling defaults 没有逐请求回传，故不能宣称可在未来逐字节重现；
3. 当前只有一个固定 candidate order、一次 hosted-API sweep，未量化 candidate-order 或 provider-run 随机性；它支持可审计 baseline，不支持将 Qwen/GPT 指标差异归因为模型能力本身；
4. Transformer / PDPP 的训练 checkpoint 不在本地，需要在服务器完成重训并保存。

## 10. Table V 论文与历史实验一致性审计（全量前）

**审计来源：** 论文 `OEPP-TIP.pdf` 的 Table V 与 §V-C、`OEPP_server/LLM_latest/4olatest_T=4_image3_{base,novel}.py`、`LLM_latest/llm_prase.ipynb`、历史 `T=4_*_image3_results.json`，以及当前 manifest / runner / evaluator。

### 已确认一致的部分

- **论文目标行：** Table V 的 `ChatGPT-4o-latest, T=4, Num of images=3` 为 Base SR/Acc/mIoU `1.87/31.78/39.43`、Novel `5.94/35.62/57.48`。按历史 `LLM_latest` 结果和 notebook 的实现重算后精确复现该行的四舍五入值；
- **样本分母：** 当前 Base 855 / Novel 1,297 manifest 与历史 `T=4_*_image3_results.json` 的 `(sequence index, vid, start/end, normalized GT)` 集合完全一致；Base 缺视频的 33 个滑窗没有进入任一历史结果或当前视觉 manifest；
- **图像协议：** 旧脚本和当前 extractor 都以最长边 512 的 JPEG 传输 `start_f+[0,1,2]` 与 `end_f+[-2,-1,0]` 六帧。论文 §V-C 对三图设置的描述亦一致；
- **候选池：** server `data_prepare` 与本 fork 的 Base/Novel action pool 字节内容一致；Base→Base、Novel→Novel 的 split-specific pool 选择一致；
- **指标数学：** 历史 notebook 对 action 名小写、去空白后计算逐位置 Acc、全序列 SR、逐样本 set mIoU；在所有样本固定 $T=4$ 时，当前逐样本 Acc 平均与历史的逐 action Acc micro-average 数值相同。

### 剩余且不可消除的实质差异

1. **Qwen prompt 不是历史 ChatGPT 调用的字节级复现。** Base/Novel 已分别保留各自历史 user instruction 和 `max_tokens=4096`，但 Qwen 必须将三个历史 system messages 合并为一个，并在第六张图后增加严格四行输出指令，才能可靠地遵循候选池；
2. **采样策略不同。** 历史 GPT-4o 脚本没有固定 temperature 且尝试传递随机 seed；SiliconFlow Chat Completions 未在当前官方文档中声明该 endpoint 的 seed 控制，因此 Qwen 固定 `temperature=0` 以保证可复现；
3. **Provider preprocessing 不同。** 当前明确记录 SiliconFlow `image_detail=auto`；其官方文档将其说明为 low-resolution preprocessing。历史 GPT proxy 未指定 `detail`，故视觉 tokenization 不能视作相同；
4. **像素级相同无法回溯证明。** 历史脚本未保存实际请求图片，只能由代码确认相同时间点、顺序、512 resize 和 JPEG 方式。当前保存 frame hash，后续 Qwen 结果可复审，但不能声明与历史瞬时帧 JPEG 二进制相同。
### 全量前结论

**目前可比较：** 相同的 Table V sliding-window 样本、六帧时间协议、split pool、$T=4$ 和 SR/Acc/mIoU 定义。

**尚不可宣称完全复现：** provider/model、Qwen-specific system-message serialization、末尾格式约束、采样控制和 provider image preprocessing；其余可控的 Table V 样本、帧、pool、split prompt 与主评分已经对齐。

**已落实：** 全量 Qwen 使用 split-specific prompt（Novel 移除历史 latest 中没有的 repetition instruction）；主表使用论文兼容的 normalized action-wise SR/Acc/mIoU，严格 candidate/format compliance 与 API failure rate 作为并列审计报告。API error、缺失预测或无法恢复四个编号动作的响应仍保留分母并计零；四个格式正确但 out-of-pool 的动作按论文兼容 scorer 逐位置计分。

### 10.1 已实施的历史对齐修正

- 新增 `tablev_t4_3x3_legacy_base_v2.json` 与 `tablev_t4_3x3_legacy_novel_v2.json`。Base 保留历史 repetition instruction，Novel 与 `LLM_latest` 一致地移除它；runner 通过 `expected_split` 拒绝 prompt/config 与 manifest split 不匹配的运行；
- Base manifest 重新保留历史 pool 的全部 122 个原始 action text，Novel 保留 55 个；parser 对 whitespace-only alias 解析为同一 normalized name，但不删除 prompt 中的原始候选；
- `ParseResult`、response JSONL 和 prediction JSONL 现在保存结构正确的原始 action texts。`evaluate.py --scoring-mode paper_compatible` 对其执行历史的大小写/空白归一化、逐位置 Acc 和 set mIoU；`strict` 仍保留为格式/候选池合规审计；
- 用历史 `LLM_latest` 的完整 T=4、3-image 输出回放：paper-compatible 模式复现 Base `1.871345/31.783626/39.428014` 和 Novel `5.936777/35.620663/57.482469`（SR/Acc/mIoU），与论文 Table V 的两位小数完全一致；strict replay 单独保留 37 Base、56 Novel 条 candidate-compliance failure；
- 剩余且已记录的不可消除差异只有 Qwen-specific 单 system serialization、末尾四行格式指令、temperature 0 的可复现性选择，以及 SiliconFlow `image_detail=auto` 与历史 GPT proxy 未指定 detail 的 provider preprocessing 差异。

## 11. Qwen3-VL-32B 对齐后全量运行状态

- **已启动：** 后台进程 `qwen3vl32-tablev-full` 以串行顺序运行 Base，完成 Base 后自动运行 Novel；不会并发写 workspace-wide API ledger；
- **固定运行名：** `siliconflow-qwen3vl32-base-full`、`siliconflow-qwen3vl32-novel-full`；运行监督状态写入 `runs/siliconflow-qwen3vl32-full/launch.json`，安全进度快照写入同目录 `progress.json`；
- **审批与预算：** 启动 ledger 为 8；持久化 handoff 后最终 ledger 上限为 2,161（新全量尝试至多 2,153：2,152 样本 + 1 条 handoff reserve，无自动 retry）。按用户提供的 2 CNY / 1,000 requests，预计新增请求为 4.306 CNY，ledger 上限为 4.322 CNY；
- **持久化 handoff：** 初始非持久 supervisor 在 ledger 85 时被受控停止并迁移为 `persist=true`；当时 Base 有 76 条完成 response、无 API failure，保留 1 条额外 call 以重新覆盖停止时在途请求，防止 Base 样本遗漏；
- **运行完成：** `2026-07-25T21:57:43+00:00`，串行 supervisor 正常退出（exit 0）。Base 855 / 855、Novel 1,297 / 1,297 均获得 provider response，合计 2,152 / 2,152，API failure 为 0；最终 ledger 为 2,161。
- **严格格式审计（未计入 API failure）：** Base 782 条 strict parse success、73 条 strict parse failure；Novel 1,240 条 strict parse success、57 条 strict parse failure。两种评分的最终结果见 §12，绝不将它们混合。
- **固定全量协议：** Base/Novel split-specific historical prompts、原始 pool text、`max_tokens=4096`、`temperature=0`、`image_detail=auto`、无自动 retry。

## 12. Qwen3-VL-32B 全量指标

**评测输入：** `tablev_t4_3x3` 的 Base 855 / Novel 1,297 个 observation，以及各 split 的完整 `predictions.jsonl`。每个 split 独立评分；未合并不同 split、prompt、candidate order 或 scoring mode。

### 12.1 论文兼容主指标

| Split | Samples | SR | Acc | mIoU |
|---|---:|---:|---:|---:|
| Base | 855 | 1.52 | 27.22 | 31.35 |
| Novel | 1,297 | 3.93 | 29.74 | 51.77 |

`paper_compatible` 对格式正确的 raw four-action response 使用历史 Table V 的 action-name 大小写/空白归一化、逐位置 Acc、全序列 SR 和逐样本 set mIoU。无 API failure；每个样本仍在分母内。相对 §10 中回放的 ChatGPT-4o-latest 历史行，Base 的 SR/Acc/mIoU 分别低 0.35/4.56/8.08 个百分点，Novel 分别低 2.00/5.88/5.71 个百分点。该比较共享 Table V 样本、帧和评分协议，但不应表述为 provider/model 的调用配置复现，原因见 §10。

### 12.2 严格合规审计

| Split | Samples | SR | Acc | mIoU | Strict failures | Failure rate | API failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 855 | 1.52 | 25.99 | 29.98 | 73 | 8.54% | 0 |
| Novel | 1,297 | 3.93 | 28.87 | 50.29 | 57 | 4.39% | 0 |

严格失败为候选池/格式不合规，因此 strict scorer 对该样本整条输出计零；它们不等同于 API failure。全部 855 个 Base、1,297 个 Novel response 都提供了可按论文兼容规则评分的 raw action sequence，故主指标不省略这些样本，也不语义修复输出。

### 12.3 可复核产物

- Base：`runs/siliconflow-qwen3vl32-base-full/metrics_{paper_compatible,strict}.json`、`per_sample_{paper_compatible,strict}.jsonl`、`summary_{paper_compatible,strict}.md`；
- Novel：`runs/siliconflow-qwen3vl32-novel-full/metrics_{paper_compatible,strict}.json`、`per_sample_{paper_compatible,strict}.jsonl`、`summary_{paper_compatible,strict}.md`；
- 全量 supervisor：`runs/siliconflow-qwen3vl32-full/launch.json`；运行账本：`runs/api_call_ledger.jsonl`。

## 13. 论文写入审查

### 13.1 决定

**可作为“Qwen3-VL-32B 的 hosted-API Table V baseline”写入论文，附带方法与限制披露；不可写作 Qwen 与历史 GPT-4o 的严格受控、模型能力因果比较，也不可写作历史 GPT 调用的字节级复现。**

### 13.2 已排除的运行级问题

- Base 855 / Novel 1,297 的 manifest、request、response 与 prediction 的 `sample_id` 集合完全相同、各自唯一；无缺失或额外样本；
- 全部 2,152 个 response 的 returned model string 均为 `Qwen/Qwen3-VL-32B-Instruct`，response ID 唯一，`finish_reason=stop`，并携带 usage；没有截断、API failure 或重试替代成功样本；
- Base strict 73 条、Novel strict 57 条失败全部是 `action is not in the candidate pool`。每条仍有四个编号 action text；不存在长度、编号或停止原因导致的结构性失败；
- config snapshot 固定了 split、$T=4$、3+3、split pool、原始候选顺序、`temperature=0`、`max_tokens=4096`、prompt version 和 data-URL image transport。完整证据在 §12.3。

### 13.3 与历史 GPT 结果的关系

相同 observation 的 paired sample bootstrap（10,000 resamples；仅衡量测试样本抽样不确定性，不衡量 hosted API 随机性）显示：Base Acc 差值 `-4.56 pp`（95% CI `[-6.43, -2.66]`）、mIoU `-8.08 pp`（`[-10.14, -6.02]`）；Novel SR `-2.00 pp`（`[-3.47, -0.54]`）、Acc `-5.88 pp`（`[-7.59, -4.12]`）、mIoU `-5.71 pp`（`[-7.12, -4.32]`）。Base SR 差值 `-0.35 pp` 的 CI `[-1.52, 0.82]` 跨越零。

这些差异不能由 strict output failure 单独解释：切换至 strict 后 Base/Novel Acc 分别再下降 `1.23/0.87 pp`，mIoU 分别再下降 `1.37/1.49 pp`，SR 不变。因而“Qwen 在本协议下低于历史 GPT 行”是可报告的观察结果；它不是模型家族间的受控能力排序。

### 13.4 必须披露的限制

1. 写明 provider、endpoint、模型 ID、运行日期、prompt version、Base/Novel action-pool size（122/55）、$T=4$、3+3 frame protocol、`temperature=0`、`max_tokens=4096`、source-order candidate list、Base 33 个缺视频排除和 Base/Novel 分母；
2. 主表使用 `paper_compatible` 指标时，必须同表注或附录同时给出 strict 指标与 out-of-pool failure rate（Base 8.54%，Novel 4.39%）；
3. 说明 Qwen 和历史 GPT 的 provider、system-message serialization、尾部格式指令、采样设置及 image preprocessing 不同（§10）；
4. SiliconFlow 文档将 `detail` 定义为 `auto`/`low`/`high` 图像预处理控制，并在 response 中只保证返回 model string、usage 和 completion ID；当前结果没有权重 hash 或实际 `auto` 图像分辨率，故以 hosted API snapshot 而非可冻结权重报告；
5. API 模型训练数据不可得，无法排除 OEPP 或近似程序视频的训练数据重合；将结果表述为 zero-shot API evaluation，不能声明 contamination-free generalization。

### 13.5 发表前建议

当前结果无需因发现运行错误而重跑。若论文需要“Qwen 与 GPT-4o 的公平比较”或“输出格式可靠”的强结论，必须另做匹配 prompt/image preprocessing 的 GPT 对照、固定 candidate-order 敏感性和随机抽样重复请求；这些应作为新实验版本，不能与本 §12 基线合并。

外部方法依据：[SiliconFlow Vision API](https://docs.siliconflow.com/en/userguide/capabilities/vision)、[SiliconFlow Chat Completions API](https://docs.siliconflow.com/en/api-reference/chat-completions/chat-completions)、[Qwen3-VL-32B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)。

## 14. OpenRouter Gemini 3.1 Flash Lite 可用性检查

- **模型检索：** `2026-07-26` 使用当前 `OPENROUTER_API_KEY` 查询 `GET https://openrouter.ai/api/v1/models`，账户可见 `google/gemini-3.1-flash-lite`。返回 metadata 声明其为 `text+image+file+audio+video->text`，支持 `temperature`、`max_tokens`、`seed` 与 structured outputs，context 为 1,048,576、最大 completion 为 65,536。该时的 advertised prompt/completion 单价为 `$0.25/$1.50` 每百万 token；image 与 internal reasoning 字段也单独计价，不能把这些价格投影为已有 Qwen run 的实际成本。
- **安全配置与协议：** 新增的 Base/Novel checked-in config 保持 `api_enabled: false`、`max_calls: 0`，使用现有 split-specific Table V prompt、3+3 JPEG data URLs、原始 action-pool order 和 legacy numbered-action parser。真实 key 只存在环境变量；启用的两份 pilot config 位于 gitignored `private/`。
- **预算修复：** 原 `run_api.py` 以工作区全局 ledger 长度限制新运行，已有 Qwen 的 2,161 ledger entries 会使小型 Gemini pilot 零调用退出。现在仍保留全局审计 ledger，但仅统计同一 `run_name` 的 attempts；回归测试证明既有 Qwen record 不消耗新 Gemini run 的 `max_calls`。
- **Base pilot：** 运行 `openrouter-gemini31flashlite-base-pilot-20260726`，使用 `temperature=0`、`max_tokens=512`、无自动 retry、最多 2 calls。两条 Base observation 均在发送六张已 hash 的 JPEG 后返回 HTTP 403：`This model is not available in your region.`；没有 provider completion、usage、parsed prediction 或可报告指标。
- **停止条件：** 403 为非 retryable provider-availability failure，未重复发送，也没有启动 Novel pilot。该结果仅证明当前 key/region 无法访问这个模型，不能解释为模型质量、视觉能力或格式能力。保持 model ID 不变，等待区域可用性恢复或用户明确选择并批准另一个 account-visible Gemini text-and-image model；不得自动 fallback。

外部接口依据：[OpenRouter API reference](https://openrouter.ai/docs/api-reference/overview)、[OpenRouter models API](https://openrouter.ai/api/v1/models)。
