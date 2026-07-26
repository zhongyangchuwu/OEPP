# 2026-07-26 汇报材料

**用途：** 8 点项目状态汇报。所有结论均区分已完成结果、正在执行的 API 任务和未验证推断。

## 一句话结论

- **Table V API 基线：** 历史 ChatGPT-4o-latest 论文行已回放复现；Qwen3-VL-32B hosted-API 基线已完整完成；Gemini 3 Flash Preview 已通过 Japan VPN 的 2 Base + 2 Novel 六图 transport/format pilot，正式 Base 855 条已完成收集，Novel 正在由持久顺序任务继续执行。
- **Experiment 4：** Transformer、MLP、PDPP 三个 fresh 200-epoch run 均已完成并导出；最终 raw artifacts 已下载本地并与服务器 SHA-256 一致。
- **不能现在宣称：** Gemini 全量 SR/Acc/mIoU，或 Gemini/Qwen/GPT 的模型能力排序；Gemini Novel 与全量审计尚未完成。

## 1. 实验协议与可比性

| Track | 固定协议 | 主输出 | 不应混同 |
|---|---|---|---|
| Table V hosted API | `T=4`、3 start + 3 goal JPEG、Base→Base / Novel→Novel pool、分 split 评测 | SR、Acc、mIoU | provider、模型、prompt version、候选顺序、`paper_compatible` / `strict` |
| Experiment 4 embedding | VideoCLIP、split 1、`T=3`、`is_pad=1`、seed 42、fresh 200 epochs | cosine、MSE、candidate top-1、margin、GT rank | 不能与 Table V API SR/Acc/mIoU 直接比较 |

Table V Base pool 有 122 actions、Novel pool 有 55 actions；可用 observation 为 Base 855、Novel 1,297。Experiment 4 的导出窗口为 Base 1,138、Novel 1,691。

## 2. Table V：历史与 Qwen 已完成结果

| Model / source | Split | SR | Acc | mIoU | 状态 |
|---|---|---:|---:|---:|---|
| ChatGPT-4o-latest 历史输出回放 | Base | 1.87 | 31.78 | 39.43 | 论文行精确复现 |
| ChatGPT-4o-latest 历史输出回放 | Novel | 5.94 | 35.62 | 57.48 | 论文行精确复现 |
| Qwen/Qwen3-VL-32B-Instruct `paper_compatible` | Base | 1.52 | 27.22 | 31.35 | 855 / 855，0 API failure |
| Qwen/Qwen3-VL-32B-Instruct `paper_compatible` | Novel | 3.93 | 29.74 | 51.77 | 1,297 / 1,297，0 API failure |
| Qwen strict compliance | Base | 1.52 | 25.99 | 29.98 | 73 out-of-pool failures，8.54% |
| Qwen strict compliance | Novel | 3.93 | 28.87 | 50.29 | 57 out-of-pool failures，4.39% |

**汇报口径：** Qwen 是特定 provider、日期、prompt、candidate order 与图像预处理下的 hosted-API zero-shot baseline；与历史 GPT 共享同一窗口、帧协议、split pool 与评分数学，但不是受控因果能力比较。

## 3. Gemini 3 Flash Preview：实时状态

**模型与路径：** `google/gemini-3-flash-preview` through OpenRouter; OpenRouter routing metadata returned `provider=Google` on pilot success.

| 阶段 | 结果 | 证据 |
|---|---|---|
| 非 Japan route | 403 `This model is not available in your region.` | 1 Base smoke failure |
| Japan VPN Base smoke | 成功；4 行、in-pool、`finish_reason=stop` | 1 / 1 parsed |
| Japan VPN transport pilot | Base 2 / 2、Novel 2 / 2 provider success、legacy parser success、0 API failure | 4 requests；Google provider |
| 正式 full run | Base 855 / 855 completed | `gemini3flash-full` nonblocking log |
| 正式 Novel run | 未形成完成日志或指标 | persistent sequential process 仍在运行；不报告指标 |

Pilot total reported usage / cost：Base 14,778 prompt + 60 completion tokens、`$0.007569`；Novel 14,264 prompt + 53 completion tokens、`$0.007291`。这些是 transport/format evidence，不是质量结果或全量成本预测。

**当前风险：** 可用性依赖 Japan VPN route；模型是 preview；OpenRouter provider routing 与上游版本不可冻结。待 Novel 完成后，必须审计 Base/Novel request-response-prediction coverage、returned model、provider、finish reason、usage、parse failures 与两种评分，再公布 Gemini 指标。

## 4. Experiment 4：validation-selected training checkpoints

| Model | Selected epoch | Validation SR | Validation Acc | Validation mIoU / mIoU1 |
|---|---:|---:|---:|---:|
| Transformer | 152 | 25.82% | 55.61% | 59.48% |
| MLP | 145 | 27.63% | 56.21% | 61.14% |
| PDPP | 168 / 200 | 30.58% | 56.93% | 68.80% |

所有 checkpoint 仅以 validation 选择；Base/Novel test 未参与选择。PDPP `last.pt` 到 epoch 200，但 epoch 168 的 validation SR/Acc 更高，因此使用 epoch 168 的 EMA `best.pt`。

## 5. Experiment 4：最终 embedding 汇总

`Correct` 是候选 action cosine top-1 的逐 step 平均；不是 Table V Acc，也不是上述 validation Acc。

| Model | Split | Cosine ↑ | MSE ↓ | Candidate top-1 ↑ | Candidate margin ↑ |
|---|---|---:|---:|---:|---:|
| Transformer | Base | 0.5590 | 0.01530 | 0.5501 | 0.01987 |
| Transformer | Novel | 0.3410 | 0.01952 | 0.3739 | -0.10268 |
| MLP | Base | 0.5426 | 0.01799 | 0.5565 | 0.01323 |
| MLP | Novel | 0.3288 | 0.02069 | 0.3566 | -0.11931 |
| PDPP | Base | **0.8012** | **0.00842** | **0.5691** | -0.03192 |
| PDPP | Novel | **0.5997** | **0.01748** | 0.3369 | -0.14173 |

**可说的观察：**

1. PDPP 在连续 embedding cosine 与 MSE 上，Base 和 Novel 都优于 direct Transformer / MLP。
2. PDPP 的 Base candidate top-1 也最高；Novel candidate top-1 则由 Transformer 最高。
3. PDPP 的 overall candidate margin 在 Base / Novel 都为负；连续 embedding 接近 GT 并不自动意味着候选 action separation 最强。
4. Novel 均明显弱于 Base；这是跨 split 泛化差异的观察，不可归因于某一个模型组件。

## 6. 本地 artifact 复核

最终汇报只使用以下三个目录，**不混入 calibration / legacy smoke**：

```text
embedding_results/attention_t3_seed42_run1/
embedding_results/mlp_t3_seed42_run1/
embedding_results/pdpp_t3_seed42_run1/
```

每个 final run 的已验证 contract：

| Check | Base | Novel |
|---|---:|---:|
| predicted / GT tensor | `(1138, 3, 768)` | `(1691, 3, 768)` |
| CSV window-step rows | 3,414 | 5,073 |
| pre-specified figures | 7 | 7 |

本地六个 final raw NPZ 的 SHA-256 与服务器同名文件逐一一致。PDPP `run1_repeat` 与 `run1` 的 Base/Novel predicted 及 GT tensors 逐元素完全相同，证明 fixed `sampling_seed=42` 的 exporter 复现性。

**图表入口：** `embedding_results/<final-run>/figures/`。汇报主图建议依次选择：`base_vs_novel.png`、`cosine_by_step.png`、`mse_by_step.png`、`candidate_margin_by_step.png`；保留其余三张作为附录，不按结论筛选。

## 7. 建议的 7 页汇报结构

1. **目标与协议**：Table V hosted API + Experiment 4 embedding 是两条独立证据链。
2. **数据与审计**：T=4 3+3 API、T=3 embedding、pool / frame / split / checkpoint 规则。
3. **历史 GPT 与 Qwen baseline**：表 2，附 strict compliance caveat。
4. **Gemini 当前进度**：Japan VPN pilot passed；Base full completed；Novel live；明确“无全量指标”。
5. **Experiment 4 training validity**：fresh initialization、validation-only checkpoint selection、PDPP epoch 168 / 200。
6. **Embedding result table**：表 5；解释 continuous fit 与 discrete candidate separation 的差异。
7. **风险与下一步**：Gemini completion audit、provider/region reproducibility、图表完整审阅、论文表述边界。

## 8. 高概率问题与答复

- **为什么 Gemini 还没有分数？** 全量 Novel 尚未完成并审计；提前公布是把不完整样本错误当作结果。
- **为什么 Qwen 有两套指标？** `paper_compatible` 回放论文历史 action-name scoring；`strict` 把候选池外或格式无效输出完整计零，二者回答不同问题。
- **PDPP 为什么 cosine 更高但 Novel top-1 较低？** 连续 embedding loss 与候选 pool 的最近邻决策边界不同；margin 和 rank 是必要的补充诊断。
- **下载到本地是否可信？** 六个 final raw NPZ 与服务器 SHA-256 一致，且每个结果目录通过 shape、row count、figure count contract；这强于仅检查目录存在。

## 9. 汇报后待办

1. 让 persistent Gemini full process 自然完成；不在执行中改变 VPN、模型、prompt、candidate order 或配置。
2. 完成后运行 Base/Novel coverage、API/parse failure、provider / finish reason / usage 审计与 `paper_compatible`、`strict` 双评分。
3. 将 Gemini 作为独立 OpenRouter/Japan-route hosted-API baseline 报告，不与 Qwen/GPT 输出混合。
4. 依据完整预指定图表写 Experiment 4 的科学解释；当前仅报告已观察到的数值。
