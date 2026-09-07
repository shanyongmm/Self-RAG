# 临床知识库对比评测：Naive RAG vs Self/Corrective RAG

> 面向当前 5 篇临床文档（高血压指南 2025 / 糖尿病指南 2022 / 医疗器械分类调整 2026 / 基本药物目录 2018+2026）的 **可复现**、**抗失效** 对比评测。
> 亮点指标 = **成本-效果（Cost-Effectiveness）**：Self-RAG 用更少的生成上下文 token 拿到等于或更好的答案质量。

---

## 1. 为什么旧评测不能用了

| 旧数据集 | 问题 |
| --- | --- |
| `qa_eval.jsonl`（客服库） | 文档不同，README 里引用的 best_report 数字（Precision 33%→87% 等）来自**旧客服知识库**，与当前临床 KB 无关。 |
| `qa_eval_medical.jsonl`（期刊题，20 条） | 题目指向的期刊文章根本不在当前 5 篇文档里；`expected_chunk_ids` 还是某次旧 MinerU 解析的 id。文档重新解析后 `block_index` 全变，**逐条 0 命中** → 2026-08-29 那次实跑 Self-RAG 反而不如 Naive（答案 0.45 vs 0.55），纯属数据集失配，不是真实水平。 |

结论：对比必须 **锚定当前文档的真实原文**，并能在每次重新解析后**自动重算 chunk id**。

---

## 2. 新方法：gold_passage 锚定 + 离线解析 chunk id

### 2.1 数据集不再写死 chunk id

种子集 `evaluation/datasets/clinical_qa_seed.jsonl` 每个 case 只写：

- `id`：用例编号；
- `partition`：所属分区 `diagnosis / device / medication`（用于校验路由）；
- `doc`：命中的源文档名（只是可读提示，不参与打分）；
- `question`：用户问题（问句刻意只用该分区关键词，保证 Self-RAG 首轮就路由进正确分区）；
- `reference_answer`：参考回答（给 reviewer 看，不直接参与判分）；
- `answer_keywords`：**判分依据** —— 全部命中才算该题回答正确；
- `gold_passages`：**1~N 句逐字摘自当前 parsed markdown 的原文片段**，是定位正确 chunk 的锚。

### 2.2 解析器 `evaluation/build_qa_dataset.py`

`build_qa_dataset.py` 用与 `scripts/ingest_partitioned.py` **完全相同的** loader + splitter + config：

1. 遍历 `data/parsed/mineru_partitions/<partition>`；
2. `load_documents_from_path()` 读 markdown；
3. `split_documents()` 按当前 `RAG_CHUNK_SIZE/OVERLAP` 重分块；
4. 把每个 `gold_passage` 做**去空白**后与每个 chunk 做“子串包含”匹配；
5. 命中的 `chunk_id` 全部收集为该 case 的 `expected_chunk_ids`（已含 `partition:` 前缀，与 Milvus 主键格式一致）；
6. **只要有一句 gold_passage 找不到就抛错退出** —— 用硬失败防止旧版那种“静默失配”。

产物：`evaluation/datasets/qa_eval_clinical.jsonl`，可直接喂给 `run_eval.py`。

> chunk_id 是确定性的：`f"{source_stem}:{block_index}:{chunk_index}:sha1(...)[:10]"`，
> 解析结果只依赖「markdown 文本 + 切分参数」，与机器/路径无关，因此离线可复算。

---

## 3. 打分口径与公平性

`evaluation/metrics.py` 在每条 case 上对两种 pipeline 算同一组指标：

| 指标 | 含义 |
| --- | --- |
| `raw_precision_at_k` | 首轮检索 top-k 里命中 gold chunk 的比例（Naive 与 Self-RAG 首轮都算） |
| `final_context_precision` | **最终进入生成**的上下文里命中 gold chunk 的比例（Self-RAG = 过滤后；Naive = top-k 原样） |
| `invalid_retrieval_rate` | `1 - final_context_precision`（无效检索率） |
| `answer_correct` | `answer_keywords` **全部**出现在模型答案里 |
| `citation_accuracy / recall` | 模型给出的 citations 与 gold chunk 的对齐情况 |
| `retry_count / latency` | Self-RAG 改写重试轮数 / 单题耗时 |

### 公平性处理

- Naive baseline 用**全库检索**（不分分区），Self-RAG 首轮**收窄到路由分区**。全库检索空间更大，等于给 baseline 一点优势 —— 这是**保守对比**：即便 baseline 更容易搜到正确片段，Self-RAG 仍要靠 grade/filter 把噪音挡在生成之外。
- 两边 `top_k` 默认都是 `.env` 的 `RAG_TOP_K=3`；若用 `--top-k` 覆盖，只会作用于 Naive，请同步把 `RAG_TOP_K` 改一致。
- 题目全部设计成单分区问题（问句只含目标分区关键词），避免“路由本身”成为差异来源，让对比聚焦在 Self-RAG 的 `grade → decide → rewrite` 机制上。

---

## 4. 成本-效果指标（本次主打的“另一个指标”）

Self-RAG 最直观的卖点是质量，但它的改写/评分也带来更多 LLM 调用与延迟。要证明它“值”，必须同时看**单位质量的成本**：

在每模式汇总里新增：

- `avg_generation_context_tokens`：最终生成阶段喂给答案模型的证据 token（问题 + 最终 sources 文本）均值。Self-RAG 过滤掉被判无关的片段后，这个值应显著小于 Naive 的整包 top-k。
- `tokens_per_correct_answer` = （全部无错误 case 的 估算输入 token + 输出 token）÷ **正确回答数**。越低越省。
- `total_generation_context_tokens` / `total_output_tokens`：无错误 case 的合计。

在模式对比里新增：

- `generation_context_token_reduction_pct`：Self-RAG 相比 Naive 生成上下文 token 降幅（%）。
- `tokens_per_correct_answer_reduction_pct`：**每得到一个正确答案省下的估算 token 比例（%）= 成本-效果 headline**。
- `total_token_reduction_pct`：总 token 均值降幅。
- `correct_answer_count_delta`：正确题数差。

**估算口径（务必写进简历/汇报，别夸大）**：
`estimate_tokens` = CJK 字符数 + 非 CJK 字符数 ÷ 4 向上取整，是对「证据载荷」的估算，**不含**系统提示词，也**不含** Self-RAG 中途 grade / rewrite / decide / memory 这些额外 LLM 调用的 token。因此它对比的是“同样生成一个答案，两边要喂多少证据”；Self-RAG 额外增加的内部调用成本体现为更高的 `avg_latency_seconds`（会如实报出，作为质量的代价维度）。两边用同一估算器，相对结论稳健。

---

## 5. 运行步骤

先决条件：服务已在跑（Milvus + PostgreSQL + 后端），`.env` 就绪，且 **Milvus 里的 chunk 与当前 `data/parsed/mineru_partitions` 一致**。

```bash
# 0) 保险起见重入库一次（用已解析 md，不重新调用 MinerU；会按当前切分重建行）
#    若之前 re-parse 过但没重入库，这一步不可省，否则 expected id 仍会对不上 DB。
python -m scripts.ingest_partitioned --no-parse

# 1) 解析种子集 -> qa_eval_clinical.jsonl（gold_passage -> 当前 chunk_id）
python -m evaluation.build_qa_dataset

# 2) 先冒烟跑 3 条，确认打分正常（推荐）
python -m evaluation.run_eval --mode both --limit 3

# 3) 全量正式跑（默认自动用 qa_eval_clinical.jsonl）
python -m evaluation.run_eval --mode both
# 也可以显式指定：
# python -m evaluation.run_eval --dataset evaluation/datasets/qa_eval_clinical.jsonl --mode both
```

产物在 `evaluation/results/`：

- `eval_report_<ts>.json` —— 汇总：`modes`（两套指标）+ `comparison`（含成本-效果降幅）。
- `eval_details_<ts>.jsonl` —— 逐题明细（含 trace），可回溯每一轮的 retrieve/grade/decide/rewrite。

终端在 both 模式会额外打印 `成本-效果一览`。

---

## 6. 怎么读报告、怎么用

预期如果 Self-RAG 正常发挥：

- `final_context_precision` Self-RAG 明显高于 Naive（尤其 `clinic-med-001/002` 这种“2018 vs 2026 长得几乎一样”的干扰对）；
- `invalid_retrieval_rate` 下降；
- `avg_generation_context_tokens` 下降（20%~50% 量级）；
- `tokens_per_correct_answer_reduction_pct` **为负**（负 = 省）且与质量指标同向 —— 这就是你要的“成本-效果”卖点；
- `answer_accuracy` Self-RAG ≥ Naive（持平或更高都行，成本端必须有优势）；
- `avg_latency_seconds` Self-RAG 更高 —— 汇报时诚实说明：用延迟换“更少证据 + 更高质量”。

简历/README 建议口径示例：

> 在 14 道临床问答题上，Self/Corrective RAG 相较 Naive RAG 用约 **-XX% 的生成上下文 token** 达到 **+YY pp 的答案准确率** / 相同准确率，最终上下文精确率提升 **+ZZ pp**，无效检索率下降 **-WW%**；代价是单题延迟增加 **+NNs**（改写重试带来的内部调用）。

跑完把真实数字填进去。

---

## 7. 新增/修改题目

1. 打开对应的 parsed markdown（`data/parsed/mineru_partitions/<分区>/<doc>/<doc>.md`），**逐字**复制一段能回答问题的话到 `gold_passages`（可容忍空白差异，但不能容忍字符差异；标点、数字、特殊连字符必须一致）。
2. `question` 保证只含该分区关键词（见 `knowledge_partitions.json` 的 `route_keywords`），避免误路由到别的分区。
3. `answer_keywords` 选 2~4 个**必然出现在标准答案里**且不易被同义词干扰的片段。
4. 重新跑 `build_qa_dataset.py` —— 任何一句对不上都会**报错**而不是静默产出坏数据。

---

## 8. 已知边界

- 暂不支持“应拒答”类用例：现有打分按 `answer_keywords` 命中率判对，无法给“正确拒答”计分；要加需在 dataset 里加 `unanswerable` 标记并改打分逻辑（见 `metrics.score_eval_result`）。
- 指标不含 grade/rewrite 中间调用的 token（见第 4 节口径）。
- 如果将来某篇文档被替换，请同步修订对应 seed 的 `gold_passages` 后重跑解析。
