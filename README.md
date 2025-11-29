# VALM：Vision-Anchored Latent Memory

本仓库实现了一套围绕 **视觉渲染 → OCR 锚点 → 语义熵压缩 → 跨模态检索** 的 Agent Memory 系统。目标是在 LoCoMo / InfiniteBench / AgentBench 等常见 benchmark 上，以更低的 Token 成本保持高召回与高可解释性，为 ACL 论文与 Demo 提供完整可复现的栈。

---

## 目录概览

- `src/valm/`：核心库，包含 VisualSketchBuffer、DeepSeek-OCR 封装、Semantic Entropy Estimator、HyperTokenizer、VALMStore、跨模态检索引擎以及 HyperToken 注入模块。
- `scripts/`：
  - `convert_locomo.py`：将官方 LoCoMo json 转换为本项目统一 JSONL。
  - `run_benchmark_suite.py`：批量跑 benchmark（LoCoMo / InfiniteBench / AgentBench）。
- `configs/benchmarks.yaml`：定义需要跑的 benchmark 数据集列表。
- `data/`：外部数据存放处，详细结构见下文。
- `models/`：放置本地的 DeepSeek-OCR、ViT 等模型权重（支持 Tsinghua/HF 镜像或手动下载）。
- `tests/`：基础单元测试。

---

## 数据与目录约定

```
data/
└── external/
    ├── locomo-github/          # 官方仓库镜像（保持原貌）
    └── locomo/
        ├── raw/
        │   ├── locomo10.json   # 官方提供的 10 个长对话
        │   └── ...             # 其他原始文件
        └── processed/
            └── locomo10.jsonl  # 运行 convert_locomo.py 后得到
```

获取 LoCoMo 的推荐流程：

```bash
# 1. 克隆/下载官方项目（可通过脚本 download_benchmarks.py 或手动）
# 2. 将 locomo10.json 复制到 data/external/locomo/raw/
cp data/external/locomo-github/data/locomo10.json data/external/locomo/raw/

# 3. 运行转换脚本
python3 scripts/convert_locomo.py \
  --input data/external/locomo/raw/locomo10.json \
  --output data/external/locomo/processed/locomo10.jsonl
```

其他 benchmark（InfiniteBench、AgentBench 等）也遵循相同结构：先放入 `raw/`，再转换到 `processed/`，最后更新 `configs/benchmarks.yaml`。

---

## 环境准备

1. Python >= 3.10，建议创建独立虚拟环境。
2. 安装依赖（根据 `requirement.md`，含 `transformers`, `torch`, `pillow`, `sentence-transformers`, `tqdm`, `numpy`, `pyyaml` 等）。
3. 准备模型权重：
   - `models/deepseek-ocr`：DeepSeek-OCR 的本地权重（推荐使用镜像下载）。
   - `models/vit-base`：VIST/ViT backbone。
4. GPU：建议至少 1×A100 / 80GB，用于 `--mode real`（VIST + DeepSeek-OCR）。

---

## 核心组件速览

| 组件 | 文件 | 说明 |
| --- | --- | --- |
| VisualSketchBuffer / VIST backend | `src/valm/visualizer.py`、`src/valm/vision.py`、`src/valm/vist_backend.py` | 文本→视觉 Token：支持纯文本划块、VIST 多分辨率 patch、统计 saliency/unique_ratio/布局信息。 |
| DeepSeekOCRClient / AnchorGenerator | `src/valm/ocr.py`、`src/valm/anchor.py` | 批量调用 DeepSeek-OCR（GPU）或 regex fallback，输出 Anchor `{text, entities, timestamp}`。内置 FlashAttention2 兼容 stub。 |
| SemanticEntropyEstimator | `src/valm/entropy.py` | 根据 token 文本特征估算语义熵，决定 bit allocation。 |
| HyperTokenizer | `src/valm/hyper_token.py` | 将 VisualToken + Anchor 编码为 HyperToken，保存 importance/entropy/bits/timestamp，并依据阈值丢弃或高保真 boost。 |
| VALMStore | `src/valm/store.py` | 统一保存 tokens/anchors/hyper tokens；维护 topic 索引、key-value、alert、event summary，并把结构化事件转换成伪 HyperToken。 |
| VALMRetrievalEngine | `src/valm/retrieval.py` | 文本重叠 + 残差多头匹配；按 importance 预筛；结合 topic/time bonus；事件 HyperToken 也可参与检索。 |
| HyperTokenInjector + ConstantStateController | `src/valm/injection.py`、`src/valm/state/controller.py` | 将 HyperToken 量化向量转成 pseudo K/V，并维护恒定记忆状态（推理阶段可用）。 |
| Benchmark Runner / Judges | `src/valm/bench/runner.py`、`src/valm/bench/judges.py` | ingest → retrieve → judge 流程；LoCoMoJudge 会优先输出结构化事件/alert/key-value。 |

---

## 快速体验：LoCoMo Benchmark

1. **转换数据**  
   参见前文 `convert_locomo.py`。

2. **单 benchmark 运行**  

```bash
# synthetic（快速 sanity check）
PYTHONPATH=src python3 -m valm.bench.runner \
  --data data/external/locomo/processed/locomo10.jsonl \
  --mode synthetic \
  --benchmark locomo \
  --max 10      # 可选：仅跑前 10 条

# real 模式（VIST + DeepSeek-OCR + GPU）
PYTHONPATH=src python3 -m valm.bench.runner \
  --data data/external/locomo/processed/locomo10.jsonl \
  --mode real \
  --benchmark locomo
# 输出示例：
#   Avg Compression Ratio: 76.7x
#   Retrieval Accuracy (Hit@1): 0.1%
#   Avg Match Score: ...
# 过程中会打印各 topic 的 raw/hyper/importance 统计
```

3. **批量运行**  

```bash
python3 scripts/run_benchmark_suite.py --config configs/benchmarks.yaml
# 结果会写入 artifacts/benchmarks/<timestamp>_summary.json
```

---

## 检索与结构化策略

- 结构化索引（LoCoMo）：
  - `event_summary` 解析成 `{text, date, timestamp}`；既能直接回答“日期/时间”问题，也会作为伪 HyperToken 参与检索。
  - `_kv_pattern` 抽取 `key_x_y is val`。
  - `_alert_pattern` 识别带时间戳的 ALERT 行。
- `BenchmarkJudge` 会根据问题类型：
  - 优先返回结构化答案（日期、key-value、alert）。
  - 若无结果，fallback 到跨模态检索。
- `VALMRetrievalEngine`：
  - 只对 importance ≥ 阈值的 HyperToken 进行 residual 打分。
  - 根据 Query 的时间提示过滤候选，允许 ±30 天窗口。

---

## 常见命令速查

| 命令 | 作用 |
| --- | --- |
| `python3 scripts/convert_locomo.py --input ... --output ...` | 生成 LoCoMo JSONL |
| `PYTHONPATH=src python3 -m valm.bench.runner --data ... --mode real --benchmark locomo` | 跑单个 benchmark（real 模式） |
| `python3 scripts/run_benchmark_suite.py --config configs/benchmarks.yaml` | 依配置跑多 benchmark |
| `python3 -m compileall src/valm` | 快速校验 Python 语法 |

---

## 当前状态 & TODO

- ✅ 视觉渲染 / DeepSeek-OCR / HyperTokenizer 主线打通，结构化事件被纳入检索。
- ✅ LoCoMo 全量 ingest + real 模式 benchmark 可运行，压缩比 ~76×。
- ⏳ Hit@1 仍偏低，需要继续优化结构化匹配、时间对齐与检索打分（已有统计日志用于排查）。
- ⏳ InfiniteBench / AgentBench 数据转换与评测尚在推进中。
- ⏳ 需要补充：Ablation、效率对比、GRPO 记忆策略、HyperToken 注入到推理模型的 Demo。

---

## 交接指南

1. 阅读 `README.md` + `requirement.md` 了解愿景、术语和规划。
2. 准备数据：先在 `data/external/<benchmark>/raw/` 放入官方文件，再运行对应转换脚本生成 `processed/*.jsonl`。
3. 快速验证：`PYTHONPATH=src python3 -m valm.bench.runner --data ... --mode synthetic --max 5`（无需 GPU），确认 pipeline 可通。
4. 实测 GPU：切换 `--mode real`，需要准备 DeepSeek-OCR/ViT 权重，默认读取 `models/`。
5. 功能开发：参考“核心组件速览”表定位文件；修改配置时留意 `src/valm/config.py`。
6. Benchmark：修改或新增 `configs/benchmarks.yaml` 项后，使用 `scripts/run_benchmark_suite.py` 批量跑，生成 `artifacts/benchmarks/*.json`。
7. 记录日志：Benchmark runner 会输出每个 topic 的 raw/hyper/importance/drop，便于分析；结构化命中 vs 检索命中可在 judge 中查看。

如需进一步同步，可在 `development_plan.md` 更新每周目标或在 Issue 中记录 Pending 项目，方便团队并行推进。

## 研发规划（摘要）

- **当前冲刺**：完善 VALM Memory OS（视觉化 + OCR + HyperToken Store），让 LoCoMo/InfiniteBench/AgentBench 有可复现的 baseline。
- **下一阶段**：接入 MEM1 Constant State Controller，完成跨模态检索 + HyperToken 注入闭环，并产出 ACL 论文初稿。
- **长期目标**：在 Benchmark 上形成可量化领先（召回、FLOPs、幻觉率），提交 ACL，提供 Demo / 复现实验脚本。

如需更详细的技术路线与里程碑，请查看 `requirement.md` 与 `development_plan.md`。

---

## FAQ

1. **为什么使用视觉 + OCR？**  
   视觉渲染能在高熵区域自动突出重点，结合 OCR 锚点可保持符号语义，同时为跨模态检索提供“截图式 token”。

2. **HyperToken 与传统向量有何区别？**  
   HyperToken 通过多级残差量化（ARVQ）记录更紧凑的 latent 记忆，并携带 entropy/importance/timestamp 等 metadata，检索和注入都可以利用这些结构化信息。

3. **缺失模型怎么办？**  
   可将 `models/deepseek-ocr` 与 `models/vit-base` 放到本地后再运行 `--mode real`。若只需要流程，先用 `--mode synthetic` 验证。

---

欢迎贡献：如果你发现 bug/性能瓶颈/新的 benchmark 线索，请直接提交 Issue 或 PR，或更新 `development_plan.md` 中的日程。希望和你一起，把 VALM 打造为下一代 Agent Memory 的标准范式！
