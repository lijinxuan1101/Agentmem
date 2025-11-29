## Agentmemory Requirement（VIST + DeepSeek-OCR 专项）

### 1. 项目使命

- **核心目标**：基于 VIST（Vision-centric Token Compression in LLM, arXiv:2502.00791）与 DeepSeek-OCR，提出并落地一种新的 Agent Memory 优化方案，让少量 Token 覆盖更多信息密度，同时在 LoCoMo / InfiniteBench / AgentBench 等同行常用 Benchmark 上实现可量化的超越，并最终以 ACL 论文形式发布。
- **终端产出**：
  - ValM（Vision-Anchored Latent Memory）或等价创新方案的完整技术栈；
  - Benchmark 成绩（>SOTA）与 Ablation；
  - ACL 论文 + 复现实验脚本 + Demo。

### 2. 核心思想：从 Text-centric 到 Vision-centric Memory

> 把“把文本塞进 KV Cache”的旧范式，升级为“把文本画成图，再按语义熵自适应压缩”的新范式。

| 层级 | 传统方案 | 我们的范式 | 说明 |
| --- | --- | --- | --- |
| 表示层 | Text Token 作为唯一记忆单位 | Visual Token + OCR Anchor + HyperToken | 记忆单位换底座，实现 Representation Level Innovation |
| 算法层 | 固定 bit 量化或 Attention 剪枝 | 语义熵驱动的可变比特率压缩（Semantic-Entropy VBR） | 高熵 Token 分配更多 RVQ 层，低熵直接极限压缩或丢弃 |
| 系统层 | 文本向量检索 | 文本粗排 → 视觉残差精排（跨模态检索） | Query Flow 与倒排结构完全重写 |
| 工程层 | RAG Pipeline + VectorDB | VALM Memory OS（Visual Sketch Buffer / Anchor Store / Latent Store / Timeline DB） | 各层以 HyperToken 为核心协同运行 |

该范式迁移带来三大原创特性：

1. **Representation-level**：记忆基本单元是视觉 Token，不是文本；DeepSeek-OCR 提供锚点，但目的是构造“可解释的潜在记忆”，并非调用 OCR 功能。
2. **Algorithm-level**：以语义熵为控制器，动态决定每个视觉 Token 的量化深度（ARVQ 层数），实现语义保真与成本之间的最优平衡。
3. **System-level**：检索路径采用“文本粗排 + 视觉残差精排 + HyperToken 注入”的三段式跨模态逻辑，支撑时序/结构问题的符号推理。

### 3. 创新方向：VALM（Vision-Anchored Latent Memory）

1. **Vision-centric Token Compression**
   - 将长程对话、日志或多页面文档渲染为多分辨率视觉画布，采用 VIST 的 Slow-Fast 编码结构生成高密度 Visual Tokens。
   - 利用 PVE + Frequency Mask 机制抑制高频词，突出低频语义，目标是实现 ≥2.5× 的 Token 缩减。
2. **DeepSeek-OCR Anchoring**
   - 对 Visual Tokens 执行逆向 OCR 重构，提取关键文本片段/实体/时间标签，形成 `(Visual Token, Text Anchor, Timestamp)` 三元组。
   - Anchors 支撑可解释检索，确保视觉压缩后的信息仍具备符号语义入口。
3. **Latent HyperToken**
   - 模仿 MemGen，将视觉 + Anchor 表示输入自适应残差矢量量化（ARVQ）得到固定长度 HyperToken。
   - HyperToken 作为新的记忆原语，可在注意力层直接注入，保持 O(1) 推理成本并与 MEM1 式恒定状态控制器结合。
4. **VALM Memory OS**
   - 写入：视觉渲染 → OCR 锚定 → HyperToken 化 → Metadata（主题、显著性、结构位置信息）。
   - 检索：语义查询命中 Anchor，按需加载 HyperToken；时序/结构问题调用 Timeline/Document Graph；必要时回溯原图或文本。

### 4. Benchmark 成功标准

| Benchmark | 目标指标 | VALM 贡献点 |
| --- | --- | --- |
| LoCoMo | Temporal Reasoning ≥ 85；Adversarial QA 幻觉率 < 10% | 将多轮对话渲染为“对话卡片时间轴”，HyperToken+Timestamp 支持精准时序检索 |
| InfiniteBench | Retrieve.KV / PassKey 召回 ≥ 95%；FLOPs 下降 ≥ 2× | VIST+OCR 压缩长上下文，HyperToken 注入避免注意力稀释 |
| AgentBench | OS/Web 子任务成功率 ≥ 80%；循环率 < 5% | 多窗口网页数据视觉化后统一存入 VALM，Goal State + HyperToken 提升长期规划一致性 |

### 5. 研发里程碑

1. **W1-W2**：完成数据与评测脚手架；定义 VALM Metadata Schema。
2. **W3-W5**：实现 Visual Sketch Buffer + VIST 编码；部署 DeepSeek-OCR Anchoring。
3. **W6-W8**：完成 Latent HyperTokenizer（ARVQ）与 VALM Store；建立可视化可解释面板。
4. **W9-W12**：与 MEM1 Constant State Controller 打通；在 LoCoMo/InfiniteBench 上做首轮对比。
5. **W13-W16**：GRPO 训练记忆策略；迭代优化 Benchmark 成绩；准备消融实验。
6. **W17-W20**：产出 ACL 论文初稿、复现实验、公开 Demo。

### 6. ACL 论文规划

- **题目草案**：Vision-Anchored Latent Memory for Long-Horizon Agents.
- **亮点**：将视觉压缩、OCR 锚点和潜在记忆注入统一成 VALM 原语；显著降低 Token 成本并提升多 Benchmark 表现。
- **稿件结构**：问题动机 → VALM 方法 → 数学细节（ARVQ、Anchor Schema、注入策略） → Benchmark 实验与 Ablation → 讨论与未来工作。

### 7. 风险与对策

| 风险 | 描述 | 缓解 |
| --- | --- | --- |
| 视觉渲染吞吐不足 | 长文本渲染导致延迟 | 批处理渲染 + GPU 并行 + 缓存 |
| OCR 锚点错误 | 锚点偏差影响检索 | 使用双路 OCR + 人工校验样本；训练 Anchor 置信度模型 |
| HyperToken 注入不稳定 | 可能破坏原模型推理 | 在 LoRA/Adapter 层插入桥接模块，分阶段蒸馏 |
| Benchmark 泛化 | 仅在特定任务有效 | 引入自建仿真数据，验证跨领域效果 |

### 8. 最终验收

- VALM 工程栈开源（或内部可复现）；
- LoCoMo / InfiniteBench / AgentBench 报告达到目标；
- ACL 论文完成投稿；
- Demo 展示“少量 Token 表示海量记忆”与“视觉+符号可解释检索”。

> 以上即本项目的最新 Requirement。所有研发、Benchmark 与论文撰写均需围绕 VALM 创新路线推进，确保“视觉压缩 + OCR 锚点 + 潜在 HyperToken”这一 Agent Memory 新原语真正落地，并在业界常用 Benchmark 上形成可量化领先优势。