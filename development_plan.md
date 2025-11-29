## VALM 开发蓝图（Vision-Anchored Latent Memory）

> 以“视觉化记忆单位 + 语义熵自适应压缩 + 跨模态残差检索”为核心，构建全新的 Agent Memory 架构，并在 LoCoMo / InfiniteBench / AgentBench 打榜夺冠，输出 ACL 论文。

---

## 1. North Star

- **范式迁移**：文本→视觉→潜在 HyperToken，彻底摆脱“KV 塞文本”的旧路。
- **量化策略**：语义熵驱动 ARVQ，可变比特率确保高熵事实保真、低熵背景极限压缩。
- **检索机制**：文本粗排 + 视觉残差精排 + HyperToken 注入，形成跨模态记忆闭环。
- **成功定义**：
  - LoCoMo Temporal ≥ 85；Adversarial QA 幻觉率 < 10%。
  - InfiniteBench Retrieve.KV / PassKey 召回 ≥ 95%，推理 FLOPs ↓ ≥ 2×。
  - AgentBench OS/Web 成功率 ≥ 80%，循环率 < 5%。
  - 论文：ACL 主会/Findings 投稿 + 完整复现实验。

---

## 2. 架构解剖

### 2.1 表示层：视觉记忆原语

1. **Visual Sketch Buffer**
   - 输入：长对话/日志/网页 DOM。
   - 输出：多分辨率画布（局部卡片 + 全局 Mindmap）。
   - 技术：VIST Slow-Fast 编码 + PVE/Frequency Mask；生成 Visual Tokens。
2. **DeepSeek-OCR Anchor Layer**
   - 双路 OCR（高精度 + 快速）生成锚点。
   - 每个 Visual Token 绑定 `{Text Span, Entities, Timestamp, Layout}`。
   - 产出 Anchor Graph（支持时间/结构索引）。

### 2.2 算法层：语义熵 VBR

1. **Semantic Entropy Estimator**
   - 多候选生成 + 意义聚类 → 估算每个 Visual Token 的语义熵 $H_{sem}$。
2. **Adaptive RVQ（ARVQ）**
   - Bit Allocation：`bit = f(H_{sem}, Saliency, Task Prior)`。
   - 高熵 Token：多级 RVQ（8~12 codebooks）；低熵 Token：1~2 级或硬丢弃。
3. **Latent HyperTokenizer**
   - 将视觉向量 + Anchor Embedding 拼接后送入 ARVQ。
   - 输出固定长度 HyperToken（例如 32×D）。
   - Metadata：显著性、来源画布、时间戳、Anchor ID。

### 2.3 系统层：VALM Memory OS

| 模块 | 功能 | 关键点 |
| --- | --- | --- |
| Memory Ingest | Kafka → Visual 化 → Anchor → HyperToken | 异步流水，写时注释 |
| Memory Store | Hot (Redis) / Warm (VectorDB) / Cold (对象存储) | HyperToken + Anchor Graph 分层落盘 |
| Retrieval Router | 文本 Query → 粗排 VS → 视觉残差精排 | Residual Code Matching，避免文本漂移 |
| HyperToken Injector | 将检索到的 HyperToken 注入 LLM | Adapter Hook + MEM1 Constant State Controller |
| Timeline & Graph Engine | 处理时序/结构问题 | Python Executor + TimeScale / Graph DB |
| Observability | Token 成本、熵分布、检索命中率 | Prometheus + Grafana 面板 |

### 2.4 工程层：工具链

- **语言**：Python + Rust（渲染/编码性能关键路径）。
- **框架**：Ray 处理批量渲染与 ARVQ 编码；Lightning/RLlib 训练策略。
- **存储**：Timescale（时间轴）、Neo4j/Postgres（Anchor Graph）、Milvus（粗排向量）、MinIO（视觉资产）。
- **监控**：W&B 记录实验，Grafana 监控吞吐与熵分配。

---

## 3. 交互流程

1. **写入**
   - Step1：原始文本/网页 → Visual Sketch Buffer（GPU 渲染）。
   - Step2：DeepSeek-OCR 读取画布，生成锚点 + 布局。
   - Step3：语义熵估计 → bit allocation → ARVQ 编码 → HyperToken。
   - Step4：存储：HyperToken（Latent Store）+ Anchor Graph（Structured）+ 原图（Cold Cache）。

2. **检索**
   - Query：LLM 生成的 Retrieval Prompt。
   - 粗排：文本向量 + Metadata 过滤（topic/time）。
   - 精排：High-Recall HyperToken Residual Matching（多路 codebook 相似度）。
   - 注入：选定 HyperToken 注入 Transformer（前几层 cross attention 或 MEM1 状态）。
   - 反射：MUSE Loop 将结果写回，更新 HyperToken 权重。

3. **策略**
   - GRPO 或 DPO 训练“保留/遗忘”策略。
   - Reward：任务准确率 - λ×Token 成本 + γ×熵覆盖度（覆盖关键高熵对象）。

---

## 4. 研发冲刺计划

| 周期 | 目标 | 关键任务 | 验收 |
| --- | --- | --- | --- |
| W1-W2 | 基建 & 数据 | Benchmark 工具链、数据协议、熵估计原型 | CLI 评测 + 熵探针报告 |
| W3-W4 | 视觉记忆原语 | Visual Sketch Buffer、VIST 编码流水线 | 300k token 文档→视觉 token Demo |
| W5-W6 | OCR Anchoring | 双路 OCR、Anchor Graph、时间戳标准化 | Anchor 质量≥98% 字符准确 |
| W7-W8 | ARVQ HyperTokenizer | 实现语义熵驱动 bit allocation、HyperToken Store | 60× 压缩比，BLEU loss < 5% |
| W9-W10 | Retrieval Router | 粗排/精排联调、Residual Matching | LoCoMo retrieval latency < 1.2× baseline |
| W11-W12 | HyperToken 注入 + MEM1 | Adapter Hook、恒定状态控制器 | 长对话推理 O(1) Token 成本 |
| W13-W15 | Benchmark 打榜一轮 | LoCoMo / InfiniteBench / AgentBench 初版结果 | 指标 ≥ 80% 目标 |
| W16-W18 | GRPO 策略 + Ablation | 策略训练、消融实验、可靠性测试 | Token 成本下降 30%，性能不降 |
| W19-W20 | 论文与 Demo | ACL 初稿、复现脚本、可视化 Demo | 提交内审，准备投稿 |

---

## 5. 技术要点详解

### 5.1 Semantic-Entropy Estimator

- 多样本生成：对输入片段进行多温度采样，构建语义簇。
- 语义聚类：Sentence-BERT + Agglomerative Clustering，获取簇概率。
- 熵值计算：$H_{sem} = -\sum_{c} p(c)\log p(c)$。
- 归一化：结合显著性、时间权重，得到最终 bit allocation score。

### 5.2 Residual Matching Engine

- 存储：每个 HyperToken 保存 RVQ 码字索引序列（codebook ids）。
- 检索：给定 Query，先在文本空间选 Top-K，再比较 codebook 序列的残差距离。
- 优势：视觉 latent 保留大范围语义；残差比较能捕捉结构/位置差异。

### 5.3 HyperToken Injection

- Hook 点：Transformer 第 2-4 层 cross attention + MEM1 常量状态向量。
- 机制：将 HyperToken 解码为 pseudo-KV，对齐维度后直接拼接到 KV cache。
- 稳定性：使用 Adapter 进行蒸馏，确保注入不会破坏主干推理。

---

## 6. Benchmark & 实验矩阵

1. **LoCoMo**
   - 任务：Temporal QA、Adversarial QA、Needle Retrieval。
   - 实验：有/无 VALM；仅视觉/仅文本/双路；HyperToken 注入层数对比。
2. **InfiniteBench**
   - 任务：Retrieve.KV、PassKey、Sorting。
   - 实验：长上下文长度扫描（50k→300k）；FLOPs 统计；注意力稀释对比。
3. **AgentBench**
   - 任务：OSWorld、WebShop、DB-Guru。
   - 实验：Goal State only vs. Goal State + HyperToken；多窗口同步读写。
4. **Ablation**
   - Remove entropy control / remove OCR anchor / remove residual matching。
   - 记录 Token 成本、召回率、幻觉率、推理延迟。

---

## 7. ACL 交付

- **写作分工**：Method（表示+算法+系统）/ Experiment / Theory / Appendix。
- **时间节点**：
  - W16：定稿实验数据；
  - W18：初稿 + 图表；
  - W19：外部评审；
  - W20：定稿 & 提交。
- **Artifact**：代码仓库、配置文件、数据脚本、Demo 视频。

---

## 8. 风险矩阵

| 风险 | 描述 | 缓解 |
| --- | --- | --- |
| 渲染性能瓶颈 | 视觉化成本高 | GPU 批渲染 + 共享缓存 + wasm renderer |
| OCR 锚点噪声 | 锚点错配导致检索噪声 | 置信度过滤 + 人工校验样本 + Active Learning |
| HyperToken 注入不稳 | 破坏推理概率分布 | Adapter Bridge + 蒸馏 + Safety Monitor |
| 熵估计偏差 | bit allocation 失衡 | 在线校准：监控重建误差，动态调整熵阈值 |
| Benchmark 迭代慢 | 数据量巨大 | 自动化评测流水线 + 子集抽样快检 |

---

## 9. 立即行动（Day 0 Checklist）

1. 创建 `valm/` mono-repo，包含 `visualizer`, `anchor`, `entropy`, `hyper_token`, `retriever`, `injection`, `bench` 模块。
2. 实现 Mini Demo：1 万 Token 对话 → 视觉画布 → HyperToken → 注入 → 回答 Needle QA。
3. 启动 Observability 面板，实时监控熵分布、bit allocation、Token 成本。
4. 拉通 ACL 写作 Outline，确保研发/论文同步。

> 本蓝图即刻生效。任何新增需求需评估是否符合“视觉-熵-残差”范式；否则视为范围蔓延。全员围绕 VALM 原语推进，争取在 20 周内完成打榜与投稿。


