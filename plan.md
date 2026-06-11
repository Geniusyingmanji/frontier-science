# Frontier-Science — 实施计划 (plan.md)

> 多学科、开放式**科学优化**基准：用冻结的确定性 Oracle 给出**连续可提升**的奖励，
> 评测 Agent 在固定预算内能发现的最优科学解，而非单次问答的对/错。
>
> 出发点（工程版）：[EinsiaLab/Frontier-Engineering](https://github.com/EinsiaLab/Frontier-Engineering)
> 本仓库：[Geniusyingmanji/frontier-science](https://github.com/Geniusyingmanji/frontier-science)

---

## 0. 一页纸概览（TL;DR）

- **范式**：继承 Frontier-Engineering 的 *Generative Optimization*——Agent 迭代修改一段**可运行的科学代码**，
  每次修改后由**冻结验证器**返回连续分数，评测“预算内能达到的最优解”。
- **科学版的三点扩展**：① 多保真度 Oracle（fast proxy / exact）；② 科学约束可行性计量（feasibility / constraint violations）；③ 多目标 Pareto 前沿追踪。
- **填补的空白**：现有科学基准要么是**离散问答**（SFE / SGI-Bench / SciEvalKit / ScienceAgentBench，metric 为 EM / 选择题 / LLM-judge），
  要么是**单一领域的优化基准**（PMO=小分子、Design-Bench、Olympus=化学实验、MADE / MatBench-Discovery=材料）。
  **没有一个统一框架横跨蛋白质 / 分子 / 晶体 / 量子 / 符号回归 / 神经科学 / 地球科学，并以冻结 Oracle 给连续可提升奖励。** 这正是 Frontier-Science 的定位。
- **复用**：直接复用 Frontier-Engineering 的 `frontier_eval/` driver + `UnifiedTask` 契约 + OpenEvolve / ABMCTS / ShinkaEvolve 算法框架。
  **新增一个科学任务 = 新建一个 benchmark 目录 + 一个 conf YAML，无需改动 harness 代码。**
- **初期模型**：本地 **Azure 免 key 的 GPT-5.5**（`codex-azure-mi -p azure_uami`，或 127.0.0.1:9876 的 OpenAI 兼容代理）用于打通 + smoke。

---

## 1. 背景与定位

### 1.1 Frontier-Engineering 的核心范式

| 要素 | 含义 |
|---|---|
| 可执行验证器 | 每个任务有**确定性、可本地运行**的 Oracle，无需 LLM judge |
| 连续奖励 | `combined_score` 为浮点数，能被有意义地持续提升 |
| 轨迹 > 单次 | 评测**预算内发现的最优解**，而非单次 pass/fail |

工程版已覆盖 24 个领域（GPU kernel、量子电路、结构优化、运筹学、机器人、电池、无线信道等），证明该范式可规模化。

### 1.2 为什么做 Frontier-Science（基于调研的定位）

科学发现的本质（提出假设→实验→反馈→改进）与生成式优化高度同构。但现有“科学 + LLM”评测存在清晰空白：

**A. 离散问答类（上海 AI Lab 生态为主）——“知道 / 理解 / 能跑”**

| 基准 | 任务性质 | Metric 类型 | 是否迭代优化 |
|---|---|---|---|
| **SFE**（Scientists' First Exam，上海 AI Lab / Intern-S1 评测集） | 多模态科学 VQA，830 题，5 大领域 | LLM-judge 0–10 + BERTScore（**离散单次**） | 否 |
| **Intern-S1 / Intern-S1-Pro** | 科学多模态基础模型，报告于 SFE、ChemBench、MatBench、MicroVQA 等 | 选择题 / 分类 / 抽取（accuracy / EM） | 否 |
| **SGI-Bench**（Scientific General Intelligence） | 多学科科学推理，10 学科 4 类任务 | EM + 步骤级正确率 + LLM-judge（**单次、有步骤分**） | 否 |
| **SciEvalKit** | 上海 AI Lab 统一科学评测工具箱（ProteinLMBench / ChemBench / MaScQA…） | 选择题正确率 / 单测通过率 / 代码语义相似度 | 否 |
| **ScienceAgentBench**（OSU） | 数据驱动科研流水线，102 真实任务 | 执行结果 pass/fail + rubric 部分分 | 否（单次产出） |

→ 这些回答的是“**是否知道 / 理解 / 能复现** AlphaFold3”。

**B. 单领域优化类——有连续指标，但领域孤岛**

| 基准 | 领域 | Metric | 局限 |
|---|---|---|---|
| **PMO / mol_opt** | 小分子 | top-10 property 的 oracle-call AUC（样本效率） | 仅小分子 |
| **Design-Bench** | 材料 / 机器人形态 | 黑盒标量属性 | 离线数据集，非活体 Oracle |
| **Olympus** | 化学实验规划 | 黑盒函数值（产率 / 性质） | 化学实验为主 |
| **MADE / MatBench-Discovery** | 晶体 / 材料 | 发现加速因子、稳定结构 F1 | 仅材料 |
| **ProteinGym** | 蛋白突变效应 | Spearman ρ / AUROC | **预测**任务，非设计优化 |

→ 这些**各自只覆盖一个学科**，且多为预测 / 离线优化。

**C. Frontier-Science（拟建）填补的交集**

> **统一的、多学科的、以冻结确定性 Oracle 给连续可提升奖励的开放式科学优化基准。**
> 回答的是“**能否发现比 AlphaFold3 / Strassen / 默认参数化更好的变体**”。

最关键的比喻：

- SGI-Bench / SFE 问“是否知道 / 理解 AlphaFold3”
- ScienceAgentBench 问“是否能跑 AlphaFold3”
- **Frontier-Science 问“能否发现比 AlphaFold3 更好的变体”**

> 注：部分上述基准的精确 arXiv 号在调研中可能有偏差，正式写作前需逐一核对引用；本表的**定位与 metric 类型分类**是稳定结论。

---

## 2. 任务类型的明确定义

### 2.1 正式定义

一个 Frontier-Science 任务是一个四元组 `(P₀, O, C, B)`：

- **`P₀` 初始程序**：一段弱但可运行的科学基线代码（如随机突变 + 爬山、NSGA-II、朴素有限元）。Agent 在预算内**只能编辑这一个文件**。
- **`O` 冻结 Oracle**：确定性、可本地运行的评估器（物理 / 生化模拟器、预训练代理模型、标注数据集）。Agent **看不到** Oracle 代码与答案。
- **`C` 科学约束**：物理 / 化学 / 生物合法性（自然语言描述 + 可选验证器）。
- **`B` 交互预算**：评估次数 / wall-clock / token 上限。

输出：Oracle 返回 `metrics.json`，主指标 `combined_score ∈ [0,1]`（经归一化），可被持续提升。

### 2.2 任务准入门槛（每个候选任务必须满足）

1. **连续可提升**：metric 不是 0/1，基线与 SoTA 之间有明显、可被算法缩小的差距。
2. **确定性 Oracle**：同输入同输出（或固定随机种子），**无需 LLM judge**，无网络依赖。
3. **可本地运行**：Easy < 5s / CPU；Medium < 2min；Hard 可用 GPU 但单次 < 2min（迭代期用 fast proxy）。
4. **黑盒安全**：存在“查表 / 读答案 / 改评估器”可作弊路径时，必须能通过目录隔离 + 只读约束封堵。
5. **科学意义**：改进对应真实科研价值（有可引用的 SoTA 参考分）。

### 2.3 相对工程版的三个科学创新

**① 多保真度 Oracle（Multi-Fidelity）**
```
fast_eval  (<30s, 代理模型, 如 ESMFold/pLDDT 通过率)  → 迭代期使用
exact_eval (1–5min, 精确计算, 如 sc-TM / DFT 代理)     → 最终提交评估
```
科学任务天然有 proxy/exact 之分；工程版无此机制。

**② 科学约束可行性计量**
```json
{ "valid": true, "combined_score": 0.847,
  "feasibility_rate": 1.0, "constraint_violations": 0 }
```

**③ 多目标 Pareto 前沿追踪**
```json
{ "pareto_hypervolume": 0.73, "objective_values": [0.85, 0.71] }
```

### 2.4 难度分级

- **Easy**：CPU < 5s，开源基线多，改进空间明显。
- **Medium**：< 2min，需领域知识设计策略，基线与 SoTA 有差距。
- **Hard**：需 GPU 或较长时间，搜索空间大，接近真实科研前沿。

---

## 3. 评测方式

### 3.1 metrics.json 统一 schema

Oracle（`evaluator.py`）对每个候选程序返回：

```json
{
  "combined_score": 0.0,        // 主指标, 越大越好, 失败时 -1e18
  "valid": 1.0,                 // 是否产出合法结果
  "feasibility_rate": 1.0,      // [科学扩展] 约束满足率
  "constraint_violations": 0,   // [科学扩展] 违反次数
  "pareto_hypervolume": null,   // [科学扩展] 多目标任务主指标
  "objective_values": [],       // [科学扩展] 各目标分
  "fidelity": "exact",          // [科学扩展] fast | exact
  "runtime_s": 0.0, "timeout": 0.0
}
```
> 复用 Frontier-Engineering `run_eval.py` 的契约：evaluator 返回 dict（或 `EvaluationResult`），`combined_score` 必填，非法用 `-1e18`。

### 3.2 跨任务归一化（与工程版一致）

```
normalized_score = (score - baseline_score) / (reference_sota_score - baseline_score)
```
- `baseline_score`：`initial_program` 的初始分
- `reference_sota_score`：文献引用的 SoTA 分
→ 不同任务分数可直接比较。

### 3.3 Leaderboard

- **主指标：Average Rank**（跨任务平均排名，越小越好；避免高分任务主导）。
- 附加榜：各领域均分、各难度档均分、**Feasibility 榜**（约束满足率）、多目标任务的 Hypervolume 榜。

### 3.4 防作弊（黑盒设计）

- Agent **只能看到** `agent_files.txt` 列出的文件（README / Task.md / 待编辑程序 / constraints.txt）与返回的 `metrics.json`。
- Agent **看不到** `evaluator.py` / `verification/` / `eval_command.txt` 实现与 Oracle 答案。
- 数据集类任务（ProteinGym DMS、NLB、DUD-E）放**隔离目录**，test split 严格分离，防查表。
- evaluator 在 `tempfile` 沙箱中拷贝候选文件运行；只读文件由 `readonly_files.txt` 声明。

---

## 4. 代码与文件结构（对齐 Frontier-Engineering main 分支）

### 4.1 关键复用结论（已核对工程版源码）

工程版用一个**通用任务驱动 `UnifiedTask`（`name: unified`）** 读取 benchmark 目录里的元数据文件来评测，
因此 **绝大多数新任务不需要写 harness 侧 Python**，只需提供 benchmark 目录的契约文件 + 一个 conf YAML。

### 4.2 仓库整体结构

```
frontier-science/
├── plan.md                      # 本文件
├── README.md / README_zh-CN.md
├── init.sh                      # 环境引导 (复用工程版)
├── frontier_eval/               # 评测 driver (从 Frontier-Engineering 移植/子模块)
│   ├── cli.py  env.py  registry.py
│   ├── registry_tasks.py        # 仅在需要自定义 Task 时登记; 通用任务走 UnifiedTask
│   ├── registry_algorithms.py   # openevolve / abmcts / shinkaevolve
│   ├── tasks/{base.py, unified/}# UnifiedTask: 读 benchmark 目录契约文件
│   ├── algorithms/base.py
│   └── conf/{config.yaml, task/*.yaml, algorithm/*.yaml, llm/*.yaml}
├── benchmarks/                  # ★ 新增科学任务都在这里
│   ├── ComputationalBiology/{RNAStructureDesign, ProteinFitnessDMS, ProteinInverseFolding}/
│   ├── DrugDiscovery/{MolecularPropertyOptimization, ADMETMultiObjective, VirtualScreeningRanking}/
│   ├── MaterialsScience/{CrystalBandgapOptimization, ElectrolyteFormulation}/
│   ├── Physics/{QuantumErrorDecoder, NumericalPDESolver, GravitationalWaveTemplate}/
│   ├── Neuroscience/{NeuralDecodingOptimization, SpikeSortingOptimization}/
│   ├── ScientificComputing/{SymbolicRegressionFeynman, MatrixAlgorithmDiscovery}/
│   └── EarthScience/{AtmosphericParamTuning}/
├── baseline_archive/            # 各任务最优历史代码与分数
├── scripts/{bootstrap, env, batch, analysis, ops}/
└── runs/                        # Hydra 运行输出
```

### 4.3 单任务目录契约（每个 `<Domain>/<Task>/`）

```
<Task>/
├── README.md  Task.md  (+ _zh-CN)   # Agent 可见的任务说明
├── frontier_eval/                    # ★ 评测契约 (Agent 不可见实现)
│   ├── initial_program.txt           # 指向初始程序文件 (如 scripts/init.py)
│   ├── candidate_destination.txt     # Agent 编辑的目标文件路径
│   ├── eval_command.txt              # {python} frontier_eval/run_eval.py --candidate {candidate} --metrics-out metrics.json
│   ├── eval_cwd.txt                  # 评测工作目录 (通常 ".")
│   ├── constraints.txt               # 科学/编辑约束 (自然语言)
│   ├── readonly_files.txt            # 只读: references/ verification/ frontier_eval/ ...
│   ├── agent_files.txt               # Agent 可见文件白名单
│   ├── copy_files.txt  artifact_files.txt
│   ├── fast_eval_command.txt         # ★[科学扩展] 多保真度: 迭代期代理评估
│   ├── metadata.yaml                 # ★[科学扩展] 见 4.5
│   ├── run_eval.py                   # 通用入口 (复用工程版): 加载本地 evaluator.evaluate()
│   └── evaluator.py                  # ★ 任务私有: 沙箱跑候选 → 调 Oracle → 产出 metrics
├── scripts/init.py                   # ★ 初始程序 (弱基线, Agent 复制并改写)
├── verification/                     # Oracle 真值/精确评估 (只读, 隔离)
├── runtime/                          # 任务运行辅助 (只读)
└── references/                       # 数据集/配置/参考资料 (按需隔离)
```

### 4.4 新增一个任务（无需改 harness）

1. 在 `benchmarks/<Domain>/<Task>/` 按 4.3 放好契约文件 + `scripts/init.py` 弱基线 + `verification/` 的 Oracle。
2. 新建 `frontier_eval/conf/task/<task>.yaml`：
   ```yaml
   name: unified
   benchmark: ComputationalBiology/RNAStructureDesign
   metadata_dir: frontier_eval
   runtime:
     env_name: frontier-sci-main      # 该任务的 uv/conda 环境
   ```
3. 运行：
   ```bash
   python -m frontier_eval task=rna_structure_design algorithm=openevolve llm=azure_gpt55
   ```
> 仅当任务需要特殊 Task 逻辑（自定义 `initial_program_path` / `evaluate_program`）时，才在 `frontier_eval/tasks/` 写子类并登记到 `registry_tasks.py`；科学任务优先用 `UnifiedTask`。

### 4.5 单任务 `metadata.yaml`（科学扩展）

```yaml
domain: ComputationalBiology
task: ProteinInverseFolding
difficulty: hard                  # easy | medium | hard
oracle_type: neural_folding_model # neural_folding_model | physical_sim | dataset_oracle | analytical
gpu_required: true
eval_time_seconds: 45
multifidelity:
  fast: "pLDDT>70 通过率 (ESMFold 单次)"
  exact: "sc-TM score (100 结构均值)"
objectives: [sc_TM_score]         # 多目标任务在此列多个
reference_baseline: ProteinMPNN
reference_score: 0.55
reference_sota: 0.65              # LigandMPNN
science_metric: sc_TM_score
citation: "Hsu et al., 2022; Dauparas et al., 2022"
```

### 4.6 算法 / LLM 集成

- **算法框架**：复用工程版 `conf/algorithm/{openevolve,abmcts,shinkaevolve}.yaml`，迭代修改 `scripts/init.py`。
- **本地 Azure 免 key GPT-5.5**：新增 `conf/llm/azure_gpt55.yaml`，初期用于打通与 smoke。两条调用路径：
  - Codex CLI：`codex-azure-mi -p azure_uami exec ...`（managed-identity 自动刷新 token，model=gpt-5.5）。
  - OpenAI 兼容代理：`http://127.0.0.1:9876`（OpenEvolve / ABMCTS 的 `openai_compatible` LLM 直接指向它）。

---

## 5. 任务清单与难度分布

| # | 任务 | 领域 | 难度 | GPU | Oracle | 单次评估 | 阶段 |
|---|---|---|---|---|---|---|---|
| 1 | RNAStructureDesign | 计算生物 | Easy | 否 | ViennaRNA | <5s | **P1 种子** |
| 2 | SymbolicRegressionFeynman | 科学计算 | Easy | 否 | 解析解比对 | <1s | **P1 种子** |
| 3 | MolecularPropertyOpt (single) | 药物发现 | Easy | 否 | RDKit/TDC | <1s | **P1 种子** |
| 4 | QuantumErrorDecoder | 物理 | Easy | 否 | Stim | <10s | **P1 种子** |
| 5 | ProteinFitnessDMS (GFP) | 计算生物 | Medium | 可选 | ProteinGym | <5s | P2 |
| 6 | MolecularPropertyOpt (multi) | 药物发现 | Medium | 否 | TDC MPO | <2s | P2 |
| 7 | ADMETMultiObjective | 药物发现 | Medium | 否 | ADMET-AI | <2s | P2 |
| 8 | CrystalBandgapOptimization | 材料 | Medium | 是 | MACE-MP-0 | <10s | P2 |
| 9 | NumericalPDESolver | 物理 | Medium | 否 | 解析解验证 | <30s | P2 |
| 10 | NeuralDecodingOptimization | 神经科学 | Medium | 否 | NLB 2021 | <30s | P2 |
| 11 | SpikeSortingOptimization | 神经科学 | Medium | 否 | SpikeInterface | <2min | P3 |
| 12 | ProteinFitnessDMS (TEM-1) | 计算生物 | Hard | 否 | ProteinGym | <10s | P3 |
| 13 | VirtualScreeningRanking | 药物发现 | Hard | 是 | DUD-E | <30s | P3 |
| 14 | ElectrolyteFormulationOpt | 材料 | Hard | 是 | MD 代理 | <1min | P3 |
| 15 | ProteinInverseFolding | 计算生物 | Hard | 是 | ESMFold | <2min | P3 |
| 16 | GravitationalWaveTemplate | 物理 | Hard | 否 | LALSuite | <2min | P3 |
| 17 | MatrixAlgorithmDiscovery | 科学计算 | Hard | 可选 | FLOP+数值 | <1min | P3 |
| 18 | AtmosphericParamTuning | 地球科学 | Hard | 是 | ClimSim | <2min | P3 |

各任务的科学背景、子任务、初始基线、评估公式、Oracle 细节见 `附录 B`（从原始构想迁移，逐任务细化为可执行规格）。

---

## 6. 分阶段路线图

### Phase 0 — 打通 harness（1 周）
- [ ] 将 Frontier-Engineering 的 `frontier_eval/` driver 移植/子模块化进本仓库；跑通工程版任一 `UnifiedTask` smoke。
- [ ] 配好本地 Azure GPT-5.5：`conf/llm/azure_gpt55.yaml` 指向 9876 代理；用 OpenEvolve 跑 1 次 1-iter smoke 验证回路。
- [ ] 产出 `conf/task` + `conf/llm` 模板，确认“新增任务=加目录+加 YAML”路径成立。

### Phase 1 — 4 个 Easy 种子任务（2–3 周）
- 选 #1 RNA / #2 SymbolicRegression / #3 MolOpt-single / #4 QuantumDecoder（全 CPU、依赖轻：ViennaRNA、TDC+RDKit、Stim）。
- 每个任务交付：弱基线 `scripts/init.py`、私有 `evaluator.py`、契约文件、`metadata.yaml`、baseline 分数。
- 用 GPT-5.5 + OpenEvolve 跑 25-eval，验证 `combined_score` 能从 baseline 有意义提升、归一化合理、防作弊有效。
- 里程碑：4 任务 × 3 算法（openevolve/abmcts/shinkaevolve）的最小可复现榜单。

### Phase 2 — 扩到 Medium（覆盖 6 领域，3–4 周）
- 加 #5–#10；落地**多保真度**（fast_eval 代理）与**Pareto**（#6 多目标）。
- 引入 feasibility 计量（#7 ADMET 约束）。

### Phase 3 — Hard + Leaderboard + 论文（4–6 周）
- 加 #11–#18（GPU / 数据集类）；完善隔离与 test-split 分离。
- 实现 Average Rank 榜单 + 各领域/难度/Feasibility 子榜。
- 跨模型对比（GPT-5.5 / Claude / 开源），形成 v1 问题集与技术报告。

---

## 7. 风险与开放问题

| 风险 | 缓解 |
|---|---|
| 重资产/依赖（ESMFold、MACE、ClimSim、LALSuite）安装难 | 按 Frontier-Eng 的 `scripts/bootstrap/fetch_task_assets.py` 思路做资产引导；Hard 任务延后到 P3 |
| 数据集类任务可“查表”作弊 | 隔离目录 + test split 分离 + 只读约束 + 沙箱评估 |
| 代理模型（fast）与精确 Oracle 排序不一致 | metadata 记录二者相关性；最终榜以 exact 为准，fast 仅迭代期 |
| GPT-5.5 token / 速率限制 | 9876 代理自愈旧会话；P0/P1 用小 iter 预算；批量走 `scripts/batch` |
| 引用准确性（部分基准 arXiv 号待核） | 论文阶段逐条核对 DBLP/arXiv |

---

## 附录 A — 完整任务样例骨架（`benchmarks/Physics/QuantumErrorDecoder/`）

```
QuantumErrorDecoder/
├── Task.md                         # 表面码 d=5/7 解码, 目标降低 logical_error_rate
├── scripts/init.py                 # 弱基线: 最近邻查找解码器 (logical_error≈2e-3)
├── frontier_eval/
│   ├── initial_program.txt         # -> scripts/init.py
│   ├── candidate_destination.txt   # scripts/init.py
│   ├── eval_command.txt            # {python} frontier_eval/run_eval.py --candidate {candidate} --metrics-out metrics.json
│   ├── eval_cwd.txt                # .
│   ├── constraints.txt             # 只改 scripts/init.py; 保持 decode() 签名与输出契约
│   ├── readonly_files.txt          # verification/ frontier_eval/ references/
│   ├── agent_files.txt             # README.md Task.md scripts/init.py frontier_eval/constraints.txt
│   ├── metadata.yaml               # difficulty: easy, oracle_type: physical_sim, science_metric: logical_error_rate
│   ├── run_eval.py                 # 复用工程版
│   └── evaluator.py                # 沙箱跑候选 decode() → Stim 生成 1e4 综合征 → 算 logical_error_rate
│                                   #   combined_score = 1 - logical_error_rate (在 p=0.1% 下)
└── verification/evaluator.py       # Stim 真值评估 (只读, Agent 不可见)
```

## 附录 B — 各任务详细规格（迁移自原始构想，逐条细化）

> 原 `Frontier-science.md` 第五节（任务详细设计）整体迁移到这里，作为每个任务实现时的规格来源；
> 实现时把“科学背景 / 任务设定 / 初始基线 / 评估指标 / Oracle 细节 / 难度区分”落成可执行的
> `scripts/init.py`（弱基线）+ `verification/evaluator.py`（Oracle）+ `metadata.yaml`（参考分）。
> 各任务的 `reference_baseline` / `reference_sota` 已知值见第 5 节与原始构想（如 ProteinInverseFolding：ProteinMPNN 0.55 → LigandMPNN 0.65；QuantumErrorDecoder：MWPM 5e-4 → NN 2e-4；NeuralDecoding：LFADS 0.82 → NDT2 0.88）。
```
