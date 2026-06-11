# Frontier\-science

## 一、背景与动机

### 1\.1 Frontier\-Eng 的核心范式

Frontier\-Eng 提出了 **Generative Optimization** 这一评测范式：



- Agent 在固定交互预算内**迭代修改可运行的工程代码**

- 每次修改后获得来自**冻结验证器**的反馈

- 评测重点是**预算内能发现的最优解**，而非单次通过率

- 使用**连续奖励信号**代替二元 pass/fail



这一范式在工程领域取得了成功（47 个任务，覆盖 GPU kernel、量子电路、结构优化、运筹学等）。



### 1\.2 为什么要做 Frontier\-Science？

科学发现的本质与工程优化高度一致，但现有 Science Benchmark 存在明显空白：

|现有 Benchmark|任务性质|评测方式|信号类型|
|---|---|---|---|
|SGI\-Bench（上海 AI Lab）|科学问答 / 推理|LLM judge \+ EM|离散（正确/错误）|
|SciEvalKit（上海 AI Lab）|多学科知识评测|代码执行 \+ 人工|离散/连续|
|ScienceAgentBench|数据驱动科研流水线|代码执行|离散|
|**Frontier\-Science（拟建）**|**迭代优化科学实验**|**冻结物理/生化 Oracle**|**连续优化轨迹**|

最关键的比喻：

- SGI\-Bench 问"是否知道 AlphaFold3"

- SciEvalKit 问"是否理解 AlphaFold3"

- ScienceAgentBench 问"是否能跑 AlphaFold3"

- **Frontier\-Science 问"能否发现比 AlphaFold3 更好的变体"**





## 二、设计原则



### 2\.1 继承自 Frontier\-Eng 的三大核心



1\. **可执行验证器**：每个任务有确定性、可本地运行的 Oracle，无需 LLM judge

2\. **连续奖励信号**：\`combined\_score\` 为 \[0, 1\] 浮点数，能有意义地提升

3\. **轨迹 \> 单次**：评测预算内发现的最优解，使用同样的 OpenEvolve / ABMCTS / ShinkaEvolve 算法框架



### 2\.2 Frontier\-Science 的三个创新



**① 多保真度评测（Multi\-Fidelity Oracle）**



```Plain Text
fast_eval（<30s，代理模型）→ 迭代期间使用
exact_eval（1-5min，精确计算）→ 最终提交评估
```



科学任务天然有 proxy/exact 的区分（如 ESMFold 快速折叠 vs\. AlphaFold3 精确预测），Frontier\-Eng 没有这个机制。



**② 科学约束违反计量**



```JSON
{
  "valid": true,
  "combined_score": 0.847,
  "feasibility_rate": 1.0,        // 约束满足率（物理合法性、化学合法性等）
  "constraint_violations": 0      // 违反约束次数
}
```



**③ Pareto 前沿追踪（多目标任务专用）**



```JSON
{
  "pareto_hypervolume": 0.73,      // 多目标优化任务的主指标
  "objective_values": [0.85, 0.71] // 各目标分数
}
```



### 2\.3 难度分级



全部任务按难度分为三档，确保生态多样性：



\- **Easy（入门）**：Oracle 在 CPU 上 \< 5 秒，有大量开源基线，改进空间明显

\- **Medium（标准）**：Oracle \< 2 分钟，需要领域知识设计策略，基线与 SoTA 有差距

\- **Hard（挑战）**：Oracle 需要 GPU 或较长时间，搜索空间大，接近真实科研前沿



---



## 三、技术架构



### 3\.1 基于 Frontier\-Eng 扩展



```Plain Text
仓库结构（在 Frontier-Eng 基础上新增 branch: frontier-science）

benchmarks/
  [原有 Frontier-Eng 任务...]
  # 新增 Frontier-Science 任务：
  ComputationalBiology/
    ├── ProteinFitnessDMS/
    ├── ProteinInverseFolding/
    └── RNAStructureDesign/
  DrugDiscovery/
    ├── MolecularPropertyOptimization/
    ├── ADMETMultiObjective/
    └── VirtualScreeningRanking/
  MaterialsScience/
    ├── CrystalBandgapOptimization/
    └── ElectrolyteFormulation/
  Physics/
    ├── QuantumErrorDecoder/
    ├── NumericalPDESolver/
    └── GravitationalWaveTemplate/
  Neuroscience/
    ├── NeuralDecodingOptimization/
    └── SpikeSortingOptimization/
  ScientificComputing/
    ├── SymbolicRegressionFeynman/
    └── MatrixAlgorithmDiscovery/
  EarthScience/
    └── AtmosphericParamTuning/
```



### 3\.2 统一 Metadata 格式（每个任务目录下）



```Plain Text
benchmarks/<Domain>/<Task>/frontier_eval/
├── initial_program.txt       # 科学基线（弱但可运行，如朴素遗传搜索、随机突变）
├── eval_command.txt          # 黑盒评估命令（Agent 只能看到分数，看不到评估代码）
├── eval_cwd.txt
├── constraints.txt           # 科学约束描述（自然语言 + 可选验证器路径）
├── fast_eval_command.txt     # 可选：代理评估命令（多保真度）
└── metadata.yaml             # 新增：难度、领域、Oracle 类型、参考文献
```



`metadata.yaml` 示例：



```YAML
domain: ComputationalBiology
task: ProteinInverseFolding
difficulty: hard
oracle_type: neural_folding_model   # neural_folding_model | physical_sim | dataset_oracle | analytical
gpu_required: true
eval_time_seconds: 45               # 单次评估时间（秒）
reference_baseline: ProteinMPNN     # 已知竞争方法
reference_score: 0.71               # 参考基线得分
science_metric: sc_TM_score
citation: "Hsu et al., 2022; Liu et al., 2023"
```



---





## 四、完整任务列表与难度分布



|\#|任务名|领域|难度|GPU|Oracle 类型|单次评估时间|
|---|---|---|---|---|---|---|
|1|RNAStructureDesign|计算生物学|Easy|否|ViennaRNA 折叠|\< 5s|
|2|SymbolicRegressionFeynman|科学计算|Easy|否|解析解精确比对|\< 1s|
|3|MolecularPropertyOptimization \(single\)|药物发现|Easy|否|RDKit / TDC|\< 1s|
|4|QuantumErrorDecoder|物理|Easy|否|Stim 仿真器|\< 10s|
|5|ProteinFitnessDMS \(GFP\)|计算生物学|Medium|可选|ProteinGym 数据集|\< 5s|
|6|MolecularPropertyOptimization \(multi\)|药物发现|Medium|否|TDC MPO|\< 2s|
|7|ADMETMultiObjective|药物发现|Medium|否|ADMET\-AI|\< 2s|
|8|CrystalBandgapOptimization|材料科学|Medium|是|MACE\-MP\-0|\< 10s|
|9|NumericalPDESolver|物理|Medium|否|解析解验证|\< 30s|
|10|NeuralDecodingOptimization|神经科学|Medium|否|NLB 2021 数据集|\< 30s|
|11|SpikeSortingOptimization|神经科学|Medium|否|SpikeInterface|\< 2min|
|12|ProteinFitnessDMS \(TEM\-1\)|计算生物学|Hard|否|ProteinGym 数据集|\< 10s|
|13|VirtualScreeningRanking|药物发现|Hard|是|DUD\-E 数据集|\< 30s|
|14|ElectrolyteFormulationOptimization|材料科学|Hard|是|MD 代理模型|\< 1min|
|15|ProteinInverseFolding|计算生物学|Hard|是|ESMFold|\< 2min|
|16|GravitationalWaveTemplate|物理|Hard|否|LALSuite|\< 2min|
|17|MatrixAlgorithmDiscovery|科学计算|Hard|可选|FLOP 计数 \+ 数值验证|\< 1min|
|18|AtmosphericParamTuning|地球科学|Hard|是|ClimSim|\< 2min|



## 五、任务详细设计



### 4\.1 计算生物学（ComputationalBiology）



---



#### Task 1 — `RNAStructureDesign`

**难度：Easy** \| Oracle: ViennaRNA（CPU \< 1s）



**科学背景**：RNA 反折叠问题（Inverse Folding）是给定目标二级结构，设计能折叠到该结构的核酸序列。在疫苗设计、RNA 治疗剂开发中具有重要应用。



**任务设定**：

- 输入：一组目标 RNA 二级结构（点括号表示法），来自 RNAInvBench 数据集（100 个结构，覆盖 50\-200 nt）

- Agent 需要实现一个序列设计算法（如 MCTS、遗传算法、基于 LLM 的采样）

- 评估器使用 ViennaRNA 折叠生成的序列，计算与目标的结构匹配度

**初始基线**：随机核苷酸采样 \+ 贪心局部搜索



**评估指标**：

```Plain Text
combined_score = 1 - (平均结构汉明距离 / 序列长度)
```



**Oracle 细节**：ViennaRNA Python 绑定（`RNA` 包），完全本地，无网络依赖，评估 100 个结构，每次 \< 5s



**难度区分**：Easy 结构（无假结）/ Medium 结构（有假结，需 pknotted 扩展）



---



#### Task 2 — `ProteinFitnessDMS`

**难度：Medium** \| Oracle: ProteinGym DMS 数据集（CPU \< 5s）



**科学背景**：蛋白质适应度景观导航是药物设计和蛋白质工程的核心问题。Deep Mutational Scanning（DMS）数据提供了真实测量的突变体适应度，可作为确定性 Oracle。



**任务设定**：

- 三个子任务（难度递增）：

    - `GFP`：绿色荧光蛋白荧光强度优化，237 AA，54,025 个已知突变体

    - `AAV`：腺相关病毒 DNA 包装效率优化，28 AA 功能片段

    - `TEM-1`：β\-内酰胺酶活性优化，263 AA，高组合复杂度

- Agent 实现一个序列优化算法（贝叶斯优化、进化策略、序列设计网络）

- Oracle 直接查表（DMS 数据集）或使用训练好的代理模型

**初始基线**：随机突变 \+ 单点突变爬山法



**评估指标**：

```Plain Text
combined_score = (top-128 候选的平均 fitness) / max_known_fitness
```



**Oracle 细节**：ProteinGym 数据集完全公开，代理模型（MLP on ESM\-2 embeddings）推理 \< 1ms/seq



---



#### Task 3 — `ProteinInverseFolding`

**难度：Hard** \| Oracle: ESMFold 本地推理（GPU，\~10s/seq）



**科学背景**：给定蛋白质骨架结构，设计能折叠到该结构的氨基酸序列（蛋白质逆折叠）。AlphaFold2/3 的出现使得正向折叠（序列 → 结构）高度可靠，而逆向设计仍然极具挑战性。



**任务设定**：

- 输入：CATH 4\.3 数据集中 100 个多样化蛋白结构（覆盖 α、β、α/β 等折叠类型）

- Agent 实现逆折叠算法（基于 GNN、LLM、MCTS 或混合方法）

- Oracle 使用 ESMFold 对设计序列进行结构预测，计算 sc\-TM score

**初始基线**：ProteinMPNN（已有 Python 实现）作为初始程序模板，sc\-TM ≈ 0\.55



**评估指标**：

```Plain Text
combined_score = mean_sc_TM_score（100 个结构上的平均自洽 TM 分）
```



**参考水平**：ProteinMPNN ≈ 0\.55，ESM\-IF1 ≈ 0\.60，LigandMPNN ≈ 0\.65



**Oracle 细节**：ESMFold 官方实现，A100 单卡推理，100 个结构约 15\-20 分钟完整评测，迭代期使用 fast proxy（pLDDT \> 70 通过率）



---



### 4\.2 药物发现（DrugDiscovery）



---



#### Task 4 — `MolecularPropertyOptimization`

**难度：Easy（单目标）/ Medium（多目标）** \| Oracle: RDKit \+ TDC（CPU \< 1s）



**科学背景**：先导化合物优化是药物开发的核心环节，需要在多个药物化学性质之间取得平衡（类药性、合成可及性、溶解度等）。



**子任务设定**：

- `single_qed`：最大化 QED（类药性，单目标，Easy）

- `penalized_logP`：最大化惩罚 logP（疏水性，单目标，Easy）

- `jnk3_gsk3b`：双靶点活性联合优化（JNK3 \+ GSK\-3β，Medium）

- `mpo_ranolazine`：Ranolazine MPO（5 属性联合，Medium）

**初始基线**：

- 单目标：基于 SMILES 的随机变异（REINVENT\-style 贪心搜索）

- 多目标：NSGA\-II on fingerprint space

**评估指标**：

```Plain Text
# 单目标
combined_score = top-10 分子的平均 oracle 分数

# 多目标
combined_score = Pareto 超体积（归一化参考点为全零）
```



**Oracle 细节**：TDC `oracle('qed')`, `oracle('jnk3')` 等，完全本地，RDKit 验证化学合法性



---



#### Task 5 — `ADMETMultiObjective`

**难度：Medium** \| Oracle: ADMET\-AI（CPU \< 2s）



**科学背景**：ADMET 性质（吸收、分布、代谢、排泄、毒性）是药物开发中的筛选关卡，90% 以上的临床失败与 ADMET 问题相关。



**任务设定**：

- 设计分子同时优化 6 个 ADMET 属性：

    - 口服吸收（Caco\-2 渗透率）

    - 血脑屏障穿透（BBB\-）

    - CYP 酶抑制（CYP3A4 安全）

    - 急性毒性（LD50 \> 1000 mg/kg）

    - hERG 心脏安全（非 hERG 阻断剂）

    - 水溶性（logS \> \-4）

- Oracle 使用 ADMET\-AI 开源模型（TDC 集成），完全本地推理

**评估指标**：

```Plain Text
combined_score = 满足所有 6 个约束的分子中，QED 的平均值
feasibility_rate = 满足全部约束的分子占比
```



---



#### Task 6 — `VirtualScreeningRanking`

**难度：Hard** \| Oracle: DUD\-E 数据集 \+ 分子对接代理（GPU \< 30s）



**科学背景**：虚拟筛选是从化合物库中高效识别活性化合物的关键步骤，改进排序算法可以大幅提升先导化合物发现效率。



**任务设定**：

- 靶蛋白：DUD\-E 数据集中的 10 个代表性靶点（EGFR、VEGFR2 等）

- Agent 需要实现一个对化合物库的排序算法（基于图神经网络、3D 特征、集成方法等）

- Oracle：已知 DUD\-E 活性/非活性标注

**评估指标**：

```Plain Text
combined_score = mean AUROC（10 个靶点）
```



---



### 4\.3 材料科学（MaterialsScience）



---



#### Task 7 — `CrystalBandgapOptimization`

**难度：Medium** \| Oracle: MACE\-MP\-0 神经网络势（GPU，\< 10s）



**科学背景**：带隙工程是半导体材料设计的核心，寻找特定带隙（如 1\.1\-1\.5 eV 光伏最优区间）且稳定的晶体结构是材料发现的关键任务。



**任务设定**：

- 目标：设计带隙在 \[1\.1, 1\.5\] eV 之间、形成能 \< 0 eV/atom（热力学稳定）的晶体结构

- Agent 实现晶体结构生成与搜索算法（进化算法、扩散模型采样、组成空间搜索）

- Oracle：MACE\-MP\-0 代理模型（替代 DFT），支持快速结构弛豫和性质预测

**初始基线**：简单组合搜索（A₁B₁X₃ 钙钛矿空间随机采样）



**评估指标**：

```Plain Text
combined_score = 找到的满足约束结构中，|Eg - 1.3|的倒数（归一化）× 稳定性满足率
```



---



#### Task 8 — `ElectrolyteFormulationOptimization`

**难度：Hard** \| Oracle: MD 代理模型（GPU，\< 1min）



**科学背景**：锂离子电池电解液配方优化（溶剂比例 \+ 添加剂选择）需要平衡离子电导率、电化学稳定窗口、SEI 成膜质量，是电池材料研究的核心瓶颈。



**任务设定**：

- 搜索空间：5 种溶剂（EC、DMC、EMC、PC、FEC）的比例组合 \+ 3 种添加剂（VC、LiDFOB、LiTFSI）的浓度

- Oracle：基于分子动力学模拟训练的代理神经网络（已预训练好）

- 目标：离子电导率 \> 10 mS/cm，氧化分解电压 \> 4\.5 V，还原分解电压 \< 0\.5 V

**评估指标**：

```Plain Text
combined_score = Pareto_hypervolume(离子电导率, 电化学稳定窗口宽度)
```



---



### 4\.4 物理科学（Physics）



---



#### Task 9 — `QuantumErrorDecoder`

**难度：Easy** \| Oracle: Stim 量子电路模拟器（CPU，\< 10s）



**科学背景**：量子纠错（QEC）解码器的性能直接决定容错量子计算机的资源开销。表面码是最有前景的 QEC 码之一，改进解码算法可以显著降低逻辑错误率，是通往实用量子计算的关键步骤（Google 2024 年首次演示低于阈值的表面码）。



**任务设定**：

- 基于 Stim（Google 开源量子稳定子电路仿真器）

- 任务：为 d=5 和 d=7 表面码实现解码算法（Union\-Find、MWPM、神经网络解码器）

- Oracle：Stim 生成 10,000 个随机错误综合征，评测解码算法的逻辑错误率

**初始基线**：简单最近邻查找解码器，逻辑错误率 ≈ 2×10⁻³



**评估指标**：

```Plain Text
combined_score = 1 - logical_error_rate（归一化，越低越好）
# 在物理错误率 p=0.1% 下评测
```



**参考水平**：MWPM ≈ 逻辑错误率 5×10⁻⁴，神经网络解码器 ≈ 2×10⁻⁴



---



#### Task 10 — `NumericalPDESolver`

**难度：Medium** \| Oracle: 解析解验证（CPU，\< 30s）



**科学背景**：偏微分方程（PDE）的数值求解是计算物理、计算流体力学的基础。改进自适应网格剖分策略、预条件子选择、多重网格加速等，是科学计算的长期研究课题。



**子任务**：

- `poisson_2d`：二维泊松方程，自适应网格，最小化误差与计算量的乘积

- `heat_equation`：含复杂边界条件的热方程，改进时间积分方案

- `navier_stokes_lid`：方腔驱动流（Re=1000），改进压力\-速度耦合算法

**初始基线**：FEniCS 标准有限元（均匀网格，无预条件）



**评估指标**：

```Plain Text
combined_score = 1 / (L2_relative_error × FLOPs / FLOPs_baseline)
# 相对于基线，越高越好
```



---



#### Task 11 — `GravitationalWaveTemplate`

**难度：Hard** \| Oracle: LALSuite 匹配滤波（CPU，\< 2min）



**科学背景**：引力波探测（LIGO/Virgo）依赖模板匹配滤波，模板库的覆盖效率（最小匹配度 vs 模板数量）直接决定探测灵敏度和计算代价。这是 LIGO 科学合作组织的长期工程问题。



**任务设定**：

- 目标：构建覆盖 BBH（双黑洞）参数空间的引力波波形模板库

- 约束：最小 match ≥ 0\.97，模板数量最少

- Oracle：LALSuite Python 绑定，计算模板覆盖 faithfulness

**评估指标**：

```Plain Text
combined_score = min_match_achieved / template_count_ratio
```



---



### 4\.5 神经科学（Neuroscience）



---



#### Task 12 — `NeuralDecodingOptimization`

**难度：Medium** \| Oracle: NLB 2021 数据集（CPU，\< 30s）



**科学背景**：脑机接口（BCI）的神经解码算法决定了残疾人用运动意图控制假肢的精度。Neural Latents Benchmark（NLB）是 NeurIPS 2021 提出的标准化神经解码评测框架，包含真实灵长类动物运动皮层记录。



**任务设定**：

- 数据集：`MC_Maze`（108 种到达运动，180 个神经元，\>2000 trials）

- Agent 实现神经群体动力学建模 \+ 运动轨迹解码算法

- 评测：手部速度轨迹的 R²（归一化 R\-squared），以及神经活动预测的 co\-smoothing

**初始基线**：线性解码器（Wiener Filter），R² ≈ 0\.65



**评估指标**：

```Plain Text
combined_score = 0.7 × velocity_R2 + 0.3 × co_smoothing_R2
```



**参考水平**：LFADS ≈ 0\.82，NDT2 ≈ 0\.88



---



#### Task 13 — `SpikeSortingOptimization`

**难度：Medium** \| Oracle: SpikeInterface \+ MEArec 合成数据（CPU，\< 2min）



**科学背景**：Spike Sorting（尖峰分类）是从多电极阵列记录中分离单个神经元信号的关键步骤，是所有神经科学电生理实验的必要预处理流程。现有算法在高密度 Neuropixels 探针上的性能仍有较大提升空间。



**任务设定**：

- 合成数据：MEArec 生成的 Neuropixels 风格数据（100 神经元，512 通道，噪声等级 10 μV）

- Agent 实现 spike sorting 算法（模板匹配、聚类、神经网络检测）

- Oracle：SpikeInterface 提供精确率/召回率、SI 指标

**初始基线**：经典模板匹配算法（Kilosort2 配置）



**评估指标**：

```Plain Text
combined_score = F1_score（spike train comparison，delta=1.3 ms）
```



---



### 4\.6 科学计算（ScientificComputing）



---



#### Task 14 — `SymbolicRegressionFeynman`

**难度：Easy** \| Oracle: 解析解精确比对（CPU \< 1s）



**科学背景**：符号回归是从实验数据中发现科学规律（方程）的自动化方法，是 AI 科学发现（AI Feynman）的核心技术之一。Feynman Symbolic Regression Database 包含 120 个来自费曼物理讲义的方程，是该领域的标准基准。



**任务设定**：

- 输入：来自费曼数据库中 30 个经典物理公式生成的噪声数据（分为 Easy 10 个 / Hard 20 个）

- Agent 实现符号回归算法（遗传编程、LLM 引导搜索、神经符号混合方法）

- Oracle：精确符号匹配（等价化简后匹配）\+ 解析解验证

**初始基线**：简单多项式拟合（R² 约 0\.7 on Easy，0\.3 on Hard）



**评估指标**：

```Plain Text
combined_score = (精确恢复率 × 1.0 + 近似恢复率（R²>0.99）× 0.5) / num_equations
```



**参考水平**：GPSR ≈ 60% exact（noiseless），AI Feynman 2\.0 ≈ 85% exact（noiseless）



---



#### Task 15 — `MatrixAlgorithmDiscovery`

**难度：Hard** \| Oracle: 数值正确性验证 \+ FLOP 计数器（CPU/GPU，\< 1min）



**科学背景**：矩阵运算是科学计算的基础原语。AlphaEvolve（2025）发现了比 Strassen（1969）更优的 4×4 复数矩阵乘法算法（48 vs 49 次标量乘法）。Frontier\-Science 将这一思路扩展为通用矩阵算法发现 benchmark。



**子任务**：

- `complex_matmul_4x4`：4×4 复数矩阵乘法，最小化标量乘法次数（复现 AlphaEvolve 结果）

- `symmetric_eigenvalue`：对称矩阵特征分解，最小化 FLOP 同时保证精度

- `sparse_matmul_structured`：结构稀疏矩阵乘法，最大化稀疏利用率

**初始基线**：标准朴素实现（$O(n^3)$）



**评估指标**：

```Plain Text
combined_score = (correctness_rate × baseline_flops) / actual_flops
# 同时满足：相对误差 < 1e-10（数值精度约束）
```



---



### 4\.7 地球科学（EarthScience）



---



#### Task 16 — `AtmosphericParamTuning`

**难度：Hard** \| Oracle: ClimSim 代理模型（GPU，\< 2min）



**科学背景**：全球气候模型（GCM）的参数化方案调优是气候预测的核心挑战之一。ClimSim（NeurIPS 2024）提供了基于 E3SM 气候模型的机器学习代理，使 GPU 上的快速评测成为可能。



**任务设定**：

- 调优对象：对流参数化的 8 个关键参数（如 CAPE 触发阈值、夹卷率等）

- Oracle：ClimSim 神经网络代理，对 6 个月大气状态预测与再分析数据（ERA5）对比

- 目标：最小化温度和降水的 RMSE，同时维持大气守恒律

**初始基线**：E3SM 默认参数配置



**评估指标**：

```Plain Text
combined_score = 1 - RMSE_normalized（温度 0.6 + 降水 0.4 加权）
```



---





## 六、评测框架与 Leaderboard 设计



### 6\.1 评分标准化



所有任务的 `combined_score` 归一化为 \[0, 1\]，定义如下：



```Plain Text
normalized_score = (score - baseline_score) / (reference_sota_score - baseline_score)
```



其中：

- `baseline_score`：`initial_program.txt` 的初始分数

- `reference_sota_score`：从已发表文献中引用的 SoTA 方法分数

这使得不同任务的分数可以直接比较（类似 Frontier\-Eng 的归一化思路）。



### 6\.2 Leaderboard 指标



参考 Frontier\-Eng 使用 **Average Rank（平均排名）** 作为主指标（而非平均分，避免某些任务得分主导整体排名）：



```Plain Text
leaderboard_score = Average Rank across all tasks（越小越好）
```



额外展示：

- 各领域平均分（ComputationalBiology / DrugDiscovery / Physics / \.\.\.）

- 各难度档平均分（Easy / Medium / Hard）

- 约束满足率排名（Feasibility Leaderboard）

### 6\.3 防作弊机制



延续 Frontier\-Eng 的黑盒设计：

- Agent **只能看到** `initial_program.txt`（初始代码）和评估返回的 `metrics.json`

- Agent **看不到** `eval_command.txt` 的实现细节和 Oracle 代码

- 数据集（如 DMS 数据、NLB 数据）存放在**隔离目录**，Agent 无法直接读取答案

- ProteinGym 类任务的 "test split" 严格分离，防止查表

---

