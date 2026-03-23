## Slide Outline for Interview

---

## Slide 1: Self Introduction

<! not do this>

## Slide 2: Research Experience 1 - Genetic Algorithm

<! not do this>

## Slide 3: Research Experience 2 - Microstate Sequence Analysis

<! not do this>

## Slide 4: Towards the Structural Analysis of Genome

## Slide 5: DNA Linguistics - Introduction



## Slide 6: DNA Linguistics - Generative Grammar



## Slide 7: DNA Linguistics - Situation & Conclusion

**标题**：DNA 语言学的现状

**已有工作**：
- 上下文无关文法（CFG）：RNA 二级结构
- 随机上下文无关文法（SCFG）：基因结构预测
- 但：**真正的生物学规则是上下文敏感的**

**上下文敏感的例子**：
```
剪接供体 (GT) 必须紧跟在 Exon 之后
剪接受体 (AG) 必须紧跟在 CDS 之前
起始密码子 (ATG) 必须紧跟在 UTR5 之后
```

**挑战**：上下文敏感文法（CSG）解析是指数复杂度的！

当前的DNA语言学研究对于统计学性质或信息论性质上的研究。解析复杂。



## Slide x: DNA Linguistics is Valuable but Parsing is Complex

**标题**：价值 vs. 复杂度

| 维度           | 状态                      |
| -------------- | ------------------------- |
| **生物学价值** | 极高（结构决定功能）      |
| **形式化难度** | 高（上下文敏感）          |
| **解析复杂度** | PSPACE-complete（指数）   |
| **现状**       | 大多用 CFG 近似，丢失精度 |

**核心矛盾**：我们需要 CSG 的表达力，但付不起指数代价

**本文目标**：用代数结构**压缩**状态空间，用神经网络**加速**解析

---

## Slide 8: Objectives

## Slide x: Group and Symmetry Algebra

## Slide x: Group and Symmetry Algebra for Compression

## Slide x: gLM

## Slide x: Proposal 1 - Milley Parser + Group Compression

**标题**：方案一：Milley 解析器 + 群压缩

**Milley 解析器**：
- 限制：fan-out ≤ 5
- 复杂度：O(n⁵)
- 局限：仍只能处理 ~1,000 bp

**群压缩（Level 1）**：
```
输入长度: n
群作用: k-mer 压缩 (k=3)
压缩后: m = n/k

复杂度: O(n⁵) → O((n/k)⁵) = O(n⁵/k⁵)
```
**k=3 时：243× 加速！**

**关键**：压缩是**代数的**，不是近似的——保留所有合法结构

---

## Slide 10: Proposal 1 - Group Compression (Continued)

**标题**：群压缩的数学基础

**群 G** = DNA 对称群
- 互补对称 (Z₂)
- 反向对称 (Z₂)
- 阅读框架循环 (Z₃)

**陪集空间 G/G_i** = 密码子轨道
- 大小: 64 → 10 (压缩！)
- 轨道: {ATG, TGA, GAT, ...} 共享同一功能

**Wedderburn 分解（Level 2）**：
```
ℂ[G] ≅ ⊕_{ρ} M_{d_ρ}(ℂ)
```
- |Ĝ| 个独立块
- 块间零通信并行
- 加速因子: |Ĝ| = 12（默认）

**总加速**：k⁵ × |Ĝ| = 243 × 12 = 2,916×

---

## Slide 11: Proposal 2 - LLM Neural Parser

**标题**：方案二：LLM 神经解析器

**核心思想**：让 LLM 直接预测语法树，GE-CSG 负责验证

```
序列 → LLM → 语法树（预测）
                ↓
           GE-CSG 验证（快）
                ↓
           通过 → 输出树
           失败 → 反馈修正
```

**优势**：
- 解析复杂度：O(n²)（LLM） + O(n)（验证）
- 可微端到端训练
- 保持符号保证

**关键**：GE-CSG 从**解析器**转变为**验证器**

---

## Slide 12: Mathematical Basis - Differentiable Parser

**标题**：可微解析器的数学基础

**目标**：构建态射 Φ: ℳ → Δ(𝒯)

```
ℳ (流形)     →     Δ(𝒯) (树空间上的概率分布)
LLM 嵌入          语法树分布
```

**实现**：
1. LLM 编码序列 → 嵌入 E ∈ ℝ^{n×d}
2. 投影层 → 终结符概率 P ∈ ℝ^{n×|Σ|}
3. 可微 CYK → 树概率分布

**可微性来源**：
- softmax 替代 argmax
- 概率乘法替代布尔与
- 可学习规则权重

---

## Slide 13: Mathematical Basis - Training Without Trees

**标题**：无需真实树的自训练

**核心洞察**：GE-CSG 本身就是**树生成器**！

**训练流程**：

| 阶段         | 数据            | 目标             |
| ------------ | --------------- | ---------------- |
| 1. 生成      | GE-CSG 生成     | 合法序列（正例） |
| 2. 对比学习  | 正例 + 突变负例 | 训练投影层       |
| 3. EM 自训练 | 真实无标注序列  | 提升泛化能力     |

**关键**：不需要真实树，只需要 GE-CSG 定义的"合法性"

---

## Slide 14: A Remaining Problem - Why Generative Grammar When We Have LLM?

**标题**：为什么还需要生成文法？

**本质回答**：实在界 vs. 符号界

| LLM (实在界) | GE-CSG (符号界) |
| ------------ | --------------- |
| 发现统计模式 | 编码形式规则    |
| 概率预测     | 确定性验证      |
| 黑箱         | 可解释          |
| 不可控       | 可控制          |
| 从数据学习   | 从规则推理      |

**两者不是替代，是互补**

**拉康三界**：
- 实在界：LLM 捕捉隐式结构
- 符号界：GE-CSG 显式编码规则
- 结合 = 神经符号学

---

## Slide 15: A Remaining Problem - The Neuro-Symbolic Answer

**标题**：神经符号学的答案

**我们需要符号界，因为**：

1. **可控性**：规则可以精确规定，保证行为
2. **可验证性**：可以证明序列符合规则
3. **创造性**：可以定义新的规则，生成未见的序列
4. **可解释性**：规则可读、可检查、可修改

**类比**：
- LLM = 直觉（快速但可能错）
- GE-CSG = 逻辑（慢但保证正确）

**最佳方案**：直觉 + 逻辑 = 神经符号系统

---

## Slide 16: Evaluation Plan

**标题**：评估计划

| 任务         | 数据     | 对比基线              | 评估指标           |
| ------------ | -------- | --------------------- | ------------------ |
| 剪接位点验证 | GENCODE  | DNABERT, SpliceAI     | 精确率、召回率、F1 |
| 基因结构预测 | Ensembl  | Augustus, GENSCAN     | 精确率、召回率、F1 |
| 变异致病性   | ClinVar  | CADD, REVEL           | AUC, 精确率        |
| 合成序列生成 | 人工设计 | 文法采样 vs. LLM 采样 | 合法性、多样性     |

**假设**：
- LLM 单独：高召回率，但假阳性多
- GE-CSG 单独：精度高，但覆盖有限
- 结合：高精度 + 高召回率 + 可验证

---

## Slide 17: Conclusion

**标题**：结论

**本文贡献**：

1. **代数框架**：群结构编码 DNA 固有对称性
2. **复杂度降低**：O(n⁵) → O(n²) 预测 + O(n) 验证
3. **神经符号融合**：LLM 发现 + GE-CSG 验证
4. **可解释性**：符号规则提供可读的解析树

**核心信息**：

> "GE-CSG 不是 LLM 的替代，而是它的**符号补充**——让 LLM 从'猜'变成'证'。"

**未来工作**：
- 扩展文法覆盖更多生物学现象
- 在真实临床数据上验证
- 探索更高效的神经符号架构

