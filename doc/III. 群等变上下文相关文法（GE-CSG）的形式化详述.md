#### I. 本体论基础 (Ontological Foundation)

本框架摒弃了以孤立的“静态碱基”为研究对象的传统视角，确立了**“关系即本体”**的原则。基因组序列不被视为一串离散状态的堆砌，而是被定义为**群作用（对称性变换）在物质载体上留下的破缺轨迹**。长程依赖不再是跨越物理空间的概率关联，而是同一全局对称群在不同局部的等变映射。

#### II. 代数基元与复合群 (Algebraic Primitives & The Composite Group)

**1. 字母表与群作用**

定义基础基因组字母表 $\Sigma = \{A, T, G, C\}$。在此之上定义复合对称群 $G_{\text{genome}}$，它是刻画基因组拓扑与结构变换的代数核心：

$$G_{\text{genome}} = \mathbb{Z}_2^{\text{comp}} \times \mathbb{Z}_2^{\text{rev}} \times S_k \times \mathbb{Z}_n$$

- $\mathbb{Z}_2^{\text{comp}}$：互补群，作用为 $A \leftrightarrow T, G \leftrightarrow C$。
- $\mathbb{Z}_2^{\text{rev}}$：反向群，作用为序列的空间镜像反转 $w \mapsto w^R$。
- $S_k$：对称群，描述 $k$ 个功能模块（如外显子）的置换。
- $\mathbb{Z}_n$：循环群，描述串联重复序列的移位对称性。

**2. 轨道代表元 (Orbit Representative)**

文法的终结符不再是单一字符 $x \in \Sigma$，而是群 $G$ 作用下的轨道 $[\alpha]_G$：

$$[\alpha]_G = \{g \cdot \alpha \mid g \in G\}$$

在逻辑解析层，轨道内的一切元素被视为结构同构的“同一类”，从而将信息规模进行商空间（Quotient Space）压缩。



## III. 群等变上下文相关文法（GE-CSG）的形式化详述

### 1. 基础载体：字母表与符号空间

在 GE-CSG 中，我们区分“物理符号”与“代数单词”。

- **物理字母表 $\Sigma_{base}$**：

  这是基因组的原始观测集合，即 $\Sigma_{base} = \{A, T, G, C\}$。

- **等变单词（Equivariant Terminals） $\Sigma$**：

  文法中的终结符 $\Sigma$ 实际上是定义在 $\Sigma_{base}$ 上的**符号向量空间**。

  每一个终结符 $x \in \Sigma$ 都关联着一个**轨道标识**。例如，当我们写下符号 $a$ 时，它在代数上等同于轨道 $[\alpha]_G$ 的代表元。

- **非终结符（Non-terminals） $V$**：

  $V$ 是抽象结构的集合（如 `Gene`, `Promoter`, `Loop`）。在 GE-CSG 中，$V$ 中的每个元素 $A$ 都是群 $G$ 的一个**表示空间（Representation Space）**。这意味着非终结符不仅是一个标签，它还承载了某种对称性张量。

### 2. 五元组 $\mathcal{G} = (G, \Sigma, V, S, R)$ 的精确定义

#### (1) 复合对称群 $G$ (The Symmetry Kernel)

$G$ 是由生物学对称操作构成的群。

$$G = G_{loc} \rtimes G_{glob}$$

它包含了前文定义的复合群（互补 $\mathbb{Z}_2$、反向 $\mathbb{Z}_2$、置换 $S_k$ 等）。群 $G$ 通过一个映射 $\phi: G \times (\Sigma \cup V)^* \to (\Sigma \cup V)^*$ 作用于符号串上。这种作用具有**左分配律**：$g \cdot (s_1 s_2 \dots s_n) = (g \cdot s_1) (g \cdot s_2) \dots (g \cdot s_n)$。

#### (2) 起始符号 $S$ (The Initial High-Symmetry State)

$S \in V$。在生成逻辑中，$S$ 代表了系统的**全对称态**。所有的生成过程（即对称性破缺过程）都从 $S$ 开始演化。

#### (3) 规则集 $R$ (The Equivariant Rule Set)

这是 GE-CSG 的灵魂。每条规则 $r \in R$ 必须是一个**等变映射**。

------

### 3. 产生式规则的解构：$\xi A \eta \to \xi \gamma \eta$

我们必须详细定义这些符号串的构成：

- **核心项 $A$**：

  $A \in V$。它是规则的主体，代表正在发生的结构变换。

- **上下文（Context） $\xi$ 与 $\eta$**：

  $\xi, \eta \in (\Sigma \cup V)^*$。

  - **$\xi$ (xi)**：左侧上下文（上游环境）。
  - **$\eta$ (eta)**：右侧上下文（下游环境）。
  - **重要特性**：在 CSG 中，$\xi$ 和 $\eta$ 充当**守恒环境**。它们在规则作用前后不发生改变，但它们的存在是 $A \to \gamma$ 发生的充要条件。

- **生成项 $\gamma$ (gamma)**：

  $\gamma \in (\Sigma \cup V)^+$。这是 $A$ 在环境 $\{\xi, \eta\}$ 下演化出的新结构。

### 4. 等变约束条件 (Equivariance Constraint)

这是 GE-CSG 区别于传统文法的硬性数学约束。对于规则 $r: \xi A \eta \to \xi \gamma \eta$，我们定义群 $G$ 对规则的作用为：

$$g \cdot r = (g \cdot \xi)(g \cdot A)(g \cdot \eta) \to (g \cdot \xi)(g \cdot \gamma)(g \cdot \eta)$$

**规则集的闭包属性：**

$$\forall g \in G, \quad r \in R \implies g \cdot r \in R$$

### 5. 存储压缩：轨道代表元规则集 $R/G$

利用这种等变性，我们不需要显式存储所有变体。

- **轨道规则 $r_{orb}$**：我们只存储规则轨道的代表元。

- **Burnside 引理的应用**：

  有效规则数 $|R/G|$ 的计算公式为：

  $$|R/G| = \frac{1}{|G|} \sum_{g \in G} |\text{fix}(g)|$$

  其中 $\text{fix}(g)$ 是在群元素 $g$ 作用下保持不变的规则集合。在实际的基因组解析中，这意味着：**如果我们定义了互补对称，我们只需写下正链的语法规则，负链的逻辑通过群作用自动推导生成。**

------

### 6. 语义映射：从形式到生物物理

我们将上述符号统一映射到基因组动力学：

| **符号**           | **形式定义**   | **生物物理意义**                           |
| ------------------ | -------------- | ------------------------------------------ |
| **$\Sigma$**       | 等变终结符空间 | 带有轨道相位的碱基序列                     |
| **$V$**            | 群表示空间     | 染色质的高级拓扑结构（如 Loop, TAD）       |
| **$\xi, \eta$**    | 上下文边界     | 顺式作用元件（Cis-elements）及其物理微环境 |
| **$A \to \gamma$** | 结构演化映射   | 在特定环境下的序列重组或表观遗传状态跃迁   |
| **$g$**            | 群元素作用     | 空间反转、链置换、或序列重复的对称变换     |

------



#### IV. 对称性破缺与特异性评估 (Symmetry Breaking & Soft Equivariance)

为解决“基因表达取决于具体碱基（特异性）”的生物学现实，引入**软等变（Soft Equivariance）\**与\**破缺测度**：

解析不仅匹配抽象轨道，同时计算观测序列 $w[i:j]$ 与理想轨道代表元 $g \cdot \alpha$ 之间的代数距离（如编辑距离或热核概率权重）：

$$E_{\text{break}} = \text{dist}(w[i:j], g \cdot \alpha)$$

- **结构发现层**：依赖轨道进行大尺度长程关联的快速拓扑锚定（忽略微小 $E_{\text{break}}$）。
- **功能预测层**：在锚定区域内，将 $E_{\text{break}}$ 转化为生物学上的“进化选择压力”或“特异性结合能”，恢复具体碱基的物理意义。

#### V. 解析动力学与 Fourier 空间加速 (Parsing Dynamics & Fourier Acceleration)

这是突破经典 CSG 解析复杂度 $O(n^5)$ 的核心机制。

将传统的“轨道图解析（Orbit Chart Parsing）”映射至群代数 $\mathbb{C}[G]$ 中。通过有限群上的傅里叶变换（GFFT），将全局耦合的文法推导操作（时域卷积）转化为不可约表示（irreps）空间中的块对角化矩阵乘法：

$$\mathbb{C}[G] \cong \bigoplus_{\rho \in \hat{G}} M_{d_\rho}(\mathbb{C})$$

- 对于规则应用 $r_1 * r_2$，在频域中变为：$\widehat{r_1 * r_2}(\rho) = \hat{r}_1(\rho) \cdot \hat{r}_2(\rho)$。
- **计算重构**：复杂的长程依赖解析被完全解耦，分配到各个独立的不可约表示块 $\rho$ 中并行处理。
- **理论复杂度降维**：从指数级/高阶多项式，骤降至 $O\left(\sum_{\rho \in \hat{G}} d_\rho^3 \cdot P(n, \rho)\right)$。







