## Generativity from Symmetry And Symmetry Breaking: Group-Equivariant Formal Grammars for Genomic Sequence Parsing



> 不必与CFG相比。与经典方法相比即可。我们也可以用CFG。我们这里 的比较对象是”生成文法“与其他的方法。
>
> 自己的观点是 1. 在质量方面，生成文法存在潜力去捕获远程关联以及子结构间的关系。因此具有潜力去提升现有基因组解析方法的精确度。 2. 在速度方面，生成文法未必能够与其他方法相比，我们妥协地选择采取尽可能优化速度的方式，而非主张生成文法速度最优。
>
> 我们的主张：
>
> 1. 生成文法能提供精确度：（1）远程依赖，（2）上下文，（3）子结构的关系。
> 2. 虽然均为生成式模型，形式文法能够相比LLM等生成性模型，提供了形式的规则定义，这意味着可追踪性，可解释学，.
> 3. 在生成文法的范畴内，我们的群论模型可以提高速度。
>
> TODO LISTS before job interview:
>
> 1. Find papers（Milley CSG，CSG and CFG in genome analysis）
> 2. Design a proper demo grammar for subsequently analyzing human genome.
> 3. Create a simple analyzer to analyze a real fragment from a human genome.
> 4. Give the evaluation methods for each of our claims.
> 5. Preliminary Evaluation of SliceAI after Syntax Features Injection
>
> Reviewer Questions:
>
> 1. What the differences between your model and LLM.
> 2. Why you believe that your model potentially help improving analysis quality.
> 3. Is the group grammar just reduce small constant at complexity is really significant for application?
> 4. What you means "group theory based method may be more explainable".
> 5. What your method differ to another method, specifically, build the fragment that can appear at the leave of the parse tree with compression algorithms, then perform syntax analysis. It even can compress move than your $n/k$.
> 6. You mention that formal model is explainable and tractable, what you means. LLM also can perform explainability experiments, and also mathematically tractable.
> 7. Why you think "structure originally from symmetry and are generated through symmetric breaking", any support idea or previous work?
> 8. Can you actually parse a real human gene?
> 9. How do you obtain the grammar? Is it hand-designed or learned? 
> 10. What is your evaluation metric and dataset?
> 11. Why context-sensitive grammar specifically? Why not probabilistic CFG (PCFG)?
> 12. Is your $O(n^5) $ claim actually from Milley, not from your own work?









### Methods

### Research Questions

### Evaluation

#### C1: Context-Sensitive Rules Capture Biological Constraints

**What needs to be proved:** That the CSG rules (CS1–CS4) reject sequences that a CFG would accept but are biologically invalid.

**Evaluation protocol:**

Generate two sets of sequences:

- **Valid**: real annotated gene sequences (e.g. from Ensembl)
- **Invalid**: sequences where CS rules are violated — e.g. a donor site GT appearing without a preceding exon

Show that GE-CSG rejects the invalid set and accepts the valid set, while a CFG baseline accepts both. This is a **precision/recall** experiment on sequence classification.



#### C2: Long-range Dependency Potential

**What needs to be proved:**That $\mathcal{L}(\text{GE-CSG}) \not\subseteq \mathcal{L}(\text{CFG}) $ — i.e. GE-CSG can generate at least one language that no CFG can.

**Evaluation protocol:**

This is **purely theoretical** — construct a formal proof, not an experiment. Specifically:

Find one concrete GE-CSG grammar $\mathcal{G} $ such that $\mathcal{L}(\mathcal{G}) $ contains a cross-serial dependency pattern known to be outside CFL (e.g. $\{a^n b^n c^n\} $ or the copy language $\{ww\} $). The pumping lemma for CFLs then gives the proof.

No experiment needed — this is a mathematical theorem to be proved once.



#### C3: Generative Model Quality

**What needs to be proved:** That sequences sampled from PGE-CSG are statistically similar to real genomic sequences.

**Evaluation protocol:**

Three tests:

1. **Codon usage**: compare codon frequency distribution between sampled sequences and real CDS sequences (chi-squared test)
2. **Splice site statistics**: verify that GT-AG rule is satisfied in sampled sequences at the correct rate
3. **Discriminator test**: train a simple classifier (logistic regression on k-mer features) to distinguish PGE-CSG samples from (a) real sequences and (b) random sequences. A good generative model should be indistinguishable from real but clearly distinguishable from random.



#### C4: Constant Factor Speedup

**What needs to be proved:**That GE-CSG + Milley runs $k^5 \times |\hat{G}| $ times faster than Milley alone on the same input.

**Evaluation protocol:**

Direct benchmark:

- Implement both Milley CSG parser and GE-CSG + Milley parser
- Run both on sequences of length $n = 100, 200, \ldots, 5000 $ bp
- Measure wall-clock time
- Fit $T = c \cdot n^5 $ to both curves, extract constants $c_{\text{Milley}} $ and $c_{\text{GE-CSG}} $
- Verify $c_{\text{Milley}} / c_{\text{GE-CSG}} \approx k^5 \times |\hat{G}| = 2916 $

### Expect Outcomes

1. A GE-CSG parser for milley CSG grammar using for genome parsing

2. A GA-based algorithm to search good grammars.

   

### Expect Contributions

**C1: A grammar that understands biological context**

> Unlike previous computational models of DNA sequences, GE-CSG can enforce rules like "a splice donor site is only valid immediately after an exon" — reflecting real biological constraints rather than treating every position as independent.

**C2: Built to handle long-range interactions**

> Regulatory elements can influence gene expression across tens of thousands of base pairs. GE-CSG is formally capable of representing such long-range interactions — a class of biological structure that simpler models (context-free grammars, HMMs) cannot express.

**C3: A principled sequence generator**

> GE-CSG defines a probability distribution over DNA sequences grounded in molecular symmetry. One can sample from it to generate biologically plausible sequences, and verify the model against real genomic data statistically — without requiring a parser at all.

**C4: Faster analysis through molecular symmetry**

> By exploiting the natural symmetries of DNA — strand complementarity, reading frame equivalence — the computational cost of sequence analysis is reduced by a factor of nearly 3,000 compared to existing methods, extending the range of sequences that can be rigorously analysed from ~1,000 to ~10,000 base pairs: the scale of a typical human gene.

**C5 A Grammar Automatically Discovery Algorithm**

> AI? GA?
>
> 自己仅是认为可以在上述模型上引入我们的约束从而自动探索生成文法。或是从基因发现的大模型模型转换到一个近似的生成文法模型。









1. 在CSG范畴内加速 ()
2. 尝试LLM -> Syntax Tree (Neural Parser)

