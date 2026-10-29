## 目标

给定一个旧的语义码本 C\_old 和历史物品集合 D\_old，当新物品集合 D\_new 到来时，我们希望学习并更新得到新的码本 C\_new，使其满足以下两个目标：

1. **新语义分配**：新出现的语义应优先被分配到使用较少或新激活的码本区域，而不是覆盖已有的高质量历史语义区域。

2. **对齐可行性**：更新后的码本仍然能够与旧码本进行有效对齐。

3. 下游性能： 更新后下游推荐训练分数有改善

对于&nbsp;**对齐可行性**，NMI 分数是一个足够合适的评估指标 （见[SID增量训练实验](https://clouddocs.huawei.com/wapp/share/6b2db81a-8c43-40ed-a5f7-30f054bac63b)）。

剩下的问题是：

**如何衡量 新语义分配？ 最直接的衡量方式是下游任务的AUC**

***

## 衡量新语义分配

首先，我们需要在 C\_new 下定义&nbsp;**新颖样本**。这里可以复用&nbsp;**OOR 样本**&nbsp;的概念，详情见[语义偏移趋势分析](https://clouddocs.huawei.com/wapp/share/bd2ed5d7-44c3-46c1-ba27-ddb7ff4c1860)。

对于新语义分配，可以定义以下几个代理指标。

### 1. OOR 样本上的量化误差 / 重构误差

如果更新后的码本能够更好地表示新出现的语义，那么它在 OOR 样本上的量化误差或重构误差应该更低。

### 2. OOR 路径的凝聚度分数

对于一个码本路径 p，可以将其凝聚度定义为：被分配到该路径上的物品 embedding 之间的平均两两余弦相似度。

$$ \text{Cohesion}\left(p\right)=\frac{1}{{\left|p\right|}^{2}}\sum _{i,j\in p}\mathrm{c}\mathrm{o}\mathrm{s}\left({e}_{i},{e}_{j}\right) $$

其中：

* p 表示一个码本路径。

* \|p\| 表示被分配到路径 p 的样本数量。

* e\_i 和 e\_j 分别表示物品 i 和物品 j 的 embedding。

然后，整体码本凝聚度可以定义为所有至少包含两个物品的路径上的平均凝聚度：

$$ \text{CodebookCohesion}=\frac{1}{\left|{\mathcal{P}}_{\ge 2}\right|}\sum _{p\in {\mathcal{P}}_{\ge 2}}\text{Cohesion}\left(p\right) $$

其中，P\_≥2 表示至少包含两个物品的路径集合。

***

## 码本更新思路

## 思路 1：保持旧码本结构以提升对齐可行性

为了提升&nbsp;**对齐可行性**，可以在增量训练过程中冻结或软约束 C\_old 的结构。

可能的策略包括：

* 冻结整个 encoder。

* 冻结旧 centroids（码本锚点） embedding。

* 通过正则项对旧 centroids 进行软冻结。

对于软冻结，可以通过加入一个加权的 centroid drift 正则项来惩罚旧 centroids 的漂移：

$$ {R}_{\text{centroid}}=\sum _{k}{w}_{k}{∥{c}_{k}^{\text{new}}-{c}_{k}^{\text{old}}∥}_{2}^{2} $$

其中：

* w\_k 表示第 k 个 centroid 的重要性权重，可以根据该 centroid 在 D\_old 上的使用频率来估计。

这一思路与 Elastic Weight Consolidation \([arxiv.org](https://arxiv.org/pdf/1612.00796)\) 的思想类似：越重要的参数受到越强的保护，从而减少灾难性遗忘。

类似地，我们也可以加入另一个正则项，用于约束 encoder 权重的漂移....

在encoder上增加adapter:

A: 原有E\_old

B: E\_old + adapter

分类冗余样本和OOR样本, 在冗余样本上做软冻结。

***

## 思路 2：仅使用新颖物品进行训练

D\_new 中的大多数样本可能已经被 D\_old 的语义空间覆盖。这些冗余样本提供的新信息有限，但仍然可能导致 centroid 位置发生偏移，并改变 SID 分配结果。

因此，可以通过聚焦新颖样本或 OOR 样本，限制增量训练数据量。

这样做的目标是鼓励更新过程吸收新语义，而不是重新学习已经被覆盖的历史语义区域。

该思路可以有效与思路1结合。

***

## 思路 3：鼓励 SID 分配中的新颖性

新语义应该被引导到使用较少或新激活的路径上，而不是覆盖稳定的历史语义区域。这样可以在吸收新兴语义的同时，保留高质量的历史语义结构。

我们可以为 OOR 样本引入一个&nbsp;**路径稀有性奖励**。下面的损失项鼓励 OOR 样本避免被分配到旧数据中占用较高的路径上：

$$ {R}_{\text{rare}}=-\frac{1}{\left|{D}_{\text{OOR}}\right|}\sum _{x\in {D}_{\text{OOR}}}\mathrm{l}\mathrm{o}\mathrm{g}\left(1-{P}_{\text{old}}\left({\text{SID}}_{x}^{\text{new}}\right)\right) $$

其中：

* P\_old\(SID\_x^new\) 表示该新 SID 路径在旧数据 D\_old 中的概率或频率。

最小化该项会鼓励 OOR 样本被分配到旧语义空间中较少使用的路径上。

## 实验 \- 思路2

验证 idea 2：使用 Novel items 进行训练。

主要比较以下 codebooks：

* Codebook A：在 reference set 上训练得到。

* Codebook B：从 Codebook A 初始化，然后在完整 March 数据上训练 200 个 epoch。

* Codebook C：从 Codebook A 初始化，然后只在 March OOR 数据上训练 200 个 epoch；该 OOR 数据约占 March 数据的 5.7%。

* Codebook D：额外的 OOR\-only 变体，使用略小的学习率设置，即 3e\-5。

* Codebook E：额外的 OOR\-only 变体，使用极小的学习率设置，即 1e\-5。

* Codebook F:  sampling OOR\-only variant to have 30% ratio.

## 评估指标

* Reconstruction error

* Quantization error

* Cohesion score：分配到同一个 SID path 的 item embeddings 之间的平均 pairwise cosine similarity。

* NMI score：用于衡量更新后的 codebook 与 Codebook A 之间的 alignment feasibility。

# 2. 相对于 Codebook A 的 NMI 与MSE 结果

|**Comparison**|**Full\-SID Changed**|**Level 0 NMI**|**Level 1 NMI**|**Level 2 NMI**|**Avg. Reconstruction Error MSE**|**Avg. Quantization Error MSE**|
|---|---|---|---|---|---|---|
|A| | | | |0.00150815|0.00057598|
|**A vs B**|**93.31%**|**0.648**|**0.275**|**0.111**|**0.00045087**|**0.00029751**|
|A vs C&nbsp;|95.37%|0.619|0.240|0.095|0.00109685|0.00049543|
|A vs D|90.32%|0.680|0.314|0.132|0.00124202|0.00053244|
|A vs E|71.93%|0.790|0.494|0.272|0.00141866|0.00057539|
|A vs F|92.00%|0.676|0.305|0.123|0.0004371|0.00028747|

# 5. 结论

* 适当过采样OOR样本\(例如30%\) 同时改善了NMI分数与MSE分数，使得样本不仅更拟合新的语义信息，也更容易对齐旧码本。

**实验 \- 思路1：**

* A：基线模型，使用 2025 年 12 月、2026 年 1 月和 2026 年 2 月数据训练 200 个 epoch。

* B：增量训练，不冻结任何层。

* C：冻结 batch norm，并冻结 encoder\-decoder。

* D：冻结 codebook，解冻 batch norm、encoder 和 decoder。

* E：仅冻结 encoder。

* F：软正则化，对所有正则项使用统一权重w=1。

* G：软正则化，对所有正则项使用统一权重w=2。

* H：软正则化，对所有正则项使用统一权重w=100。

增量训练：始终从Codebook A 初始化，并在 2026 年 3 月数据上进行。

# 实验结果

|**Codebook**|**Full SID 匹配率**|**NMI L0**|**NMI L1**|**NMI L2**|**平均重建误差 MSE**|**平均量化误差 MSE**|
|---|---|---|---|---|---|---|
|A|&nbsp;|&nbsp;|&nbsp;|&nbsp;|0.00134275|0.00052444|
|**A vs B**|**6.69%**|**0.648**|**0.275**|**0.111**|**0.00040671**|**0.00027073**|
|A vs D|20.96%|0.825|0.456|0.198|0.00042621|0.00035733|
|A vs F|30.19%|0.810|0.514|0.275|0.00041662|0.00030679|
|A vs G|27.21%|0.815|0.496|0.249|0.00041988|0.00032202|

# 结论：

冻结参数实验显示， 软正则化实验\(Exp F\)在不牺牲MSE的情况下，极大程度上的改善了码本匹配率（6.69%==>30.19%）。

## ===

## 低LR实验 1：

|**Codebook**|**完整 SID 匹配率**|**NMI L0**|**NMI L1**|**NMI L2**|
|---|---|---|---|---|
|A1 vs B|84%|0.955|0.884|0.802|
|A1 vs C|77.27%|0.935|0.834|0.728|
|A1 vs D|85.3%|0.959|0.893|0.816|
|A1 vs E|77.16%|0.935|0.833|0.726|
|A1 vs F|84.26%|0.955|0.885|0.804|
|A1 vs G|84.4%|0.955|0.886|0.805|
|A1 vs H|85.27%|0.959|0.893|0.815|

所有增量训练均基于 A1  
训练好的码本，在 2026 年 3 月数据上训练 1 个 epoch。

* A1：baseline，在2025 年 12 月、2026 年 1 月和 2026 年 2 月数据上训练 200 个 epoch。

* B：增量训练 1 个  
epoch，不冻结参数。

* C：冻结  
batch norm，并冻结 encoder\-decoder。

* D：冻结  
codebook，解冻 batch norm，并解冻 encoder\-decoder。

* E：仅冻结  
encoder。

* F：软正则，正则项使用  
uniform 权重（w = 1）。

* G：软正则，正则项使用  
uniform 权重（w = 2）。

* H：软正则，正则项使用  
uniform 权重（w = 100）。

## 实验 2：增量训练 200 个 epoch

* Codebook  
A1：在 2025 年 12 月、2026 年 1 月和 2026 年 2 月数据上，从头训练码本 200 个 epoch。

* Codebook  
B：基于 A1 在 2026 年 3 月数据上进行增量训练 200 个 epoch，学习率降低 10 倍。

|**Codebook**|**完整 SID 匹配率**|**NMI L0**|**NMI L1**|**NMI L2**|
|---|---|---|---|---|
|A1 vs B|51.90%|0.860|0.662|0.474|
|A1 vs C|53.72%|0.867|0.675|0.492|
|A1 vs D|74.87%|0.936|0.829|0.699|
|A1 vs E|52.01%|0.857|0.658|0.475|
|A1 vs F|65.39%|0.896|0.749|0.601|
|A1 vs G| | | | |
|A1 vs H| | | | |

所有增量训练均基于 A1  
训练好的码本，在 2026 年 3 月数据上训练 200 个 epoch。

* A1：baseline，在2025 年 12 月、2026 年 1 月和 2026 年 2 月数据上训练 200 个 epoch。

* B：增量训练 200 个 epoch，不冻结参数。

* C：冻结batch norm，并冻结 encoder\-decoder。

* D：冻结codebook，解冻 batch norm，并解冻 encoder\-decoder。

* E：仅冻结encoder。

* F：软正则，正则项使用uniform 权重（w = 1）。

* G：软正则，正则项使用uniform 权重（w = 2）。

* H：软正则，正则项使用uniform 权重（w = 100）。
