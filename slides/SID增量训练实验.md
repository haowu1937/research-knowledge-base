### 02 \- 09 June

#### 1 \- 实验设置

1. 将当前目标月份设为&nbsp;`2026 年 3 月`

2. *Codebook A*：使用最近 3 个月的历史数据训练一个基础 codebook

    1. `2025 年 12 月`

    2. `2026 年 1 月`

    3. `2026 年 2 月`

3. *Codebook B*：使用基础 codebook 初始化模型，并继续使用&nbsp;`2026 年 3 月`&nbsp;的数据进行训练。这样可以得到一个增量更新后的 codebook。

4. 使用步骤 \[2\] 中训练得到的模型，在 3 月数据上生成 SID。

5. 使用步骤 \[3\] 中训练得到的模型，即基于 3 月数据增量训练后的模型，在 3 月数据上生成 SID。

    1. Codebook A 和 Codebook B 包含相同的 item，但不同模型会为这些 item 生成不同的 SID。

6. 最终得到两个 codebook：

    1. Item ID \-> SIDs，基于&nbsp;`2025 年 12 月 - 2026 年 2 月`&nbsp;训练，并在&nbsp;`2026 年 3 月`&nbsp;数据上推理得到

    2. Item ID \-> SIDs，基于&nbsp;`2026 年 3 月`&nbsp;增量训练，并在&nbsp;`2026 年 3 月`&nbsp;数据上推理得到

7. 使用&nbsp;`GR_unified`&nbsp;训练 codebook。

**实验目标**：比较以上方式得到的两个 codebook。

#### 2 \- 实验结果

##### 2.1 通过检查 item ID \-> SID 的映射进行比较

##### 2.1.1 “旧”SID 与“新”SID 的直接比较

* **99.92%**&nbsp;的 item ID 发生了 SID 变化，即长时间增量训练后的结果。

    * 是否可能是因为增量训练时间过长？

    * 当前设置与原始训练相同：200 epochs，学习率为 0.0005。

额外运行了一个短时间增量训练实验：

* 将增量训练设置为 1 epoch，学习率为 0.00005，即比原始学习率低 10 倍。

* 仍然有大约&nbsp;**18%**&nbsp;的 item 发生了 SID 变化。

##### 2.1.2 Normalized Mutual Information（NMI）和 Adjusted Rand Index（ARI）

这些指标回答的问题是：“对于同一组 item，两个聚类划分有多相似？” 这些指标关注的是哪些 item 被分到同一组，而不是具体使用了什么标签。

这与论文&nbsp;*“Mitigating Collaborative Semantic ID Staleness in Generative Retrieval”*&nbsp;中的思路类似。

**ARI（Adjusted Rand Index）**：基于 item pair 进行统计。对于每一对 item，判断 A 和 B 是否都认为它们“在一起”或“分开”。Rand Index 表示两个划分在 item pair 层面达成一致的比例。

* 1 = 聚类划分完全相同

* 0 = 与随机结果相当，即二者独立

* &lt; 0 = 比随机结果更差

**NMI（Normalized Mutual Information）**：信息论指标，用于衡量已知某个 item 在 A 中的 cluster 后，能在多大程度上推断它在 B 中的 cluster。

* 1 = A 的聚类结果可以完全决定 B 的聚类结果，即除了标签重命名外完全一致

* 0 = 二者相互独立

这两个指标都是对称的，数值范围大致在 \[0, 1\] 之间。代码如下：

`from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI`  
`# sA, sB: (N, depth) integer SID arrays for the SAME N items, in the same order`  
`for L in range(depth):`  
`    print(L, NMI(sA[:,L], sB[:,L]), ARI(sA[:,L], sB[:,L]))   # per level`

长时间增量（1个月增量 200 epoch）训练的输出结果：

```text
0 0.4885036639396838 0.2188848104529585
1 0.09971274447584165 0.0039366169262526046
2 0.03549218339647379 0.0006292697104508967
```

短时间增量（1个月 1个 epoch， 小lr）训练的输出结果：

```text
0 0.9585296025493422 0.9418386284200694
1 0.8773518007697204 0.8161913342133561
2 0.777025944971207 0.674026463249377
```

解读：两个 codebook 在较粗粒度结构上，即第 0 层 codebook 上，大约有一半程度的一致性；但在更细粒度层级上，即 Level > 0，二者的一致性越来越低，指标下降很快。

对于&nbsp;`short`&nbsp;增量训练，我们可以看到类似的模式，但指标数值明显更高。由于增量训练时间较短，基本可以理解为：item 仍然大体被聚在一起，但结合 2.1.1 的结果来看，它们的 SID label 发生了重新分配。

以上比较是直接基于 item ID \-> SID 的映射进行的。不过，我们还有&nbsp;`Encoder -> Centroids -> Decoder`&nbsp;模型结构。因此，接下来可以进一步评估 Codebook A，即基于&nbsp;`2025 年 12 月 - 2026 年 2 月`&nbsp;训练得到的 codebook，和 Codebook B，即基于&nbsp;`2026 年 3 月`&nbsp;增量训练后的 codebook，二者的 centroid vector 发生了多大变化。

##### 2.2 使用 codebook centroid vector 进行比较

* 实验中的 codebook shape 为 \[3, 512, 64\]

    * 3 表示 codebook 的层数

    * 512 表示每层有 512 个 centroid

    * 64 表示每个 centroid vector 的维度

这是一种“几何”角度的分析方法，用于回答：centroid 到底移动了多远？ 这里找到了一种叫作 Chamfer distance 的距离度量。它可以用于估计两个点云之间的相似度。

* 距离为 0 表示与自身进行比较

* 数值越大，表示差异越大

` import torch`  
`for L in range(3):`  
`    D = torch.cdist(cbA[L], cbB[L]) # (512,512)`  
`    print(L, D.min(1).values.mean(), D.min(0).values.mean())`

输出结果：

```text
0 0.1218
1 0.0744
2 0.0626
```

这一部分还需要进一步探索，才能更有把握地解释结果。但从第一眼看，这个结果似乎与上面的结果，即 2.1.2 中的结果存在一定矛盾。Chamfer distance 较小，说明 centroid 本身并没有移动太多。

#### 3 \- 可能的下一步

结论：

* 训练数据量/LR影响码本偏移程度

    * 通过OOR过滤，限制增量训练的数据量，减少增量训练对码本的冲击，从而降低后续码本对齐难度

* 冻结 encoder 和 decoder，只允许 codebook 在增量数据上重新训练

* 进一步探索 Chamfer distance
