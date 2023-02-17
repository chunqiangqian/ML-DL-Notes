# XGBoost

[资料来源](https://towardsdatascience.com/beginners-guide-to-xgboost-for-classification-problems-50f75aac5390)

XGBoost是一种集成机器学习，即，它组合了许多模型的结果，这些模型被称为`base learner`。

和随机森林一样，XGBoost使用决策树作为`base learner`。

单个决策树是低偏差、高方差模型。 他们非常擅长在任何类型的训练数据中找到关系，但很难很好地泛化看不见的数据。

然而，XGBoost使用的树与传统的决策树有点不同。 它们被称为 CART 树（分类和回归树），不是在每个“叶”节点中包含一个决策，它们包含一个实例是否属于一个组的实数值分数。 在树达到最大深度后，可以通过使用特定阈值将分数转换为类别来做出决策。

推荐的XGBoost资料，
1. [Youtube视频](https://www.youtube.com/playlist?list=PLblh5JKOoLULU0irPgs1SnKO6wqVjKUsQ)
2. [Gradient Boosting](https://www.youtube.com/playlist?list=PLblh5JKOoLUJjeXUvUE0maghNuY2_5fY6)


## XGBoost分类器的超参数

```python

>>> xgb_cl

XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.300000012, max_delta_step=0, max_depth=6,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=100, n_jobs=4, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
```

可以看到XGBoost的超参数有很多。

尽管使用默认超参数设置也可以达到的比较好的结果，但如果对以上超参数进行调节，可能在性能上得到很大的提升。

以下为最经常调节的超参数：
1. `learning_rate`：也被称为$eta$，用于控制模型使用额外`base learner`来拟合残差的速度
   * 典型值为：0.01-0.2
2. `gamma`，`reg_alpha`，`reg_lambda`：这3种参数定义了XGBoost所使用的3种类型的正则化，分别为，用与创建新的split的最小loss减少量，对leaf权重的L1正则化，对leaf权重的L2正则化
   *  `gamma`典型值：0-0.5，但与数据相关度很高
   *  `reg_alpha`和`reg_lambda`的典型值：0-1，但也取决于数据
3. `max_depth`：为树的决策节点所能达到的深度。必须为正整数
   * 典型值：1-10
4. `subsample`：被用于训练每棵树的数据占训练集的比例。如果这个值偏低，则会导致欠拟合，反之，如果过高，则会导致过拟合
   * 典型值：0.5-0.9
5. `colsample_bytree`：用于训练每棵树的特征比例。大的值意味着大部分特征被用于构建决策树
   * 典型值：0.5-0.9
