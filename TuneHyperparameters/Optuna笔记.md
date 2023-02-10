# Optuna笔记

摘自[博客](https://towardsdatascience.com/why-is-everyone-at-kaggle-obsessed-with-optuna-for-hyperparameter-tuning-7608fdca337c)

主要特点：

* 具有使用循环和条件定义类python的搜索空间的能力。
* 与平台无关的 API——您可以调整几乎任何 ML、DL 包/框架的估计器，包括 Sklearn、PyTorch、TensorFlow、Keras、XGBoost、LightGBM、CatBoost 等。
* 一大套优化算法，内置了early stopping和pruning features。
* 只需很少或无需更改代码即可轻松实现并行化。
* 内置支持对搜索结果进行可视化探索。

## Optuna基础

使用Optuna api来调节一个简单的函数$(x-1)^2+(y+3)^2$。我们知道，函数在 $x=1,y=-3$ 处达到最小值。

示例代码如下：

```python
import optuna  # pip install optuna

# 所要优化的目标函数
def objective(trial):
    # 优化参数1
    x = trial.suggest_float("x", -7, 7)
    # 优化参数2
    y = trial.suggest_float("y", -7, 7)
    return (x - 1) ** 2 + (y + 3) ** 2
```

其中，变量`trial`为optuna的特定的Trial对象，用于优化每个超参数。

其具有`suggest_float`方法用来设置超参数名称和定义优化值的范围，如

```python
x = trial.suggest_float("x", -7, 7)
```
