# 使用Optuna优化

[资料来源](https://www.analyticsvidhya.com/blog/2023/02/building-customer-churn-prediction-model-with-imbalance-dataset/)

## CatBoost

### 使用默认参数

```python
accuracy= []
recall =[]
roc_auc= []
precision = []
X= df1.drop('churn', axis=1)
y= df1['churn']
categorical_features_indices = np.where(X.dtypes != np.float)[0]
#Separate Training and Testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# With scale_pos_weight=5, minority class gets 5 times more impact and 5 times more correction than errors made on majority class.
catboost_5 = CatBoostClassifier(verbose=False,random_state=0,scale_pos_weight=5)
#Train the Model
catboost_5.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_test, y_test))
#Take Predictions
y_pred = catboost_5.predict(X_test)
#Calculate Metrics
accuracy.append(round(accuracy_score(y_test, y_pred),4))
recall.append(round(recall_score(y_test, y_pred),4))
roc_auc.append(round(roc_auc_score(y_test, y_pred),4))
precision.append(round(precision_score(y_test, y_pred),4))
model_names = ['Catboost_adjusted_weight_5']
result_df1 = pd.DataFrame({'Accuracy':accuracy,'Recall':recall, 'Roc_Auc':roc_auc, 'Precision':precision}, index=model_names)
result_df1
```

### 使用Optuna调节超参数

```python
def objective(trial):
    param = {
        "objective": "Logloss",
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "used_ram_limit": "3gb",
    }
    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)
    cat_cls = CatBoostClassifier(verbose=False,random_state=0,scale_pos_weight=1.2, **param)
    cat_cls.fit(X_train, y_train, eval_set=[(X_test, y_test)], cat_features=categorical_features_indices,verbose=0, early_stopping_rounds=100)
    preds = cat_cls.predict(X_test)
    pred_labels = np.rint(preds)
    accuracy = accuracy_score(y_test, pred_labels)
    return accuracy
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
```

```python
accuracy= []
recall =[]
roc_auc= []
precision = []
#since our dataset is not imbalanced, we do not have to use scale_pos_weight parameter to counter balance our results
catboost_5 = CatBoostClassifier(verbose=False,random_state=0,
                                 colsample_bylevel=0.09928058251743176,
                                 depth=9,
                                 boosting_type="Ordered",
                                 bootstrap_type="MVS")
catboost_5.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_test, y_test), early_stopping_rounds=100)
y_pred = catboost_5.predict(X_test)
accuracy.append(round(accuracy_score(y_test, y_pred),4))
recall.append(round(recall_score(y_test, y_pred),4))
roc_auc.append(round(roc_auc_score(y_test, y_pred),4))
precision.append(round(precision_score(y_test, y_pred),4))
model_names = ['Catboost_adjusted_weight_5_optuna']
result_df2 = pd.DataFrame({'Accuracy':accuracy,'Recall':recall, 'Roc_Auc':roc_auc, 'Precision':precision}, index=model_names)
result_df2
```

## Light GBM 

### 使用默认参数

```python
accuracy= []
recall =[]
roc_auc= []
precision = []
#Creating data for LightGBM
independent_features= df1.drop('churn', axis=1)
dependent_feature= df1['churn']
for col in independent_features.columns:
    col_type = independent_features[col].dtype
    if col_type == 'object' or col_type.name == 'category':
        independent_features[col] = independent_features[col].astype('category')
X_train, X_test, y_train, y_test = train_test_split(independent_features, dependent_feature, test_size=0.3, random_state=42)
#Creat LightGBM Classifier
lgbmc_5=LGBMClassifier(random_state=0,scale_pos_weight=5)
#Train the Model
lgbmc_5.fit(X_train, y_train,categorical_feature = 'auto',eval_set=(X_test, y_test),feature_name='auto', verbose=0)
#Make Predictions
y_pred = lgbmc_5.predict(X_test)
#Calculate Metrics
accuracy.append(round(accuracy_score(y_test, y_pred),4))
recall.append(round(recall_score(y_test, y_pred),4))
roc_auc.append(round(roc_auc_score(y_test, y_pred),4))
precision.append(round(precision_score(y_test, y_pred),4))
#Create DF of metrics
model_names = ['LightGBM_adjusted_weight_5']
result_df3 = pd.DataFrame({'Accuracy':accuracy,'Recall':recall, 'Roc_Auc':roc_auc, 'Precision':precision}, index=model_names)
result_df3
```

### 使用Optuna调节超参数

```python
def objective(trial):    
    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "dart",
        "num_leaves": trial.suggest_int("num_leaves", 2,2000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    lgbmc_adj=lgb.LGBMClassifier(random_state=0,scale_pos_weight=5,**param)
    lgbmc_adj.fit(X_train, y_train,categorical_feature = 'auto',eval_set=(X_test, y_test),feature_name='auto', verbose=0, early_stopping_rounds=100)
    preds = lgbmc_adj.predict(X_test)
    pred_labels = np.rint(preds)
    accuracy = accuracy_score(y_test, pred_labels)
    return accuracy
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
```

## 使用XGBoost

```python
accuracy= []
recall =[]
roc_auc= []
precision = []
#Since XGBoost does not handle categorical values itself, we use get_dummies to convert categorical variables into numeric variables.
df1= pd.get_dummies(df1)
X= df1.drop('churn', axis=1)
y= df1['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
xgbc_5 = XGBClassifier(random_state=0)
xgbc_5.fit(X_train, y_train)
y_pred = xgbc_5.predict(X_test)
accuracy.append(round(accuracy_score(y_test, y_pred),4))
recall.append(round(recall_score(y_test, y_pred),4))
roc_auc.append(round(roc_auc_score(y_test, y_pred),4))
precision.append(round(precision_score(y_test, y_pred),4))
model_names = ['XGBoost_adjusted_weight_5']
result_df5 = pd.DataFrame({'Accuracy':accuracy,'Recall':recall, 'Roc_Auc':roc_auc, 'Precision':precision}, index=model_names)
result_df5
```

## 比较不同算法结果

```python
result_final= pd.concat([result_df1,result_df2,result_df3,result_df4,result_df5,result_df6],axis=0)
result_final
```

## 总结

不平衡的数据集在数据科学中很常见，每当您进行数据采集时，数据的一侧会被淹没，而另一侧则只占少数。 在本文中，我们学习了如何在构建模型时处理不平衡数据集，而不是应用重采样、SMOTE、集成等个别方法。阅读本文后，让我们讨论一下我们应该记住的学习要点。

* 我们已经了解了数据质量对于获得机器学习模型的良好性能的重要性。
* 如果我们对 scale pos weight 使用极值，我们可能会过度拟合少数类，并且模型可能会做出更差的预测，因此 scale pos weight 的值应该是最优的。
* 虽然 Cat boost 和 Light GBM 可以处理分类特征，但 XGBoost 不能。 您必须在创建模型之前转换分类特征。
* scale pos weight值默认为 1。 多数类和少数类的权重相同。
* Optuna 是我们可以用来找到最佳参数以自动微调机器学习模型的框架。