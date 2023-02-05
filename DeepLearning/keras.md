# 结果可视化
## 绘制训练曲线
keras模型训练过程可以保存history变量，该变量是一个字典类型。  
我们可以将其转化为DataFrame，并进行绘图
```python
import pandas as pd
import matplotlib.pyplot as plt

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

plt.plot(hist['epoch'], hist['mse'], label='train mse') 
plt.plot(hist['epoch'], hist['val_mse'], label='val mse') 
plt.xlabel('Epochs') 
plt.ylabel('MSE') 
plt.title('Training and Validation MSE by Epoch') 
plt.legend() 
plt.show()
```

![trainingCurvePlot](pics/trainingCurvePlot.png)

### 对绘制曲线作光滑处理
对曲线进行高斯光滑处理，具体的处理函数为
```python
import numpy as np 
def smooth_curve(values, std=5): 
    width = std * 4 
    x = np.linspace(-width, width, 2 * width + 1) 
    kernel = np.exp(-(x / 5) ** 2) 

    values = np.array(values) 
    weights = np.ones_like(values) 

    smoothed_values = np.convolve(values, kernel, mode='same')
    smoothed_weights = np.convolve(weights, kernel, mode='same') 

    return smoothed_values / smoothed_weights
```