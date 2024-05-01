# fudan_nn_lecture

 **数据准备**

**`data_loader.py`** 文件包含 **`DataLoader`** 类，用于加载和预处理数据。其功能包括从文件中读取数据、打乱、分批处理以及执行数据归一化等操作，为模型的训练和测试提供准备好的数据。

 **模型架构**

模型由三个核心类组成：

1. **`nn_Layer`**：
    - 表示神经网络中的单个层。
    - 包含权重和偏置的初始化，以及前向传播方法。
2. **`ActivationFunctions`**：
    - 包含不同的激活函数和对应的梯度
3. **`Network`**：
    - 表示整个神经网络模型。
    - 包含多个层、激活函数以及前向和后向传播的方法。
    - 通过将多层组合在一起形成一个完整的网络结构。

**`model.py`** 文件中包含这些类的实现。

 **训练过程**

训练过程在 **`train.py`** 文件中实现，包含以下步骤：

1. **训练循环**：通过多次迭代（epoch）和批次（batch）进行训练。
2. **优化器**：使用反向传播和优化算法更新模型参数。
3. **超参数**：学习率和正则化等关键参数在训练开始前设置。

**模型评估**

模型评估在 **`test.py`** 文件中实现。该文件用于在测试数据上评估模型的性能。通常会使用准确率、精确度、召回率等指标来评估模型在未见过的数据上的表现。

**结果可视化**

**`plot.py`** 文件用于可视化模型结果，包括损失值、准确率以及模型参数的可视化等。通过图形化呈现，使得模型性能的变化和训练效果一目了然。

**模型持久化和加载**

**`model.pkl`** 文件包含训练好的 **`Network`** 实例，通过 Python 的 **`pickle`** 模块进行序列化保存。

**模型初始化和训练代码示例**
```python
activations = ActivationFunctions()
activation_fn_list = [activations.get_function('tanh'), activations.get_function('tanh'), activations.get_function('softmax')]
activation_fn_grad_list = [activations.get_gradient('tanh'), activations.get_gradient('tanh'), activations.get_gradient('softmax')]

model = Network([784,256,128,10], activation_fn_list, activation_fn_grad_list,0.01, 0.01)
```

**模型测试和可视化代码示例**
```python
with open("model.pkl", "rb") as file:
    best_model = pickle.load(file)

test(best_model, X_test, y_test)

visualize_network(best_model)
```

**调用脚本指令**
```bash
python train.py
python test.py 
```

**最佳模型参数地址：**
链接: https://pan.baidu.com/s/1BimrLrInVwItPKfT6DpEqw?pwd=dc2u 提取码: dc2u
