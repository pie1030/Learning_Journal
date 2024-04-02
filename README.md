# Learning_Journal <br>
As this is my first attempt to learn those intriguing CS courses by myself (actually together with my friend who helps me a lot!!!), I would like to record all the footprints and encourage myself to head on courageously whatever hard the process might be. <br>


----
## Subject
1. [Git](#jump1)
3. [Python](#jump3) 
2. [Machine Learning](#jump2)

---
### <span id="jump1">1. Git</span>

---
### <span id="jump2">2. Machine Learning</span>

#### 2.1 Pytorch
*  [torch.normal()](https://pytorch.org/docs/stable/generated/torch.normal.html#torch.normal) <br>
>   torch.normal(*mean, std, \*, generator=None, out=None*) → Tensor <br>

Returns a tensor of random numbers drawn from **separate normal distributions** whose mean and standard deviation are given.

The mean is a tensor with the mean of each output element’s normal distribution

The std is a tensor with the standard deviation of each output element’s normal distribution

The shapes of mean and std don’t need to match, but the total number of elements in each tensor need to be the same.

即 该函数返回从单独的正态分布中提取的随机数的张量，该正态分布的均值是`means`，标准差是`std`。

*   `len()` VS `shape()` <br>
1. len(tensor) <br>
 1.1. len() 返回张量在**第一个维度**上的长度（大小）。
 1.2. 对于一维张量，返回其元素的个数。
 1.3. 对于多维张量，返回第一个维度的长度。<br>
2. tensor.shape <br>
 2.1. tensor.shape是一个**元组**，表示张量在**每个**维度上的大小。
 2.2  对于一维张量，返回单元素元组，包含张量中元素的个数。
 2.3. 对于多维张量，返回元组，其中每个元素表示相应维度上的大小。<br> <br>
 ```Python
 import torch

 # 一维张量

 x = torch.tensor([1,2,3])

 print(len(x))  # output:5
 print(x.shape) # output:torch.Size([5])

 # 二维张量

y = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

print(len(y)) # output:3
print(y.shape) # outpur:torch.Size([3,3])
```
<br><br>
总结，`len()` 函数用于获取 `torch.Tensor` 的元素数量，而 `shape` 属性用于获取其维度信息。区别: `len()` 仅仅返回一个数字，而 `shape` 返回一个包含所有维度大小的元组。
<br>
___
<br>

小tips：<br>
判断每个维度的大小:<br>
对于最内层的括号,它包含的元素个数就是*最后一维*的大小<br>
对于外层的括号,它包含的*内层括号的个数*就是该维度的大小<br>
例如 [[[1, 2], [3, 4]], [[5, 6], [7, 8]]] 有**三个维度**,第一维大小为`2`(包含2个二维张量),第二维大小为`2`(每个二维张量包含2个一维张量),第三维大小为`2`(每个一维张量包含2个元素) <br>

* [torch.matmul()](https://pytorch.org/docs/stable/generated/torch.matmul.html)
>   torch.matmul(tensor1, tensor2) <br>

tensor.matmul() 是用于计算两个张量之间的乘积， 即矩阵乘法。
* tensor.reshape() <br>
tensor.reshape() :重塑张量形状，允许在不改变张量中元素的顺序的情况下,将张量重新排列成一个新的形状。<br>
此外,你还可以使用 `-1` 作为 `shape` 中的一个值,PyTorch会自动计算出该维度的大小以保持元素总数不变。<br><br>
```Python
import torch

x = torch.tensor([1, 2, 3, 4, 5, 6])
y = x.reshape(-1, 1)
print(y)
print(y.shape)

# output:

tensor([[1],
        [2],
        [3],
        [4],
        [5],
        [6]])

torch.Size([6, 1]) 
```
<br><br>

* pyplot in matplotlib 绘制散点图 <br>

import pyplot as plt；

在 plt.scatter(x, y) 中,x 和 y 是两个参数,它们分别代表了散点图中每个点的 x 坐标和 y 坐标。

具体来说:

* x 是一个数组或列表,包含了每个点在 x 轴上的坐标值。长度必须与 y 相同。
* y 是一个数组或列表,包含了每个点在 y 轴上的坐标值。长度必须与 x 相同。

这两个参数的长度相同,构成了散点图中每个点的 (x, y) 坐标对。
<br> <br>

### 数据迭代器
```Python
# 使用python的generator，构造一个生成数据集的迭代器
def data_iter(batch_size, features, labels):
    num_examples = len(features) 
    // 计算总的样本数量

    indices = list(range(num_examples)) 

    random.shuffle(indices)
    // 生成一个包含样本索引的列表(indices)并随机打乱。

    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
    // 遍历这个索引列表,每次取出一个批量的特征和标签数据。

        yield features[batch_indices], labels[batch_indices]
        // 使用 yield 关键字返回这个批量的特征和标签数据。
```

生成器函数`data_iter`,用于从给定的特征和标签数据集中生成小批量数据迭代器。

1. `def data_iter(batch_size, features, labels)`:
   - `batch_size`是一个整数参数,表示每个小批量数据的大小。
   - `features`是一个张量或数组,包含了数据集的所有特征。
   - `labels`是一个张量或数组,包含了数据集的所有标签。


2. `for i in range(0, num_examples, batch_size)`:
   - 这是一个Python循环,用于遍历数据集。`range(0, num_examples, batch_size)`生成一个从0开始,以`batch_size`为步长,直到`num_examples`为止的整数序列。

3. `batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])`:
   - 这一行从`indices`列表中获取一个小批量的索引。`indices[i: min(i + batch_size, num_examples)]`表示从`i`开始,取`min(i + batch_size, num_examples)`个索引。`torch.tensor`将这些索引转换为PyTorch张量。

4. `yield features[batch_indices], labels[batch_indices]`:
   - 这是生成器函数的关键部分。`yield`关键字用于返回一个小批量的特征和标签数据。`features[batch_indices]`使用前面获取的`batch_indices`从`features`中提取相应的特征数据,`labels[batch_indices]`则从`labels`中提取相应的标签数据。<br>
___
   **`yield` 的含义**

`yield` 在英语中意为“产生”或“产出”。在 Python 中，`yield` 关键字用于生成器函数中，表示函数执行暂停，并返回一个值。
___

**理解 `yield` 关键字**


举个例子，假设你正在做一道汤。食谱的第一步可能是切洋葱。你可以使用 `yield` 关键字来表示这一步：

```python
def make_soup():
    yield "切洋葱"
```

当调用 `make_soup` 函数时，它不会立即执行整个食谱。相反，它会在第一步暂停，并返回字符串 "切洋葱"。

你可以使用一个循环来遍历生成器函数的步骤：

```python
for step in make_soup():
    print(step)
```

这将打印出：

```
切洋葱
```


**`yield` 的好处**


* **生成序列**：生成器函数可以生成一个无限的序列，而无需一次性创建整个序列。
* **分步处理数据**：生成器函数可以逐个处理大型数据集，从而节省内存开销。
* **异步编程**：生成器函数可以与异步 I/O 操作一起使用，以避免阻塞线程。

**总结**

`yield` 关键字允许你创建生成器函数，这些函数可以按需生成值，从而提高效率和灵活性。它就像一个食谱中的步骤，让你暂停执行并提供中间结果。

总的来说,这个函数的作用是从给定的特征和标签数据集中,按照指定的`batch_size`大小,生成随机打乱顺序的小批量数据迭代器。每次调用生成器函数的`next()`方法或在循环中使用它时,都会生成一个新的小批量数据。这种方式可以有效地处理大型数据集,避免一次性加载所有数据到内存中。<br><br>

### 定义损失函数

损失函数使用均方误差，即：$\frac{1}{N}\sum_{i=1}^N\frac{1}{2}(\hat{y}^{(i)}-y^{(i)})^2$。

``` Python
def square_loss(y_hat, y):
    # ---------计算均方损失---------    
    loss = 0
    N = len(y)  # 用len（）来求数量！
    for i in range(N) :
        loss += (y_hat - y) ** 2 * 0.5
    loss = loss / N 
    pass
    # ---------计算均方损失---------
``` 
<br> <br>


### 定义优化算法（SGD:Sochastic Gradient Discend） <br>
``` Python 
def sgd(params, lr):
    with torch.no_grad(): 
        for p in params:
            p -= lr * p.grad
            p.grad.zero_()
```

在`no_grad`上下文中,使用预先计算的梯度`p.grad`和学习率`lr`**更新模型参数`params`**。在每次更新后,梯度`p.grad`被清零,为下一次迭代做准备。<br><br>

1. `torch.no_grad()`:
    - `with torch.no_grad()`:当执行进入with语句块时,会调用上下文管理器对象的__enter__()方法。一旦离开with语句块,就会调用上下文管理器对象的__exit__()方法。
   - `torch.no_grad()`是一个上下文管理器,用于临时**阻止**PyTorch构建计算图和跟踪梯度。
   - 在该上下文管理器中执行的操作将不会跟踪梯度,从而节省内存空间和计算资源。
   - 这在更新模型参数时非常有用,因为我们不需要计算梯度,只需应用*预先计算*的梯度即可。

        - 前向传播(Forward Propagation)：输入数据通过神经网络模型进行前向计算,得到模型的输出预测值。

        - 计算损失(Loss Computation)：将模型的输出预测值与真实标签进行比较,计算损失函数的值。

        - 反向传播(Backward Propagation)：根据损失函数值,利用反向传播算法自动计算每个参数的梯度。在这一步,PyTorch会自动构建计算图并跟踪梯度。

        - 梯度更新(Gradient Update)：使用优化算法(如随机梯度下降SGD)根据计算出的梯度值,更新模型的参数。

2. `params`:
   - `params`是一个包含模型可训练参数(通常是权重和偏置)的Python列表或类似的可迭代对象。
   - 这些参数需要在训练过程中根据梯度进行更新。

3. `lr`:
   - `lr`是学习率(learning rate),是一个标量值,控制着每次参数更新的步长大小。


4. `p.grad`:
   - `p.grad`是PyTorch中每个参数张量`p`的梯度。
   - 在前向传播和反向传播之后,PyTorch会自动计算每个参数的梯度,并存储在`p.grad`中。
   - 这些梯度用于更新对应的参数。

5. `p.grad.zero_()`:
   - `p.grad.zero_()`用于将参数`p`的梯度张量`p.grad`清零。
   - 这是因为PyTorch会累积梯度,如果不清零,新计算的梯度会与之前的梯度累加。
   - 在每次参数更新后,需要清零梯度,为下一次迭代做准备。<br><br>


### 训练

- 随机初始化模型参数，重复以下步骤：
- 计算梯度 $\bm g$；
- 更新参数 $(\bm w,b) \leftarrow (\bm w, b) - \alpha \bm g$，其中 $\alpha$ 为学习率。<br>
```Python
# 定义超参数
lr = 0.003
num_epochs = 3
batch_size = 10

# 初始化模型参数
w = torch.normal(0, 0.01, (2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 开始迭代
iteration = 0
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels): 
        # 在每次循环迭代中: X获取一个批量的输入特征, y获取对应这个批量输入的标签
        loss = square_loss(linearModel(X, w, b), y) 
        # 使用这个批量的X和y计算损失函数loss
        loss.backward()
        # 通过反向传播计算梯度
        sgd([w,b], lr) 
        # 并基于梯度更新模型参数w和b
        iteration += 1
        print(f'epoch {epoch+1}, iteraion {iteration}, loss {loss}')

print(f'估计的w={w.detach().numpy()}')
print(f'估计的b={b.detach().numpy()}')
```

Q: 为什么偏置参数(b)一开始要设置为0的，但是权重参数(w)不用。

A: 先从线性模型的角度来看:

y = wx + b

其中:
- w是权重参数(weight)
- x是输入特征
- b是偏置参数(bias)
- y是模型的输出


**权重w** <br>
权重w决定了输入特征x对输出y的重要性大小。我们一开始不知道哪些特征对最终结果影响更大,所以需要从随机值出发,让模型去自己学习每个特征的权重大小。 

**偏置b**  
偏置b可以理解为模型的基本输出值或常量项。比如在线性回归中,当所有输入x=0时,输出y的值就是b。通常我们会将偏置初始化为0,因为:

1) 如果设置为非0值,会增加模型的复杂度和训练难度。**从0值出发,模型更容易先捕捉输入和输出的基本关系**。

2) 在激活函数中(如sigmoid),如果偏置初始化为较大正值或负值,会使大部分神经元节点输出接近0或1,导致梯度消失困难。从0初始化可以缓解这个问题。


**例子**

比如我们有一个模型来判断一个人是否会购买某个产品,特征包括年龄(age)、收入(income)等:

y = w1 * age + w2 * income + b  

- w1和w2是权重,模型需要自己学习年龄和收入这两个特征对购买行为的影响大小
- b是偏置,可以理解为基本购买意愿。我们可以先从b=0开始,让模型自己去捕捉年龄、收入等特征与实际购买意愿之间的关系

通过不断调整w1、w2和b的值,模型可以学习得到一个能很好预测购买行为的公式。


`loss.backward()`是在PyTorch中进行反向传播(backpropagation)的关键步骤。它用于计算模型参数的梯度,以便后续根据梯度更新参数值。具体来说:

1. `loss`是一个标量,表示当前模型的损失或误差。在训练神经网络时,我们希望**最小化**(minimize)这个损失值。

2. `backward()`会自动计算`loss`相对于模型所有需要学习的参数(如权重`w`和偏置`b`)的**梯度**。也就是求取`∂loss/∂w`和`∂loss/∂b`。

3. **PyTorch会沿着构建计算图的反向链路,基于链式法则,自动计算所有参数梯度**。这个过程被称为反向传播。

4. 计算出的梯度会分别存储在`w.grad`和`b.grad`中,为后续基于梯度更新参数做好准备。

5. `sgd([w, b], lr)`这一步通常会利用梯度下降法,用当前梯度对`w`和`b`进行更新,从而使损失值`loss`减小。

所以`loss.backward()`是让PyTorch自动计算模型参数梯度的关键步骤,为优化模型参数做好准备。通过不断迭代这个过程,模型就能不断学习,使损失值最小化。





