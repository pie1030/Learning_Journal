# 作业一 Regression
## 2.1 Pytorch
*  [torch.normal()](https://pytorch.org/docs/stable/generated/torch.normal.html#torch.normal) <br>
>   torch.normal(*mean, std, \*, generator=None, out=None*) → Tensor <br>

Returns a tensor of random numbers drawn from **separate normal distributions** whose mean and standard deviation are given.

The mean is a tensor with the mean of each output element’s normal distribution

The std is a tensor with the standard deviation of each output element’s normal distribution

The shapes of mean and std don’t need to match, but the total number of elements in each tensor need to be the same.

即 该函数返回从单独的正态分布中提取的随机数的张量，该正态分布的均值是`means`，标准差是`std`。
