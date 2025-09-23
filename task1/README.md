# Task1实验报告

## 摘要
在 Bag of Words 与 N-gram 特征的基础上，构建了一个线性分类模型，通过 mini-batch 梯度下降完成训练，实现了对不同影评的情感分类。实验过程中比较了不同特征表示（BoW、N-gram）、不同损失函数与学习率对模型性能的影响，并使用可视化工具对训练过程中的 loss 与 accuracy 进行监控，从而加深了对向量化表示与基础分类器的理解。

## 方法

### 数据集：
训练集new_train.tsv:  https://fudan-nlp.feishu.cn/wiki/VhCHwZiXQicIYrkbd2ocFS6Kn9c#share-ZmBOdjQ2Tox695xImpMcAA5kn9f

测试集new_test.tsv:  https://fudan-nlp.feishu.cn/wiki/VhCHwZiXQicIYrkbd2ocFS6Kn9c#share-InjndOoL7oDUagxiX88ccXTznhd

数据规模：共有 8528 条训练数据，3309 条测试数据

### 构建流程
1. 使用 panda 库对 tsv 文件进行读写操作。

2. 基于sklearn.model_selection 库中的 train_test_split 函数，将数据划为训练集、验证集、测试集三个部分。其中train集20%划分为验证集，80%划分到训练集，test集全部为测试集。
3. 手动定义词表，并将测试集的句子转换为向量，即实现 Bag of Word 或者 N-gram 。
4. 基于torch.utils.data中的TensorDataset方法把X_train_tensor 和标签 y_train_tensor 打包在一起，再调用DataLoader为数据集划分batch，按batch读取TensorDataset，为后续按batch对数据进行操作做铺垫。
5. 自行定义前向传播，损失函数，准确率函数（这里反向传播和梯度计算直接调用了现成的）
6. 规定训练轮数，开始循环训练，每个epoch循环里面对数据集按batch操作，提高计算速度；每完成一个epoch记录一次loss和acc
7. 绘制loss和acc随epoch变化的图像，可视化分析模型分类效果


## 实验
#### 数据：
| **特征方式** | **损失函数** | **学习率** | **batch_size** | **num_epochs**|**Valid Acc** | **Test Acc** |
| ------------ | ------------ | ---------- | ------------- | ------------ |---|---|
| BoW          | CE           | 0.1      | 32          | 15         |0.38|0.39|
|N-gram(n=2）|CE|0.1|32|15|0.43|0.45|
|N-gram(n=3)|CE|0.1|32|15|0.43|0.44|
|N-gram(n=2)|CE|0.01|32|15|0.38|0.39|
|N-gram(n=2)|CE|1   |32|15|0.36|0.36|
|N-gram(n=2)|MSE|0.1|32|15|0.35|0.36|
|N-gram(n=2)|hinge|0.1|32|15|0.42|0.43|

### 一、测试 Bag of Word 与 N-gram 的性能差异
前三此实验的loss、acc随epoch变化的图像：
第一次（n=1）：
![输入图片说明](/imgs/2025-09-23/Z2ZPuyyi6pXwB5S2.png)第二次（n=2）：
![输入图片说明](/imgs/2025-09-23/jLIC2XXEztfa2Zml.png)第三次（n=3）：
![输入图片说明](/imgs/2025-09-23/IzqL6k3DDomsMYIA.png)

对比发现：Bag of Word的性能相较于N-gram差一些；而n=2与n=3的N-gram方法性能相差无几；

### 二、测试不同的损失函数、学习率对最终分类性能的影响

#### 对于学习率：

lr=1时：
![输入图片说明](/imgs/2025-09-23/HPvp3mH69ycKHAfj.png)

lr=0.01时：
![输入图片说明](/imgs/2025-09-23/JHAdUa5u3Lr0p8FE.png)

lr=0.1时：
![输入图片说明](/imgs/2025-09-23/GDz0B4KKbmXSlk0A.png)
![](blob:https://stackedit.cn/742951fc-e56e-49c1-b0d2-20e9636178eb)
结论：lr较大的时候，收敛速度快，但是由于步长太大，容易跨过最优点，导致loss值起伏较大，且降不下来；而lr较小时，收敛速度慢，可能15个epochs过后loss值还在持续下降，学习速度慢；当lr适中时，loss值已然降到最低，基本收敛到当前最优，模型效果最好。

#### 对于不同的损失函数：
cross entropy:
![输入图片说明](/imgs/2025-09-23/pBQzh1Jr5pByepX1.png)![](blob:https://stackedit.cn/742951fc-e56e-49c1-b0d2-20e9636178eb)

hinge:
![输入图片说明](/imgs/2025-09-23/VZKjEw6I4Y55Ts9y.png)

mse:

结论：
交叉熵损失函数与多分类 Hinge Loss函数效果相近，适合用于多分类任务；而均方误差函数效果较差，用于回归任务，不适合做分类

## 总结与反思
### 一、关于可视化
最开始想用tensorboard实现可视化，但是由于kaggle平台不支持，反复尝试后无果，于是改用python的matplotlib绘制图像；

### 二、关于跑实验
目前实现的代码参数or方法设置都是固定的，如果要做对比实验，需要改动多处地方，非常麻烦；后续应该把必要的变化操作包装在一起，只用修改一处的值即可，便于实验

### 三、关于模型构建
借助chatgpt解决了许多代码实现细节上的问题，比如：在循环训练的时候，既然每轮epoch最终要算总的loss，acc值，为何定义loss和acc函数的时候不直接把batch里面每一行的结果累加？反倒要求均值呢？gpt给我的解释：<![endif]-->

**1. 计算梯度时需要“平均”**

在深度学习里，我们通常希望 **不同 batch 的梯度规模大致一致**，否则 batch 大小一变，梯度就会成倍放大或缩小，训练会不稳定。

-   如果 loss 是 batch 内样本的 **总和**，那 batch 大小为 64 时的梯度大约会是 batch 大小为 32 时的 **2 倍**。
-   如果 loss 是 **平均值**，无论 batch 大小多少，梯度的数值范围都比较稳定。

👉 所以主流框架（PyTorch、TensorFlow）里 cross_entropy 默认就是返回 batch 的平均损失。

----------

**2. 平均值更直观**

看训练日志时，loss 和 acc 通常都是 **单个样本的平均水平**，更容易比较。  
比如：

-   loss=0.3 → “每个样本平均损失 0.3”
-   acc=0.85 → “预测 85% 的样本正确”

如果直接打印总和（比如 loss=2000），这个数字会随着 batch 大小和数据集规模变化，没什么参考价值。

----------

**3. 统计 epoch 指标时才用“加权和”**

虽然 loss 和 acc 在函数里是 **平均值**，但在一个 epoch 里：

-   各 batch 的大小可能不一样（最后一个 batch 经常不足满 size）。
-   如果直接把“batch 平均值”做算术平均，会偏差。

所以我们在外层累加时，要把它们乘回样本数，算“加权和”，最后再除以总样本数。

类似的还有很多

### 四、关于acc值上不去
实验发现无论怎么设置参数、方法，尽管loss值可以降到很低，模型预测的准确率一直不是很高，基本卡在30~40%；个人推测，一方面可能是参数没有调好，另一方面可能是模型过于简单（softmax 回归模型），只有logits=X⋅W+b再加上一层softmax，且分词方法简单，**BoW（Bag of Words）/ N-gram** 方法本身也属于 最基础的文本特征表示方法。模型效果难以进一步优化
