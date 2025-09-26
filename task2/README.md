# Task2实验报告

## 摘要
本实验在给定中文文本分类数据集上，基于 CNN、RNN、Transformer 与 BERT 等多种模型进行了对比研究。实验围绕损失函数、学习率、卷积核规模与数量、优化器选择以及词向量初始化方式展开，评估其对分类性能的影响。结果表明：在 CNN、RNN、Transformer 等轻量模型中，准确率始终难以突破约 50%，表现出明显的过拟合现象；而引入 GloVe 预训练词向量可一定程度提升效果，但仍不足以显著改善泛化性能。相比之下，BERT 由于具备大规模语料的预训练、动态上下文建模能力和更强的模型容量，在测试集上取得了明显优于前者的结果。实验结论表明，本任务性能瓶颈主要来自数据本身的可分性不足与模型预训练差距，而非网络结构复杂度的限制，为后续通过数据增强与更强预训练模型的探索提供了方向。

## 方法

### 数据集

1. 训练集new_train.tsv:  https://fudan-nlp.feishu.cn/wiki/VhCHwZiXQicIYrkbd2ocFS6Kn9c#share-ZmBOdjQ2Tox695xImpMcAA5kn9f

2. 测试集new_test.tsv:  https://fudan-nlp.feishu.cn/wiki/VhCHwZiXQicIYrkbd2ocFS6Kn9c#share-InjndOoL7oDUagxiX88ccXTznhd
（数据规模：共有 8528 条训练数据，3309 条测试数据）

3. glove预训练的embedding：
https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.100d.zip（100维的词向量）

### 构建流程
1. 与Task1一样，先对文本进行读取与预处理；默认为：train集20%划分为验证集，80%划分到训练集，test集全部为测试集。比例后续可调整对比。

2. 定义模型。利用库定义了embedding，CNN，RNN，以及transformer。
其中embedding词向量维度均设定为100d，可使用glove预训练好的词向量库，不必从头训练；
CNN用了三组不同大小的filter，池化后拼接了三组filter的结果；RNN使用单层双向 LSTM；transformer为小维度、少层数、无预训练的轻量级版本；（三者选一个，对比实验）
最后定义共同的全连接层与softmax激活。

3. 定义损失函数与优化器。损失函数CE和MSE二选一，优化器Adam和SGD二选一。

4. 对模型进行训练与验证，保存loss值与acc值。

5. 结果可视化，考察模型在测试集上的表现。

----
## 实验
### 参数设置：
----（以下为默认值，根据不同实验修改对应参数）----
MAX_LEN = 100              # 句子最大长度
EMBED_DIM = 100 # 词向量维度
KERNEL_SIZES = [3, 4, 5] # 卷积核大小
NUM_CHANNELS = 50 # 通道数/卷积核个数
DROPOUT = 0.3 # 随即丢弃概率
MODEL_TYPE = "CNN"   # 模型可选: CNN, RNN, TRANSFORMER
BATCH_SIZE = 32# 训练批次大小
NUM_EPOCHS = 15 # 训练轮次
LEARNING_RATE = 2e-4 # 学习率
LOSS_FUNCTION = "CrossEntropy"    # 损失函数可选: "CrossEntropy", "MSE"
OPTIMIZER = "Adam"                可选: Adam, SGD
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # 优先使用gpu
filter滑动步长设置为1


### 实验内容：
-   **一、测试不同的损失函数、学习率对最终分类性能的影响**
    1. 不同损失函数（）
    - 交叉熵损失函数：Final Test Loss=1.2147, Test Acc=0.4856
    - 均方误差损失函数：Final Test Loss=0.1462, Test Acc=0.4219
    
	    **结论**：
	    在多分类任务中，如同Task1一样，MSE的表现不如CE。
	   
    2. 学习率
    - lr = 2e-4:
    ![输入图片说明](/imgs/2025-09-24/4Z2NOIlVKTG0Xdq8.png)Final Test Loss=1.2363, Test Acc=0.4896
    - lr = 1e-3:
    ![输入图片说明](/imgs/2025-09-24/zNj7wHJCdk4lzChj.png)
    - lr = 1e-2:
    ![输入图片说明](/imgs/2025-09-24/DH8FHg8wdUYMZAOt.png)
    - lr = 1e-5:
    ![输入图片说明](/imgs/2025-09-24/90mPJ4mAtslbilSU.png)
    
	    **结论**：
	    lr较大的时候，收敛速度快，但是由于步长太大，容易跨过最优点，导致valid_acc有所下降，而valid_loss不降反升，可以看到train_loss很快就降低到低值，train_acc很快就到了0.9以上，由此推断训练形成了模型的过拟合。
	    而lr较小时，收敛速度慢，可能15个epochs过后loss值还在持续下降，学习速度慢；
	    当lr适中，比如取2e-4时，loss值已然降到最低，基本收敛到当前最优，模型效果最好。
	    
-   **二、测试卷积核个数、大小及不同的优化器对最终分类性能的影响**
    - 核个数：（**3 组卷积核**窗口大小分别是 3、4、5, lr=2e-4）
		通道数 = 50：Final Test Loss=1.2363, Test Acc=0.4896
		通道数 = 100：Final Test Loss=1.2889, Test Acc=0.4835
		通道数 = 200：Final Test Loss=1.3651, Test Acc=0.4300
		通道数 = 10：Final Test Loss=1.2007, Test Acc=0.4829
		**结论**：
		-   **越多卷积核 → 表达能力越强**  
    每个卷积核学到的是一个“模式”，比如某些关键 n-gram 组合。  
    核数多 = 可以学习到的模式多，模型更“聪明”。
		-   **过多会导致过拟合**  
    参数量增加，容易在训练集拟合得很好，验证集/测试集反而下降（你的 val acc 就可能受这个影响）。
		-   **计算开销增加**  
    卷积核越多，计算量和显存消耗也线性增加。
	- 核大小：（分别测试3，4，5）
	kernel-size=3: Final Test Loss=1.1877, Test Acc=0.4841
	kernel-size=10: Final Test Loss=1.2127, Test Acc=0.4878
	kernel-size=100: Final Test Loss=1.8224, Test Acc=0.4358
	**结论**：
	卷积核大小决定每次卷积可以看到多少词，小卷积核对短语级信息敏感，大卷积核对段落级特征更敏感。当卷积核大到100时，相当于对句子进行全连接，失去了CNN提取句子局部特征的特点。然而本实验中，卷积核大小似乎并不起决定性作用。
	- 不同优化器：
	Adam：![输入图片说明](/imgs/2025-09-25/PE6nEx7JF6Y2mkia.png)![](blob:https://stackedit.cn/a619396c-2ee9-4a71-85d2-53e8b34f25c4)
	SGD：
	调参前：![输入图片说明](/imgs/2025-09-25/fxYVYwynxUMG4oOp.png)
	调参后：
	Final Test Loss=1.2205, Test Acc=0.4720
**结论**：
	SGD更不容易过拟合，泛化性好，但收敛过慢，需要反复调参才能得到好的效果；Adam收敛速度快，泛化性较差，但不太依赖超参数的组合。
	

-   **三、测试使用 glove 预训练的embedding进行初始化对最终分类性能的影响**
    本实验默认设置为使用预训练词向量库，现改为随机初始化词向量库并训练，用作对比：
    
    - 使用glove：
    ![输入图片说明](/imgs/2025-09-25/LaPQD7qejfxkEZmF.png)
    
    - 从零开始embedding：
    ![输入图片说明](/imgs/2025-09-25/Nzxnpo5tasVhrL7d.png)
    **结论**：
    明显可以看到，目前的模型从零开始训练词向量映射效果并不好。
    
-  **四、测试 CNN 改为 RNN 、Transformer （直接调用 pytorch 中的 api）等其它模型对最终分类性能的影响**
CNN详细搭建，RNN、Transformer简单调用api：
    - CNN：
    ![输入图片说明](/imgs/2025-09-25/8fSiAjVB7Xt9tyHM.png)![](blob:https://stackedit.cn/a619396c-2ee9-4a71-85d2-53e8b34f25c4)
    - RNN：
    ![输入图片说明](/imgs/2025-09-25/lfw8BxGVnQeNcb6t.png)
    当前参数下存在过拟合
    - Transformer：
![输入图片说明](/imgs/2025-09-25/zSX0MSchIlGSWCsR.png)
效果一般
-   **五、将结果绘制成图表**

![输入图片说明](/imgs/2025-09-24/1fVHTGBcHPHN9PEJ.png)

------
## 总结与反思

### 一、关于准确率卡在50%
实验发现，无论采取哪种模型，无论怎样调参，最终acc值还是卡在50%上不去，而train-acc可以到80-90%，推测模型学习能力并不差，只是数据太弱，训练出来的效果不好
数据本身的可分性不强，应该做数据强化，这一点留到Task3尝试

### 二、改用更强的模型
![输入图片说明](/imgs/2025-09-25/8zm38Hz2SyGpJN8c.png)
### **BERT 强在哪里**

### 1. **预训练语料**

-   你的 CNN / RNN 只在当前的数据集上训练，数据量可能只有几千条，容易过拟合，泛化能力有限。
    
-   **BERT 在预训练阶段已经看过了整个 Wikipedia 和 BookCorpus**（几十亿词），学到了语言的丰富语义与上下文规律。
    
-   所以在小数据集上微调时，BERT 已经带着“先验知识”，直接就能提取比较合理的语义特征。


### 2. **上下文表示**

-   传统嵌入（比如 GloVe）是 **静态词向量**，同一个词无论语境如何，向量都一样。
    
-   RNN / CNN 在捕捉语境时也比较有限：
    
    -   CNN 偏重于 **局部 n-gram** 模式；
        
    -   RNN 可以捕捉顺序，但长距离依赖容易丢失。
        
-   **BERT 用 Transformer Encoder，能通过自注意力机制捕捉全局上下文**，每个词的表示会根据整句动态调整。


### 3. **模型规模**

-   你的 CNN / RNN：
    
    -   嵌入 100 维
        
    -   卷积核数量 50
        
    -   参数量可能几十万 ~ 几百万
        
-   **BERT-base**：
    
    -   12 层 Transformer
        
    -   隐藏维度 768
        
    -   注意力头 12
        
    -   参数量约 1.1 亿
        
-   模型容量完全不在一个量级。BERT 能拟合和表达的模式远比浅层 CNN / RNN 强。
    

----------

### 4. **优化与收敛**

-   你的模型是从随机初始化开始学的，优化难度大，容易陷入局部最优。
    
-   **BERT 已经在大语料上收敛到一个很好的参数初始化点**，微调时只需“调整”而不是“从零学”，所以收敛更快更稳定。

### 三、如果采用更深的CNN：
![输入图片说明](/imgs/2025-09-26/jOzuf3cl0UlU2V63.png)

效果并不好，过拟合更加严重；说明不是模型复杂度的问题，而是预训练的差距

### 四、关于实验方法
这次将所有可能会在实验中变动的参数or模型都放在代码开头进行了宏定义，只需要在开头修改参数或者模型类型，就能直接运行做对比，不必到代码里面找到参数or模型定义的地方去注释/取消注释繁杂的修改。但是每次实验的数据，包括参数和模型的设置，都要自己另行保存，争取在Task3改进。
