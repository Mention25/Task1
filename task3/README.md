# Task3实验报告

## 摘要
本报告旨在实现并验证 Vaswani 等人于 2017 年提出的 Transformer 模型的基础结构。基于 PyTorch 框架，模块化地构建了包含多头注意力（Multi-head Attention）、位置编码（Positional Encoding）、编码器（Encoder）和解码器（Decoder）等核心组件的模型。为全面探究 Transformer 架构的有效性与泛化能力，完成了两项子任务：多位数加法运算与德语-法语神经机器翻译及语言模型学习。

**在多位数加法任务中**，构建了一个 Encoder-Decoder 模型，用于学习形如 "A+B=C" 的序列到序列映射。实验结果表明，该模型能够成功学习到加法运算的内在算法逻辑，并在随机划分的测试集上达到了极高的准确率。这验证了 Transformer 模型处理符号化和算法规律任务的强大能力。

**在自然语言处理任务中**，基于德法平行语料库，分别实现了两种 Transformer 变体：

1.  一个标准的 **Encoder-Decoder 模型**，用于完成德语到法语的翻译任务。
    
2.  一个 **Decoder-Only 模型**（类似GPT架构），用于学习单语（法语）语言模型并进行文本生成。
    

在这些实验中，重点探究了**不同分词策略**对模型性能的影响，实现了 词级别（Word-level）、字符级别（Char-level）和字节对编码（BPE）三种分词器，并对不同词表大小（Vocabulary Size）的效果进行了对比。

综上所述，本次实验成功复现了 Transformer 模型，并证明了其在不同领域的任务上的通用性和高效性。实验结果清晰地展示了模型架构选择（Encoder-Decoder vs. Decoder-Only）和数据预处理策略（如分词方式）对最终性能的关键影响，为后续的深入研究奠定了坚实的基础。
## 方法
### 子任务1
#### 数据集：
本任务使用的数据集是**程序化自动生成的**，包含了大量多位数加法运算的文本序列。每一条数据都遵循“数字A+数字B=答案C”的格式。词汇表由10个数字（'0'-'9'）、2个运算符（'+', '='）以及3个特殊标记（`<sos>`, `<eos>`, `<pad>`）构成，总词汇量非常小且固定。
#### 构建流程：
-   参数配置：通过 argparse 或字典定义模型类型、数据划分方式、学习率、训练轮数等，方便切换实验设置。
    
-   数据生成与划分：生成所有可能的加法表达式，根据参数选择 random、length_extrapolation、carry_split 三种方式进行训练集、验证集、测试集划分。
    
-   词表构建与编码：定义符号集（0-9、+、=、pad、sos、eos），建立字符到索引的映射，将输入输出序列转为张量并进行 padding。
    
-   数据集与加载器：封装 AdditionDataset 类，返回源序列和目标序列，使用 DataLoader 批量加载并支持随机打乱。
    
-   模型搭建：根据选择构建 encoder-decoder Transformer 或 decoder-only Transformer，前者分输入输出序列，后者拼接表达式和答案进行自回归预测。
    
-   损失函数与优化器：使用交叉熵损失（忽略 pad），优化器采用 Adam 或 AdamW。
    
-   训练循环：每轮训练包括前向传播、计算 loss、反向传播和参数更新；在验证集上评估 loss 和准确率，记录日志。
    
-   测试与评估：使用训练好的模型在测试集预测答案，计算精确匹配准确率，比较不同划分方式下的结果分析泛化能力。
### 子任务2
#### 数据集：
本实验使用了公开的 **Multi30k Flickr 德法平行语料库**。该数据集包含了约3万条描述图片的简短句子，其中训练集29,000句，验证集和测试集各约1,000句。基于这份语料，分别进行了 Encoder-Decoder 翻译模型和 Decoder-Only 语言模型的实验。
#### 构建流程：
-   参数配置：在 PARAMS 里统一定义分词方式、词表大小、embedding 维度、层数、学习率、训练轮数等超参数，方便实验切换。
    
-   数据准备：指定训练/验证/测试的平行语料路径；若数据缺失则生成一个简易 dummy 数据，保证代码能跑通。
    
-   数据读取：实现 `load_parallel_corpus`，把德语和法语句子读入并一一配对。
    
-   分词器设计：
    
    -   BPE：基于 SentencePiece 训练并加载模型；
        
    -   Word：用词频统计构建有限大小的词表；
        
    -   Char：收集所有字符作为词表。  
        三种 tokenizer 都统一提供 encode/decode 接口。
        
-   数据集与加载器：封装 `TranslationDataset`，对每个样本分别调用 src_tok 和 tgt_tok 编码，返回张量；再用 DataLoader 打包成 batch。
    
-   模型搭建：
    
    -   Embedding 层 + 位置编码（用 Embedding 实现）；
        
    -   Encoder：多层 `TransformerEncoderLayer` 堆叠；
        
    -   Decoder：多层 `TransformerDecoderLayer` 堆叠；
        
    -   输出层：全连接映射到目标词表大小。  
        forward 时生成 padding mask 和 subsequent mask。
        
-   训练流程：
    
    -   `train_model`：逐 batch 前向计算、计算 loss、反向传播和优化；记录训练和验证集的 loss、准确率。
        
    -   `evaluate`：评估模型在验证/测试集的表现。
        
-   推理与展示：
    
    -   `greedy_decode`：给定源句，逐步预测目标词直到 `<eos>`；
        
    -   `show_translations`：抽样展示源句、参考翻译和模型预测；
        
    -   `show_tokenization_examples`：展示不同 tokenizer 的编码结果。
        
-   可视化：训练完成后画出 loss 和 accuracy 曲线，直观对比收敛情况。
    
-   主程序：加载数据 → 构建分词器 → 构建数据集和 DataLoader → 初始化模型、优化器、损失函数 → 训练 → 可视化与测试集评估 → 打印翻译示例。
## 实验
### 子任务1
#### 实验内容
- 参数设置：
	>parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["encdec","dec-only"], default="encdec")
    parser.add_argument("--split", choices=["random","length_extrapolation","carry_split"], default="random")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
- **尝试数据集不同的划分方式**
	- "encdec", "--split", "random", "--epochs", "5"
	![输入图片说明](/imgs/2025-09-28/EnDxBKGZqyWLZwM7.png)
	Final Test Acc: 0.9878876559769134
	**分析**： 训练集和测试集分布相似。测试几乎等价于“记忆力测试”，因为模型在训练时见过很多相似的组合。
	- "encdec", "--split", "length_extrapolation", "--epochs", "5"]
	![输入图片说明](/imgs/2025-09-28/9dy9FwfGflHFN6BF.png)
	Final Test Acc: 0.2932312130103944
	**分析**：
	   训练集用 **低位数加法**（比如 2+2、3+3）        
       测试集用 **高位数加法**（比如 4+4）
       测试数据分布和训练数据不重叠。        
       检查模型能否“学到进位规则并推广到更长位数”。
       正确率为30%，效果一般。
    
	- "encdec", "--split", "carry_split", "--epochs", "5"]
	![输入图片说明](/imgs/2025-09-28/QRe431yr1e2F8Wdb.png)
	Final Test Acc: 0.425269671284313
	**分析**：
		  训练集包含 **不产生进位的加法**（例如 123+456）
      测试集专门用 **产生进位的加法**（例如 589+412）。
       明确检验模型能否从“无进位”推广到“有进位”的规律。
		正确率为42%，还行。
		
	**结论**：
	encoder-decoder结构下，本模型对加法的推广能力一般，对进位的推广能力要稍强于高位加法；
	
- **encoder-decoder与decoder-only**
	- encoder-decoder:
	![输入图片说明](/imgs/2025-09-29/u1nSrSNGfdX90zZy.png)
	Final Test Acc: 0.9700967102334803
	
	- decoder-only:
	![输入图片说明](/imgs/2025-09-29/Hn3W18v25S1WgczU.png)
	Final Test Acc: 0.8893317606865416
	
	**结论**：不考虑推广能力，decoder-only的效果都会比encoder-decoder差一些，实验发现encoder-decoder对于进位和高位加法有一定的推广能力，而decoder-only则几乎没有。
### 子任务2
#### 参数设置：（默认设置，实验时可能变动）
	PARAMS = {
    "TOKENIZER_TYPE": "bpe",   # 可选 "word", "bpe", 或 "char" (Choose "word", "bpe", or "char")
    "VOCAB_SIZE": 29236,        # --- 控制词级别和BPE的词表大小 ---
    # --- 修改: 使用模型前缀，完整文件名将自动生成 ---
    "SRC_MODEL_PREFIX": "spm_de",
    "TGT_MODEL_PREFIX": "spm_fr",
    "MAX_LEN": 50,
    "BATCH_SIZE": 32,
    "EMBED_DIM": 128,
    "NHEAD": 4,
    "NUM_LAYERS": 2,
    "LR": 1e-2,
    "EPOCHS": 5,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}
#### 实验内容：
- **测试不同参数对模型的影响**：
	1. 学习率lr：
	- lr = 1e-4:
					Epoch 1: Train Loss=5.0373, Train Acc=0.3272, Valid Loss=3.7658, Valid Acc=0.4219
				Epoch 2: Train Loss=3.4221, Train Acc=0.4423, Valid Loss=3.2500, Valid Acc=0.4737
				Epoch 3: Train Loss=3.0227, Train Acc=0.4828, Valid Loss=2.9696, Valid Acc=0.5054
				Epoch 4: Train Loss=2.7738, Train Acc=0.5111, Valid Loss=2.7870, Valid Acc=0.5269
				Epoch 5: Train Loss=2.5884, Train Acc=0.5330, Valid 	Loss=2.6417, Valid Acc=0.5472
				Final Test Loss: 3.0195, Test Acc: 0.4915
	
	- lr = 1e-3:
![输入图片说明](/imgs/2025-09-30/oNOyM6gCYP2IQI0L.png)
		Final Test Loss: 2.3712, Test Acc: 0.5990
	
	- lr = 1e-2:
	![输入图片说明](/imgs/2025-09-30/57DpB4liqnHMfQBF.png)
	
		**结论**：Transformer适合更小的lr，实验得知lr=1e-3~5e-3时最合适，太小收敛速度慢，太大直接什么都学不到。
		
	2. 模型规模:
	- 128emb-dim / 2 层 / 4 heads：
	![输入图片说明](/imgs/2025-09-30/oNOyM6gCYP2IQI0L.png)
		Final Test Loss: 2.3712, Test Acc: 0.5990
		
	- 256emb-dim / 4 层 / 8 heads：
	![输入图片说明](/imgs/2025-09-30/VVVlD1hbt3n1TpMh.png)
	Final Test Loss: 3.2911, Test Acc: 0.4014
	
		**结论**：当前参数下，复杂模型的训练效果不如简单模型，且前者训练时间更长，大约为后者两倍。
		
	3. 序列长度 (MAX_SEQ_LEN)：
	- 50：
	![输入图片说明](/imgs/2025-09-30/oNOyM6gCYP2IQI0L.png)
		Final Test Loss: 2.3712, Test Acc: 0.5990
	- 25：
			Epoch 1: Train Loss=3.3508, Train Acc=0.4320, Valid Loss=2.6413, Valid Acc=0.5298
		Epoch 2: Train Loss=2.3055, Train Acc=0.5583, Valid Loss=2.2547, Valid Acc=0.5940
		Epoch 3: Train Loss=1.9188, Train Acc=0.6101, Valid Loss=2.0761, Valid Acc=0.6284
		Epoch 4: Train Loss=1.6596, Train Acc=0.6436, Valid Loss=1.9791, Valid Acc=0.6487
		Epoch 5: Train Loss=1.4656, Train Acc=0.6695, Valid Loss=1.9482, Valid Acc=0.6626
		Final Test Loss: 2.3482, Test Acc: 0.6028
	- 15：
	Epoch 1: Train Loss=3.4363, Train Acc=0.4144, Valid Loss=2.6558, Valid Acc=0.5206
Epoch 2: Train Loss=2.3498, Train Acc=0.5490, Valid Loss=2.2634, Valid Acc=0.5945
Epoch 3: Train Loss=1.9520, Train Acc=0.6022, Valid Loss=2.0884, Valid Acc=0.6210
Epoch 4: Train Loss=1.6852, Train Acc=0.6358, Valid Loss=2.0028, Valid Acc=0.6438
Epoch 5: Train Loss=1.4823, Train Acc=0.6620, Valid Loss=1.9812, Valid Acc=0.6495
Final Test Loss: 2.4675, Test Acc: 0.5777
	
		**结论**：数据集多为短句子，序列长度在25~50区间合适；
		太短会导致信息丢失，太长会学习到过多无关的上下文依赖。
		
- **尝试不同的tokenizer & 不同大小的vocab**：
	1. bpe：
	- vocab = 8000：
	![输入图片说明](/imgs/2025-09-30/JpZIC9Y60OBz5eSI.png)
	Final Test Loss: 2.1784, Test Acc: 0.5986
	
			1.源 (Source): eine alte waschmaschine auf der straße
			参考 (Target): une vieille machine à laver dans la rue
			预测 (Predicted): une vieille femme marchant dans la rue .
	
			2.源 (Source): eine frau posiert an einem steinigen strand , 	während ihr freund ein foto macht .
			参考 (Target): une femme pose sur une plage rocheuse , tandis que son amie prend une photo .
			预测 (Predicted): une femme pose sur une plage tandis qu&apos; elle prend une photo de son ami prend une photo .
			
	- vocab = 16000：
	![输入图片说明](/imgs/2025-09-30/JTxGBkOIpKQEuSIQ.png)
	Final Test Loss: 2.3243, Test Acc: 0.6009

	- vocab = 29236
	![输入图片说明](/imgs/2025-09-30/oNOyM6gCYP2IQI0L.png)
		Final Test Loss: 2.3712, Test Acc: 0.5990

		[德语原文]  ein kleiner gecko sitzt auf einem baumstamm .
		[德语ID序列]  [1, 7, 298, 47, 62, 29217, 160, 33, 24, 2833, 11, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ...
		[德语分词(BPE)]  '<s>', '▁ein', '▁kleiner', '▁ge', 'ck', 'o', '▁sitzt', '▁auf', '▁einem', '▁baumstamm', '▁.', '</s>'


			1.源 (Source): eine menschengruppe in einem kleinen dorf im winter .
			参考 (Target): un groupe de personnes dans un petit village en hiver .
			预测 (Predicted): un groupe de personnes dans un petit village .

			2.源 (Source): die statue vor dem gebäude
			参考 (Target): la statue devant le bâtiment
			预测 (Predicted): la statue d&apos; un bâtiment .
		
	2. word：（vocab = 29236）
	![输入图片说明](/imgs/2025-09-30/PcupKpKFaJAvEcaK.png)
	3. char：
	![输入图片说明](/imgs/2025-09-30/HjMost8UjpMbeG8S.png)
	Final Test Loss: 0.8772, Test Acc: 0.7263

			[德语原文]  ein wolkiger himmel mit blitzen über einer stadt .
			[德语ID序列]  [1, 34, 38, 43, 4, 52, 44, 41, 40, 38, 36, 34, 47, 4, 37, 38, 42, 42, 34, 41, 4, 42, 38, 49, 4, 31, 41, 38, 49, 55, 34, 43, 4, 63, 31, 34, 47, 4, 34, 38] ...
			[德语分词(Char)]  ['<sos>', 'e', 'i', 'n', ' ', 'w', 'o', 'l', 'k', 'i', 'g', 'e', 'r', ' ', 'h', 'i', 'm', 'm', 'e', 'l', ' ', 'm', 'i', 't', ' ', 'b', 'l', 'i', 't', 'z', 'e', 'n', ' ', 'ü', 'b', 'e', 'r', ' ', 'e', 'i']


			源 (Source): ein gelber hund im freien mit papier im maul .
			参考 (Target): un chien jaune dehors avec du papier dans sa gueule .
			预测 (Predicted): un chien blanc sur un trottoir dans la rue .

			源 (Source): menschengruppe , die einen turm aus menschen bildet
			参考 (Target): un groupe de personnes faisant une tour humaine
			预测 (Predicted): des gens se rassemblent dans une rue avec des cha

			源 (Source): ein motorradfahrer nimmt mit seinem motorrad an einem wettkampf teil .
			参考 (Target): un motard lors d&apos; une course avec sa moto
			预测 (Predicted): un chien marron et un homme avec un chapeau rouge

		**结论**：
		由于test-acc是以token为标准的，char的acc高于其他bpe和word是在预期内的。从给出的示例来人工分析，bpe的效果是最好的，优于word，最差是char。
	同时可以看到，vocab越大，模型的翻译效果越好
	

- **比较encoder-decoder与decoder-only的不同**:
	- encoder-decoder:
	![输入图片说明](/imgs/2025-09-30/JpZIC9Y60OBz5eSI.png)
	Final Test Loss: 2.1784, Test Acc: 0.5986
	
			1.源 (Source): eine alte waschmaschine auf der straße
			参考 (Target): une vieille machine à laver dans la rue
			预测 (Predicted): une vieille femme marchant dans la rue .
	
			2.源 (Source): eine frau posiert an einem steinigen strand , 	während ihr freund ein foto macht .
			参考 (Target): une femme pose sur une plage rocheuse , tandis que son amie prend une photo .
			预测 (Predicted): une femme pose sur une plage tandis qu&apos; elle prend une photo de son ami prend une photo .
	- decoder-only:
![输入图片说明](/imgs/2025-09-30/QZBtzqXVaqDe7Sw7.png)
	最终测试集损失 (Final Test Loss): 3.1797, 最终测试集准确率 (Test Acc): 0.3889

			[原文]  une paire de pieds avec des chaussettes sont près d&apos; un feu dans un baril .
			[ID序列]  [1, 16, 3454, 13, 1071, 57, 54, 2698, 143, 265, 6, 7983, 27, 7982, 5, 713, 48, 5, 3847, 10, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ...
			[分词(BPE)]  ['<s>', '▁une', '▁paire', '▁de', '▁pieds', '▁avec', '▁des', '▁chaussettes', '▁sont', '▁près', '▁d', '&', 'apos', ';', '▁un', '▁feu', '▁dans', '▁un', '▁baril', '▁.', '</s>']

			=== 随机文本生成示例 (Generation Examples) ===
			提示 (Prompt): 'deux jeunes filles ,'
			参考 (Original): deux jeunes filles , l&apos; une sur un petit vélo , l&apos; autre sur un petit tricycle , dans un endroit avec du sable .
			生成 (Generated): deux jeunes filles , l&apos; un en noir , l&apos; autre en t-shirt rouge et l&apos; autre en t-shirt rouge , l&apos; autre en t-shirt rouge et l&apos;

			提示 (Prompt): 'une voiture'
			参考 (Original): une voiture de sport noire avec une gopro traverse un parcours délimité par des cônes oranges .
			生成 (Generated): une voiture de course rouge et blanche est en train de regarder un autre homme en t-shirt rouge .

			提示 (Prompt): 'c&apos; est'
			参考 (Original): c&apos; est une femme souriant , portant des lunettes de soleil et des oreilles de lapin .
			生成 (Generated): c&apos; est un homme en t-shirt rouge et un pantalon noir , avec un casque rouge et un pantalon rouge et blanche .
		**结论**：在翻译任务上，encoder-decoder的效果要好于decoder-only.
		
## 总结与反思

### 一、关于语言模型任务的选取
最开始试图训练出一个给出第一个单词就可自己生成句子的语言模型，卡了挺久的；后面换成做机器翻译，简单了很多；但最开始的探索还是学到很多经验：比如模型训练语料与分词训练语料风格不一致。BPE / pretrained-BPE 需要词表和数据匹配，否则<unk>和奇怪分词会让效果更差。

### 二、实验方法的改进
本次实验将代码进行了分块，分工更加明确，易于调试；并且把每轮实验的参数设置+实验结果（loss/acc值，图像，日志）都保存到了一个文件里面，便于对比调参与后续复现；写报告时也不必跑一轮写一点，不必总是截图。

### 三、gpu与cpu
最开始的写法里，每个batch都做 tokenizer.encode，动态转成张量，数据量大时，CPU 端做 BPE 编码很耗时，GPU 在等数据，所以训练很慢。后来改进了代码，在准备数据阶段，就一次性把 train/valid/test 全部转成 token id 序列，同时做 padding；后续 DataLoader直接读数字张量，不需要再做字符串处理，这样GPU 就不会闲着。

### 四、翻译任务里的评估指标
test-acc应该如何定义？是翻译出来的句子一定要和标准答案中的每一个单词都一模一样而且位置差不多吗？并不见得。本实验对于三种分词方法，都选择做token级别的正确率评判。对于char而言，如果目标句是 `une`，字符序列是 `['u', 'n', 'e']`。模型逐个预测，如果都预测正确，那么这部分的字符准确率就是100%。这必定会导致对模型性能评估的较大偏差。所以本实验的test-acc仅仅作为参考，真正的翻译效果应该使用BLEU分数来判断。但由于编辑器在安装torchtext时遇到了麻烦，这里就步深入了。

### 五、为什么最大vocab是29236？
语料库中所有有意义的组合都已经被创建并加入词表后，`sentencepiece` 发现词汇表的总成员数量（包括初始的单个字符和所有合并后的子词）一共只有 **29,236** 个。

### 六、关于Mask
最开始的时候只加了subsequent mask，确保测试逻辑是对的；但忘记加padding mask，模型在注意力里把 `<pad>` 当成了有意义的输入，导致翻译出来的句子全是一个词。

### 七、关于代码分块
实验后期把代码放在一整块是因为这样方便问ai（）
