# Text_Classification

中文文本分类，Bert，ERNIE ，TextCNN，TextRNN，FastText，TextRCNN，BiLSTM_Attention, DPCNN, Transformer。

## 环境

python 3.7
pytorch 1.1
tqdm
sklearn
tensorboardX

## 中文数据集

[THUCNews](http://thuctc.thunlp.org/)

类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。

数据集划分：

数据集|数据量
--|--
训练集|18万
验证集|1万
测试集|1万

### 更换自己数据集

- 按照`data/THUCNews/`格式化自己的数据集
- 提前分好词，词之间用空格隔开
- 使用预训练词向量：utils.py的main函数可以提取词表对应的预训练词向量。

## 使用说明

```python
# TextCNN
python run.py --model TextCNN

# TextRNN
python run.py --model TextRNN

# TextRNN_Attention
python run.py --model TextRNN_Attention

# TextRCNN
python run.py --model TextRCNN

# FastText, embedding层是随机初始化的
python run.py --model FastText --embedding random 

# DPCNN
python run.py --model DPCNN

# Transformer
python run.py --model Transformer

# Bert
python run.py --model Bert

# bert + 其它
python run.py --model Bert_CNN

# ERNIE
python run.py --model ERNIE

```

## 对应论文

-  [Convolutional Neural Networks for Sentence Classification](http://emnlp2014.org/papers/pdf/EMNLP2014181.pdf)

- [Recurrent Neural Network for Text Classification with Multi-Task Learning](https://www.ijcai.org/Proceedings/16/Papers/408.pdf)

- [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](https://aclanthology.org/P16-2034.pdf)

- [Recurrent Convolutional Neural Networks for Text Classification](https://www.researchgate.net/publication/326185899_A_text_classification_model_using_convolution_neural_network_and_recurrent_neural_network)
- [Bag of Tricks for Efficient Text Classification](https://aclanthology.org/E17-2068.pdf)
- [Deep Pyramid Convolutional Neural Networks for Text Categorization](https://aclanthology.org/P17-1052.pdf)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://static.aminer.cn/upload/pdf/program/5bdc31b417c44a1f58a0b8c2_0.pdf)
- [ERNIE: Enhanced Representation through Knowledge Integration](https://arxiv.org/pdf/1904.09223v1.pdf)