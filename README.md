# Text_Classification

中文文本分类，Bert，BertCNN, BertRNN, BertRCNN, BertDPCNN, TextCNN，TextRNN，TextRCNN，DPCNN。

## 环境

- numpy==1.23.5
- scikit_learn==1.2.2
- torch==1.10.0
- tqdm==4.64.1
- transformers==4.18.0

## 中文数据集

### THUCNews
类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。

数据集划分：

数据集|数据量
--|--
训练集|18万
验证集|1万
测试集|1万

### 头条新闻数据集

数据规模：

共382688条，分布于15个分类中。0.7、0.15、0.15比例划分数据集，验证集，测试集。

数据处理函数可见`process_data.py` , `toutiao_cat_data.txt`可以在这里[下载](https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset)。


15个分类：

```
故事 news_story
文化 news_culture
娱乐 news_entertainment
体育 news_sports
财经 news_finance
房产 news_house
汽车 news_car
教育 news_edu 
科技 news_tech
军事 news_military
旅游 news_travel
国际 news_world
股票 stock
三农 news_agriculture
游戏 news_game
```
### 更换自己数据集

- 按照`data/THUCNews/`格式化自己的数据集
- 提前分好词，词之间用空格隔开
- 使用预训练词向量：pretrained下面更换自己的词向量



## 使用说明
Bert模型在[Bert](Bert/)下面，CNN和RNN的模型在[CNN&RNN](CNN&RNN)下面。

model_type: TextCNN, TextRNN, TextRCNN, DPCNN, Bert, BertCNN, BertRNN, BertRCNN, BertDPCNN
```python
python run.py --do_train --do_eval --model_type Bert
```


## 对应论文

-  [Convolutional Neural Networks for Sentence Classification](http://emnlp2014.org/papers/pdf/EMNLP2014181.pdf)

- [Recurrent Neural Network for Text Classification with Multi-Task Learning](https://www.ijcai.org/Proceedings/16/Papers/408.pdf)

- [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](https://aclanthology.org/P16-2034.pdf)

- [Recurrent Convolutional Neural Networks for Text Classification](https://www.researchgate.net/publication/326185899_A_text_classification_model_using_convolution_neural_network_and_recurrent_neural_network)
- [Bag of Tricks for Efficient Text Classification](https://aclanthology.org/E17-2068.pdf)
- [Deep Pyramid Convolutional Neural Networks for Text Categorization](https://aclanthology.org/P17-1052.pdf)

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://static.aminer.cn/upload/pdf/program/5bdc31b417c44a1f58a0b8c2_0.pdf)
