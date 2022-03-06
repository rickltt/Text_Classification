import pandas as pd
import re

def text_parse(text):
    # 正则过滤掉特殊符号，标点，英文，数字等
    reg_1 = '[a-zA-Z0-9’!"#$%&\'()*+,-./:：;；|<=>?@，—。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    # 去除空格
    reg_2 = '\\s+'
    text = re.sub(reg_1, ' ', text)
    text = re.sub(reg_2, ' ', text)

    # 去除换行符
    text = text.replace('\n', '')
    return text

def read_file(filename):
    text = []
    label = []
    with open(filename,encoding='utf-8') as f:
        for line in f.readlines():
            text.append(line.strip('\n').split('_!_')[3])
            label.append(line.strip('\n').split('_!_')[2])
    df = pd.DataFrame(list(zip(text,label)),columns=['text','label'])
    
    return df

if __name__ == '__main__':
    label2id={
    'news_story':0,
    'news_culture':1,
    'news_entertainment':2,
    'news_sports':3,
    'news_finance':4,
    'news_house':5,
    'news_car':6,
    'news_edu' :7,
    'news_tech':8,
    'news_military':9,
    'news_travel':10,
    'news_world':11,
    'stock':12,
    'news_agriculture':13,
    'news_game':14
    }
    filename = 'toutiao_cat_data.txt'
    df = read_file(filename)
    df.text = df.text.map(text_parse)
    df['label_id'] = df.label.map(label2id)
    df = df[['text','label_id']]

    df = df.drop(df[df['text']==" "].index)
    #0.7,0.15,0.15比例划分训练集，测试集，验证集
    df = df.sample(frac=1.0)
    rows,cols = df.shape
    split_index_1 = int(rows*0.15)
    split_index_2 = int(rows*0.3)
    
    #数据分割
    df_test = df.iloc[0:split_index_1, :]
    df_dev = df.iloc[split_index_1:split_index_2 , :]
    df_train = df.iloc[split_index_2 : rows , :]
    
    df_test.to_csv('data/toutiao_news/test.txt',sep="\t",index=False,header=None)
    df_train.to_csv('data/toutiao_news/train.txt',sep="\t",index=False,header=None)
    df_dev.to_csv('data/toutiao_news/dev.txt',sep="\t",index=False,header=None)
