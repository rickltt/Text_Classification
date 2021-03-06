# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset_bert, build_iterator, get_time_dif,build_dataset
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer, Bert, Bert_CNN , ERNIE')
    parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
    parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
    args = parser.parse_args()

    #dataset = 'data/THUCNews'  # 数据集
    dataset = 'data/toutiao_news'  # 数据集
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    
    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'data/word_embedding/embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
   
    
    print(embedding)
    x = import_module('models.' + model_name)
    #print(model_name)
    if model_name[:4] == 'Bert':
        model_name = 'Bert'
    #print(model_name)
    if model_name == 'Bert':
        config = x.Config(dataset)
    else:
        config = x.Config(dataset, embedding)

    config.model_name = model_name
    #随机数种子
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    if model_name == 'Bert':
        train_data, dev_data, test_data = build_dataset_bert(config)
    else:
        vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


    # train
    if model_name != 'Bert':
        config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
        
    #print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)