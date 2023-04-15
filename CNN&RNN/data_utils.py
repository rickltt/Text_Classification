import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

def collate_fn(batch):
    new_batch = { key: [] for key in batch[0].keys()}
    for b in batch:
        for key in new_batch:
            new_batch[key].append(b[key]) 
    for b in new_batch:
        new_batch[b] = torch.tensor(new_batch[b], dtype=torch.long)
    return new_batch

def load_embedding_dict(args):
    with open(args.embedding_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines = lines[1:]
    unk_embedding = np.random.randn(args.emb_size)
    unk_embedding = unk_embedding.astype(str)
    unk_embedding = '<UNK> ' + ' '.join(unk_embedding)
    lines.insert(0,unk_embedding)

    pad_embedding = np.random.randn(args.emb_size)
    pad_embedding = pad_embedding.astype(str)
    pad_embedding = '<PAD> ' + ' '.join(pad_embedding)
    lines.insert(0,pad_embedding)

    embedding_dict = {}
    for line in lines:
        split = line.split(" ")
        embedding_dict[split[0]] = np.array(list(map(float, split[1:])))
    word2id = {}
    for i,j in enumerate(embedding_dict.keys()):
        word2id[j] = i
    vec_mat = [i for i in embedding_dict.values()]
    vec_mat = np.array(vec_mat)
    
    return embedding_dict, word2id, vec_mat


class DataProcessor(object):
    def __init__(self, args):
        fname = os.path.join(args.data_dir, 'class.txt')
        fin = open(fname, 'r')
        labels = []
        for line in fin.readlines():
            labels.append(line.strip())
        fin.close()

        self.labels = labels
        self.id2label = {i:j for i,j in enumerate(labels) }
        self.label2id = {j:i for i,j in enumerate(labels) }
        self.num_labels = len(labels)
        self.embedding_dict, self.word2id, self.vec_mat = load_embedding_dict(args)


class TC_Dataset(Dataset):
    def __init__(self, args, mode):
        fname = os.path.join(args.data_dir, '{}.txt'.format(mode))
        fin = open(fname, 'r')
        data = fin.readlines()
        random.shuffle(data)
        data = data[:5000]
        fin.close()

        max_seq_length = args.max_seq_length
        self.embedding_dict, self.word2id, self.vec_mat = args.embedding_dict, args.word2id, args.vec_mat

        self.samples = []
        data_iterator = tqdm(data, desc="Loading: {} Data".format(mode))

        for instance in data_iterator:
            instance = instance.strip()
            sentence = instance.split('\t')[0]
            label = int(instance.split('\t')[1])
            
            token_ids = []
            for word in sentence:
                if word in self.word2id:
                    token_ids.append(self.word2id[word])
                else:
                    token_ids.append(self.word2id['<UNK>']) 

            # padding
            seq_len = len(token_ids)
            padding_length = max_seq_length - seq_len
            if seq_len < max_seq_length:
                token_ids += ([0] * padding_length)
            else:
                token_ids = token_ids[:max_seq_length]

            assert len(token_ids) == max_seq_length

            sample = {
                'input_ids': token_ids,
                'label': label
            }
            self.samples.append(sample)
    def __getitem__(self, index):
        return self.samples[index] 
    def __len__(self):
        return len(self.samples)  

