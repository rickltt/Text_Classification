import os
import torch
import random
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

def collate_fn(batch):
    new_batch = { key: [] for key in batch[0].keys()}
    for b in batch:
        for key in new_batch:
            new_batch[key].append(b[key]) 
    for b in new_batch:
        new_batch[b] = torch.tensor(new_batch[b], dtype=torch.long)
    return new_batch

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

class BERT_Dataset(Dataset):
    def __init__(self, args, mode):
        fname = os.path.join(args.data_dir, '{}.txt'.format(mode))
        fin = open(fname, 'r')
        data = fin.readlines()
        fin.close()
        self.samples = []
        data_iterator = tqdm(data, desc="Loading: {} Data".format(mode))

        tokenizer = args.tokenizer
        max_seq_length = args.max_seq_length

        for instance in data_iterator:
            instance = instance.strip()
            sentence = instance.split('\t')[0]
            label = int(instance.split('\t')[1])
            tokens = []
            word_tokens = tokenizer.tokenize(sentence)
            # Chinese may have space for separate, use unk_token instead
            if word_tokens == []:
                word_tokens = [self.tokenizer.unk_token]
            for word_token in word_tokens:
                tokens.append(word_token)

            special_tokens = 2
            if len(tokens) > max_seq_length - special_tokens:
                tokens = tokens[:(max_seq_length - special_tokens)]

            text =  ['[CLS]'] + tokens + ['[SEP]'] 
            input_ids = tokenizer.convert_tokens_to_ids(text)
            input_masks = [1] * len(input_ids)


            padding_length = max_seq_length - len(input_ids)
            input_ids += [0] * padding_length
            input_masks += [0] * padding_length
            segment_ids = [0] * len(input_ids)
           

            assert len(input_ids) == args.max_seq_length
            assert len(input_masks) == args.max_seq_length
            assert len(segment_ids) == args.max_seq_length

            item = {
                'input_ids': input_ids,
                'attention_mask': input_masks,
                'token_type_ids': segment_ids,
                'label': label,
            } 
            self.samples.append(item)

    def __getitem__(self, index):
        return self.samples[index] 
    def __len__(self):
        return len(self.samples)  

def load_embedding_dict(args):
    with open(args.embedding_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

        lines = lines[1:]
        unk_embedding = np.random.randn(300)
        unk_embedding = unk_embedding.astype(str)
        unk_embedding = '<UNK> ' + ' '.join(unk_embedding)
        lines.insert(0,unk_embedding)

        pad_embedding = np.random.randn(300)
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

