import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.embedding_dim = args.emb_size
        self.embedding = nn.Embedding(len(args.word2id), self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(args.vec_mat))
        self.embedding.weight.requires_grad = True

        self.num_labels = args.num_labels
        self.num_filters = 256  
        self.filter_sizes = (2, 3, 4) 
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.embedding_dim)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(args.dropout_prob)
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), self.num_labels)
        self.loss_fun = nn.CrossEntropyLoss()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_ids, label, mode):
        out=self.embedding(input_ids)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        logits = self.fc(out)

        loss = self.loss_fun(logits.view(-1,self.num_labels), label)

        if mode == 'train':
            return loss
        else:
            return loss, logits
        

class TextRNN(nn.Module):
    def __init__(self, args):
        super(TextRNN, self).__init__()
        self.embedding_dim = args.emb_size
        self.embedding = nn.Embedding(len(args.word2id), self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(args.vec_mat))
        self.embedding.weight.requires_grad = True

        self.loss_fun = nn.CrossEntropyLoss()

        self.num_labels = args.num_labels
        self.hidden_size = 256
        self.num_layers = 1

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, self.num_layers,
                            bidirectional=True, batch_first=True, dropout=args.dropout_prob)
        self.fc = nn.Linear(self.hidden_size * 2, self.num_labels)

    def forward(self, input_ids, label, mode):

        embed=self.embedding(input_ids)
        out, _ = self.lstm(embed)
        logits  = self.fc(out[:, -1, :])
 
        loss = self.loss_fun(logits.view(-1,self.num_labels), label)

        if mode == 'train':
            return loss
        else:
            return loss, logits

class TextRCNN(nn.Module):
    def __init__(self, args):
        super(TextRCNN, self).__init__()
        self.embedding_dim = args.emb_size
        self.embedding = nn.Embedding(len(args.word2id), self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(args.vec_mat))
        self.embedding.weight.requires_grad = True

        self.loss_fun = nn.CrossEntropyLoss()

        self.num_labels = args.num_labels
        self.hidden_size = 256
        self.num_layers = 1
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, self.num_layers,
                    bidirectional=True, batch_first=True, dropout=args.dropout_prob)
        self.maxpool = nn.MaxPool1d(args.max_seq_length)
        self.fc = nn.Linear(self.hidden_size * 2 + self.embedding_dim, self.num_labels)
    
    def forward(self, input_ids, label, mode):

        embed=self.embedding(input_ids)
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()   
        logits = self.fc(out)

        loss = self.loss_fun(logits.view(-1,self.num_labels), label)

        if mode == 'train':
            return loss
        else:
            return loss, logits
        

class DPCNN(nn.Module):
    def __init__(self, args):
        super(DPCNN, self).__init__()
        self.embedding_dim = args.emb_size
        self.embedding = nn.Embedding(len(args.word2id), self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(args.vec_mat))
        self.embedding.weight.requires_grad = True

        self.loss_fun = nn.CrossEntropyLoss()

        self.num_labels = args.num_labels
        self.hidden_size = 256
        self.num_layers = 1
        self.num_filters = 250
        self.conv_region = nn.Conv2d(1, self.num_filters, (3, self.embedding_dim), stride=1)
        self.conv = nn.Conv2d(self.num_filters, self.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.num_filters, self.num_labels)
    
    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x
    
    def forward(self, input_ids, label, mode):
        embed=self.embedding(input_ids)
        x = embed.unsqueeze(1)  # [batch_size, 250, seq_len, 1]
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters(250)]
        logits = self.fc(x)

        loss = self.loss_fun(logits.view(-1,self.num_labels), label)

        if mode == 'train':
            return loss
        else:
            return loss, logits