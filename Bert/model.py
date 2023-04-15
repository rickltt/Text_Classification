import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertModel
import torch.nn.functional as F

class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained(args.model_name_or_path)
        self.num_labels = args.num_labels
        self.fc = nn.Linear(self.bert.config.hidden_size, args.num_labels) 
        self.dropout = nn.Dropout(args.dropout_prob)
        self.loss_fun = nn.CrossEntropyLoss()
        
    def forward(self, input_ids, attention_mask, token_type_ids, label, mode):
        bert_outputs = self.bert(input_ids, attention_mask, token_type_ids) 
        cls_output = bert_outputs[1]  # bs x hidden_size
        cls_output = self.dropout(cls_output)

        logits = self.fc(cls_output) # bs x num_labels
        loss = self.loss_fun(logits.view(-1,self.num_labels), label)

        if mode == 'train':
            return loss
        else:
            return loss, logits

class BertCNN(nn.Module):
    def __init__(self, args):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(args.model_name_or_path)
        self.num_labels = args.num_labels
        self.filter_sizes = (2, 3, 4)  
        self.num_filters = 256  
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.bert.config.hidden_size)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(args.dropout_prob)
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), args.num_labels) 
        self.loss_fun = nn.CrossEntropyLoss()
    
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_ids, attention_mask, token_type_ids, label, mode):
        bert_outputs = self.bert(input_ids, attention_mask, token_type_ids)
        encoder_out = bert_outputs[0] 

        out = encoder_out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)

        logits = self.fc(out) # bs x num_labels
        loss = self.loss_fun(logits.view(-1,self.num_labels), label)

        if mode == 'train':
            return loss
        else:
            return loss, logits


class BertRNN(nn.Module):
    def __init__(self, args):
        super(BertRNN, self).__init__()
        self.bert = BertModel.from_pretrained(args.model_name_or_path)
        self.num_labels = args.num_labels
        self.rnn_hidden = 768
        self.num_layers = 2
        self.lstm = nn.LSTM(self.bert.config.hidden_size, self.rnn_hidden, self.num_layers,
                            bidirectional=True, batch_first=True, dropout=args.dropout_prob)
        
        self.dropout = nn.Dropout(args.dropout_prob)
        self.fc = nn.Linear(self.rnn_hidden * 2, args.num_labels) 
        self.loss_fun = nn.CrossEntropyLoss()
    

    def forward(self, input_ids, attention_mask, token_type_ids, label, mode):
        bert_outputs = self.bert(input_ids, attention_mask, token_type_ids)
        encoder_out = bert_outputs[0] 
        out, _ = self.lstm(encoder_out)

        out = self.dropout(out)
        logits = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state , shape: bs x num_labels

        loss = self.loss_fun(logits.view(-1,self.num_labels), label)

        if mode == 'train':
            return loss
        else:
            return loss, logits


class BertRCNN(nn.Module):
    def __init__(self, args):
        super(BertRCNN, self).__init__()
        self.bert = BertModel.from_pretrained(args.model_name_or_path)
        self.num_labels = args.num_labels
        self.rnn_hidden = 256
        self.num_layers = 2 
        self.dropout = nn.Dropout(args.dropout_prob)
        self.loss_fun = nn.CrossEntropyLoss()

        self.lstm = nn.LSTM(self.bert.config.hidden_size, self.rnn_hidden, self.num_layers,
                            bidirectional=True, batch_first=True, dropout=args.dropout_prob)
        self.maxpool = nn.MaxPool1d(args.max_seq_length)
        self.fc = nn.Linear(self.rnn_hidden * 2 + self.bert.config.hidden_size, self.num_labels)
    

    def forward(self, input_ids, attention_mask, token_type_ids, label, mode):
        bert_outputs = self.bert(input_ids, attention_mask, token_type_ids)
        encoder_out = bert_outputs[0] 
        out, _ = self.lstm(encoder_out)
        out = self.dropout(out)

        out = torch.cat((encoder_out, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()

        logits = self.fc(out) #  bs x num_labels

        loss = self.loss_fun(logits.view(-1,self.num_labels), label)

        if mode == 'train':
            return loss
        else:
            return loss, logits
        

class BertDPCNN(nn.Module):
    def __init__(self, args):
        super(BertDPCNN, self).__init__()
        self.bert = BertModel.from_pretrained(args.model_name_or_path)
        self.num_labels = args.num_labels

        self.num_filters = 250
        self.dropout = nn.Dropout(args.dropout_prob)
        self.loss_fun = nn.CrossEntropyLoss()

        self.conv_region = nn.Conv2d(1, self.num_filters, (3, self.bert.config.hidden_size), stride=1)
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
        x = x + px  # short cut
        return x
    
    def forward(self, input_ids, attention_mask, token_type_ids, label, mode):
        bert_outputs = self.bert(input_ids, attention_mask, token_type_ids)
        encoder_out = bert_outputs[0] 
        encoder_out = self.dropout(encoder_out)

        x = encoder_out.unsqueeze(1)  # [batch_size, 1, seq_len, embed]
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