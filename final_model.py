#Importing necessary libraries
import pandas as pd 
import numpy as np
import re
from ast import literal_eval
from tqdm import tqdm
from icecream import ic
import time

#Initializing torch and cuda device
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device


import torch #importing pytorch modules
import torchtext
from transformers import BertModel, BertTokenizer #importing transformer modules

"""Importing Basic English Tokenizer and Bert Tokenizer"""
#Importing Basic Tokenizer
import torchtext
from torchtext.data import get_tokenizer
tokenizer = get_tokenizer('basic_english')

from transformers import BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

"""Loading datasets, comment out the dataset you don't want to use"""
#READ ParaNMT dataset
df = pd.read_csv('para-nmt-50m-demo/para-nmt50m-exemplar_1L.csv')
df.head()

# df = pd.read_csv('quora-question-pairs/qqp_exemplar_1L.csv')
# df.head()
# df.rename(columns = {'dataX':'source','dataY':'target','c_final_set':'exemplar'}, inplace = True)


"""Deleting the NaN value rows entirely"""
df.dropna(subset=['source', 'target','pos_tagged_s','pos_tagged_t','exemplar'], inplace=True)
print('NaN value rows deleted successfully')

"""Splitting the dataset into train, validation and test"""
#For ParaNMT
df_train = df[:93000]
df_val = df[93000:96000]
df_test = df[96000:]
#For QQPos
# df_train = df[:94000]
# df_val = df[94000:97000]
# df_test = df[97000:]


"""Handling Source Sentences"""
##SOURCE TRAIN
train_source_sent = df_train['source'].values
train_tokenized_source_sent = [tokenizer(sent) for sent in train_source_sent]

##SOURCE VALID
valid_source_sent = df_val['source'].values
valid_tokenized_source_sent = [tokenizer(sent) for sent in valid_source_sent]

##SOURCE TEST
test_source_sent = df_test['source'].values
test_tokenized_source_sent = [tokenizer(sent) for sent in test_source_sent]

"""Handling Target Sentences"""

##TARGET TRAIN
#target sentences are used for content and style
train_target_sent = df_train['target'].values
train_tokenized_target_sent = [tokenizer(sent) for sent in train_target_sent]
train_tokenized_ts_sos = [['<SOS>'] + x + ['<EOS>'] for x in train_tokenized_target_sent]
train_bert_tokenized_target_sent = [bert_tokenizer.encode(sent, add_special_tokens=True, truncation = True, max_length = 15, padding = "max_length") for sent in train_target_sent]

##TARGET VALID
#target sentences are used for content and style
valid_target_sent = df_val['target'].values
valid_tokenized_target_sent = [tokenizer(sent) for sent in valid_target_sent]
valid_tokenized_ts_sos = [['<SOS>'] + x + ['<EOS>'] for x in valid_tokenized_target_sent]
valid_bert_tokenized_target_sent = [bert_tokenizer.encode(sent, add_special_tokens=True, truncation = True, max_length = 15, padding = "max_length") for sent in valid_target_sent]

##TARGET TEST
#target sentences are used for content and style
test_target_sent = df_test['target'].values
test_tokenized_target_sent = [tokenizer(sent) for sent in test_target_sent]
test_tokenized_ts_sos = [['<SOS>'] + x + ['<EOS>'] for x in test_tokenized_target_sent]
test_bert_tokenized_target_sent = [bert_tokenizer.encode(sent, add_special_tokens=True, truncation = True, max_length = 15, padding = "max_length") for sent in test_target_sent]

"""Handling Exemplar Sentences"""
##EXEMPLAR TRAIN
strain_tokenized_exmp_sent = df_train['exemplar'].values
train_tokenized_exmp_sent = []
for i in range(len(strain_tokenized_exmp_sent)):
    train_tokenized_exmp_sent.append(literal_eval(strain_tokenized_exmp_sent[i]))
train_exmp_sent = [' '.join(sent) for sent in train_tokenized_exmp_sent]
train_bert_tokenized_exmp_sent = [bert_tokenizer.encode(sent, add_special_tokens=True,truncation = True, max_length = 15, padding = "max_length") for sent in train_exmp_sent]

##EXEMPLAR VALID
svalid_tokenized_exmp_sent = df_val['exemplar'].values
valid_tokenized_exmp_sent = []
for i in range(len(svalid_tokenized_exmp_sent)):
    valid_tokenized_exmp_sent.append(literal_eval(svalid_tokenized_exmp_sent[i]))
valid_exmp_sent = [' '.join(sent) for sent in valid_tokenized_exmp_sent]
valid_bert_tokenized_exmp_sent = [bert_tokenizer.encode(sent, add_special_tokens=True,truncation = True, max_length = 15, padding = "max_length") for sent in valid_exmp_sent]

##EXEMPLAR TEST
stest_tokenized_exmp_sent = df_test['exemplar'].values
test_tokenized_exmp_sent = []
for i in range(len(stest_tokenized_exmp_sent)):
    test_tokenized_exmp_sent.append(literal_eval(stest_tokenized_exmp_sent[i]))
test_exmp_sent = [' '.join(sent) for sent in test_tokenized_exmp_sent]
test_bert_tokenized_exmp_sent = [bert_tokenizer.encode(sent, add_special_tokens=True,truncation = True, max_length = 15, padding = "max_length") for sent in test_exmp_sent]


"""Creating Dataset and DataLoaders"""
#Importing Dataset and DataLoader
from torch.utils.data import Dataset, DataLoader

#Creating vocabulary for source and target sentences
def create_vocab(tokenized):
    vocab = {}
    freq = {}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    vocab['<SOS>'] = 2
    vocab['<EOS>'] = 3
    freq['<PAD>'] = 0
    freq['<UNK>'] = 0
    freq['<SOS>'] = 0
    freq['<EOS>'] = 0
    for sent in tokenized:
        for word in sent:
            if word not in vocab:
                vocab[word] = len(vocab)
                freq[word] = 1
            else:
                freq[word] += 1
    vocab_final = {}
    vocab_final['<PAD>'] = 0
    vocab_final['<UNK>'] = 1
    vocab_final['<SOS>'] = 2
    vocab_final['<EOS>'] = 3
    for word in vocab:
        if freq[word] >= 2:
            vocab_final[word] = len(vocab_final)
    return vocab_final

#Building vocabulary from tokenized sentences
vocab = create_vocab(train_tokenized_source_sent+train_tokenized_target_sent)

#Using the vocabulary to convert tokenized sentences to indices
def token2index_dataset(tokenized):
    indices = []
    for sent in tokenized:
        index = []
        for word in sent:
            if word in vocab:
                index.append(vocab[word])
            else:
                index.append(vocab['<UNK>'])
        indices.append(index)
    return indices

#Converting tokenized sentences to indices
###TRAIN SET
train_source = token2index_dataset(train_tokenized_source_sent)
train_target = token2index_dataset(train_tokenized_target_sent)
train_target_sos = token2index_dataset(train_tokenized_ts_sos)
train_exmp = token2index_dataset(train_tokenized_exmp_sent)
###VALID SET
valid_source = token2index_dataset(valid_tokenized_source_sent)
valid_target = token2index_dataset(valid_tokenized_target_sent)
valid_target_sos = token2index_dataset(valid_tokenized_ts_sos)
valid_exmp = token2index_dataset(valid_tokenized_exmp_sent)
###TEST SET
test_source = token2index_dataset(test_tokenized_source_sent)
test_target = token2index_dataset(test_tokenized_target_sent)
test_target_sos = token2index_dataset(test_tokenized_ts_sos)
test_exmp = token2index_dataset(test_tokenized_exmp_sent)


#Function to pad sentences to max length or truncate them according to max length
def pad_sents(sents, pad_token, max_len):
    padded_sents = []
    for sent in sents:
        if len(sent) < max_len:
            padded_sents.append(sent + [pad_token] * (max_len - len(sent)))
        else:
            padded_sents.append(sent[:max_len])
    return padded_sents
#Function to pad sentences to max length or truncate them according to max length
#BUT ADDS SOS AND EOS TOKENS AS WELL
def pad_sent_sos_eos(sents, pad_token, max_len):
    padded_sents = []
    for sent in sents:
        if len(sent) < max_len:
            padded_sents.append( sent + [pad_token] * (max_len - len(sent)))
        else:
            padded_sents.append(sent[:max_len-1] + [3])
    return padded_sents

"""Creating Dataset as a class"""

class EGPGDataset(Dataset):
    def __init__(self, source, target, target_sos, bert_target, exemplar, bert_exemplar, vocab):
        self.source = source
        self.target = target
        self.target_sos = target_sos
        self.exemplar = exemplar
        self.vocab = vocab
        self.source_data = pad_sents(self.source, self.vocab['<PAD>'], 15)
        self.target_data = pad_sents(self.target, self.vocab['<PAD>'], 15)
        self.target_data_sos = pad_sent_sos_eos(self.target_sos, self.vocab['<PAD>'], 15)
        self.exemplar_data = pad_sents(self.exemplar, self.vocab['<PAD>'], 15)
        self.bert_target_data = bert_target
        self.bert_exemplar_data = bert_exemplar
        
    def __len__(self):
        return len(self.source_data), len(self.target_data), len(self.target_data_sos), len(self.exemplar_data), len(self.bert_target_data), len(self.bert_exemplar_data)

    def __getitem__(self, idx):
        source_data = torch.tensor(self.source_data[idx])
        target_data = torch.tensor(self.target_data[idx])
        target_data_sos = torch.tensor(self.target_data_sos[idx])
        exemplar_data = torch.tensor(self.exemplar_data[idx])
        bert_target_data = torch.tensor(self.bert_target_data[idx])
        bert_exemplar_data = torch.tensor(self.bert_exemplar_data[idx])
        return source_data, target_data, target_data_sos, bert_target_data, exemplar_data, bert_exemplar_data

"""Creating Dataset for train, valid and test"""
train_dataset = list(EGPGDataset(train_source, train_target, train_target_sos, train_bert_tokenized_target_sent, train_exmp, train_bert_tokenized_exmp_sent, vocab))
valid_dataset = list(EGPGDataset(valid_source, valid_target, valid_target_sos, valid_bert_tokenized_target_sent, valid_exmp, valid_bert_tokenized_exmp_sent, vocab))
test_dataset = list(EGPGDataset(test_source, test_target, test_target_sos, test_bert_tokenized_target_sent, test_exmp, test_bert_tokenized_exmp_sent, vocab))

"""Creating Dataloader for train, valid and test"""
batch_size = 32 #fixed batch size of 32
###TRAIN LOADER
train_loader = DataLoader(train_dataset, batch_size=batch_size)
train_load_source, train_load_target, train_load_target_sos , train_load_bert_target, train_load_exmp, train_load_bert_exmp = next(iter(train_loader))
##VALID LOADER
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
valid_load_source, valid_load_target, valid_load_target_sos ,valid_load_bert_target, valid_load_exmp, valid_load_bert_exmp = next(iter(valid_loader))
##TEST LOADER
test_loader = DataLoader(test_dataset, batch_size=batch_size)
test_load_source, test_load_target, test_load_target_sos ,test_load_bert_target, test_load_exmp, test_load_bert_exmp = next(iter(test_loader))

"""Getting GloVe Embeddings"""
from torchtext.vocab import GloVe
glove = GloVe(name='6B', dim=300) #300 dimensional GloVe embeddings

#Function to create embedding matrix using the glove embeddings and vocabulary of the dataset
def create_embedding_matrix(vocab, embedding_dim):
    embedding_matrix = torch.zeros((len(vocab), embedding_dim))
    for word, index in vocab.items():
        if word in glove.stoi:
            embedding_matrix[index] = glove.vectors[glove.stoi[word]]
    embedding_matrix[1] = torch.mean(embedding_matrix, dim=0)
    return embedding_matrix.detach().clone()
#initialize embedding matrix
embedding_matrix = create_embedding_matrix(vocab, 100)

"""Calling required pytorch libraries to construct the model"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
bert = BertModel.from_pretrained('bert-base-uncased')

"""Creating a Style Encoder using pretrained BERT Model"""
#Style Encoder
class StyleEncoder(nn.Module):
    def __init__(self, bert):
        super(StyleEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True) #calling model
    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        #getting last hidden state of the CLS token since it encodes the entire informaiton
        cls_hidden_state = outputs[0][:,0,:]
        return cls_hidden_state #dim: batch_size x 768
#Declaring Style Encoder
style_encoder = StyleEncoder(bert)


"""Creating a Content Encoder using a BiGRU"""
#Content Encoder
class ContentEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, embedding_matrix):
        super(ContentEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        self.linear = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.2)
    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1) #dimensions:(batch_size, hidden_dim*2) 
        hidden = self.linear(hidden) #hidden dimensions:(batch_size, hidden_dim)
        return output, hidden #output dimensions: (batch_size, seq_len, hidden_dim)
#Declaring content encoder
content_encoder = ContentEncoder(len(vocab), 300, 512, embedding_matrix)

"""Applying ScaledDotAttention to the Encoder Decoder model"""
class ScaledDotProductAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super().__init__()
        self.scale = torch.sqrt(torch.FloatTensor([query_dim])).to(device) #dimensions: (32,qd)
        self.query_linear = nn.Linear(query_dim, query_dim)
        self.key_linear = nn.Linear(key_dim, query_dim)
        self.value_linear = nn.Linear(value_dim, query_dim)
    def forward(self, query, key, value, mask=None):
        query = self.query_linear(query) #dimensions: (1, batch_size, query_dim)
        key = self.key_linear(key) #dimensions: (batch_size, src_len, query_dim)
        value = self.value_linear(value) #dimensions: (batch_size, src_len, query_dim)
        attn_weights = torch.bmm(query.permute(1,0,2), key.permute(0,2,1)).to(device) #dimensions: (b, 1, s)
        attn_weights = attn_weights / self.scale 
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e10)
        attn_weights = F.softmax(attn_weights, dim=2) #dimensions: (b,1,s)
        context = torch.bmm(attn_weights, value).to(device) #dimensions: (b,1,q)
        return context, attn_weights
attention = ScaledDotProductAttention(1024, 1024, 1024)


"""Creating a single TimeStep Decoder using a GRU"""
#Single Step Decoder Class
class Decoder(nn.Module):
    def __init__(self, vocab_dim, emb_dim, hid_dim, embedding_matrix, attention):
        super(Decoder, self).__init__()
        self.vocab_dim = vocab_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.attention = attention
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        self.gru = nn.GRU(self.emb_dim, self.hid_dim, batch_first=True)
        self.fc = nn.Linear(1024, self.vocab_dim)
        self.gru_hid = nn.Linear(self.hid_dim, 1024)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, dec_data, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(dec_data)) #embedded has dimensions (batch_size, 1, emb_dim)
        output, gru_hidden = self.gru(embedded, hidden) #output dimention:(batch_size, 1, 1280) #h: (1,b,1280)
        gru_hidden = self.gru_hid(gru_hidden)
        context, attn_weights = self.attention(gru_hidden, encoder_outputs,encoder_outputs) 
        #Dimensions of: output(b,1,q) and attw (b,1,s)
        prediction = self.fc(context) 
        return prediction, gru_hidden
#Declaring Decoder
decoder = Decoder(len(vocab), 300, 1024, embedding_matrix,attention)

# Note: We do not softmax here since we are using a CrossEntropyLoss function which applies softmax internally 

"""Declaring a Seq2Seq Model which intgrates the encoders and single step decoder"""
from torch.distributions.categorical import Categorical
class Seq2Seq(nn.Module):
    def __init__(self, vocab_dim, emb_dim, hid_dim, content_encoder, style_encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.vocab_dim = vocab_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.content_encoder = content_encoder
        self.style_encoder = style_encoder
        self.decoder = decoder
        self.concat_linear = nn.Linear(self.hid_dim, 1024)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, load_source, load_target, load_target_sos, load_bert_exmp, load_bert_target, teacher_force=True):
        batch_size = load_source.shape[0] #batch_size
        seq_len = load_target.shape[1] #seq_len
        cont_output, cont_hidden = self.content_encoder(load_source) #cont_hidden dimention = (batch_size, 512)
        style_hidden = self.style_encoder(load_bert_exmp) #style_hidden dimention = (batch_size, 768)
        #concatenating hidden states
        concat_hidden = torch.cat((cont_hidden, style_hidden), dim = 1).unsqueeze(0) 
        concat_hidden = self.concat_linear(concat_hidden) #concat_hidden dimention = (1, batch_size, 1280)
        outputs = torch.zeros(batch_size, seq_len, self.vocab_dim).to(device)
        
        #first input to the decoder is the <SOS> token
        dec_input = load_target_sos[:,0].unsqueeze(1) #dec_input dimention = (batch_size,1)
        for t in range(1,seq_len):
            dec_output, concat_hidden = self.decoder(dec_input, concat_hidden, cont_output) #(b,1) #(1,b,1280)
            outputs[:, t, :] = dec_output.squeeze(1) #dimention = (batch_size, seq_len, vocab_dim) 
            #applying teacher forcing
            if teacher_force:
                dec_input = load_target_sos[:, t].unsqueeze(1) #dimention = (batch_size, 1)
            else:
                distribution = Categorical(logits=dec_output)
                dec_input = distribution.sample()
                # dec_input = dec_output.argmax(2) #dimention = (batch_size, 1)
        return outputs, cont_hidden, style_hidden
#Declaring seq2seq model
seq2seq = Seq2Seq(len(vocab), 300, 1280, content_encoder, style_encoder, decoder)
seq2seq.to(device) #Setting device to GPU

"""Verification that the model has trainable parameters and is working"""
#Printing number of traininable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(seq2seq):,} trainable parameters')

"""Declaring Optimizer and Loss Function(s)"""
optimizer = optim.Adam(seq2seq.parameters(), lr=0.0001)
#NLL Loss + Softmax
criterion = nn.CrossEntropyLoss(ignore_index=0)

"""Constructing a general contrastive loss class"""
def contrastive_loss(a, b):
    # For both inputs a and b, we have the same shape.
    # shape: (batch_size, hidden_size)
    TAU = 0.5

    prod_aa = torch.matmul(a, a.t())
    prod_ab = torch.matmul(a, b.t())
    #scale
    sig_aa = torch.sigmoid(prod_aa)
    sig_ab = torch.sigmoid(prod_ab)
    
    eps_aa = torch.div(sig_aa, TAU)
    eps_ab = torch.div(sig_ab, TAU)

    #exponentiate the values
    exp_aa = torch.exp(eps_aa)
    exp_ab = torch.exp(eps_ab)

    sum_samples = torch.sum(exp_aa, dim=1) + torch.sum(exp_ab, dim=1)
    identity_samples = torch.diag(exp_aa)
    pos_samples = torch.diag(exp_ab)
    logits = torch.div(sum_samples - identity_samples, pos_samples)
    loss = torch.log(logits)
    return torch.sum(loss)

##You can use the above fucntion as both Style Loss and Content Loss

"""Function to train the Model"""
#TRAIN
def train(model, iterator, optimizer, criterion, clip, content_encoder, style_encoder):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(tqdm(iterator)):
        #Loading required data values
        load_source = batch[0].to(device)
        load_target = batch[1].to(device)
        load_target_sos = batch[2].to(device)
        load_bert_target = batch[3].to(device)
        load_bert_exmp = batch[5].to(device)
        optimizer.zero_grad()
        output,source_cont_hidden,exemplar_style_hidden = seq2seq(load_source, load_target, load_target_sos,load_bert_exmp, load_bert_target,teacher_force=True)
        output = output[1:].view(-1, output.shape[2])
        target = load_target_sos[1:].view(-1)
        nll_loss = criterion(output, target)
        #Calculating Content Loss
        target_cont_output, target_cont_hidden = content_encoder(load_target) #dimension of target_cont_hidden = (batch_size, hid_dim)
        content_loss = contrastive_loss(target_cont_hidden,source_cont_hidden)+contrastive_loss(source_cont_hidden,target_cont_hidden)
        #Calculating Style Loss
        target_style_hidden = style_encoder(load_bert_target) #dimension hidden = (batch_size, hid_dim)
        style_loss = contrastive_loss(target_style_hidden,exemplar_style_hidden)+contrastive_loss(exemplar_style_hidden,target_style_hidden)
        loss = nll_loss+0.1*(content_loss+style_loss) #Combining the 3 losses as specified
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

"""Function to perform validation on the Model"""
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator)):
            load_source = batch[0].to(device)
            load_target = batch[1].to(device)
            load_target_sos = batch[2].to(device)
            load_bert_target = batch[3].to(device)
            load_bert_exmp = batch[5].to(device)
            output,source_cont_hidden,exemplar_style_hidden = seq2seq(load_source, load_target, load_target_sos, load_bert_exmp, load_bert_target,teacher_force=True)
            output = output[1:].view(-1, output.shape[2])
            target = load_target_sos[1:].view(-1)
            nll_loss = criterion(output, target)
            #Calculating Content Loss
            target_cont_output, target_cont_hidden = content_encoder(load_target) #dimension of target_cont_hidden = (batch_size, hid_dim)
            content_loss = contrastive_loss(target_cont_hidden,source_cont_hidden)+contrastive_loss(source_cont_hidden,target_cont_hidden)
            #Calculating Style Loss
            target_style_hidden = style_encoder(load_bert_target) #dimension hidden = (batch_size, hid_dim)
            style_loss = contrastive_loss(target_style_hidden,exemplar_style_hidden)+contrastive_loss(exemplar_style_hidden,target_style_hidden)
            loss = nll_loss+0.1*(content_loss+style_loss)
            # loss = nll_loss
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

"""Function to calculate the time taken for each epoch"""
def epoch_time (start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

"""Training the Model"""
N_EPOCHS = 20 #Number of epochs
CLIP = 1 #gradient clipping
best_valid_loss = float('inf') #setting best validation loss to infinity
for epoch in range(N_EPOCHS):
    start_time = time.time()
    #training the model
    train_loss = train(seq2seq, train_loader, optimizer, criterion, CLIP, content_encoder, style_encoder)
    #validating the model
    valid_loss = evaluate(seq2seq, valid_loader, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    #saving the best model based on lowest validation loss
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(seq2seq.state_dict(), 'm_para1.pt')
        print("BEST MODEL CHECKPOINT SAVED\n")
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}') 

"""Loading the best model which was saved previously during training"""
seq2seq.load_state_dict(torch.load('m_para1.pt'))
test_loss = evaluate(seq2seq, test_loader, criterion)
print(f'Test Loss: {test_loss:.3f}')

"""Now we generate sentences using the test data"""

#Inverse the vocab dictionary from (word to idx) to (idx to word)
vocab_inv = {v: k for k, v in vocab.items()}
#Function to convert indexes to tokens for a list
def index2token_dataset(indices):
    tokenized = []
    for index in indices:
        sent = []
        for word in index:
            sent.append(vocab_inv[word])
        tokenized.append(sent)
    return tokenized


"""Generating sentences using the test data"""
seq2seq.eval()
#Creating lists to store the generated sentences and more
source_data = []
target_data = []
exemplar_data = []
generated_data = []
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        #Loading the inputs from the dataloader
        load_source = batch[0].to(device)
        load_target = batch[1].to(device)
        load_target_sos = batch[2].to(device)
        load_bert_target = batch[3].to(device)
        load_bert_exmp = batch[5].to(device)
        output = seq2seq(load_source, load_target, load_target_sos, load_bert_exmp, load_bert_target, teacher_force=False)
        #Getting word with highest probability
        output = output.argmax(2) #dimention = (batch_size, seq_len)
        load_source = load_source.tolist()
        load_target = load_target.tolist()
        output = output.tolist()
        #using bert tokenizer, convert index to word
        for i in range(len(load_source)):
            source_data.append(load_source[i])
            target_data.append(load_target[i])
            exemplar_data.append(bert_tokenizer.decode(load_bert_exmp[i]))
            generated_data.append(output[i])

#Using the above functions to convert indexes to tokens
source_list = index2token_dataset(source_data)
target_list = index2token_dataset(target_data)
generated_list = index2token_dataset(generated_data)
source_list = [' '.join(sent) for sent in source_list]
target_list = [' '.join(sent) for sent in target_list]
generated_list = [' '.join(sent) for sent in generated_list]

"""Snippet to view all the sentences for that particular index"""
# #print examples
i=10
print('Source: ', source_list[i])
print('Target: ', target_list[i])
print('Exemplar: ', exemplar_data[i])
print('Generated: ', generated_list[i])
print('----------------------------------')

"""Calculating the ROUGEL score"""

from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rougeL'])

# a dictionary that will contain the results
results = {'precision': [], 'recall': [], 'fmeasure': []}

# for each of the hypothesis and reference documents pair
for (h, r) in zip(target_list,generated_list):
    # computing the ROUGE
    score = scorer.score(h, r)
    # separating the measurements
    precision, recall, fmeasure = score['rougeL']
    # add them to the proper list in the dictionary
    results['precision'].append(precision)
    results['recall'].append(recall)
    results['fmeasure'].append(fmeasure)

#calculating the average over all ROUGEL scores
avg_precision = sum(results['precision']) / len(results['precision'])
avg_recall = sum(results['recall']) / len(results['recall'])
avg_fmeasure = sum(results['fmeasure']) / len(results['fmeasure'])
print("Average Precision: ", avg_precision)
print("Average Recall: ", avg_recall)
print("Average F-Measure: ", avg_fmeasure)

"""Saving the lists into a csv fil format just for reference and possible later use"""
save_df = pd.DataFrame({'source': source_list, 'target': target_list, 'generated': generated_list, 'exemplar': exemplar_data})
save_df.to_csv('para-nmt-50m-demo/m_para1ltf.csv', index=False)





