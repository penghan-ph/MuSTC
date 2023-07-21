# AttentionRNNTime

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.TimeEncoder import TimeEncoder

class Self_Attention_Muti_Head(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self,input_dim,dim_k,dim_v,nums_head):
        super(Self_Attention_Muti_Head,self).__init__()
        assert dim_k % nums_head == 0
        assert dim_v % nums_head == 0
        self.q = nn.Linear(input_dim,dim_k)
        self.k = nn.Linear(input_dim,dim_k)
        self.v = nn.Linear(input_dim,dim_v)
        
        self.nums_head = nums_head
        self.dim_k = dim_k
        self.dim_v = dim_v
        self._norm_fact = 1 / math.sqrt(dim_k)
        
    def forward(self,x):
        # print('=====attention in: ',x.shape)
        Q = self.q(x).reshape(x.shape[0],x.shape[1],-1,self.dim_k // self.nums_head).permute(2,0,1,3)
        K = self.k(x).reshape(x.shape[0],x.shape[1],-1,self.dim_k // self.nums_head).permute(2,0,1,3)
        V = self.v(x).reshape(x.shape[0],x.shape[1],-1,self.dim_v // self.nums_head).permute(2,0,1,3)
        atten = nn.Softmax(dim=-1)(torch.matmul(Q,K.permute(0,1,3,2))) # Q * K.T() # batch_size * seq_len * seq_len
        output = torch.matmul(atten,V).reshape(x.shape[0],x.shape[1],-1) # Q * K.T() * V # batch_size * seq_len * dim_v
        # print('=====attention finish: ',output.shape)
        # x = output.tranpose(0,1)
        # print('=====attention tranpose finish: ',x.shape)
        return output, atten

class ART(nn.Module):
    def __init__(self, dim_in, dim_encoder, dim_attention, dim_k, dim_v, nums_head, dim_out, node):
        super(ART, self).__init__()
        # dim_in = 365
        self.node = node
        self.encoder = TimeEncoder(dim_in, dim_encoder, node)
        self.fc_attention = nn.Linear(dim_encoder, dim_attention)
        self.attention = Self_Attention_Muti_Head(dim_attention, dim_k, dim_v, nums_head)
        self.fc = nn.Linear(dim_v, dim_out)
        
    def forward(self, x):
        # print('=====ART in: ',x.shape)
        # x = bs*n*dim_in
        encoder_out = self.encoder(x)
        fc_out = self.fc_attention(encoder_out)
        attention_out, attention_score = self.attention(fc_out) #bs*n*dim_v
        # print('=====ART attention finish: ',x.shape)
        y = self.fc(attention_out) #bs*n*dim_out
        # print('=====ART fc finish: ',x.shape)
        return y, encoder_out, attention_score