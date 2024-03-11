import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 实现transformer模型
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, max_len, embed_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(max_len, embed_size)
        
    def forward(self, input_ids, pos_ids):
        """
        input_ids/pos_ids: batch_size * seq_len
        return: batch_size * seq_len * embed_size
        """
        word_embed = self.word_embedding(input_ids)
        pos_embed = self.pos_embedding(pos_ids)
        # 将词嵌入和位置嵌入相加得到嵌入层输出
        return word_embed + pos_embed

# 缩放点乘注意力
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys, values, attention_mask):
        """
        queries/keys/values: batch_size * seq_len * hidden_size
        attention_mask: batch_size * seq_len * seq_len
        return: batch_size * seq_len * hidden_size
        """
        d = queries.size(-1)
        # 根据点乘注意力的矩阵形式计算注意力分数，除以查询向量或键向量维度的平方根，即为缩放点乘注意力
        scores = torch.bmm(queries, torch.transpose(keys, 1, 2)) / np.sqrt(d)
        # 将掩码为0的位置的注意力分数设为一个大负数，根据softmax函数的性质，这些注意力分数归一化后接近0
        scores[attention_mask == 0] = -1e6
        self.attention_weights = F.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), values)
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.W_o = nn.Linear(hidden_size, hidden_size)
        self.attention = ScaledDotProductAttention(dropout)
    
    def transpose_qkv(self, states):
        # 将长度为hidden_size的向量分成num_heads个长度相等的向量
        states = states.reshape(states.shape[0], states.shape[1], self.num_heads, self.hidden_size // self.num_heads)
        states = torch.permute(states, (0, 2, 1, 3))
        return states.reshape(-1, states.shape[2], states.shape[3])
    
    # 与transpose_qkv的变换相反
    def transpose_output(self, states):
        states = states.reshape(-1, self.num_heads, states.shape[1], states.shape[2])
        states = torch.permute(states, (0, 2, 1, 3))
        return states.reshape(states.shape[0], states.shape[1], -1)
    
    def forward(self, queries, keys, values, attention_mask):
        """
        querys/keys/values: batch * seq_len * hidden_size
        attention_mask: batch * seq_len * seq_len
        return:
        """
        # (batch_size * num_heads) * seq_len * (hidden_size / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))
        # 重复张量的元素，用以支持多个注意力头的运算
        # (batch_size * num_heads) * seq_len * seq_len
        attention_mask = torch.repeat_interleave(attention_mask, repeats=self.num_heads, dim=0)
        # (batch_size * num_heads) * seq_len * (hidden_size / num_heads)
        output = self.attention(queries, keys, values, attention_mask)
        # batch * seq_len * hidden_size
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)

# 两个简单的前向层
class PositionWiseFNN(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

# 层归一化
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        # 一个小量用于数值稳定（防止除0）
        self.eps = eps
        
    def forward(self, hidden_states):
        mean = torch.mean(hidden_states, -1, keepdim=True)
        std = torch.std(hidden_states, -1, keepdim=True)
        return self.gamma * (hidden_states - mean) / (std + self.eps) + self.beta

# 将两个输入相加并归一化
class AddNorm(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(hidden_size)
        
    def forward(self, X, Y):
        return self.layer_norm(self.dropout(Y) + X)
    
# 一个完整的transformer层
class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, intermediate_size):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.add_norm1 = AddNorm(hidden_size, dropout)
        self.fnn = PositionWiseFFN(hidden_size, intermediate_size)
        self.add_norm2 = AddNorm(hidden_size, dropout)
    
    def forward(self, X, attention_mask):
        Y = self.add_norm1(X, self.self_attention(X, X, X, attention_mask))
        return self.add_norm2(Y, self.fnn(Y))
