import tensorflow as tf 

from layers.feed_forward import * 
from layers.layer_norm import * 



class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, att_dropout=0.1, residual_dropout=0.1, scale=True):
        super(MultiHeadAttention, self).__init__()  
        self.num_heads = num_heads
        self.d_model = d_model 
        self.att_dropout = att_dropout 
        self.residual_dropout = residual_dropout 
        self.scale = scale 

        assert d_model % self.num_heads == 0 

        self.depth = d_model // self.num_heads 

        self.c_attn = Conv1d(self.d_model, self.d_model * 3) 
        self.c_proj = Conv1d(self.d_model, self.d_model) 


    def multihead_attention(self, q, k, v, training, mask=None):
        """
        scaled-dot attention
        实际应用中， 训练阶段seq_len_q = seq_len_k = seq_len_v, 推理阶段seq_len_q != seq_len_k = seq_len_v 
        :param q: shape=(B, num_heads, seq_len_q, depth)
        :param k: shape=(B, num_heads, seq_len_k, depth)
        :param v: shape=(B, num_heads, seq_len_v, depth)
        :param training:
        :param mask: shape=(B, 1, seq_len_q, seq_len_q)
        :return 
        """
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        if self.scale:
            dk = tf.case(tf.shape(k)[-1], tf.float32)
            matmul_qk = matmul_qk/tf.math.sqrt(dk)
        
        if mask is not None:
            matmul_qk = matmul_qk * (mask * -1e9)
        
        attention_weights = tf.nn.softmax(matmul_qk, axis=-1) 

        if training:
            attention_weights = tf.nn.dropout(attention_weights, rate=self.att_dropout, name="attn_dropout")
        
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

        

