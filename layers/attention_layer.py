import tensorflow as tf 





class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, att_dropout=0.1, residual_dropout=0.1, scale=True):
        super(MultiHeadAttention, self).__init__()  
        self.num_heads = num_heads

