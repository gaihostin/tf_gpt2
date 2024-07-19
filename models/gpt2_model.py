import os 
from tensorflow.python.framework import tensor_shape 

from layers.attention_layer import * 
from layers.


class Gpt2(tf.keras.Model):
    def __init__(self, num_layers, 
                 d_model, 
                 num_heads, 
                 dff, 
                 max_seq_len, 
                 vocab_size, 
                 optimizer="adam", 
                 learning_rate=1e-3,
                 rev_embedding_projection=True, 
                 grad_clip=False, 
                 clip_value=1.0):
        super(Gpt2, self).__init__()
        pass 
    

