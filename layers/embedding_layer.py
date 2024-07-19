import tensorflow as tf 
import math 
from tensorflow.python.framework import tensor_shape 


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size, initializer=None, stddev=0.01, mean=0.0):
        super(EmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size 
        self.embedding_size = embedding_size 
        self.stddev = stddev
        self.mean = mean 
        self.initializer = initializer 

        if self.initializer is None:
            self.initializer = tf.random_normal_initializer(mean=self.mean,
                                                            stddev=self.stddev)
        
    
    def build(self, input_shape):
        with tf.name_scope("embedding_weights"):
            self.embedding_weights = self.add_weight(
                "weights",
                shape=[self.vocab_size, self.embedding_size],
                dtype="float32",
                initializer=self.initializer
            )
        super(EmbeddingLayer, self).build(input_shape)

    def call(self, inputs, mode="embedding", scale=False):
        if mode=="embedding":
            pass 

    
    def embedding(self, inputs, scale=False):
        with tf.name_scope("embedding"):
            mask = tf.cast(tf.not_equal(inputs, 0), tf.float32)
            inputs = tf.cast(inputs, tf.int32)
            embeddings = tf.nn.embedding_lookup(self.embedding_weights, inputs)
            embeddings *= tf.expand_dims(mask, -1)

            if scale:
                embeddings *= self.embedding_size ** 0.5 
            return embeddings 
    
    def projection(self, inputs):
        """
        embedding映射的逆操作， 用于在解析token的时候替代输出层的全连接
        : param inputs：shape=(B, seq_len, embedding_size)
        : return: shape=(B, seq_len, vocab_size)
        """
        with tf.name_scope("output_layer"):
            batch_size = tf.shape(inputs)[0]
            seq_len = tf.shape(inputs)[1]

            h_flat = tf.reshape(inputs, [-1, self.embedding_size])
            logits = tf.matmul(h_flat, self.embedding_weights, transpose_b=True)

            return tf.reshape(logits, [batch_size, seq_len, self.vocab_size])
        
    
class PositionEmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self, position_seq, pos_embedding_size, trainable=True, stddev=0.02, mean=0.0):
        super(PositionEmbeddingLayer, self).__init__()
        self.position_seq = position_seq
        self.hidden_size = pos_embedding_size 
        self.trainable = trainable 
        self.stddev = stddev 
        self.mean = mean 

        if trainable:
            self.position_embedding = EmbeddingLayer(self.position_seq, self.hidden_size, 
                                                     stddev=self.stddev, mean=self.stddev)
        
    def call(self, inputs, start=1):
        with tf.name_scope("pos_embedding"):
            if self.trainable:
                batch_size = tf.shape(inputs)[0]
                batch_seq = tf.shape(inputs)[1]

                positions = tf.reshape(tf.tile(tf.range(start, batch_seq + start), [batch_size]), 
                                       [batch_size, batch_seq])
                
                positions = tf.cast(positions, tf.int32)

                position_mask = tf.cast(tf.not_equal(inputs, 0), tf.int32)

                positions *= position_mask 

                return self.position_embedding(positions)
            else:
                pass 

    @staticmethod
    def get_position_sinusoid(seq_len, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
        position = tf.cast(tf.range(seq_len), tf.float32)
        num_timescales = hidden_size // 2 

        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) / 
            (tf.cast(num_timescales, tf.float32) -1))
        
        inv_timescales = min_timescale * tf.exp(
            tf.cast()
        )



