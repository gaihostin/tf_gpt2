import os 
import sys 
from typing import Dict, Tuple, List 
import tensorflow as tf 

from tqdm import tqdm 


def get_tf_gpt_parse(version: str= 'v1', data_type: str='done'):
    def _get_parse(serialized_example):
        data_fields = {
            'inputs': tf.io.VarLenFeature(tf.int64),
            'targets': tf.io.VarLenFeature(tf.int64)
        }

        parsed = tf.io.parse_single_example(serialized_example, data_fields)

        inputs = tf.sparse.to_dense(parsed["inputs"])
        targets = tf.sparse.to_dense(parsed["targets"])

        inputs = tf.cast(inputs, tf.int32)
        outputs = tf.case(outputs, tf.int32)
        return inputs, outputs

    def _get_raw_parse(serialized_example):
        """
        输入仅有tokenid的时候使用这个parse
        """

        data_fields = {
            "truncated_token_ids": tf.io.VarLenFeature(tf.int64)
        }

        parsed = tf.io.parse_single_example(serialized_example, data_fields)
        token_ids = tf.sparse.to_dense(parsed["truncated_token_ids"])

        inputs = token_ids[:-1]
        targets = token_ids[1:]

        inputs = tf.cast(inputs, tf.int32)
        targets = tf.cast(targets, tf.int32)
        return inputs, targets 
    
    if version == 'v1':
        if data_type == 'done':
            return _get_parse
        elif data_type == 'raw':
            return _get_raw_parse
        raise ValueError("未定义的data_type类型")
    else:
        raise ValueError("未定义的gpt类型")







