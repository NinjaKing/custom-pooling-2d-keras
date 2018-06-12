import numpy as np
from time import time
import math

from keras.engine.topology import Layer
from keras.layers import InputSpec
from keras import backend as K
from keras.utils import conv_utils

import tensorflow as tf
from tensorflow.python.framework import ops  
from tensorflow.python.ops import gen_nn_ops

import random
random.seed(91)

class MixedPooling2D(Layer):
    def __init__(self, pool_size=(2, 2), padding='same', data_format='channels_last', **kwargs):
        super(MixedPooling2D, self).__init__(**kwargs)
        
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = conv_utils.normalize_tuple(pool_size, 2, 'strides')
        self.padding = padding
        self.data_format = 'NHWC' if data_format=='channels_last' else 'NCHW' 
        self.input_spec = InputSpec(ndim=4)
        self.alpha = random.uniform(0, 1)
        self.alpha_frequencies = np.zeros(2)
       
    def build(self, input_shape):
        super(MixedPooling2D, self).build(input_shape)
        
    def _pooling_function(self, x, name=None):
        """
        Mixed Pooling
        """
        max_pool = K.pool2d(x, self.pool_size, strides=self.strides, padding=self.padding, pool_mode="max")
        avg_pool = K.pool2d(x, self.pool_size, strides=self.strides, padding=self.padding, pool_mode="avg")
        
        def _train_pool(max_pool, avg_pool):
            self.alpha = random.uniform(0, 1)
            self.alpha_frequencies[0] += self.alpha
            self.alpha_frequencies[1] += 1 - self.alpha
            
            return self.alpha * max_pool + (1 - self.alpha) * avg_pool
        
        def _test_pool(max_pool, avg_pool):
            return K.switch(K.less(self.alpha_frequencies[0], self.alpha_frequencies[1]), avg_pool, max_pool)
        
        outs = K.in_train_phase(_train_pool(max_pool, avg_pool), _test_pool(max_pool, avg_pool))
        
        return outs

    def compute_output_shape(self, input_shape):
        r, c = input_shape[1], input_shape[2]
        sr, sc = self.strides
        num_r =  math.ceil(r/sr) if self.padding == 'same' else r//sr
        num_c = math.ceil(c/sc) if self.padding == 'same' else c//sc
        return (input_shape[0], num_r, num_c, input_shape[3])
    
    def call(self, inputs):
        output = self._pooling_function(inputs)
        return output
    
    def get_config(self):
        config = {
                  'pool_size': self.pool_size,
                  'strides': self.strides
                }
        base_config = super(MixedPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items())) 