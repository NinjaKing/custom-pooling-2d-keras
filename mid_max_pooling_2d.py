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
from tensorflow.python.ops import array_ops

import random
random.seed(91)

class MidMaxPooling2D(Layer):
    def __init__(self, pool_size=(2, 2), padding='SAME', data_format='channels_last', **kwargs):
        super(MidMaxPooling2D, self).__init__(**kwargs)
        
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = conv_utils.normalize_tuple(pool_size, 2, 'strides')
        self.padding = padding
        self.data_format = 'NHWC' if data_format=='channels_last' else 'NCHW' 
        self.input_spec = InputSpec(ndim=4)
        self.alpha = random.uniform(0, 1)
        self.alpha_frequencies = np.zeros(2)
       
    def build(self, input_shape):
        super(MidMaxPooling2D, self).build(input_shape)
        
    def _pooling_function(self, x, name=None):
        #b = K.shape(x)[0]
        input_shape = K.int_shape(x)
        b, r, c, channel = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        
        pr, pc = self.pool_size 
        sr, sc = self.strides
        
        # compute number of windows
        num_r =  math.ceil(r/sr) if self.padding == 'SAME' else r//sr
        num_c = math.ceil(c/sc) if self.padding == 'SAME' else c//sc
        
        def _mid_pool(inputs, is_train):
            input_shape = inputs.shape
            batch = input_shape[0]

            # reshape
            w = np.transpose(inputs, (0, 3, 1, 2)) 
            w = np.reshape(w, (batch*channel, r, c))

            def pool(x):
                # sort
                size = pr*pc
                x_flat = np.reshape(x, [size])
                x_sorted = np.argsort(-x_flat)
                p = x_sorted[math.ceil(size/2)]
                
                mid_matrix = np.zeros((pr, pc))
                mid_matrix[p//pr, p%pc] = 1.0

                return mid_matrix

            re = np.zeros((batch*channel, num_r, num_c), dtype=np.float32)
            # extract with pool_size
            for i in range(num_r):
                for j in range(num_c):
                    crop = np.zeros((batch*channel, pr, pc))
                    crop[:,:,:] = w[:, i*sr:i*sr+pr, j*sc:j*sc+pc]

                    # pool
                    outs = np.array(list(map(pool, crop)))
                    
                    if is_train:
                        re[:, i, j] = (crop * outs).max(axis=(1,2))
                    else:
                        re[:, i, j] = (crop * outs).sum(axis=(1,2))

            # reshape
            re = np.reshape(re, (batch, channel, num_r, num_c))
            re = np.transpose(re, (0, 2, 3, 1))
            
            return re

        def custom_grad(op, grad):
            if self.data_format=='NHWC':
                ksizes=[1, self.pool_size[0], self.pool_size[1], 1]
                strides=[1, self.strides[0], self.strides[1], 1]
            else:
                ksizes=[1, 1, self.pool_size[0], self.pool_size[1]]
                strides=[1, 1, self.strides[0], self.strides[1]]
                
            #return gen_nn_ops.max_pool_grad(
            return gen_nn_ops.avg_pool_grad(
                array_ops.shape(op.inputs[0]),
                grad,
                ksizes,
                strides,
                self.padding,
                data_format=self.data_format
            ), K.tf.constant(0.0)

        def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
            # Need to generate a unique name to avoid duplicates:
            rnd_name = 'MidPooling2DGrad' + str(np.random.randint(0, 1E+8))

            K.tf.RegisterGradient(rnd_name)(grad)  
            g = K.tf.get_default_graph()
            with g.gradient_override_map({"PyFunc": rnd_name}):
                return K.tf.py_func(func, inp, Tout, stateful=stateful, name=name)
        
        def _mid_range_pool(x, name=None):
            with ops.name_scope(name, "mod", [x]) as name:
                z = py_func(_mid_pool,
                            [x, K.learning_phase()],
                            [K.tf.float32],
                            name=name,
                            grad=custom_grad)[0]
                z.set_shape((b, num_r, num_c, channel))

                return z
        
        """
        Mixed Pooling
        """
        max_pool = K.pool2d(x, self.pool_size, strides=self.strides, padding=self.padding.lower(), pool_mode="max")
        mid_pool = _mid_range_pool(x, name)
        
        def _train_pool(max_pool, mid_pool):
            self.alpha = random.uniform(0, 1)
            self.alpha_frequencies[0] += self.alpha
            self.alpha_frequencies[1] += 1 - self.alpha
            
            return self.alpha * max_pool + (1 - self.alpha) * mid_pool
        
        def _test_pool(max_pool, mid_pool):
            return K.switch(K.less(self.alpha_frequencies[0], self.alpha_frequencies[1]), mid_pool, max_pool)
        
        outs = K.in_train_phase(_train_pool(max_pool, mid_pool), _test_pool(max_pool, mid_pool))
        
        return outs 

    def compute_output_shape(self, input_shape):
        r, c = input_shape[1], input_shape[2]
        sr, sc = self.strides
        num_r =  math.ceil(r/sr) if self.padding == 'SAME' else r//sr
        num_c = math.ceil(c/sc) if self.padding == 'SAME' else c//sc
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