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

class AcceptRejectPooling2D(Layer):
    def __init__(self, pool_size=(2, 2), padding='SAME', data_format='channels_last', **kwargs):
        super(AcceptRejectPooling2D, self).__init__(**kwargs)
        
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = conv_utils.normalize_tuple(pool_size, 2, 'strides')
        self.padding = padding
        self.data_format = 'NHWC' if data_format=='channels_last' else 'NCHW'
        self.input_spec = InputSpec(ndim=4)
       
    def build(self, input_shape):
        super(AcceptRejectPooling2D, self).build(input_shape)
        
    def _tf_pooling_function(self, x, name=None):
        #b = K.shape(x)[0]
        input_shape = K.int_shape(x)
        b, r, c, channel = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        
        pr, pc = self.pool_size 
        sr, sc = self.strides
        
        # compute number of windows
        num_r =  math.ceil(r/sr) if self.padding == 'SAME' else r//sr
        num_c = math.ceil(c/sc) if self.padding == 'SAME' else c//sc
        
        def _pool(inputs, is_train):
            # ensure inputs is positive 
            inputs[inputs < 0] = 0
            if np.sum(inputs) == 0:
                inputs = np.random.rand(inputs.shape)
            
            input_shape = inputs.shape
            batch = input_shape[0]

            # reshape
            w = np.transpose(inputs, (0, 3, 1, 2)) 
            w = np.reshape(w, (batch*channel, r, c))

            def pool(x):                
                if np.sum(x) == 0:
                    cache = np.zeros((pr, pc))
                    cache[0, 0] = 1.0
                    return cache

                x_prob = x / np.sum(x)
                
                if is_train:
                    ### in forward pass
                    # sort
                    size = pr*pc
                    x_prob = np.reshape(x_prob, [size])
                    x_sorted = np.argsort(-x_prob)
                    
                    while True:
                        y = random.random()
                        h = random.randint(0, size - 1)
                        p = x_sorted[h]
                        
                        if x_prob[p] >= y:
                            break                     
                    
                    pool_matrix = np.zeros((pr, pc))
                    pool_matrix[p//pr, p%pc] = 1.0

                    return pool_matrix
                else:
                    return x_prob

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
            return gen_nn_ops.max_pool_grad_v2(
                op.inputs[0],
                op.outputs[0],
                grad,
                ksizes,
                strides,
                self.padding,
                data_format=self.data_format
            ), K.tf.constant(0.0)

        def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
            # Need to generate a unique name to avoid duplicates:
            rnd_name = 'AcceptRejectPooling2DGrad' + str(np.random.randint(0, 1E+8))

            K.tf.RegisterGradient(rnd_name)(grad)  
            g = K.tf.get_default_graph()
            with g.gradient_override_map({"PyFunc": rnd_name}):
                return K.tf.py_func(func, inp, Tout, stateful=stateful, name=name)
        
        with ops.name_scope(name, "mod", [x]) as name:
            z = py_func(_pool,
                        [x, K.learning_phase()],
                        [K.tf.float32],
                        name=name,
                        grad=custom_grad)[0]
            z.set_shape((b, num_r, num_c, channel))
            
            return z

    def compute_output_shape(self, input_shape):
        r, c = input_shape[1], input_shape[2]
        sr, sc = self.strides
        num_r =  math.ceil(r/sr) if self.padding == 'SAME' else r//sr
        num_c = math.ceil(c/sc) if self.padding == 'SAME' else c//sc
        return (input_shape[0], num_r, num_c, input_shape[3])
    
    def call(self, inputs):
        output = self._tf_pooling_function(inputs)
        return output
    
    def get_config(self):
        config = {
                  'pool_size': self.pool_size,
                  'strides': self.strides
                }
        base_config = super(AcceptRejectPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items())) 