from keras import backend as K
from theano import tensor as T
from theano.tensor.signal.conv import conv2d
from keras.engine.topology import Layer
from keras import regularizers
from keras import constraints
from keras import activations
from keras import initializations
import functools

import numpy as np

import pdb


class SDLayer(Layer):
    def __init__(self, phi=None, init='glorot_uniform',
                dim_ordering='th', activation=None, border_mode='same', 
                W_constraint=None, W_regularizer=None, **kwargs):
        self.dim_ordering = dim_ordering
        self.W_regularizer = regularizers.get(W_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.initial_weights = phi
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.border_mode = border_mode

        super(SDLayer, self).__init__(**kwargs)

    def build(self,input_shape):

        self.W_shape = (3,3,1,1)  # 3 filters of size 3x1x1. The weights are actually inv(S.V.)

        self.W = self.add_weight(self.W_shape,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint,
                                 initializer=functools.partial(self.init,
                                                               dim_ordering='th'),
                                 trainable=True)
        
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            #del self.initial_weights

        super(SDLayer, self).build(input_shape)

    def call(self,I,mask=None):
        
	PHI     = self.W
	PHI_INV = T.nlinalg.matrix_inverse(K.reshape(self.W,(3,3)))
	PHI_INV = K.reshape(PHI_INV,(3,3,1,1))
	
	
	mask  = (1.0 - (I > 0.)) * 255.0
	I = I + mask  # this image contains 255 wherever it had 0 initially

	I_OD = - T.log10(I/255.0)
	

	A = K.conv2d(I_OD,PHI_INV, border_mode='same')
	A = self.activation(A)    

        return A
    
    def get_output_shape_for(self, input_shape):
	
        return (input_shape[0],input_shape[1],input_shape[2],input_shape[3])
        
