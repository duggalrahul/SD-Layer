'''
This file implements the energy layer specified in the following paper

Andrearczyk, Vincent, and Paul F. Whelan.
"Using filter banks in convolutional neural networks for texture classification."
Pattern Recognition Letters 84 (2016): 63-69.
'''

from keras import backend as K
import theano
from theano.tensor.signal.conv import conv2d
from keras.engine.topology import Layer
from keras import regularizers
from keras import constraints
import pdb


class energyPool(Layer):
    def __init__(self, **kwargs):

        super(energyPool, self).__init__(**kwargs)
    
    def call(self,X,mask=None):

        return K.mean(X,axis=[2,3]).dimshuffle([0,1,'x','x'])

    def get_output_shape_for(self, input_shape):
        
        return (input_shape[0],input_shape[1],1,1)
        
