import os    
os.environ['THEANO_FLAGS'] = "device=cuda"    
import theano

import sys
sys.path.insert(0, '../convnets-keras')

from keras import backend as K
from theano import tensor as T
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
import numpy as np
from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.callbacks import CSVLogger, ModelCheckpoint, LambdaCallback
from keras.regularizers import l2
from keras.constraints import maxnorm
from macenko import *
from scipy.misc import imread

from energypool import *
from SDLayer import *

def OD_init(input_shape, ref_img_path):
    '''This function initialized the SDLayer with Stain-Matrix obtained via SVD.'''
    squeeze_percentile = 99.9
    query = imread(ref_img_path) / 255.0
    phi,a = GetWedgeMacenko(query, squeeze_percentile)
    init = phi
    return [np.reshape(init,input_shape)]

def embed_image(img):
    ''' embeds the image in a 400x400 box.'''
    n,c,h,w = img.shape
    
    square_side = 400
    
    left_pad = (square_side + 1 - w) // 2
    right_pad = (square_side - w) // 2
    top_pad = (square_side + 1 - h) // 2
    bottom_pad = (square_side - h) // 2         
    return K.asymmetric_spatial_2d_padding(img,top_pad=top_pad, bottom_pad=bottom_pad,\
                                           left_pad=left_pad, right_pad=right_pad, dim_ordering='th')

def embed_image_output_shape(input_shape):
    
    square_side = 400
    input_shape = list(input_shape)
    
    input_shape[2] = square_side
    input_shape[3] = square_side
    
    return tuple(input_shape)  


def get_model(model_type, learning_rate, decay, ref_img_path): 
	'''This function builds and returns a CNN model.'''

	inputs = Input(shape=(3,None,None))

	embed = Lambda(embed_image, output_shape=embed_image_output_shape,
		                                         name='embed')(inputs)

	OD_img = SDLayer(activation='tanh', phi=OD_init((3,3,1,1), ref_img_path))(embed)

	conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
		                   name='conv_1', init='he_normal')(OD_img)

	conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
	conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
	conv_2 = ZeroPadding2D((2,2))(conv_2)
	conv_2 = merge([
	    Convolution2D(128,5,5,activation="relu",init='he_normal', name='conv_2_'+str(i+1))(
		splittensor(ratio_split=2,id_split=i)(conv_2)
	    ) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")

	conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
	conv_3 = crosschannelnormalization()(conv_3)
	conv_3 = ZeroPadding2D((1,1))(conv_3)
	conv_3 = Convolution2D(384,3,3,activation='relu',name='conv_3',init='he_normal')(conv_3)

	conv_4 = ZeroPadding2D((1,1))(conv_3)
	conv_4 = merge([
	    Convolution2D(192,3,3,activation="relu", init='he_normal', name='conv_4_'+str(i+1))(
		splittensor(ratio_split=2,id_split=i)(conv_4)
	    ) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

	conv_5 = ZeroPadding2D((1,1))(conv_4)
	conv_5 = merge([
	    Convolution2D(128,3,3,activation="relu",init='he_normal', name='conv_5_'+str(i+1))(
		splittensor(ratio_split=2,id_split=i)(conv_5)
	    ) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")
	
	if model_type == 'tcnn':
	    dense_1 = energyPool(name='energy_pool')(conv_5)
	elif model_type == 'alexnet':
	    dense_1 = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)	    
	else:
	    print(model_type+' currently not implemented')
	
	dense_1 = Flatten(name="flatten")(dense_1)   
	dense_1 = Dense(4096, activation='relu',name='dense_1_rahul',init='he_normal')(dense_1)
	dense_2 = Dropout(0.5)(dense_1)
	dense_2 = Dense(4096, activation='relu',name='dense_2_rahul',init='he_normal')(dense_2)
	dense_3 = Dropout(0.5)(dense_2)
	dense_3 = Dense(2,name='dense_3_rahul',init='he_normal')(dense_3)

	prediction = Activation("softmax",name="softmax")(dense_3)

	model = Model(input=inputs, output=prediction)

	for l in model.layers:
	    l.trainable = True
	    
	sgd = SGD(lr=learning_rate, decay=decay, momentum=0.9, nesterov=True)
	model.compile(loss='mse',
		      optimizer=sgd,
		      metrics=['accuracy'])
    
	return model

