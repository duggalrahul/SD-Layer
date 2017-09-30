import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.misc import imsave, imread
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from keras import backend as K
from theano import tensor as T

def evaluate_model(model,folder_path,class_idx):
    y_true,y_pred = [],[]

    for (a, b, filenames) in os.walk(folder_path):    
        for f in filenames:
            fname = folder_path + f
            img = imread(fname)
            img = np.transpose(img,[2,0,1])
            img = img.reshape((1,) + img.shape)

            y_true = y_true + [class_idx]            
            y_pred = y_pred + [np.argmax(model.predict(img)[0])]  

    return y_pred,y_true 
    

def evaluate_model_and_append(alexnet,folder_path,log_file_name,performance,history): 

    blast_path = folder_path + '/Blast/'
    hemat_path = folder_path + '/Hemat/'

    y_pred_blast,y_true_blast = evaluate_model(alexnet,blast_path,0)
    y_pred_hemat,y_true_hemat = evaluate_model(alexnet,hemat_path,1)

    y_pred = y_pred_blast + y_pred_hemat
    y_true = y_true_blast + y_true_hemat        
         
        
    try:     
        performance['train_loss']	   += history.history['loss']
	performance['train_acc_batchwise'] += history.history['acc']       
        performance['test_acc']  	   += [accuracy_score(y_true,y_pred)]
        performance['test_f1']   	   += [f1_score(y_pred,y_true,average='binary')]
        performance['test_prec'] 	   += [precision_score(y_pred,y_true,average='binary')]
        performance['test_rec']  	   += [recall_score(y_pred,y_true,average='binary')]

    except:   
        with open(log_file_name, 'w') as f:
            f.write('train_loss,train_acc_batchwise,test_acc,test_f1,test_prec,test_rec\n')               

        performance['train_loss']	    = history.history['loss']
	performance['train_acc_batchwise']  = history.history['acc']      
        performance['test_acc']  	    = [accuracy_score(y_true,y_pred)]
        performance['test_f1']   	    = [f1_score(y_pred,y_true,average='binary')]
        performance['test_prec'] 	    = [precision_score(y_pred,y_true,average='binary')]
        performance['test_rec']  	    = [recall_score(y_pred,y_true,average='binary')]

    with open(log_file_name, 'a') as f:
        f.write('{0},{1},{2},{3},{4},{5}\n'.format(performance['train_loss'][-1],\
		                                           performance['train_acc_batchwise'][-1],\
		                                           performance['test_acc'][-1],\
		                                           performance['test_f1'][-1],\
		                                           performance['test_prec'][-1],\
		                                           performance['test_rec'][-1]))
                
    return performance

def plot_performance(performance, path):
    full_path = path + 'plots/'    
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        
    plt.figure()
    plt.plot(performance['train_loss'])
    plt.title('Loss v/s Epochs')
    plt.ylabel('M.S.E Loss')
    plt.xlabel('Epoch')
    plt.legend(['train'], loc='upper left')    
    plt.savefig(full_path+'loss.png', bbox_inches='tight')
        
    plt.figure()
    plt.plot(performance['test_acc'])
    plt.title('Accuracy v/s Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['test'], loc='upper left')    
    plt.savefig(full_path+'test_acc.png', bbox_inches='tight')    

    plt.figure()
    plt.plot(performance['test_f1'])
    plt.title('F1-score v/s Epochs')
    plt.ylabel('F1-binary')
    plt.xlabel('Epoch')
    plt.legend(['test'], loc='upper left')    
    plt.savefig(full_path+'f1.png', bbox_inches='tight')
    
    plt.figure()
    plt.plot(performance['test_prec'])
    plt.title('Precision v/s Epochs')
    plt.ylabel('Precision-binary')
    plt.xlabel('Epoch')
    plt.legend(['test'], loc='upper left')    
    plt.savefig(full_path+'precision.png', bbox_inches='tight')
    
    plt.figure()
    plt.plot(performance['test_rec'])
    plt.title('Recall v/s Epochs')
    plt.ylabel('Recall-binary')
    plt.xlabel('Epoch')
    plt.legend(['test'], loc='upper left')    
    plt.savefig(full_path+'recall.png', bbox_inches='tight')
    
    plt.close('all')
    
    
    
def plot_deconvolution(model,layer_name,img):
    img_orig = img
    img = np.transpose(img,[2,0,1])
    img = img.reshape((1,) + img.shape)

    PHI = K.variable(model.get_layer(layer_name).get_weights()[0])

    PHI_INV = T.nlinalg.matrix_inverse(K.reshape(PHI,(3,3)))
    PHI_INV = K.reshape(PHI_INV,(3,3,1,1))

    I = K.variable(img)
    mask  = (1.0 - (I > 0.)) * 255.0
    I = I + mask  # this image contains 255 wherever it had 0 initially

    I_OD = - T.log10(I/255.0)


    A = K.conv2d(I_OD,PHI_INV, border_mode='same')


    PHI_1 = K.zeros((3,3,1,1))
    PHI_2 = K.zeros((3,3,1,1))
    PHI_3 = K.zeros((3,3,1,1))

    PHI_1 = T.set_subtensor(PHI_1[:,0,:,:],PHI[:,0,:,:])
    PHI_2 = T.set_subtensor(PHI_2[:,1,:,:],PHI[:,1,:,:])
    PHI_3 = T.set_subtensor(PHI_3[:,2,:,:],PHI[:,2,:,:])

    I_OD_1 = K.conv2d(A,PHI_1, border_mode='same')
    I_OD_2 = K.conv2d(A,PHI_2, border_mode='same')
    I_OD_3 = K.conv2d(A,PHI_3, border_mode='same')

    I_OD = K.concatenate([I_OD_1,I_OD_2,I_OD_3],axis=1)
    I_OD = I_OD.eval()

    I_OD_1 = np.reshape(I_OD[:,[0,1,2],:,:],I_OD[:,[0,1,2],:,:].shape[1:]).transpose(1,2,0)
    I_OD_2 = np.reshape(I_OD[:,[3,4,5],:,:],I_OD[:,[3,4,5],:,:].shape[1:]).transpose(1,2,0)
    I_OD_3 = np.reshape(I_OD[:,[6,7,8],:,:],I_OD[:,[6,7,8],:,:].shape[1:]).transpose(1,2,0)

    I_RGB_1 = (np.power(10,-I_OD_1) * 255).astype(int)
    I_RGB_2 = (np.power(10,-I_OD_2) * 255).astype(int)
    I_RGB_3 = (np.power(10,-I_OD_3) * 255).astype(int)

    I_RGB_1[I_RGB_1>255] = 255
    I_RGB_2[I_RGB_2>255] = 255
    I_RGB_3[I_RGB_3>255] = 255
    
    plt.figure()     
    
    plt.subplot(('141'))
    plt.axis('off')
    plt.title('Orig. Img')
    plt.imshow(img_orig)
    plt.subplot(('142'))
    plt.axis('off')
    plt.title('Stain 1')
    plt.imshow(I_RGB_1.astype(np.uint8))
    plt.subplot('143')
    plt.axis('off')
    plt.title('Stain 2')
    plt.imshow(I_RGB_2.astype(np.uint8))
    plt.subplot('144')
    plt.axis('off')
    plt.title('Stain 3')
    plt.imshow(I_RGB_3.astype(np.uint8))
