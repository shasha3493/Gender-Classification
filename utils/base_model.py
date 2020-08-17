# Importing relevant libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Convolution2D, ZeroPadding2D, Activation,MaxPooling2D
import scipy.io
from scipy.io import loadmat

def create_base_model(mat_file):

    '''
    
    Creates the network based on http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf 
    and loads the pretrained weights passed as mat file

    Parameters:
    mat_file: file path for .mat file containing the pre-trained weights

    Returns:
    model: model architecture with pre-trained loaded 

    '''
    # Building architecture based on the paper

    # base model
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), name= 'conv1_1'))
    model.add(Activation('relu', name='relu1_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), name= 'conv1_2'))
    model.add(Activation('relu', name='relu1_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2), name='pool1'))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), name= 'conv2_1'))
    model.add(Activation('relu', name='relu2_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), name= 'conv2_2'))
    model.add(Activation('relu', name='relu2_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2), name='pool2'))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), name= 'conv3_1'))
    model.add(Activation('relu', name='relu3_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), name= 'conv3_2'))
    model.add(Activation('relu', name='relu3_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), name= 'conv3_3'))
    model.add(Activation('relu', name='relu3_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2), name='pool3'))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), name= 'conv4_1'))
    model.add(Activation('relu', name='relu4_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), name= 'conv4_2'))
    model.add(Activation('relu', name='relu4_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), name= 'conv4_3'))
    model.add(Activation('relu', name='relu4_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2), name='pool4'))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), name= 'conv5_1'))
    model.add(Activation('relu', name='relu5_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), name= 'conv5_2'))
    model.add(Activation('relu', name='relu5_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), name= 'conv5_3'))
    model.add(Activation('relu', name='relu5_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2), name='pool5'))
    
    model.add(Convolution2D(4096, (7, 7), name= 'fc6'))
    model.add(Activation('relu', name='relu6'))
    model.add(Dropout(0.5, name='dropout6'))
    model.add(Convolution2D(4096, (1, 1), name= 'fc7'))
    model.add(Activation('relu', name='relu7'))
    model.add(Dropout(0.5, name='dropout7'))
    model.add(Convolution2D(2622, (1, 1), name= 'fc8'))
    model.add(Activation('relu'))
    model.add(Flatten())

    # Citation: https://sefiks.com/2019/07/15/how-to-convert-matlab-models-to-keras/
    
    data = loadmat(mat_file, matlab_compatible=False, struct_as_record=False)

    # reference model
    net = data['net'][0][0]

    ref_model_layers = net.layers

    # base model's layers names
    base_model_layer_names = [layer.name for layer in model.layers]

    num_of_ref_model_layers = ref_model_layers[0].shape[0]

    #Iterate on reference model layers and if any layer exists in both reference and b
    # base model, then we can copy the layer weights from reference model to base model.
    for i in range(num_of_ref_model_layers):
        ref_model_layer = ref_model_layers[0][i][0][0].name[0]

        if ref_model_layer in base_model_layer_names:
            # we just need to set convolution and fully connected weights
            if ref_model_layer.find("conv") == 0 or ref_model_layer.find("fc") == 0:
                base_model_index = base_model_layer_names.index(ref_model_layer)
                
                weights = ref_model_layers[0][i][0][0].weights[0,0]
                bias = ref_model_layers[0][i][0][0].weights[0,1]
                
                model.layers[base_model_index].set_weights([weights, bias[:,0]])

    return model

