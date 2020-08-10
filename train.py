# Importing relevant libraries
import os
import argparse
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from utils.data_prep import data_prep
from utils.base_model import create_base_model
from utils.plot_history import plot_training_history
from utils.preprocess import subtract_mean
import scipy.io

# Reading input from the terminal
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=13, help="seed for reproducibiity of results")
parser.add_argument("--weights_path", type=str, default="./pretrained_weights/vgg_face.mat", help="path to mat file with pretrained weights")
parser.add_argument("--train_dir", type=str, default="./data/train_data/", help="path train dataset")
parser.add_argument("--val_dir", type=str, default="./data/val_data/", help="path validation dataset")
parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
parser.add_argument("--checkpoint_path", type=str, default="./best_model/", help="path to save the best performing model")
parser.add_argument("--epochs", type=int, default=20, help="Epochs")
parser.add_argument("--plot_history", type=str, default="./results/history.jpeg", help="path to save the plot of training history to")


opt = parser.parse_args()



# Setting seed
tf.random.set_seed(opt.seed)

# Creating a directory struture for training, validation and testing
data_prep()

# Pretrained model
base_model = create_base_model(opt.weights_path)


# train and validation data directory
train_dir = opt.train_dir
val_dir = opt.val_dir


datagen_train = ImageDataGenerator(preprocessing_function = subtract_mean)
datagen_val = ImageDataGenerator(preprocessing_function = subtract_mean)

# batch_size and number of classes
batch_size = opt.batch_size
num_classes = 2

# Generator for train and validation dataset
generator_train = datagen_train.flow_from_directory(directory=train_dir,
                                                    target_size=(224, 224),
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    save_to_dir=None)

generator_val = datagen_val.flow_from_directory(directory=val_dir,
                                                  target_size=(224,224),
                                                  batch_size=batch_size,
                                                  shuffle=False)




# Extracting only upto 5th convolution layer from the base model to be used as feature extractor
transfer_layer = base_model.get_layer('pool5')
conv_model = Model(inputs=base_model.input,
                   outputs=transfer_layer.output)



# Start a new Keras Sequential model.
new_model = Sequential()

# Add the convolutional part of the VGG16 model from above.
new_model.add(conv_model)

# Flatten the output of the VGG16 model because it is from a convolutional layer.
new_model.add(Flatten())

# Add a dense (aka. fully-connected) layer.
new_model.add(Dense(2048, activation='relu'))

# Add Batch Normalization
new_model.add(BatchNormalization())

# Add dropout
new_model.add(Dropout(0.1))

# Add a dense (aka. fully-connected) layer.
new_model.add(Dense(1024, activation='relu'))

# Add Batch Normalization
new_model.add(BatchNormalization())

# new_model.add(Dropout(0.1))
new_model.add(Dropout(0.1))

# Add the final layer for the actual classification.
new_model.add(Dense(num_classes, activation='softmax'))

optimizer = Adam(lr=1e-5)

flag = False

# Fine tuning only the last convolution layer
for i, layer in enumerate(conv_model.layers):
    if layer.name == 'conv5_1':
        flag = True
    if not flag:
        layer.trainable = False


loss = 'categorical_crossentropy'
metrics = ['categorical_accuracy']

new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

epochs = opt.epochs
steps_per_epoch = generator_train.n/batch_size
steps_test = generator_val.n / batch_size



# Directory too store the best model
checkpoint_path = opt.checkpoint_path


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    monitor='val_categorical_accuracy',
    mode='max',
    save_best_only=True)

# Training and Validation
history = new_model.fit(x=generator_train,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=generator_val,
                        validation_steps=steps_test,
                        callbacks = [model_checkpoint_callback])
                        
# Plotting train and val history
plot_training_history(history, opt.plot_history)