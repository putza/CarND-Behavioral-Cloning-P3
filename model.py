#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Model training script

This file contains all the necessary classes to train a self driving car model.
The file itself is executable on all machines with the correct libraries installed.


Example:
    Training of a model based on a input data path::

        $ ./model.py -p path to data

    This will create a model.h5 hdf file.

Todo:
    * Implement

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""


import argparse
import os

import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import cv2

import sklearn
from sklearn.model_selection import train_test_split

# Tensorflow import
import tensorflow as tf
#import tensorflow.keras as keras
from tensorflow.python.client import device_lib

#############################################
# Keras Import
#############################################
# Keras model definition
import keras
from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers import Flatten, Dense, Lambda, Activation
from keras.layers.core import Dropout
from keras.layers.core import Reshape
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
# Keras training
from keras.models import Model, load_model
from keras.optimizers import Adagrad
from keras.callbacks import EarlyStopping, ModelCheckpoint

import logging

# Create custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def get_device_list(self, tf_device_type=None):
	"""List all available compute devices (GPU, CPU)

	Args:
		device_type (string, optional): Filter for computer devices.
			Should be either "CPU" or "
	"""
	local_device_protos = device_lib.list_local_devices()
	if tf_device_type:
		return [x.name for x in local_device_protos if x.device_type == tf_device_type]
	else:
		return [x.name for x in local_device_protos]

class DriveData(object):
    """Management class for training and validation data.

    Class to import the steering file as well as the camera images and
    present them memory efficiently.
    Attributes:
      attr1 (str): Description of `attr1`.
      attr2 (:obj:`int`, optional): Description of `attr2`.
    """

    def __init__(self, path='./data', filename='augmented.csv', batchsize=1024):
        """Initializer of the DriveData object

        Args:
            path (str): Path to the data directory `param1`.
        """

        self._batchsize = batchsize
        self._filename = filename

        path = os.path.expanduser(path)
        self._path = path

        # Load data and create generators
        self._load_data()


    def _load_data(self):
        """Load the driving logfile and split in training vs validation set.

        TODO:
            * Add left and right image to dataset
            * Add flipped center image to dataset

        Args:
            path (str): Path to the data directory `param1`.
        """

        logger.debug('DriveData._load_csv: Load csv file')
        path = self._path

        # Read data
        df = pd.read_csv(os.path.join(path,self._filename))
        #df["im_sel"] = 'center'
        #df["im_flip"] = False


        train_samples, validation_samples = train_test_split(df, test_size=0.2)

        # Create attributes:
        self._samples = df
        self._samples_train = train_samples
        self._samples_validation = validation_samples

        self.generator_train = self.generator(samples=self._samples_train)
        self.generator_validation = self.generator(samples=self._samples_validation)

    def get_size(self):
        return len(self._samples_train), len(self._samples_validation)

    def generator(self,samples=None):
        """Generator to only hold batch_size of data in memory

        TODO:
            * Trim images -> Or do this with a cropping layer in keras

        Args:
            samples (Dataframe): Dataframe containing sample info.
            batch_size (int): Batch size for the training
        """

        print("="*50)
        print(" Generate batch data")
        print("-"*50)

        im_output_flip = True
        path = self._path
        batch_size = self._batchsize
        num_samples = len(samples)
        num_samples_flipped = 0
        print(" Batch size = {}, Sample Size = {}".format(batch_size,num_samples))

        while 1: # Loop forever so the generator never terminates
          sklearn.utils.shuffle(samples) # Wow, this work for pandas DataFrames !!!
          for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for index, batch_sample in batch_samples.iterrows():
              # name = './IMG/'+batch_sample[0].split('/')[-1] # Add if becessart
              im_col = batch_sample["im_sel"]
              image = cv2.imread(os.path.join(path, batch_sample[im_col].lstrip()))
              image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

              if batch_sample['im_flip']:
                if im_output_flip:
                  logger.info('Saving first flipped image')
                  print('Saving first flipped image')
                  plt.imsave('examples/im_flip_original.png',image)
                image = sp.fliplr(image)
                if im_output_flip:
                  plt.imsave('examples/im_flip_flipped.png',image)
                  im_output_flip = False

              angle = float(batch_sample['steering'])
              images.append(image)
              angles.append(angle)

            # trim image to only see section with road
            X_train = sp.array(images)
            y_train = sp.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


class KerasCNN(object):
  """Preprocessing and training of Keras cnn models for autonomous driving.

  Class to import the steering file as well as the camera images and
  present them memory efficiently.
  Attributes:
    attr1 (str): Description of `attr1`.
    attr2 (:obj:`int`, optional): Description of `attr2`.
  """

  def __init__(self, tf_device='/gpu:0', tf_device_type='CPU', input_shape=(160,320,3),model='nvidia'):

    logger.info('Initialising KerasCNN Class')

    self._input_shape = input_shape
    # Initialize Keras
    if not tf_device:
      logger.info('Detecting tf device')
      tf_device = self.get_device_list(tf_device_type=tf_device_type)
    self.keras_initialize(tf_device=tf_device)

    if model == 'nvidia':
      logger.info('KerasCNN.init: Use nvidia model')
      self._model_name = model
      self.model_input()
      self.model_nvidia_original()


  def get_device_list(self, tf_device_type=None):
    """List all available compute devices (GPU, CPU)

    Args:
      device_type (string, optional): Filter for computer devices.
        Should be either "CPU" or "
    """
    local_device_protos = device_lib.list_local_devices()
    if tf_device_type:
      return [x.name for x in local_device_protos if x.device_type == tf_device_type]
    else:
      return [x.name for x in local_device_protos]

  def keras_initialize(self, tf_device=None):

    self._keras_device = tf_device

    with tf.device(tf_device):
      self._model = Sequential()

  def model_input(self, cropping=((40,25), (0,0)), resizing=True, new_size=(66,200)):
    """Create input layers for model

    Args:
      cropping (list, optional): Pixel to be cropped:
        ((y_top, y_bottom), (x_left, e_right))
      resizing (bool, optional): Resize to nvidia model expected input.
        default: True
      new_size (list, optional): New image size. Only active is resizing=True.
        default: (66,200)


    """
    with tf.device(self._keras_device):
      self._model.add(Cropping2D(cropping=cropping, input_shape=self._input_shape))
      if resizing:
        self._model.add(Lambda(lambda x: tf.image.resize_images(x, new_size))) # resize image
      self._model.add(Lambda(lambda x: x/255.0 -0.5)) # normalization

  def model_nvidia_original(self, init = 'glorot_normal', activation = 'relu', dropout=(0.5,0.7),
                                  layers_dense=(100,50,10,1)):
    """ NVIDIA CNN for self driving (Modified to reduce DOF)

    Args:
      init (string, optional): Method to initialize weights.
        Default: 'glorot_normal'
      activation (string, optional): Activation function for CNN
        Default: 'relu'
      dropout (list, optional): Dropout rate for conv and dense layers.
        Defaut: ('0.5','0.7'), 1st element: Conv, 2nd element: dense
      layers_dense (list, optional): Size of the dense layers. Constructs as many dense
        layers as specified in the list. Default: (100,50,10,1)
    """
    with tf.device(self._keras_device):
      # Convnet
      self._model.add(Conv2D(24,(5,5), strides=(2,2), padding='valid', kernel_initializer=init,activation=activation))
      self._model.add(Dropout(dropout[0]))

      self._model.add(Conv2D(36,(5,5), strides=(2,2), padding='valid', kernel_initializer=init,activation=activation))
      self._model.add(Dropout(dropout[0]))

      self._model.add(Conv2D(48,(5,5), strides=(2,2), padding='valid', kernel_initializer=init,activation=activation))
      self._model.add(Dropout(dropout[0]))

      self._model.add(Conv2D(64,(3,3), kernel_initializer=init,activation=activation))
      self._model.add(Dropout(dropout[0]))

      self._model.add(Conv2D(64,(3,3), kernel_initializer=init,activation=activation))
      self._model.add(Dropout(dropout[0]))

      # Dense Layers
      self._model.add(Flatten())
      for dense_size in layers_dense:
        self._model.add(Dense(dense_size, kernel_initializer=init))
        if (dense_size !=1):
          self._model.add(Dropout(dropout[1]))


  def run_training(self, train_data=None, train_size=None,
                   val_data=None, val_size=None,
                   batch_size=1024, nb_epoch=100):

    if not train_data:
      logger.error('No training data provided')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

    with tf.device(self._keras_device):
      self._model.compile(loss='mse', optimizer='Adagrad')

      history_object = self._model.fit_generator(train_data, steps_per_epoch=train_size,
            validation_data=val_data, validation_steps=val_size,
            nb_epoch=nb_epoch)

    self._model.save('model_' + self._model_name + '.h5')
    print(history_object.history.keys())
    # Plot the training and validation loss for each epoch
    history_ = history_object.history
    print(history_.keys())
    sp.save('history_obj_origin.npy', history_)

    plt.figure()
    plt.plot(history_object.history['loss'],'-o')
    plt.plot(history_object.history['val_loss'],'-o')
    plt.title('Mean Squared Error of model ...')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.ylim([0, 0.1])
    plt.show()






def main():

    ########################################################################################
    # Parse the commandline
    ########################################################################################

    parser = argparse.ArgumentParser(description='Train DNN model for self driving cars')
    parser.add_argument('-p','--path', dest='path',
                        default='./data',
                        help='file to the data directory (default: ./data)'
    )
    parser.add_argument('-f','--filename', dest='fname',
                        default='augmented.csv',
                        help='name of data csv file (default augmented.csv)'
    )
    parser.add_argument('-m','--model', dest='model',
                        default='nvidia',
                        help='Name of the CNN model (default nvidia)'
    )
    parser.add_argument('-d','--device', dest='tf_device',
                        default='nvidia',
                        help='Name of the CNN model (default nvidia)'
    )
    parser.add_argument('-b','--batch_size', dest='batchsize',
                        default=1024,
                        help='Vatchsize for tensorflow (default 1024)'
    )

    args = parser.parse_args()

    print("="*50)
    print("= Execution of main script")
    print("-"*50)
    print(args.path)

    # Show Tensorflow Devices:
    print("-"*50)
    print("Tensorflow Devices:")
    print(device_lib.list_local_devices())
    print("-"*50)

    # Load the data and create CNN class:
    mydata = DriveData(path=args.path,filename=args.fname,batchsize=args.batchsize)
    mycnn  = KerasCNN(tf_device=args.tf_device,model=args.model)

    mycnn.get_device_list()

    # Prepare training and validation data
    size_training, size_validation = mydata.get_size()
    size_training = float(size_training / mydata._batchsize)
    size_validation = float(size_validation / mydata._batchsize)
    print("Training steps = {}, validation steps = {}".format(size_training,size_validation))

    # run training
    mycnn.run_training(train_data=gen_training, train_size=size_training,
                       val_data=gen_validation, val_size=size_validation,
                       nb_epoch=25)




########################################################################################
# Execute the main function if this file is called as main
########################################################################################
if __name__== "__main__":
    main()



