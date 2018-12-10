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
import pandas as pd
import cv2

import sklearn
from sklearn.model_selection import train_test_split
#import tensorflow as tf
#import keras


import logging

# Create custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class DriveData(object):
    """Management class for training and validation data.

    Class to import the steering file as well as the camera images and
    present them memory efficiently.
    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.
    """

    def __init__(self, path='./data'):
        """Initializer of the DriveData object

        Args:
            path (str): Path to the data directory `param1`.
        """

        path = os.path.expanduser(path)
        self._load_data(path=path)

        # Create attributes:
        self._path = path
        self.generator_train = self.generator(samples=self._samples_train,
            batch_size=32,path=path
        )

    def _load_data(self, path='./data'):
        """Load the driving logfile and split in training vs validation set.

        TODO:
            * Add left and right image to dataset
            * Add flipped center image to dataset

        Args:
            path (str): Path to the data directory `param1`.
        """

        logger.debug('DriveData._load_csv: Load csv file')

        # Read data
        df = pd.read_csv(os.path.join(path,'driving_log.csv'))
        df["im_sel"] = 'center'
        df["im_manip"] = 'none'


        train_samples, validation_samples = train_test_split(df, test_size=0.2)

        # Create attributes:
        self._samples = df
        self._samples_train = train_samples
        self._samples_validation = validation_samples

        self.generator_train = self.generator(samples=self._samples_train,
            batch_size=32,path=path
        )

        self.generator_validation = self.generator(samples=self._samples_validation,
            batch_size=32,path=path
        )

    def generator(self,samples=None, batch_size=32,path='./data'):
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

        num_samples = len(samples)
        print(f" Batch size = {batch_size}, Sample Size = {num_samples}")

        while 1: # Loop forever so the generator never terminates
            sklearn.utils.shuffle(samples) # Wow, this work for pandas DataFrames !!!
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset+batch_size]

                images = []
                angles = []

                for index, batch_sample in batch_samples.iterrows():
                    # name = './IMG/'+batch_sample[0].split('/')[-1] # Add if becessart
                    im_col = batch_sample["im_sel"]
                    image = cv2.imread(os.path.join(path, batch_sample[im_col]))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

    def __init__(self):
        print("Hallo")


def main():

    ########################################################################################
    # Parse the commandline
    ########################################################################################

    parser = argparse.ArgumentParser(description='Train DNN model for self driving cars')
    parser.add_argument('-p','--path', dest='path',
                        default='./data',
                        help='file to the data directory (default: ./data)'
    )

    args = parser.parse_args()
    print(args.path)

    # Load the data:

    mydata = DriveData(path=args.path)


########################################################################################
# Execute the main function if this file is called as main
########################################################################################
if __name__== "__main__":
    main()



