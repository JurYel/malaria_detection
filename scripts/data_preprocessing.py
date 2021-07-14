import numpy as np
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
import tensorflow as tf
from helper_functions import *

class MalariaPreprocessing():
    def __init__(self, ds_malaria, ds_info, buffer_size=27558, batch_size=64):
        self.ds_malaria = ds_malaria
        self.ds_info = ds_info
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.base_path = "models/malaria_detection/"
        self.num_epochs = 20
        self.ds_train = ds_malaria['train']

    def preprocess(self, test_split=0.2, val_split=0.1, color_mode='grayscale'):

        if color_mode == 'grayscale':
            train_and_validation_data = self.ds_train.map(normalize_image)
        
        elif color_mode == 'rgb':
            train_and_validation_data = self.ds_train.map(scale)

        num_validation_samples = val_split * self.ds_info.splits['train'].num_examples
        num_validation_samples = tf.cast(num_validation_samples, tf.int64)

        num_test_samples = test_split * self.ds_info.splits['train'].num_examples
        num_test_samples = tf.cast(num_test_samples, tf.int64)

        train_and_validation_data = train_and_validation_data.shuffle(self.buffer_size)

        train_data = train_and_validation_data.skip(num_validation_samples + num_test_samples)
        test_data = train_and_validation_data.take(num_test_samples)
        validation_data = train_and_validation_data.take(num_validation_samples)

        train_data = train_data.batch(self.batch_size)
        test_data = test_data.batch(num_test_samples)
        validation_data = validation_data.batch(num_validation_samples)

        return train_data, test_data, validation_data

    def config(self):
        # show configurations of dataset
        print(self.ds_info)

    def show_examples(self):
        print(tfds.show_examples(self.ds_info, self.ds_train))




if __name__ == "__main__":
    ds_malaria, ds_info = tfds.load(name='malaria', with_info=True, as_supervised=True)
    
    BUFFER_SIZE = 27_558
    BATCH_SIZE = 64
    
    malaria_ds = MalariaPreprocessing(ds_malaria, ds_info, BUFFER_SIZE, BATCH_SIZE)

    # malaria_ds.config()
    # malaria_ds.show_examples()

    train_data, test_data, validation_data = malaria_ds.preprocess(color_mode='grayscale')
    
    iterator = train_data.__iter__()
    next_elem = iterator.get_next()
    pt = next_elem[1]
    print(pt.numpy())

