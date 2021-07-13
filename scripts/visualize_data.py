import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow_datasets as tfds
import cv2
from data_preprocessing import MalariaPreprocessing

class Visualize():
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels


    def plot_data_imbalance(self):
        labels_train = pd.Series(self.train_labels, name='label')

        plt.figure(figsize=(10, 12))
        sns.barplot(x=labels_train.value_counts().index,
                    y=labels_train.value_counts())

        plt.show()

    def plot_blood_smears(self):
        plt.rcParams['figure.figsize'] = (10.0, 10.0)
        plt.subplots_adjust(wspace=0.2, hspace=0.2)

        for i in range(0, 25):
            rand_idx = np.random.randint(0, self.train_labels.value_counts().sum())
            img = self.train_images[rand_idx]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(5,5,i+1)
            plt.imshow(img)
            plt.grid(False)
            plt.axis('off')

        plt.show()

    
if __name__ == '__main__':
    ds_malaria, ds_info = tfds.load('malaria', with_info=True, as_supervised=True)

    BUFFER_SIZE = 27_558
    BATCH_SIZE = 64

    config = MalariaPreprocessing(ds_malaria, ds_info, BUFFER_SIZE, BATCH_SIZE)

    train_rgb, test_rgb, val_rgb = config.preprocess(color_mode='rgb')

    for images, labels in train_rgb:
        train_images = images.numpy()
        train_labels = labels.numpy()

    visualize = Visualize(train_images, train_labels)
    visualize.plot_data_imbalance()