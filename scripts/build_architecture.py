import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import plot_model
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from helper_functions import *

class ConvNetMalaria():
    def __init__(self, num_classes=2):
        self.conv1 = layers.Conv2D(filters=64, kernel_size=(5,5), activation=tf.nn.relu)
        self.conv2 = layers.Conv2D(filters=64, kernel_size=(3,3), activation=tf.nn.relu)
        self.conv3 = layers.Conv2D(filters=128, kernel_size=(3,3), 
                                    activation=tf.nn.relu,
                                    padding='same'
                                )
        self.maxpool = layers.MaxPooling2D(pool_size=(2,2))
        self.flatten = layers.Flatten()
        self.out = layers.Dense(num_classes)

    def build(self, shape=(100, 100, 1)):
        inputs = keras.Input(shape=shape, name='img')
        x = self.conv1(inputs)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.maxpool(x)

        x = self.flatten(x)
        output = self.out(x)
        logistic_output = tf.nn.sigmoid(output)

        model = keras.Model(inputs, logistic_output, name='ConvNetMalaria')
        
        return model

    def plot_model_arch(self, model_path="models/malaria_detection/", window_pos=(50, 100)):
        model_diagram_path = model_path

        model = self.build(shape=(100, 100, 1))

        # generate plot 
        plot_model(model, to_file=model_diagram_path + 'malaria_detection-plot.png',
                    show_shapes=True,
                    show_layer_names=True)
        
        # show the plot
        img = mpimg.imread(model_diagram_path + 'malaria_detection-plot.png')
        fig = plt.figure(figsize=(30, 15))
        plt.axis('off')
        plt.imshow(img)

        # move_figure(fig, window_pos[1], window_pos[0])

if __name__ == '__main__':
    conv_net = ConvNetMalaria()

    model = conv_net.build()
    print(model.summary())