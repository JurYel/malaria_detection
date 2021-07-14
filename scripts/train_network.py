import io
import itertools
import pickle
import os
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.compat.v1.keras import backend as K
from data_preprocessing import MalariaPreprocessing
from build_architecture import ConvNetMalaria
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as tf_hub

print("TF Version: ", tf.__version__)
print("Eager Mode: ", tf.executing_eagerly())
print("Hub Version: ", tf_hub.__version__)
print("GPU is: ", "AVAILABLE" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

class NeuralNetworkTrain:
    def __init__(self, base_path="models/malaria_detection/",
                 num_classes=2, shape=(100, 100, 1)):
        self.base_path = base_path
        self.num_classes = num_classes
        self.shape = shape

    @tf.function
    @tf.autograph.experimental.do_not_convert
    def train_network(self, pretrained_model_name, new_model_name, 
                        new_lr, num_epochs, ds_malaria, ds_info,
                        convNet_malaria):
        # calling other supporting classes to get the training
        config = MalariaPreprocessing(ds_malaria, ds_info)
        train_data, test_data, validation_data = config.preprocess(color_mode='grayscale')

        if pretrained_model_name is None:
            # Compile model and start training from epoch 1

            # build model architecture
            conv_model = convNet_malaria
            print(convNet_malaria.summary())
            
            # set Adam Optimizer and loss function
            opt = keras.optimizers.Adam(lr=1e-3)
            loss_fn = keras.losses.BinaryCrossentropy()
            binary_acc = keras.metrics.BinaryAccuracy(name='accuracy')
            auc = keras.metrics.AUC(name='auc')

            conv_model.compile(loss=loss_fn,
                                optimizer=opt,
                                metrics=[
                                    keras.metrics.BinaryAccuracy(name='accuracy'),
                                    keras.metrics.AUC(name='AUC')
                                ])

        elif pretrained_model_name is not None:
            conv_model = load_model(pretrained_model_name)
            if new_lr is None:
                old_lr = K.get_value(conv_model.optimizer.lr)
                new_lr = old_lr / 10
                K.set_value(conv_model.optimizer.lr, new_lr)

            else:
                old_lr = K.get_value(conv_model.optimizer.lr)
                K.set_value(conv_model.optimizer.lr, new_lr)

            print("Changed Learning Rate: {0} to {1}".format(old_lr, new_lr))

        elif (pretrained_model_name is None) and (convNet_malaria is None):
            raise Exception("Model is required for training session.")

        # list of callbacks
        checkpoint_filepath = os.path.join(self.base_path, "malaria_weights-{epoch:02d}.h5")
        callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    mode='auto',
                    min_delta=0,
                    patience=10,
                    verbose=1,
                    restore_best_weights=True
                ),

                keras.callbacks.ModelCheckpoint(
                    checkpoint_filepath,
                    monitor='val_acc',
                    mode='max',
                    verbose=1,
                    save_best_only=True
                )
        ]

        # check number of epochs
        if num_epochs is None:
            num_epochs = 20

        print("\n\n*****************TRAINING START*****************\n")

        model_train = conv_model.fit(
                            train_data,
                            epochs=num_epochs,
                            steps_per_epoch = train_data.cardinality() / num_epochs,
                            validation_data = validation_data,
                            validation_steps = validation_data.cardinality() / num_epochs,
                            max_queue_size=(config.batch_size * 2),
                            # batch_size=(config.batch_size),
                            callbacks=callbacks
                        )

        # save model weights and history

        pickle_path = self.save_model(model_train, conv_model, new_model_name)

        pickle_in = open(pickle_path, "rb")
        saved_history = pickle.load(pickle_in)
        pickle_in.close()

        print("\n\n*****************TRAINING COMPLETE*****************\n")

        return 

    def save_model(self,model_train, conv_model, model_nm):
        conv_model.save(filepath=os.path.join(self.base_path, model_nm))
        
        pickle_path = os.path.join(self.base_path, "malaria_history.pickle")
        pickle_out = open(pickle_path, "wb")
        pickle.dump(model_train.history, pickle_out)
        pickle_out.close()

        return pickle_path


if __name__ == '__main__':
    img_width, img_height = 100, 100

    ds_malaria, ds_info = tfds.load('malaria', with_info=True, as_supervised=True)

    base_path = "models/malaria_detection/"
    Train_NN = NeuralNetworkTrain(base_path, num_classes=2, shape=(100, 100, 1))

    ConvNet = ConvNetMalaria(num_classes=2)
    conv_model = ConvNet.build(shape=(img_width, img_height, 1))

    # start training
    Train_NN.train_network(num_epochs=20, pretrained_model_name=None, 
                            new_lr=None,convNet_malaria=conv_model, 
                            new_model_name="malaria_model.h5",
                            ds_malaria=ds_malaria, ds_info=ds_info)
    