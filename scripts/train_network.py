import tensorflow as tf
import io
import itertools
import pickle
import os
import tensorflow_datasets as tfds
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras import backend as K
from data_prepocessing import MalariaPreprocessing
from build_architecture import ConvNetMalaria
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model


class NeuralNetworkTrain:
    def __init__(self, base_path="models/malaria_detection/",
                 num_classes=2, shape=(100, 100, 1)):
        self.base_path = base_path
        self.num_classes = num_classes
        self.shape = shape

    def model_plot_history(self, model_hist, style='ggplot'):
        num_epochs = len(model_hist.history['loss'])

        plt.style.use(style)

        fig, ax = plt.subplots(1, 3, figsize=(10, 12))

        line1 = ax[0].plot(model_hist.history['accuracy'], label='Train Accuracy')
        line2 = ax[0].plot(model_hist.history['val_accuracy'], label='Test Accuracy')
        plt.setp(line1, linewidth=2.0)
        plt.setp(line2, linewidth=2.0)
        ax[0].set_title("Model Accuracy")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Accuracy")

        line1 = ax[1].plot(model_hist.history['loss'], label='Train Loss')
        line2 = ax[2].plot(model_hist.history['val_loss'], label='Test Loss')
        plt.setp(line1, linewidth=2.0)
        plt.setp(line2, linewidth=2.0)
        ax[1].set_title("Model Loss")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Loss")

        line1 = ax[2].plot(model_hist.history['AUC'], label='Train AUC')
        line2 = ax[2].plot(model_hist.history['val_AUC'], label='Test AUC')
        plt.setp(line1, linewidth=2.0)
        plt.setp(line2, linewidth=2.0)
        ax[2].set_title("Model AUC")
        ax[2].set_xlabel("Epochs")
        ax[2].set_ylabel("AUC")

        fig.window.manager.set_window_title("Train vs Test")
        
        for i in range(0, 2):
            ax[i].set_xlim(x_lim=(0, num_epochs))
            ax[i].legend()

        plt.show()

    def train_network(self, pretrained_model_name, new_model_name, 
                        new_lr, num_epochs, ds_malaria, ds_info):
        # calling other supporting classes to get the training
        config = MalariaPreprocessing(ds_malaria, ds_info)
        train_data, test_data, validation_data = config.preprocess(color_mode='grayscale')

        if pretrained_model_name is None:
            # Compile model and start training from epoch 1

            # build model architecture
            ConvNet = ConvNetMalaria(self.num_classes)
            conv_model = ConvNet.build(shape=self.shape)
            
            # set Adam Optimizer and loss function
            opt = keras.optimizers.Adam(lr=1e-3)
            loss_fn = keras.losses.BinaryCrossentropy()

            conv_model.compile(loss=loss_fn,
                                optimizer=opt,
                                metrics=[
                                    keras.metrics.BinaryAccuracy(name='accuracy'),
                                    keras.metrics.AUC(name='AUC')
                                ])

        else:
            conv_model = load_model(pretrained_model_name)
            if new_lr is None:
                old_lr = K.get_value(conv_model.optimizer.lr)
                new_lr = old_lr / 10
                K.set_value(conv_model.optimizer.lr, new_lr)

            else:
                old_lr = K.get_value(conv_model.optimizer.lr)
                K.set_value(conv_model.optimizer.lr, new_lr)

            print("Changed Learning Rate: {0} to {1}".format(old_lr, new_lr))

        
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
                            steps_per_epoch = train_data.__len__() / num_epochs,
                            validation_data = validation_data,
                            validation_steps = validation_data.__len__() / num_epochs,
                            max_queue_size=(config.batch_size * 2),
                            callbacks=callbacks
                        )

        # save model weights and history

        pickle_path = self.save_model(model_train, conv_model, new_model_name)

        pickle_in = open(pickle_path, "rb")
        saved_history = pickle.load(pickle_in)
        pickle_in.close()

        print("\n\n*****************TRAINING COMPLETE*****************\n")

        self.model_plot_history(saved_history)

        return 

    def save_model(self,model_train, conv_model, model_nm):
        conv_model.save(filepath=os.path.join(self.base_path, model_nm))
        
        pickle_path = os.path.join(self.base_path, "malaria_history.pickle")
        pickle_out = open(pickle_path, "wb")
        pickle.dump(model_train.history, pickle_out)
        pickle_out.close()

        return pickle_path

if __name__ == '__main__':
    ds_malaria, ds_info = tfds.load('malaria', with_info=True, as_supervised=True)

    base_path = "models/malaria_detection/"
    Train_NN = NeuralNetworkTrain(base_path, num_classes=2, shape=(100, 100, 1))

    # start training
    Train_NN.train_network(num_classes=2, num_epochs=20,
                            new_model_name="malaria_model.h5",
                            ds_malaria=ds_malaria, ds_info=ds_info)
    