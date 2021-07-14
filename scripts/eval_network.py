import tensorflow as tf
import numpy as np
from data_preprocessing import MalariaPreprocessing
from tensorflow import keras
import tensorflow_datasets as tfds
import os

class NeuralNetworkEval:
    def __init__(self):
        pass

    def evalNetwork(self, train_model_path, config):
        # load train and test data
        train_data, test_data, _ = config.preprocess(color_mode='grayscale')

        # load pretrained model to test Accuracy
        print("Loading model: {0}".format(train_model_path))
        model = keras.models.load_model(train_model_path)

        # evaluate model against unseen data (test)
        print("Evaluating model against test data")
        (train_loss, train_acc, train_auc) = model.evaluate(train_data)
        (test_loss, test_acc, test_auc) = model.evaluate(test_data)

        print("\n\n====================TRAIN EVALUATION====================\n\n")

        print("Train Loss: {:.2f}%".format(train_loss * 100))
        print("Train Accuracy: {:.2f}%".format(train_acc * 100))
        print("Train AUC: {:.2f}%".format(train_auc * 100))

        print("\n\n====================TEST EVALUATION====================\n\n")

        print("Test Loss: {:.2f}%".format(test_loss * 100))
        print("Test Accuracy: {:.2f}%".format(test_acc * 100))
        print("Test AUC: {:.2f}%".format(test_auc * 100))

        print("\n++++++FINAL TEST ACCURACY: {:.2f}%++++++".format(test_acc * 100))

        print("\n\n***********MODEL EVALUATION COMPLETE***********")

        return 


if __name__ == '__main__':
    ds_malaria, ds_info = tfds.load('malaria',with_info=True,as_supervised=True)

    config = MalariaPreprocessing(ds_malaria, ds_info)

    NN_eval = NeuralNetworkEval()
    NN_eval.evalNetwork(os.path.join(config.base_path, "malaria_model.h5"), config)
    