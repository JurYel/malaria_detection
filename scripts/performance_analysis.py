import numpy as np
import tensorflow as tf
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras

class PerformanceAnalysis():
    def __init__(self, base_path="models/malaria_detection/", 
                classes=['parasitized','uninfected'], conv_model=None):
        self.base_path = base_path
        self.classes = classes
        self.conv_model = conv_model

    def model_plot_performance(self, model_hist, style='ggplot'):
        accuracy = model_hist['accuracy']
        val_accuracy = model_hist['val_accuracy']
        loss = model_hist['loss']
        val_loss = model_hist['val_loss']
        auc = model_hist['AUC']
        val_auc = model_hist['val_AUC']
 
        num_epochs = range(1, len(accuracy) + 1)

        plt.style.use(style)

        fig, ax = plt.subplots(1, 3, figsize=(10, 12))

        line1 = ax[0].plot(accuracy, label='Train Accuracy')
        line2 = ax[0].plot(val_accuracy, label='Test Accuracy')
        plt.setp(line1, linewidth=2.0)
        plt.setp(line2, linewidth=2.0)
        ax[0].set_title("Model Accuracy")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Accuracy")

        line1 = ax[1].plot(loss, label='Train Loss')
        line2 = ax[2].plot(val_loss, label='Test Loss')
        plt.setp(line1, linewidth=2.0)
        plt.setp(line2, linewidth=2.0)
        ax[1].set_title("Model Loss")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Loss")

        line1 = ax[2].plot(auc, label='Train AUC')
        line2 = ax[2].plot(val_auc, label='Test AUC')
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
        return

    def display_cm(self, y_pred):
        print("="*12, end='')
        print("Confusion Matrix", end='')
        print("="*12, end='\n\n')

        print(confusion_matrix(self.classes, y_pred))

        # confusion matrix with visualization
        plt.figure(figsize=(10, 12))
        cm = confusion_matrix(self.classes, y_pred)
        cm = np.around(cm.astype('float') / np.sum(axis=1)[:, np.newaxis], decimals=2)

        plt.imshow(cm, interpolation='nearest')
        plt.title("Confusion Matrix")
        plt.colorbar()

        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=90)
        plt.yticks(tick_marks, self.classes)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        return

    def display_classification_report(self, y_pred):
        print("="*12, end='')
        print("Classification Report", end='')
        print("="*12, end='\n\n')
        print(classification_report(self.classes, y_pred, target_names=classes))

        return