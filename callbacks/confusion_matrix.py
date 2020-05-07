import tensorflow as tf
import datetime
import os
import io
from matplotlib import pyplot as plt
import numpy as np
import itertools


class ConfusionMatrix(tf.keras.callbacks.Callback):
    def __init__(self, test_data, model, logdir):
        super().__init__()
        self.test_data = test_data
        self.model = model
        self.fw = tf.summary.create_file_writer(os.path.join(logdir, 'cm'))
        self.epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch

    def on_train_end(self, logs=None):
        log_confusion_matrix(self.model, self.test_data, self.fw, self.epoch)


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.title("Confusion matrix")
    tick_marks = np.arange(len(class_names))

    # Rotate the tick labels and set their alignment
    plt.xticks(tick_marks, class_names, rotation=45,
               ha="right", rotation_mode="anchor")
    plt.yticks(tick_marks, class_names, rotation=45,
               ha="right", rotation_mode="anchor")

    # Normalize the confusion matrix
    cm = np.around(cm.astype('float') / cm.sum(axis=1)
                   [:, np.newaxis], decimals=2)

    # Create colorbar
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.YlOrRd)
    plt.colorbar()

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return figure


def log_confusion_matrix(model, test_data, fw, epoch):
    # Use the model to predict the values from the validation dataset.
    test_predictions_raw = model.predict(test_data)
    test_predictions = np.argmax(test_predictions_raw, axis=1)

    test_labels = []
    gestures = {}

    # Extract relevant data
    for samples_batch, labels_batch in test_data.take(-1):
        gestures_batch = samples_batch['gesture']
        for gesture, label in zip(gestures_batch.numpy(), labels_batch.numpy()):
            gestures[label] = gesture.decode("utf-8")
            test_labels.append(label)

    class_names = [gestures[key] for key in sorted(gestures.keys())]

    # Calculate the confusion matrix.
    cm = tf.math.confusion_matrix(test_labels, test_predictions).numpy()

    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=class_names)
    cm_image = plot_to_image(figure)

    # Log the confusion matrix as an image summary.
    with fw.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)
