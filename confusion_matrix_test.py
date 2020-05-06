import os
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # nopep8
os.environ["KMP_AFFINITY"] = "noverbose"  # nopep8
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # nopep8

from datasets import normalized_human_gestures as human_gestures
import random
from statistics import mean
from textwrap import wrap
import re
import itertools
import tfplot
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(3)


def run_model(batch_size, reps, epoch):
    subjects_accuracy = []
    subject_paths = human_gestures.subject_paths

    for i in range(len(subject_paths)):
        k_fold = []
        result = []
        print(f'\nSubject {i + 1}/{len(subject_paths)}')

        for n in range(1):
            model = _model()

            train, val, test = human_gestures.get_data(
                human_gestures.subject_paths[8], n, batch_size)

            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', min_delta=0.0001, restore_best_weights=True,
                patience=10)

            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            model.fit(train,
                      validation_data=val,
                      epochs=epoch,
                      callbacks=[early_stop])
            
            plot_confusion_matrix(test=test, model=model)
        
            result = model.evaluate(test)
            k_fold.append(result[-1])

        average = mean(k_fold)
        print(f'\nmean accuracy: {average}')
        subjects_accuracy.append(average)

    total_average = mean(subjects_accuracy)

    return (total_average, subjects_accuracy)


def _model():
    return tf.keras.Sequential([
        human_gestures.get_feature_layer([
            # 'subject_gender',
            # 'subject_age',
            # 'subject_fitness',
            # 'subject_handedness',
            # 'subject_wrist_circumference',
            # 'subject_forearm_circumference',
            # 'repetition',
            'readings',
            # 'wrist_calibration_iterations',
            # 'wrist_calibration_values',
            # 'arm_calibration_iterations',
            # 'arm_calibration_values'
        ]),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(18, activation='softmax')
    ])


def plot_confusion_matrix(test,model):
    correct_labels = []

    for _, label_batch in test.take(-1):
        correct_labels.append(label_batch.numpy())
    
    correct_labels = tf.concat(correct_labels, axis=0)
    # print(test_labels)

    predicted_labels = model.predict_classes(test, batch_size=None)
    # print(predicted_label)

    con_matrix = tf.math.confusion_matrix(labels=correct_labels, predictions=predicted_labels, num_classes=18).numpy()
    con_matrix = con_matrix.astype('float') / con_matrix.sum(axis=1)[:, np.newaxis]



    fig, ax = plt.subplots()
    im = ax.imshow(con_matrix, interpolation='nearest', cmap=plt.cm.YlOrRd)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(con_matrix.shape[1]),
           yticks=np.arange(con_matrix.shape[0]),
        #    xticklabels=labels_name, yticklabels=labels_name,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")    

# Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = con_matrix.max() / 2.
    for i in range(con_matrix.shape[0]):
        for j in range(con_matrix.shape[1]):
            ax.text(j, i, format(con_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if con_matrix[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_model(1024, 1, 100)
