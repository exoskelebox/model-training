from __future__ import absolute_import
import os
from .model import Model
from human_gestures import HumanGestures
import random
from statistics import mean
import tensorflow as tf
import kerastuner as kt
from datetime import datetime
from callbacks import ConfusionMatrix
from utils.data_utils import split
import tqdm


class PNN(Model):
    def __init__(self):
        self.name = 'pnn'
        self.logdir = os.path.join(
            'logs', '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"), self.name]))

    def run_model(self, batch_size, epochs, hp=kt.HyperParameters()):
        subjects_accuracy = []
        summary_writer = tf.summary.create_file_writer(
            self.logdir)

        for subject_index, (subject_repetitions, remainder) in enumerate(HumanGestures(batch_size).subject_datasets(flatten_remainder=False), start=1):
            subject_logdir = os.path.join(self.logdir, f's{subject_index}')
            subject_summary_writer = tf.summary.create_file_writer(
                subject_logdir)
            k_fold = []
            result = []

            columns = self.build(hp=kt.HyperParameters(),
                                 num_columns=(1 + len(remainder)))

            print('Subject {}/{}'.format(subject_index, len(columns)))

            csi = [i for i in range(1, len(columns))]
            r = random.Random()
            state = r.getstate()
            r.shuffle(remainder)
            r.setstate(state)
            r.shuffle(csi)

            for column_index, column_repetitions in enumerate(remainder, start=1):
                col_val, col_train = next(column_repetitions)
                column_subject = csi[column_index-1] if csi[column_index -
                                                            1] < subject_index else csi[column_index-1] + 1

                print('Column {}/{}'.format(column_index, len(columns)))

                col_logdir = os.path.join(
                    subject_logdir, 'columns', '-'.join(['column', f'c{column_index}', f'cs{column_subject}']))
                col_tensorboard = tf.keras.callbacks.TensorBoard(
                    log_dir=col_logdir)

                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', min_delta=0.0001, restore_best_weights=True,
                    patience=10)

                col_model = columns[column_index]
                col_model.fit(
                    col_train.shuffle(2**14),
                    validation_data=col_val,
                    epochs=epochs,
                    callbacks=[early_stop, col_tensorboard]
                )

                # Freeze the layers
                for layer in col_model.layers[1:]:
                    layer.trainable = False

            col_weights = [col.get_weights() for col in columns]

            for rep_index, (val, train) in enumerate(subject_repetitions, start=1):
                rep_logdir = os.path.join(subject_logdir, f'r{rep_index}')
                rep_summary_writer = tf.summary.create_file_writer(rep_logdir)

                for index, weight in enumerate(col_weights):
                    columns[index].set_weights(weight)

                print('Repetition {}'.format(rep_index))

                tensorboard = tf.keras.callbacks.TensorBoard(
                    log_dir=rep_logdir)

                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', min_delta=0.0001, restore_best_weights=True,
                    patience=10)

                model = columns[-1]
                val, test = split(val)

                cm = ConfusionMatrix(test, model, rep_logdir)
                model.fit(
                    train.shuffle(2**14),
                    validation_data=val,
                    epochs=epochs,
                    callbacks=[early_stop, tensorboard, cm]
                )

                result = model.evaluate(test)
                k_fold.append(result[-1])

                with rep_summary_writer.as_default():
                    tf.summary.text('rep_accuracy', str(
                        result[-1]), step=rep_index)

                model.save(os.path.join(rep_logdir, 'model.h5'))

            average = mean(k_fold)
            print('\nmean_accuracy: {:.4f}'.format(average))
            subjects_accuracy.append(average)

            with subject_summary_writer.as_default():
                tf.summary.text('sub_accuracy', str(
                    average), step=subject_index)

        total_average = mean(subjects_accuracy)

        with summary_writer.as_default():
            tf.summary.text('model_accuracy', str(
                (total_average, subjects_accuracy)))

        return (total_average, subjects_accuracy)

    def build(self, hp=kt.HyperParameters(), num_columns=20):
        exponent = hp.Int('exponent',
                          min_value=4,
                          max_value=10,
                          default=6,
                          step=1)
        adapter_exponent = hp.Int('adapter_exponent',
                                  min_value=2,
                                  max_value=6,
                                  default=4,
                                  step=1)
        dropout = hp.Float('dropout',
                           min_value=0.0,
                           default=0.2,
                           max_value=0.5,
                           step=0.1)

        columns = []

        # Input layer
        inputs = HumanGestures.feature_inputs()

        feature_layer = HumanGestures.feature_layer(['readings'])(inputs)

        for i in tqdm.trange(num_columns, desc='Building columns'):

            # Hidden 1
            x = tf.keras.layers.Dense(
                2**exponent, activation='relu', name='dense_1_{}'.format(i))(feature_layer)
            x = tf.keras.layers.Dropout(0.2, name='dropout_1_{}'.format(i))(x)

            ada_x = [tf.keras.layers.Dense(2**adapter_exponent, activation='relu', name='adapter_1_{}_{}'.format(
                i, j))(columns[j].get_layer('dense_1_{}'.format(j)).output) for j in range(i)]

            x = tf.keras.layers.concatenate(
                [x, *ada_x], name='concat_1_{}'.format(i)) if ada_x else x

            # Hidden 2
            x = tf.keras.layers.Dense(
                2**exponent, activation='relu', name='dense_2_{}'.format(i))(x)
            x = tf.keras.layers.Dropout(0.2, name='dropout_2_{}'.format(i))(x)

            ada_x = [tf.keras.layers.Dense(2**adapter_exponent, activation='relu', name='adapter_2_{}_{}'.format(
                i, j))(columns[j].get_layer('dense_2_{}'.format(j)).output) for j in range(i)]

            x = tf.keras.layers.concatenate(
                [x, *ada_x], name='concat_2_{}'.format(i)) if ada_x else x

            # Output
            outputs = tf.keras.layers.Dense(
                18, activation='softmax', name='output_{}'.format(i))(x)

            model = tf.keras.models.Model(
                inputs=inputs.values(), outputs=outputs)

            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    hp.Choice('learning_rate',
                              values=[1e-2, 1e-3, 1e-4], default=1e-3)),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

            columns.append(model)

        return columns
