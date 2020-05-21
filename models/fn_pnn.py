from __future__ import absolute_import
import os
from .model import Model
from human_gestures import HumanGestures
import random
from statistics import mean
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import kerastuner as kt
from datetime import datetime
from callbacks import ConfusionMatrix
from utils.data_utils import split
import tqdm


class PNN(Model):
    def __init__(self):
        self.name = 'pnn'

    def run_model(self, batch_size, epochs, hp=kt.HyperParameters()):
        subjects_accuracy = []
        log_folder = os.path.join(
            'logs', '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"), f'pnn']))

        for subject_index, (subject_repetitions, remainder) in enumerate(HumanGestures(batch_size).subject_datasets(flatten_remainder=False), start=1):
            subject_logdir = os.path.join(log_folder, f's{subject_index}')
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

                col_model = columns[column_index]['model']
                col_model.fit(
                    col_train.shuffle(2**14),
                    validation_data=col_val,
                    epochs=epochs,
                    callbacks=[early_stop, col_tensorboard]
                )

                # Freeze the layers
                for layer in col_model.layers[1:]:
                    layer.trainable = False

            col_weights = [col['model'].get_weights() for col in columns]

            for rep_index, (val, train) in enumerate(subject_repetitions, start=1):

                for index, weight in enumerate(col_weights):
                    columns[index]['model'].set_weights(weight)

                print('Repetition {}'.format(rep_index))

                rep_logdir = os.path.join(subject_logdir, f'r{rep_index}')

                tensorboard = tf.keras.callbacks.TensorBoard(
                    log_dir=rep_logdir)

                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', min_delta=0.0001, restore_best_weights=True,
                    patience=10)

                model = columns[-1]['model']
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

                model.save(os.path.join(rep_logdir, 'model.h5'))

            average = mean(k_fold)
            print('\nmean_accuracy: {:.4f}'.format(average))
            subjects_accuracy.append(average)

            subject_average = tf.summary.create_file_writer(
                os.path.join(subject_logdir, 'subject_average'))
            with subject_average.as_default():
                tf.summary.text(f"subject_{subject_index}_average", str(
                    subjects_accuracy), step=0)

        total_average = mean(subjects_accuracy)

        model_average = tf.summary.create_file_writer(
            os.path.join(log_folder, 'model_average'))
        with model_average.as_default():
            tf.summary.text(f"model_average", str(
                (total_average, subjects_accuracy)), step=1)

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

        feature_layer = HumanGestures().feature_layer([
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
        ])(inputs)

        for index in tqdm.trange(num_columns, desc='Building columns'):
            column = {}
            # 1st hidden layer
            column['layer_1'] = layers.Dense(2**exponent, activation='relu')
            column['layer_1_output'] = column['layer_1'](feature_layer)

            column['dropout_1'] = layers.Dropout(dropout)
            column['dropout_1_output'] = column['dropout_1'](
                column['layer_1_output'])

            # 2nd hidden layer
            column['layer_1_adapters'] = [layers.Dense(
                2**adapter_exponent, activation='relu') for _ in range(index)]
            column['layer_1_adapters_output'] = [column['layer_1_adapters'][i](
                columns[i]['layer_1_output']) for i in range(index)]

            layer_2_input = layers.concatenate([column['dropout_1_output'], *column['layer_1_adapters_output']]
                                               ) if column['layer_1_adapters_output'] else column['dropout_1_output']
            column['layer_2'] = layers.Dense(2**exponent, activation='relu')
            column['layer_2_output'] = column['layer_2'](layer_2_input)

            column['dropout_2'] = layers.Dropout(dropout)
            column['dropout_2_output'] = column['dropout_2'](
                column['layer_2_output'])
            # 3rd hidden layer
            column['layer_2_adapters'] = [layers.Dense(
                2**adapter_exponent, activation='relu') for _ in range(index)]
            column['layer_2_adapters_output'] = [column['layer_2_adapters'][i](
                columns[i]['layer_2_output']) for i in range(index)]

            layer_3_input = layers.concatenate([column['dropout_2_output'], *column['layer_2_adapters_output']]
                                               ) if column['layer_2_adapters_output'] else column['dropout_2_output']

            """
            column['layer_3'] = layers.Dense(2**exponent, activation='relu')
            column['layer_3_output'] = column['layer_3'](layer_3_input)

            column['dropout_3'] = layers.Dropout(dropout)
            column['dropout_3_output'] = column['dropout_3'](
                column['layer_3_output'])

            # Output layer
            column['layer_3_adapters'] = [layers.Dense(
                2**adapter_exponent, activation='relu') for _ in range(index)]
            column['layer_3_adapters_output'] = [column['layer_3_adapters'][i](
                columns[i]['layer_3_output']) for i in range(index)]
            output_layer_input = layers.concatenate(
                [column['dropout_3_output'], *column['layer_3_adapters_output']]) if column['layer_3_adapters_output'] else column['dropout_3_output']
            """

            column['output_layer'] = layers.Dense(18, activation='softmax')
            column['output'] = column['output_layer'](layer_3_input)

            column['model'] = keras.models.Model(
                inputs=inputs.values(), outputs=column['output'])
            column['model'].compile(
                optimizer=tf.keras.optimizers.Adam(
                    hp.Choice('learning_rate',
                              values=[1e-2, 1e-3, 1e-4], default=1e-3)),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
            columns.append(column)

        return columns
