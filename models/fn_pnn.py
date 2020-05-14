from __future__ import absolute_import
import os
from .model import Model
from datasets import normalized_human_gestures as human_gestures
import random
from statistics import mean
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import kerastuner as kt
from datetime import datetime
from callbacks import ConfusionMatrix


class PNN(Model):
    def __init__(self):
        self.subject_paths = human_gestures.subject_paths
        self.feature_layer = human_gestures.get_feature_layer([
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
        ])

    def run_model(self, batch_size, epochs, hp=kt.HyperParameters()):
        subjects_accuracy = []

        for subject_index, subject_path in enumerate(self.subject_paths):
            k_fold = []
            result = []
            print(f'\nSubject {subject_index + 1}')

            for rep_index, _ in enumerate(next(os.walk(subject_path))[-1]):
                columns = self.build()
                other_subjects = [s for s in self.subject_paths if s != subject_path]
                random.shuffle(other_subjects)

                for column_index, column_subject_path in enumerate(other_subjects):
                    print(f'\nSubject {subject_index + 1}, rep {rep_index + 1}, column {column_index + 1}')
                    col_logdir = os.path.join(
                        'logs', '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"), f'pnn_column', f's{subject_index}', f'r{rep_index}', f'c{column_index}', f'cs{self.subject_paths.index(column_subject_path)}']))
                    col_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=col_logdir)

                    early_stop = tf.keras.callbacks.EarlyStopping(
                        monitor='val_accuracy', min_delta=0.0001, restore_best_weights=True,
                        patience=10)

                    col_data = human_gestures.get_data(
                        subject_path, rep_index, batch_size)
                    col_train, col_val, col_test = (d.map(lambda x,y: (self.feature_layer(x), y)) for d in col_data)

                    col_model = columns[column_index]['model']

                    # Check the trainable status of the individual layers
                    for layer in col_model.layers:
                        print(layer, layer.trainable)

                    col_model.fit(
                        col_train,
                        validation_data=col_val,
                        epochs=epochs,
                        callbacks=[early_stop, col_tensorboard]
                    )

                    # Freeze the layers
                    for layer in col_model.layers:
                        layer.trainable = False

                data = human_gestures.get_data(
                    subject_path, rep_index, batch_size)
                train, val, test = (d.map(lambda x,y: (self.feature_layer(x), y)) for d in data)

                logdir = os.path.join(
                    'logs', '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"), 'cpnn', f's{subject_index}', f'r{rep_index}']))
                tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logdir)

                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', min_delta=0.0001, restore_best_weights=True,
                    patience=10)

                model = columns[-1]['model']
                model.fit(
                    train,
                    validation_data=val,
                    epochs=epochs,
                    callbacks=[early_stop, tensorboard]
                )

                result = model.evaluate(test)
                k_fold.append(result[-1])

                savepath = '.'.join([logdir, 'h5'])
                model.save(savepath)

            average = mean(k_fold)
            print(f'\nmean accuracy: {average}')
            subjects_accuracy.append(average)

        total_average = mean(subjects_accuracy)

        return (total_average, subjects_accuracy)

    def build(self, hp=kt.HyperParameters()):
        exponent = hp.Int('exponent',
                          min_value=4,
                          max_value=10,
                          default=6,
                          step=1)
        adapter_exponent = hp.Int('exponent',
                          min_value=2,
                          max_value=6,
                          default=4,
                          step=1)
        dropout = hp.Float('dropout',
                           min_value=0.0,
                           default=0.2,
                           max_value=0.5,
                           step=0.1)

        print('Building columns: 0/20', end='', flush=False)

        # Input layer
        model_input = layers.Input(name='readings', shape=(15,), dtype='float32')
        columns = []

        for index, _ in enumerate(self.subject_paths):
            
            print(f'\rBuilding columns: {index+1}/20', end='', flush=True)
            column = {}
            # 1st hidden layer
            column['layer_1'] = layers.Dense(2**exponent, activation='relu')
            column['layer_1_output'] = column['layer_1'](model_input)
            
            column['dropout_1'] = layers.Dropout(dropout)
            column['dropout_1_output'] = column['dropout_1'](column['layer_1_output'])

            # 2nd hidden layer
            column['layer_1_adapters'] = [layers.Dense(2**adapter_exponent, activation='relu') for _ in range(index)]
            column['layer_1_adapters_output'] = [column['layer_1_adapters'][i](columns[i]['layer_1_output']) for i in range(index)]
            
            layer_2_input = layers.concatenate([column['dropout_1_output'], *column['layer_1_adapters_output']]) if column['layer_1_adapters_output'] else column['dropout_1_output']
            column['layer_2'] = layers.Dense(2**exponent, activation='relu')
            column['layer_2_output'] = column['layer_2'](layer_2_input)

            column['dropout_2'] = layers.Dropout(dropout)
            column['dropout_2_output'] = column['dropout_2'](column['layer_2_output'])

            # 3rd hidden layer
            column['layer_2_adapters'] = [layers.Dense(2**adapter_exponent, activation='relu') for _ in range(index)]
            column['layer_2_adapters_output'] = [column['layer_2_adapters'][i](columns[i]['layer_2_output']) for i in range(index)]

            layer_3_input = layers.concatenate([column['dropout_2_output'], *column['layer_2_adapters_output']]) if column['layer_2_adapters_output'] else column['dropout_2_output']
            column['layer_3'] = layers.Dense(2**exponent, activation='relu')
            column['layer_3_output'] = column['layer_3'](layer_3_input)

            column['dropout_3'] = layers.Dropout(dropout)
            column['dropout_3_output'] = column['dropout_3'](column['layer_3_output'])

            # Output layer
            column['layer_3_adapters'] = [layers.Dense(2**adapter_exponent, activation='relu') for _ in range(index)]
            column['layer_3_adapters_output'] = [column['layer_3_adapters'][i](columns[i]['layer_3_output']) for i in range(index)]
            output_layer_input = layers.concatenate([column['dropout_1_output'], *column['layer_1_adapters_output']]) if column['layer_3_adapters_output'] else column['dropout_3_output']

            column['output_layer'] = layers.Dense(18, activation='softmax')
            column['output'] = column['output_layer'](output_layer_input)
            
            column['model'] = keras.models.Model(inputs=model_input, outputs=column['output'])        
            column['model'].compile(
                optimizer=tf.keras.optimizers.Adam(
                hp.Choice('learning_rate',
                          values=[1e-2, 1e-3, 1e-4], default=1e-3)),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
            columns.append(column)

        return columns
