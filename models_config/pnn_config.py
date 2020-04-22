import tensorflow as tf
from models.pnn import PNN_Column, PNN_Model
from datasets import normalized_human_gestures as human_gestures

feature_layer = human_gestures.get_feature_layer([
    'subject_gender',
    'subject_age',
    'subject_fitness',
    'subject_handedness',
    'subject_wrist_circumference',
    'subject_forearm_circumference',
    'repetition',
    'readings',
    'wrist_calibration_iterations',
    'wrist_calibration_values',
    'arm_calibration_iterations',
    'arm_calibration_values'
])


def pnn_model(generation_index, columns):
    return PNN_Model(feature_layer=feature_layer, columns=columns)

def create_column(generation_index, layer_info):
    return PNN_Column(layer_info, generation=generation_index)

def pnn_config():
    adapters = {'type': tf.keras.layers.Dense, 'units': 16, 'activation': 'relu'}
    
    core = [
    {'type': tf.keras.layers.Dense, 'units': 64, 'activation': 'relu'},
    {'type': tf.keras.layers.Dense, 'units': 64, 'activation': 'relu'},
    {'type': tf.keras.layers.Dense, 'units': 18, 'activation': 'softmax'}]

    return {'core': core, 'adapters': adapters}