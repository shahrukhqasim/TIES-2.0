import tensorflow as tf





def get_num_parameters(scope=None):
    total_parameters = 0
    for variable in tf.trainable_variables(scope):
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters
