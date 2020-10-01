import hls4ml

from qkeras import *

__all__=['_output_shape','_layer_number','_alpha']

def _output_shape(prediction1,prediction2):
    return len(prediction1[0])==len(prediction2)

def _layer_number(model,hls_model):
    return len(model.layers) + 1 == len(hls_model.get_layers())

def _alpha(model,hls_model):
    return (list(hls_model.get_layers())[1].attributes['class_name'] == model.layers[1].__class__.__name__) & (list(hls_model.get_layers())[2].attributes['class_name'] == 'Alpha')