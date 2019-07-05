
class OptimizerPass(object):
    def __init__(self):
        pass

    def match(self, node):
        raise NotImplementedError
    
    def transform(self, model, node):
        raise NotImplementedError

optimizer_map = {}

def register_pass(name, opt_cls):
    if name in optimizer_map:
        raise Exception('Optimization pass {} already registered'.format(name))
    
    if type(name) in [list, tuple]:
        for n in name:
            optimizer_map[n] = opt_cls
    else:    
        optimizer_map[name] = opt_cls

def get_optimizer(name):
    return optimizer_map[name]()

def optimize_model(model, passes):
    optimizers = [get_optimizer(opt_pass) for opt_pass in passes]
    optimization_done = False
    while not optimization_done:
        for opt in optimizers:
            for node in model.graph.values():
                if opt.match(node):
                    res = opt.transform(model, node)
                    if res:
                        break
            else:
                continue
            break
        else:
            optimization_done = True

from passes.nop import EliminateLinearActivation
from passes.bnbinary import MergeBatchNormAndBinaryTanh, QuantizeBinaryDenseOutput
from passes.tnternary import MergeBatchNormAndTernaryTanh, QuantizeTernaryDenseOutput

register_pass('eliminate_linear_activation', EliminateLinearActivation)
register_pass('merge_batch_norm_binary_tanh', MergeBatchNormAndBinaryTanh)
register_pass('quantize_binary_dense_output', QuantizeBinaryDenseOutput)
register_pass('merge_batch_norm_ternary_tanh', MergeBatchNormAndTernaryTanh)
register_pass('quantize_ternary_dense_output', QuantizeTernaryDenseOutput)
