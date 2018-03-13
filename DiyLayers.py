import tensorflow as tf
from tensorlayer.layers import *

class AddLayer(Layer):
    def __init__(self,
                 layer=[],
                 name="add_layer"):
        Layer.__init__(self,name=name)
        self.inputs = []
        print("  [TL-David] AddLayer")
        for l in layer:
            self.inputs.append(l.outputs)
        self.outputs = tf.add(self.inputs[0],self.inputs[1])

        self.all_layers = list(layer[0].all_layers)
        self.all_params = list(layer[0].all_params)
        self.all_drop = dict(layer[0].all_drop)

        for i in range(1, len(layer)):
            self.all_layers.extend(list(layer[i].all_layers))
            self.all_params.extend(list(layer[i].all_params))
            self.all_drop.update(dict(layer[i].all_drop))

        self.all_layers = list_remove_repeat(self.all_layers)
        self.all_params = list_remove_repeat(self.all_params)


