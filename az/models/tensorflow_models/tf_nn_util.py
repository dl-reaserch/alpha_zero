import os
import numpy as np
import time
import tensorflow as tf
import json
from az.preprocessing.preprocessing import Preprocess


class NeuralNetBase(object):
    """Base class for neural network classes handling feature processing, construction
        of a 'forward' function, etc.
        """

    # keep track of subclasses to make generic saving/loading cleaner.
    # subclasses can be 'registered' with the @neuralnet decorator
    subclasses = {}

    def __init__(self,feature_list,model_config, **kwargs):
        """create a neural net object that preprocesses according to feature_list and uses
        a neural network specified by keyword arguments (using subclass' create_network())

        optional argument: init_network (boolean). If set to False, skips initializing
        self.model and self.forward and the calling function should set them.
        """
        self.preprocessor = Preprocess(feature_list)
        kwargs['input_dim'] = self.preprocessor.output_dim
        if kwargs.get('init_network', True):
            # self.__class__ refers to the subclass so that subclasses only
            # need to override create_network()

            self.model = self.create_network(**kwargs)
            self._init_conf(model_config)
            self.saver.save(self.session, model_config.get('ckpt_path'), global_step=self.get_global_step())

            # self.model = self.__class__.create_network(**kwargs)

    def _init_conf(self,model_config):
        object_specs = {
            'class': self.__class__.__name__,
            'az_ckpt_path': os.path.dirname(model_config.get('ckpt_path')),
            'feature_list': self.preprocessor.feature_list,
            'weights_path': model_config.get('weights_path')
        }
        with open(model_config.get('model_file'), 'w') as f:
            json.dump(object_specs, f)

    def save_ckpts(self, file_path, steps):
        self.saver.save(self.session, file_path,global_step=steps)


    def load_ckpts(self, file_path):
        self.saver.restore(self.session, tf.train.latest_checkpoint(file_path))

def neuralnet(cls):
    """Class decorator for registering subclasses of NeuralNetBase
        """
    NeuralNetBase.subclasses[cls.__name__] = cls
    return cls
