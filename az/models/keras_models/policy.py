from keras.layers import convolutional, Input
from keras.layers.core import Activation,Flatten
from keras.models import Sequential

from az.models.keras_models.nn_util import Bias,NeuralNetBase,neuralnet


@neuralnet
class CNNPolicy(NeuralNetBase):
    pass

    @staticmethod
    def create_network(**kwargs):
        """construct a convolutional neural network.

                Keword Arguments:
                - input_dim:             depth of features to be processed by first layer (no default)
                - board:                 width of the go board to be processed (default 19)
                - filters_per_layer:     number of filters used on every layer (default 128)
                - filters_per_layer_K:   (where K is between 1 and <layers>) number of filters
                                         used on layer K (default #filters_per_layer)
                - layers:                number of convolutional steps (default 12)
                - filter_width_K:        (where K is between 1 and <layers>) width of filter on
                                         layer K (default 3 except 1st layer which defaults to 5).
                                         Must be odd.
        """

        defaults = {
            "board":9,
            "filters_per_layer":128,
            "layers":12,
            "filter_width_1":5
        }
        # copy defaults, but override with anything in kwargs
        params = defaults
        params.update(kwargs)
        # create the network:
        # a series of zero-paddings followed by convolutions
        # such that the output dimensions are also board x board
        network = Sequential()
        # create first layer
        network.add(convolutional.Convolution2D(
            input_shape=(params["input_dim"],params['board'],params['board']),
            nb_filter = params.get('filters_per_layer_1',params['filters_per_layer']),
            nb_row = params['filter_width_1'],
            nb_col = params['filter_width_1'],
            init = 'uniform',
            activation='relu',
            border_mode='same'
        ))

        # create all other layers
        for i in range(2,params['layers']+1):
            # use filter_width_K if it is there, otherwise use 3
            filter_key = 'filter_width_%d'% i
            filter_width = params.get(filter_key,3)

            # use filters_per_layer_K if it is there, otherwise use default value
            filter_count_key = "filters_per_layer_%d" % i
            filter_nb = params.get(filter_count_key,params['filters_per_layer'])

            network.add(convolutional.Convolution2D(
                nb_filter = filter_nb,
                nb_row = filter_width,
                nb_col = filter_width,
                init = 'uniform',
                activation='relu',
                border_mode = 'same'
            ))

        # the last layer maps each <filters_per_layer> feature to a number
        network.add(convolutional.Convolution2D(
            nb_filter = 1,
            nb_row = 1,
            nb_col = 1,
            init = 'uniform',
            border_mode = 'same'
        ))
        # reshape output to be board x board
        network.add(Flatten())
        # add a bias to each board location
        network.add(Bias())
        network.add(Activation('softmax'))
        return network


@neuralnet
class ResnetPolicy(CNNPolicy):
    """Residual network architecture as per He at al. 2015
        """
    @staticmethod
    def create_network(**kwargs):
        """construct a convolutional neural network with Resnet-style skip connections.
                Arguments are the same as with the default CNNPolicy network, except the default
                number of layers is 20 plus a new n_skip parameter

                Keword Arguments:
                - input_dim:             depth of features to be processed by first layer (no default)
                - board:                 width of the go board to be processed (default 19)
                - filters_per_layer:     number of filters used on every layer (default 128)
                - layers:                number of convolutional steps (default 20)
                - filter_width_K:        (where K is between 1 and <layers>) width of filter on
                                        layer K (default 3 except 1st layer which defaults to 5).
                                        Must be odd.
                - n_skip_K:             (where K is as in filter_width_K) number of convolutional
                                        layers to skip with the linear path starting at K. Only valid
                                        at K >= 1. (Each layer defaults to 1)

                Note that n_skip_1=s means that the next valid value of n_skip_* is 3

                A diagram may help explain (numbers indicate layer):

                   1        2        3           4        5        6
                I--C--B--R--C--B--R--C--M--B--R--C--B--R--C--B--R--C--M  ...  M --R--F--O
                    \__________________/ \___________________________/ \ ... /
                        [n_skip_1 = 2]          [n_skip_3 = 3]

                I - input
                B - BatchNormalization
                R - ReLU
                C - Conv2D
                F - Flatten
                O - output
                M - merge

                The input is always passed through a Conv2D layer, the output of which
                layer is counted as '1'.  Each subsequent [R -- C] block is counted as
                one 'layer'. The 'merge' layer isn't counted; hence if n_skip_1 is 2,
                the next valid skip parameter is n_skip_3, which will start at the
                output of the merge

                """

        defaults = {
            "board": 9,
            "filters_per_layer": 128,
            "layers": 20,
            "filter_width_1": 5
        }
        # copy defaults, but override with anything in kwargs
        params = defaults
        params.update(kwargs)
        # create the network using Keras' functional API,
        # since this isn't 'Sequential'
        model_input = Input(shape=(params['input_dim'],params['board'],params['board']))
        # create first layer
        convolution_path = convolutional.Convolution2D(
            input_shape=(),
            nb_filter = params['filters_per_layer'],
            nb_row = params['filter_width_1'],
            nb_col = params['filter_width_1'],
            init = 'uniform',
            activation='linear',
            border_name = 'same'
        )(model_input)





