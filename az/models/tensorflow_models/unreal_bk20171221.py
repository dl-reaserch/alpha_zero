import tensorflow as tf
import os
from az.models.tensorflow_models.tf_nn_util import NeuralNetBase, neuralnet
import numpy as np
import time
import json


@neuralnet
class PolicyValueNet(NeuralNetBase):
    # next_batch,input_dim=18,board=9,residual_filters =128,residual_blocks =6,

    def create_network(self, **kwargs):
        print("PolicyValueNet create network")
        defaults = {
            "board": 19,
            "filters_per_layer": 128,
            "layers": 12,
            "filter_width_1": 5,
            "residual_blocks": 6,
            "residual_filters": 128
        }
        params = defaults
        params.update(kwargs)

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
        # config = tf.ConfigProto(gpu_options=gpu_options)
        self.session = tf.Session()
        self.board = params['board']
        self.input_dim = 17
        self.filters_per_layer = params['filters_per_layer']
        self.layers = params['layers']
        self.residual_blocks = params['residual_blocks']
        self.residual_filters = params['residual_filters']
        # For expporting
        self.weights = []
        self.gpu_model = params['gpu_model']
        if self.gpu_model:
            self.data_format = 'NCHW'
        else:
            self.data_format = 'NHWC'
        # TF variables

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.X_ = tf.placeholder(tf.float32, [None, 17*self.board *self.board])
        self.Y_ = tf.placeholder(tf.float32, [None, self.board * self.board + 1])
        self.z_ = tf.placeholder(tf.float32, [None, 3])

        self.training = tf.placeholder(tf.bool)
        self.batch_norm_count = 0
        self.y_conv, self.z_conv = self.construct_net(self.X_)

        # Calculate loss on policy head
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y_, logits=self.y_conv)
        self.policy_loss = tf.reduce_mean(cross_entropy)

        # Loss on value head
        z_ = self.z_[0] * 1 + self.z_[1] * 0 + self.z_[2] * -1
        self.value_loss = tf.reduce_mean(tf.squared_difference(z_, self.z_conv))

        # Regularizer
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
        reg_variables = tf.trainable_variables()
        self.reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

        loss = self.value_loss - 1.0 * self.policy_loss + self.reg_term
        self.opt_op = tf.train.MomentumOptimizer(learning_rate=params['learning_rate'], momentum=params['momentum'],
                                                 use_nesterov=True)

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train_op = self.opt_op.minimize(loss, global_step=self.global_step)

        correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.Y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.policy_accuracy = tf.reduce_mean(correct_prediction)

        self.avg_policy_loss = None
        self.avg_value_loss = None
        self.avg_reg_term = None
        self.time_start = None

        self.test_writer = tf.summary.FileWriter(
            os.path.join(os.getcwd(), "smy_az_logs/test"), self.session.graph)

        self.train_writer = tf.summary.FileWriter(
            os.path.join(os.getcwd(), "smy_az_logs/train"), self.session.graph
        )
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.session.run(self.init)
        print("create network done!!")
        return self

    def estimate_value(self, X):
        return self.session.run(self.z_conv, feed_dict={self.X_: X})

    def estimate_policy(self, X):
        return self.session.run(self.y_conv, feed_dict={self.X_: X})

    def train_model(self, X, Y, z):
        policy_loss, mes_loss, reg_term, _ = self.session.run(
            [self.policy_loss, self.value_loss, self.reg_term, self.train_op],
            feed_dict={self.training: True, self.X_: X, self.Y_: Y, self.z_: z}
        )
        return policy_loss, mes_loss

    def calculate_policy_accuracy(self, X, Y):
        return self.session.run([self.policy_accuracy], feed_dict={self.training: False, self.X_: X, self.Y_: Y})

    def calculate_value_loss(self, X, Z):
        return self.session.run([self.value_loss], feed_dict={self.training: False, self.X_: X, self.Z_: Z})

    def get_global_step(self):
        return tf.train.global_step(self.session, self.global_step)

    def process(self, next_batches, model_json):
        with open(model_json, 'r') as f:
            object_specs = json.load(f)
        # Run training for this batch
        policy_loss, value_loss, reg_term, _ = self.train_model(X=next_batches[0], Y=next_batches[1], Z=next_batches[2])
        steps = self.get_global_step()
        # Keep running averages
        # XXX:use built-in support like tf.moving_average_variables?
        # Google's paper scales MSE by 1/4 to a [0, 1] range, so do the same to
        # get comparable values.
        value_loss = value_loss / 4.0
        decay = 0.99
        if self.avg_policy_loss:
            self.avg_policy_loss = decay * self.avg_policy_loss + (1 - decay) * policy_loss
        else:
            self.avg_policy_loss = policy_loss
        if self.avg_value_loss:
            self.avg_value_loss = decay * self.avg_value_loss + (1 - decay) * value_loss
        else:
            self.avg_value_loss = value_loss
        if self.avg_reg_term:
            self.avg_reg_term = decay * self.avg_reg_term + (1 - decay) * reg_term
        else:
            self.avg_reg_term = reg_term

        if steps % 100 == 0:
            time_end = time.time()
            speed = 0
            if self.time_start:
                elapsed = time_end - self.time_start
                speed = len(next_batches[0]) * (100.0 / elapsed)
            print("step {}, policy loss={:g} mse={:g} reg={:g} ({:g} pos/s)".format(
                steps, self.avg_policy_loss, self.avg_mse_loss, self.avg_reg_term, speed))
            train_summaries = tf.Summary(value=[
                tf.Summary.Value(tag="Avg Policy Loss", simple_value=self.avg_policy_loss),
                tf.Summary.Value(tag="Avg Value Loss", simple_value=self.avg_value_loss)
            ])
            self.train_writer.add_summary(train_summaries, steps)
            self.time_start = time_end
        if steps % 2000 == 0:
            sum_policy_accuracy = 0
            sum_value_loss = 0
            for _ in range(0, 10):
                summaried_policy_accuracy = self.calculate_policy_accuracy(X=next_batches[0],Y=next_batches[1])
                summaried_value_loss = self.calculate_value_loss(X=next_batches[0],Z=next_batches[2])
                sum_policy_accuracy += summaried_policy_accuracy
                sum_value_loss += summaried_value_loss
            sum_policy_accuracy /= 10.0
            # Additionally rescale to [0, 1] so divide by 4
            sum_value_loss /= (4.0 * 10.0)
            test_summaries = tf.Summary(value=[
                tf.Summary.Value(tag="Sum_Policy_Accuracy", simple_value=sum_policy_accuracy),
                tf.Summary.Value(tag="Sum_Value_Loss", simple_value=sum_value_loss)])
            self.test_writer.add_summary(test_summaries, steps)
            print("step {}, Sum_Policy_Accuracy={:g}%, Sum_Value_Loss={:g}".format(
                steps, sum_policy_accuracy * 100.0, sum_value_loss))

            save_path = self.save_ckpts(object_specs['az_ckpt_path'], steps)
            print("Model saved in file: {}".format(save_path))
            self.save_weights(object_specs['weights_path'])
            print("Leela weights saved to {}".format(object_specs['weights_path']))

    def save_weights(self, filename):
        with open(filename, "w") as file:
            # Version tag
            file.write("1")
            for weights in self.weights:
                # Newline unless last line (single bias)
                file.write("\n")
                work_weights = None
                # Keyed batchnorm weights
                if isinstance(weights, str):
                    work_weights = tf.get_default_graph().get_tensor_by_name(weights)
                elif weights.shape.ndims == 4:
                    # Convolution weights need a transpose
                    #
                    # TF (kYXInputOutput)
                    # [filter_height, filter_width, in_channels, out_channels]
                    #
                    # XiaoAgz/cuDNN/Caffe (kOutputInputYX)
                    # [output, input, filter_size, filter_size]
                    work_weights = tf.transpose(weights, [3, 2, 0, 1])
                elif weights.shape.ndims == 2:
                    # Fully connected layers are [in, out] in TF
                    #
                    # [out, in] in Leela
                    #
                    work_weights = tf.transpose(weights, [1, 0])
                else:
                    # Biases, batchnorm etc
                    work_weights = weights
                nparray = work_weights.eval(session=self.session)
                wt_str = [str(wt) for wt in np.ravel(nparray)]
                file.write(" ".join(wt_str))

    def get_batchnorm_key(self):
        result = "bn" + str(self.batch_norm_count)
        self.batch_norm_count += 1
        return result

    def conv_block(self, inputs, filter_size, input_channels, output_channels):
        W_conv = self.weight_variable([filter_size, filter_size, input_channels, output_channels])
        b_conv = self.bn_bias_variable([output_channels])
        self.weights.append(W_conv)
        self.weights.append(b_conv)
        # The weights are internal to the batchnorm layer, so apply
        # a unique scope that we can store, and use to look them back up
        # later on.
        weight_key = self.get_batchnorm_key()

        self.weights.append(weight_key + "/batch_normalization/moving_mean:0")
        self.weights.append(weight_key + "/batch_normalization/moving_variance:0")

        with tf.variable_scope(weight_key):
            if tf.get_variable_scope().name == weight_key + "/batch_normalization":
                reuse = True
            else:
                reuse = None
            h_bn = tf.layers.batch_normalization(
                tf.nn.bias_add(self.conv2d(inputs, W_conv
                                           ), b_conv, data_format=self.data_format),
                epsilon=1e-5, axis=1, reuse=reuse, center=False, scale=False, training=self.training
            )
        h_conv = tf.nn.relu(h_bn)
        return h_conv

    def residual_block(self, inputs, channels):
        # First convnet
        orig = tf.identity(inputs)
        W_conv_1 = self.weight_variable([3, 3, channels, channels])
        b_conv_1 = self.bn_bias_variable([channels])
        self.weights.append(W_conv_1)
        self.weights.append(b_conv_1)
        weight_key_1 = self.get_batchnorm_key()
        self.weights.append(weight_key_1 + "/batch_normalization/moving_mean:0")
        self.weights.append(weight_key_1 + "/batch_normalization/moving_variance:0")

        # Second convnet
        W_conv_2 = self.weight_variable([3, 3, channels, channels])
        b_conv_2 = self.bn_bias_variable([channels])
        self.weights.append(W_conv_2)
        self.weights.append(b_conv_2)
        weight_key_2 = self.get_batchnorm_key()
        self.weights.append(weight_key_2 + "/batch_normalization/moving_mean:0")
        self.weights.append(weight_key_2 + "/batch_normalization/moving_variance:0")

        with tf.variable_scope(weight_key_1):
            if tf.get_variable_scope().name == weight_key_1 + "/batch_normalization":
                reuse = True
            else:
                reuse = None
            h_bn1 = tf.layers.batch_normalization(
                tf.nn.bias_add(self.conv2d(inputs, W_conv_1), b_conv_1, data_format=self.data_format),
                epsilon=1e-5, axis=1, reuse=reuse, center=False, scale=False, training=self.training
            )
        h_out_1 = tf.nn.relu(h_bn1)

        with tf.variable_scope(weight_key_2):
            if tf.get_variable_scope().name == weight_key_2 + "/batch_normalization":
                reuse = True
            else:
                reuse = None
            h_bn2 = tf.layers.batch_normalization(
                tf.nn.bias_add(self.conv2d(h_out_1, W_conv_2), b_conv_2, data_format=self.data_format),
                epsilon=1e-5, axis=1, reuse=reuse, center=False, scale=False, training=self.training)

        h_out_2 = tf.nn.relu(tf.add(h_bn2, orig))
        return h_out_2

    def construct_net(self, planes):
        # Network structure

        # NCHW format
        # batch, 18 channels, 9 x 9
        x_planes = tf.reshape(planes, [-1, self.input_dim, self.board, self.board])
        # Input convolution
        flow = self.conv_block(x_planes,
                               filter_size=3,
                               input_channels=self.input_dim,
                               output_channels=self.residual_filters)
        # Residual tower
        for _ in range(0, self.residual_blocks):
            flow = self.residual_block(flow, self.residual_filters)

        # Policy head
        conv_pol = self.conv_block(flow, filter_size=1, input_channels=self.residual_filters, output_channels=2)
        h_conv_pol_flat = tf.reshape(conv_pol, [-1, 2 * self.board * self.board])
        W_fc1 = self.weight_variable([2 * self.board * self.board, self.board * self.board + 1])
        b_fc1 = self.bias_variable([self.board * self.board + 1])
        self.weights.append(W_fc1)
        self.weights.append(b_fc1)
        h_fc1 = tf.add(tf.matmul(h_conv_pol_flat, W_fc1), b_fc1)

        # Value head
        conv_val = self.conv_block(flow, filter_size=1, input_channels=self.residual_filters, output_channels=1)
        h_conv_val_flat = tf.reshape(conv_val, [-1, 1 * self.board * self.board])
        W_fc2 = self.weight_variable([1 * self.board * self.board, 128])
        b_fc2 = self.bias_variable([128])
        self.weights.append(W_fc2)
        self.weights.append(b_fc2)
        h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_conv_val_flat, W_fc2), b_fc2))
        W_fc3 = self.weight_variable([128, 3])
        b_fc3 = self.bias_variable([3])
        self.weights.append(W_fc3)
        self.weights.append(W_fc3)
        # h_fc3 = tf.nn.tanh(tf.add(tf.matmul(h_fc2, W_fc3), b_fc3))
        h_fc3 = tf.add(tf.matmul(h_fc2, W_fc3), b_fc3)

        e_h_fc3 = h_fc3[0] * 1 + h_fc3[1] * 0 + h_fc3[2] * -1

        return h_fc1, e_h_fc3

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.0, shape=shape)
        return tf.Variable(initial)

    # No point in learning bias weights as they are cancelled
    # out by the BatchNorm layers's mean adjustment.
    def bn_bias_variable(self, shape):
        initial = tf.constant(0.0, shape=shape)
        return tf.Variable(initial, trainable=False)

    # NCHW -> [batch, channels, height, width].
    # default NHWC -> [batch, height, width, channels].
    def conv2d(self, x, W):

        return tf.nn.conv2d(x, W, data_format=self.data_format, strides=[1, 1, 1, 1], padding='SAME')

