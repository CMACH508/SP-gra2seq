#   Copyright 2022 Sicong Zang
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#   P.S. We thank Guoyao Su, Yonggang Qi and et al. for the codes of sketch cropping
#        in https://github.com/sgybupt/SketchHealer.
#
"""SP-gra2seq model structure file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import rnn
import Cnn

class Model(object):
    def __init__(self, hps, reuse=tf.AUTO_REUSE):
        self.hps = hps
        # tf.set_random_seed(1024)
        with tf.variable_scope('SP-gra2seq', reuse=reuse):
            self.config_model()
            self.build_RPCLVQ()

    def config_model(self):
        """ Model configuration """
        self.global_ = tf.get_variable(name='num_of_steps', shape=[], initializer=tf.ones_initializer(dtype=tf.float32), trainable=False)
        self.input_seqs = tf.placeholder(tf.float32, [self.hps.batch_size, self.hps.max_seq_len + 1, 5], name="input_seqs")
        self.input_graphs = tf.placeholder(tf.float32, [self.hps.batch_size, self.hps.graph_number, self.hps.graph_picture_size,
                                                        self.hps.graph_picture_size, 1], name="input_graphs")
        self.input_masks = tf.placeholder(tf.float32, [self.hps.batch_size, self.hps.graph_number - 1, self.hps.graph_number - 1], name="input_masks")
        self.sequence_lengths = tf.placeholder(tf.int32, [self.hps.batch_size], name="seq_len")
        # Cluster centroids
        self.de_mu = tf.get_variable(name="latent_mu", shape=[self.hps.num_mixture, 512],
                                     initializer=tf.random_uniform_initializer(minval=0., maxval=5., dtype=tf.float32), trainable=False)

        # Decoder used data (input_x => rnn => output_x)
        self.input_x = tf.identity(self.input_seqs[:, :self.hps.max_seq_len, :], name='input_x')
        self.output_x = self.input_seqs[:, 1:self.hps.max_seq_len + 1, :]

        # Decoder cell configuration
        if self.hps.dec_model == 'lstm':
            cell_fn = rnn.LSTMCell
        elif self.hps.dec_model == 'layer_norm':
            cell_fn = rnn.LayerNormLSTMCell
        elif self.hps.dec_model == 'hyper':
            cell_fn = rnn.HyperLSTMCell
        else:
            assert False, 'please choose a respectable cell'

        # Dropout configuration
        use_input_dropout = False if self.hps.use_input_dropout == 0 else True
        use_output_dropout = False if self.hps.use_output_dropout == 0 else True
        use_recurrent_dropout = False if self.hps.use_recurrent_dropout == 0 else True

        cell = cell_fn(
              self.hps.dec_rnn_size,
              use_recurrent_dropout=use_recurrent_dropout,
              dropout_keep_prob=self.hps.recurrent_dropout_prob)

        if use_input_dropout:
            tf.logging.info('Dropout to input w/ keep_prob = %4.4f.', self.hps.input_dropout_prob)
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.hps.input_dropout_prob)

        if use_output_dropout:
            tf.logging.info('Dropout to output w/ keep_prob = %4.4f.', self.hps.output_dropout_prob)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.hps.output_dropout_prob)

        self.cell = cell

    def build_RPCLVQ(self):
        target = tf.reshape(self.output_x, [-1, 5])
        self.x1_data, self.x2_data, self.pen_data = tf.split(target, [1, 1, 3], 1)

        with tf.variable_scope('encoders') as enc_param_scope:
            self.p_mu, self.p_sigma2, self.hyper_mask, self.component_z, self.p_alpha, self.q_mu = self.gcn_encoder(self.input_graphs)
            self.batch_z = self.get_z(self.p_mu, self.p_sigma2)

        with tf.variable_scope('decoder') as dec_param_scope:
            fc_spec = [('tanh', self.hps.z_size, self.cell.state_size, 'init_state')]
            fc_net = Cnn.FcNet(fc_spec, self.batch_z)
            self.initial_state = fc_net.fc_layers[-1]

            dec_input = tf.concat([self.input_x, tf.tile(tf.expand_dims(self.batch_z, axis=1), [1, self.hps.max_seq_len, 1])], axis=2)

            self.dec_out, self.final_state = self.rnn_decoder(dec_input, self.initial_state)
            self.pi, self.mux, self.muy, self.sigmax, self.sigmay, self.corr, self.pen, self.pen_logits = self.dec_out

        # Loss function
        self.gaussian_loss = self.calculate_gaussian_loss(self.p_alpha, self.component_z, tf.stop_gradient(self.q_mu), tf.stop_gradient(self.hyper_mask))
        self.lil_loss = self.get_lil_loss(self.pi, self.mux, self.muy, self.sigmax, self.sigmay, self.corr,
                                          self.pen_logits, self.x1_data, self.x2_data, self.pen_data)
        self.loss = self.lil_loss + 0.25 * self.gaussian_loss
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.lr = (self.hps.learning_rate - self.hps.min_learning_rate) * \
                      (self.hps.decay_rate ** self.global_) + self.hps.min_learning_rate
            optimizer = tf.train.AdamOptimizer(self.lr)
            gvs = optimizer.compute_gradients(self.loss)

            g = self.hps.grad_clip
            for i, (grad, var) in enumerate(gvs):
                if grad is not None:
                    gvs[i] = (tf.clip_by_norm(grad, g), var)
            self.train_op = optimizer.apply_gradients(gvs)

            # Update the GMM parameters
            self.update_gmm = tf.assign(self.de_mu, self.q_mu)

    def gcn_encoder(self, graphs):
        graphs = tf.reshape(graphs, [-1, self.hps.graph_picture_size, self.hps.graph_picture_size, 1])
        conv_specs = [
            ('relu', (2, 2), [1, 2, 2, 1], 8),
            ('relu', (2, 2), [1, 2, 2, 1], 32),
            ('relu', (2, 2), [1, 2, 2, 1], 64),
            ('relu', (2, 2), [1, 2, 2, 1], 128),
            ('relu', (2, 2), [1, 2, 2, 1], 256),
            ('relu', (2, 2), [1, 2, 2, 1], 512),
            ('no', (2, 2), [1, 2, 2, 1], 512),
        ]
        cn1 = Cnn.ConvNet(conv_specs, graphs, self.hps.is_training)
        cn1_out = tf.nn.max_pool(cn1.conv_layers[-1], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # batch_size * 21, 1, 1, 512
        whole_cn1_out, part_cn1_out = tf.split(tf.reshape(cn1_out, [self.hps.batch_size, self.hps.graph_number, 512]),
                                               [1, self.hps.graph_number - 1], axis=1)

        # Compute the synonymous proximity
        r = tf.reduce_sum(tf.tile(tf.expand_dims(self.de_mu, axis=0), [self.hps.batch_size * (self.hps.graph_number - 1), 1, 1])
                          * tf.tile(tf.reshape(part_cn1_out, [self.hps.batch_size * (self.hps.graph_number - 1), 1, 512]),
                                    [1, self.hps.num_mixture, 1]), axis=2) \
            / (tf.norm(tf.tile(tf.expand_dims(self.de_mu, axis=0), [self.hps.batch_size * (self.hps.graph_number - 1), 1, 1]), axis=2)
               * tf.norm(tf.tile(tf.reshape(part_cn1_out, [self.hps.batch_size * (self.hps.graph_number - 1), 1, 512]),
                                 [1, self.hps.num_mixture, 1]), axis=2) + 1e-30)
        p_alpha = tf.one_hot(tf.argmax(r, axis=1), self.hps.num_mixture, axis=1)
        # Create adjacency matrix
        coff = tf.reduce_sum(tf.tile(tf.expand_dims(part_cn1_out, axis=1), [1, self.hps.graph_number - 1, 1, 1])
                             * tf.tile(tf.expand_dims(part_cn1_out, axis=2), [1, 1, self.hps.graph_number - 1, 1]), axis=3) \
               / (tf.norm(tf.tile(tf.expand_dims(part_cn1_out, axis=1), [1, self.hps.graph_number - 1, 1, 1]), axis=3)
                  * tf.norm(tf.tile(tf.expand_dims(part_cn1_out, axis=2), [1, 1, self.hps.graph_number - 1, 1]), axis=3) + 1e-30)

        winner_1 = tf.one_hot(tf.argmax(coff, axis=2), self.hps.graph_number - 1, axis=2)
        winner_2 = tf.one_hot(tf.argmax(coff - coff * winner_1, axis=2), self.hps.graph_number - 1, axis=2)
        winner_3 = tf.one_hot(tf.argmax(coff - coff * (winner_1 + winner_2), axis=2), self.hps.graph_number - 1, axis=2)
        reg_coff = coff * (winner_1 + winner_2 * 0.5 + winner_3 * 0.2)

        slide_mask, _ = tf.split(self.input_masks, [1, self.hps.graph_number - 2], axis=2)
        hyper_mask = tf.reshape(slide_mask, [self.hps.batch_size * (self.hps.graph_number - 1), 1])
        masked_coff = tf.concat([tf.concat([tf.ones([self.hps.batch_size, 1, 1]), tf.zeros([self.hps.batch_size, 1, self.hps.graph_number - 1])], axis=2),
                                 tf.concat([0.5 * slide_mask, reg_coff * self.input_masks], axis=2)], axis=1)
        coff_adjs = masked_coff / (tf.tile(tf.reduce_sum(masked_coff, axis=2, keepdims=True), [1, 1, self.hps.graph_number]) + 1e-10)

        # Update cluster centroids
        q_mu_new = tf.reduce_sum(tf.tile(tf.expand_dims(tf.reshape(part_cn1_out, [self.hps.batch_size * (self.hps.graph_number - 1), 512]),
                                                        axis=1), [1, self.hps.num_mixture, 1])
                                 * tf.tile(tf.expand_dims(p_alpha * tf.tile(hyper_mask, [1, self.hps.num_mixture]), axis=2), [1, 1, 512]), axis=0) \
                   / (tf.tile(tf.expand_dims(tf.reduce_sum(p_alpha * tf.tile(hyper_mask, [1, self.hps.num_mixture]), axis=0), axis=1), [1, 512]) + 1e-10)
        q_mu = self.de_mu * 0.75 + q_mu_new * 0.25

        bn1_out = tf.contrib.layers.batch_norm(tf.nn.relu(cn1_out),
                                               decay=0.9, epsilon=1e-05, center=True, scale=True,
                                               updates_collections=None, is_training=self.hps.is_training)

        def kaiming_init(size, activate):
            in_dim = size[0]
            if activate == 'relu':
                kaiming_stddev = tf.sqrt(2. / in_dim)
            elif activate == 'tanh':
                kaiming_stddev = tf.sqrt(5. / (3. * in_dim))
            elif activate == 'sigmoid':
                kaiming_stddev = tf.sqrt(1. / in_dim)
            else:
                kaiming_stddev = tf.sqrt(1. / in_dim)
            return tf.random_normal(shape=size, stddev=kaiming_stddev)

        weight_1 = tf.get_variable(name="gcn_weight_1", dtype=tf.float32, initializer=kaiming_init([512, 512], 'relu'))
        weight_2 = tf.get_variable(name="gcn_weight_2", dtype=tf.float32, initializer=kaiming_init([512, 512], 'relu'))

        gcn_out_0 = tf.reshape(tf.matmul(coff_adjs, tf.reshape(bn1_out, [-1, self.hps.graph_number, 512])), [-1, 512])
        gcn_out_1 = tf.nn.relu(tf.matmul(gcn_out_0, weight_1))
        gcn_out_2 = tf.nn.relu(tf.matmul(gcn_out_1, weight_2))
        gcn_out = tf.reduce_sum(tf.reshape(gcn_out_0 + gcn_out_2, [self.hps.batch_size, self.hps.graph_number, 512]), axis=1)
        bn2_out = tf.nn.tanh(tf.contrib.layers.batch_norm(gcn_out, decay=0.9, epsilon=1e-05, center=True, scale=True,
                                                          updates_collections=None, is_training=self.hps.is_training))

        fc_spec_mu = [('no', 512, self.hps.z_size, 'fc_mu')]
        fc_net_mu = Cnn.FcNet(fc_spec_mu, bn2_out)
        p_mu = fc_net_mu.fc_layers[-1]

        fc_spec_sigma2 = [('no', 512, self.hps.z_size, 'fc_sigma2')]
        fc_net_sigma2 = Cnn.FcNet(fc_spec_sigma2, bn2_out)
        p_sigma2 = fc_net_sigma2.fc_layers[-1]

        return p_mu, p_sigma2, hyper_mask, tf.reshape(part_cn1_out, [self.hps.batch_size * (self.hps.graph_number - 1), 512]), p_alpha, q_mu

    def rnn_decoder(self, inputs, initial_state):
        # Number of outputs is end_of_stroke + prob + 2 * (mu + sig) + corr
        num_mixture = 20
        n_out = 3 + num_mixture * 6

        with tf.variable_scope('decoder'):
            output, last_state = tf.nn.dynamic_rnn(
                self.cell,
                inputs,
                initial_state=initial_state,
                time_major=False,
                swap_memory=True,
                dtype=tf.float32)

            output = tf.reshape(output, [-1, self.hps.dec_rnn_size])
            fc_spec = [('no', self.hps.dec_rnn_size, n_out, 'fc')]
            fc_net = Cnn.FcNet(fc_spec, output)
            output = fc_net.fc_layers[-1]

            out = self.get_mixture_params(output)
            last_state = tf.identity(last_state, name='last_state')
            self.output = output
        return out, last_state

    def get_z(self, mu, sigma2):
        """ Reparameterization """
        sigma = tf.exp(sigma2 / 2)
        eps = tf.random_normal((self.hps.batch_size, self.hps.z_size), 0.0, 1.0, dtype=tf.float32)
        z = tf.add(mu, tf.multiply(sigma, eps), name='z_code')
        return z

    def calculate_gaussian_loss(self, p_alpha, p_mu, q_mu, mask):
        p_mu = tf.tile(tf.reshape(p_mu, [self.hps.batch_size * (self.hps.graph_number - 1), 1, 512]), [1, self.hps.num_mixture, 1])
        q_mu = tf.tile(tf.reshape(q_mu, [1, self.hps.num_mixture, 512]), [self.hps.batch_size * (self.hps.graph_number - 1), 1, 1])
        return tf.reduce_sum(0.5 * p_alpha * tf.tile(mask, [1, self.hps.num_mixture]) * tf.reduce_sum((p_mu - q_mu) ** 2, axis=2)) \
               / tf.reduce_sum(mask)

    def get_density(self, x1, x2, mu1, mu2, s1, s2, rho):
        norm1 = tf.subtract(x1, mu1)
        norm2 = tf.subtract(x2, mu2)
        s1s2 = tf.multiply(s1, s2)
        z = (tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) -
             2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2))
        neg_rho = 1 - tf.square(rho)
        result = tf.exp(tf.div(-z, 2 * neg_rho))
        denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(neg_rho))
        result = tf.div(result, denom)
        return result

    def get_lil_loss(self, pi, mu1, mu2, s1, s2, corr, pen_logits, x1_data, x2_data, pen_data):
        result0 = self.get_density(x1_data, x2_data, mu1, mu2, s1, s2, corr)
        epsilon = 1e-6
        result1 = tf.multiply(result0, pi)
        result1 = tf.reduce_sum(result1, axis=1, keep_dims=True)
        result1 = -tf.log(result1 + epsilon)  # Avoid log(0)

        masks = 1.0 - pen_data[:, 2]
        masks = tf.reshape(masks, [-1, 1])
        result1 = tf.multiply(result1, masks)

        result2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pen_logits, labels=pen_data)
        result2 = tf.reshape(result2, [-1, 1])

        if not self.hps.is_training:
            result2 = tf.multiply(result2, masks)
        return tf.reduce_sum(tf.reshape(result1 + result2, [self.hps.batch_size, -1])) / self.hps.batch_size

    def get_mixture_params(self, output):
        pen_logits = output[:, 0:3]
        pi, mu1, mu2, sigma1, sigma2, corr = tf.split(output[:, 3:], 6, 1)

        pi = tf.nn.softmax(pi)
        pen = tf.nn.softmax(pen_logits)

        sigma1 = tf.exp(sigma1)
        sigma2 = tf.exp(sigma2)
        corr = tf.tanh(corr)

        r = [pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits]
        return r
