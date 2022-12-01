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
"""SP-gra2seq generating process file."""

import random
import os
import json
import numpy as np
import tensorflow as tf
import utils
import glob
from PIL import Image
from seq2svg import draw_strokes
from model import Model
import scipy.misc
from scipy.linalg import block_diag
import re
from svg2png import exportsvg

def sample(sess, sample_model, z, gen_size=1, seq_len=250, temperature=0.24, greedy_mode=False):
    """ Sample a sequence from a pre-trained model """

    def adjust_pdf(pi_pdf, temp):
        """ Adjust the pdf of pi according to temperature """
        pi_pdf = np.log(pi_pdf) / temp
        pi_pdf -= pi_pdf.max()
        pi_pdf = np.exp(pi_pdf)
        pi_pdf /= pi_pdf.sum()
        return pi_pdf


    def get_pi_idx(x, pdf, temp=1.0, greedy=False):
        """ Sample from a pdf, optionally greedily """
        if greedy:
            return np.argmax(pdf)
        pdf = adjust_pdf(np.copy(pdf), temp)
        accumulate = 0
        for i in range(0, pdf.size):
            accumulate += pdf[i]
            if accumulate >= x:
                return i
        tf.logging.info('Error with sampling ensemble.')
        return -1


    def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0, greedy=False):
        """ Sample from a 2D Gaussian """
        if greedy:
            return mu1, mu2
        mean = [mu1, mu2]
        s1 *= temp * temp
        s2 *= temp * temp
        cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
        x = np.random.multivariate_normal(mean, cov, 1)
        return x[0][0], x[0][1]


    def get_seqs(z, seq_len, greedy, temp):
        """ Generate sequences according to latent vector """
        feed = {sample_model.batch_z: z}
        input_state = sess.run(sample_model.initial_state, feed)

        strokes = np.zeros((seq_len, len(z), 5), dtype=np.float32)
        input_x = np.zeros((len(z), 1, 5), dtype=np.float32)
        input_x[:, 0, 2] = 1  # initially, we want to see beginning of new stroke

        for seq_i in range(seq_len):
            feed = {sample_model.initial_state: input_state,
                    sample_model.input_x: input_x,
                    sample_model.batch_z: z
                    }

            dec_out, out_state = sess.run([sample_model.dec_out, sample_model.final_state], feed)

            pi, mux, muy, sigmax, sigmay, corr, pen, pen_logits = dec_out
            input_state = out_state

            # generate stroke position from Gaussian mixtures
            idx = get_pi_idx(random.random(), pi[0], temp, greedy)

            next_x1, next_x2 = sample_gaussian_2d(mux[0][idx], muy[0][idx],
                                                  sigmax[0][idx], sigmay[0][idx],
                                                  corr[0][idx], np.sqrt(temp), greedy)
            # generate stroke pen status
            idx_eos = get_pi_idx(random.random(), pen[0], temp, greedy)
            # eos = np.zeros((len(z), 1, 5))
            # for r, eos_i in enumerate(idx_eos):
            #     eos[r, 0, eos_i] = 1

            eos = np.zeros(3)
            eos[idx_eos] = 1

            strokes[seq_i, :, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]

            input_x = np.array([next_x1, next_x2, eos[0], eos[1], eos[2]], dtype=np.float32)
            input_x = input_x.reshape([1, 1, 5])

            # s = utils.seq_5d_to_3d(np.reshape(strokes, [seq_len, 5]))
            # filepath1 = './sample/%d.svg' % seq_i
            # draw_strokes(s, filepath1, 48, margin=1.5, color='black')
            # print(s)

        return utils.seq_5d_to_3d(np.reshape(strokes, [seq_len, 5]))


    # Generate a batch of sketches based on one latent vector
    gen_strokes = []
    for i in range(gen_size):
        sketch = get_seqs(z, seq_len, greedy_mode, temperature)
        gen_strokes.append(sketch)
    return gen_strokes


def load_model_params(model_dir):
    model_params = utils.get_default_hparams()
    with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        model_config = json.dumps(json.load(f))
        model_params.parse_json(model_config)
    return model_params


def modify_model_params(model_params):
    model_params.use_input_dropout = 0
    model_params.use_recurrent_dropout = 0
    model_params.use_output_dropout = 0
    model_params.is_training = False
    model_params.batch_size = 1
    return model_params

def sort_paths(paths):
    idxs = []
    for path in paths:
        idxs.append(int(re.findall(r'\d+', path)[-1]))

    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            if idxs[i] > idxs[j]:
                tmp = idxs[i]
                idxs[i] = idxs[j]
                idxs[j] = tmp

                tmp = paths[i]
                paths[i] = paths[j]
                paths[j] = tmp
    return paths



def main():
    FLAGS = tf.app.flags.FLAGS
    # Dataset directory
    tf.app.flags.DEFINE_string(
        'data_dir',
        './QuickDraw/',
        'The directory in which to find the dataset specified in model hparams.')
    # Checkpoint directory
    tf.app.flags.DEFINE_string(
        'model_dir', 'sketch_model',
        'Directory to store model checkpoints, tensorboard.')
    # Output dir
    tf.app.flags.DEFINE_string(
        'output_dir', 'sample',
        'Directory to store the generate sketches.')
    # Number of generated sketches per category
    tf.app.flags.DEFINE_integer(
        'num_per_category', 2500,
        'Number of generated sketches per category')
    # Masking probability
    tf.app.flags.DEFINE_float(
        'prob', 0.1,
        'Masking probability')

    model_dir = FLAGS.model_dir
    data_dir = FLAGS.data_dir
    SVG_DIR = FLAGS.output_dir
    sample_num = FLAGS.num_per_category
    mask_prob = FLAGS.prob

    model_params = load_model_params(model_dir)
    model_params = modify_model_params(model_params)

    sample_model_params = utils.copy_hparams(model_params)
    sample_model_params.max_seq_len = 1
    sample_model = Model(sample_model_params)

    # open session
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    sess.run(tf.global_variables_initializer())

    # load model from checkpoint
    utils.load_checkpoint(sess, model_dir)

    # Load random seed for sketch healing
    seed = np.load('./random_seed.npy')
    seed_id = 0

    if not os.path.exists(SVG_DIR):
        os.makedirs(SVG_DIR)

    for category in range(len(model_params.categories)):
        print(model_params.categories[category])
        # load dataset
        raw_data = utils.load_data(data_dir, model_params.categories[category], 70000)
        _, _, test_set, _ = utils.preprocess_data(raw_data,
                                                  model_params.batch_size,
                                                  0.,
                                                  0.)
        model_params.max_seq_len = test_set.max_seq_length
        model = Model(model_params)

        index = np.arange(len(test_set.strokes))
        # np.random.shuffle(index)
        for cnt in range(sample_num):
            ori_seqs, seqs, labels, seq_len = test_set._get_batch_from_indices(index[cnt:cnt + 1])

            graphs = []
            adj_mask = []
            for i in range(len(seqs)):
                data = np.copy(ori_seqs[i])
                _graph_tensor, _graph_len, mask_id, seed_id = utils.make_graph_(data, seed, seed_id, graph_num=model.hps.graph_number,
                                                                                graph_picture_size=model.hps.graph_picture_size,
                                                                                mask_prob=mask_prob, train=False)

                graphs.append(_graph_tensor)
                if _graph_len == (model.hps.graph_number - 1):
                    temp_adj = np.ones([model.hps.graph_number - 1, model.hps.graph_number - 1])
                else:
                    temp_adj = np.concatenate([np.concatenate([np.ones([_graph_len + 1, _graph_len + 1]),
                                                               np.zeros([model.hps.graph_number - 2 - _graph_len, _graph_len + 1])], axis=0),
                                               np.zeros([model.hps.graph_number - 1,
                                                         model.hps.graph_number - 2 - _graph_len])], axis=1)
                for id in mask_id:
                    temp_adj[id, :] = 0
                    temp_adj[:, id] = 0
                adj_mask.append(temp_adj)

            feed = {
                model.input_seqs: seqs,
                model.sequence_lengths: seq_len,
                model.input_graphs: np.stack(graphs, axis=0),
                model.input_masks: np.stack(adj_mask, axis=0)
            }
            z = sess.run(model.p_mu, feed)

            # generated images saved path
            path = os.path.join(SVG_DIR)
            if not os.path.exists(path):
                os.makedirs(path)

            # generate strokes
            stroke = sample(sess, sample_model, z, 1, model_params.max_seq_len)
            filepath1 = os.path.join(path, '%d_%d.svg' % (category, cnt))
            draw_strokes(stroke[0], filepath1, 225, margin=1.5, color='black')
            filepath3 = os.path.join(path, 'stroke_%d_%d.npy' % (category, cnt))
            np.save(filepath3, np.expand_dims(stroke[0], axis=0))
            filepath2 = os.path.join(path, 'code_%d_%d.npy' % (category, cnt))
            np.save(filepath2, z)

    exportsvg(SVG_DIR, SVG_DIR, 'png')

if __name__ == '__main__':
    main()
