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
""" Calculate the metric Ret"""

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
from sample import sample
import scipy.misc
import re


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
    model_params.max_seq_len = 1

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

model_dir = 'sketch_model'
data_dir = '/disk2/zangsicong/datasets/QuickDraw/'
SVG_DIR = './sample/'

def main():
    model_params = load_model_params(model_dir)
    model_params = modify_model_params(model_params)
    model = Model(model_params)

    for label in range(len(model_params.categories)):
        seq_paths = glob.glob('./sample/stroke_%d_*.npy' % label)
        seq_paths = sort_paths(seq_paths)
        if label == 0:
            seq = np.array(seq_paths)
        else:
            seq = np.hstack((seq, np.array(seq_paths)))

    sample_size = len(seq)  # Number of samples for retrieval

    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    sess.run(tf.global_variables_initializer())
    utils.load_checkpoint(sess, model_dir)

    seed = np.load('./random_seed.npy')
    seed_id = 0

    # create the codes for origins
    code_data = []

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
        # model = Model(model_params, reuse=True)

        index = np.arange(len(test_set.strokes))
        # np.random.shuffle(index)
        for cnt in range(2500):
            ori_seqs, seqs, labels, seq_len = test_set._get_batch_from_indices(index[cnt:cnt + 1])

            graphs = []
            adj_mask = []
            for i in range(len(seqs)):
                data = np.copy(ori_seqs[i])
                _graph_tensor, _graph_len, mask_id, seed_id = utils.make_graph_(data, seed, seed_id, graph_num=model.hps.graph_number,
                                                                                graph_picture_size=model.hps.graph_picture_size, mask_prob=0.0, train=False)

                graphs.append(_graph_tensor)
                if _graph_len == (model.hps.graph_number - 1):
                    temp_adj = np.ones([model.hps.graph_number - 1, model.hps.graph_number - 1])
                else:
                    temp_adj = np.concatenate([np.concatenate([np.ones([_graph_len + 1, _graph_len + 1]),
                                                               np.zeros([model.hps.graph_number - 2 - _graph_len, _graph_len + 1])], axis=0),
                                               np.zeros([model.hps.graph_number - 1, model.hps.graph_number - 2 - _graph_len])], axis=1)
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
            code = sess.run(model.p_mu, feed)
            code_data.append(np.squeeze(code, axis=0))

    code_data = np.array(code_data)  # Real codes for original sketches

    # create the codes for generations
    for i in range(len(code_data)):
        seq_data = np.load(seq[i])
        graphs, _graph_len, mask_id, _ = utils.make_graph_(seq_data[0], seed, seed_id, graph_num=model.hps.graph_number,
                                                           graph_picture_size=model.hps.graph_picture_size, mask_prob=0.0, train=False)
        if _graph_len == (model.hps.graph_number - 1):
            temp_adj = np.ones([model.hps.graph_number - 1, model.hps.graph_number - 1])
        else:
            temp_adj = np.concatenate([np.concatenate([np.ones([_graph_len + 1, _graph_len + 1]),
                                                       np.zeros([model.hps.graph_number - 2 - _graph_len, _graph_len + 1])], axis=0),
                                       np.zeros([model.hps.graph_number - 1, model.hps.graph_number - 2 - _graph_len])], axis=1)
        for id in mask_id:
            temp_adj[id, :] = 0
            temp_adj[:, id] = 0

        feed = {
            model.input_graphs: np.expand_dims(graphs, axis=0),
            model.input_masks: np.expand_dims(temp_adj, axis=0)
        }
        z = sess.run(model.p_mu, feed)  # Codes of the samples

        if i == 0:
            batch_z = z
        else:
            batch_z = np.concatenate([batch_z, z], axis=0)  # Codes for generations

    # Begin retrieval
    top_1 = 0.
    top_10 = 0.
    top_50 = 0.

    temp_sample_size = int(sample_size / 20)
    for ii in range(20):  # reduce the risk of memory out
        real_code = np.tile(np.reshape(code_data, [sample_size, 1, model_params.z_size]), [1, temp_sample_size, 1])
        fake_code = np.tile(np.reshape(batch_z[temp_sample_size * ii:temp_sample_size * (ii + 1), :], [1, temp_sample_size, model_params.z_size]), [sample_size, 1, 1])
        distances = np.average((real_code - fake_code) ** 2, axis=2)  # Distances between each two codes, sample_size * sample_size

        for n in range(50):
            temp_index = np.argmin(distances, axis=0)
            for i in range(temp_sample_size):
                distances[temp_index[i], i] = 1e10
            if n == 0:
                top_n_index = np.reshape(temp_index, [1, -1])
            else:
                top_n_index = np.concatenate([top_n_index, np.reshape(temp_index, [1, -1])], axis=0)

        for i in range(temp_sample_size):
            if top_n_index[0, i] == i + temp_sample_size * ii:
                top_1 += 1.
            for k in range(10):
                if top_n_index[k, i] == i + temp_sample_size * ii:
                    top_10 += 1.
                    break
            for k in range(50):
                if top_n_index[k, i] == i + temp_sample_size * ii:
                    top_50 += 1.
                    break

    print("Top 1 Ret: " + str(float(top_1 / sample_size)))
    print("Top 10 Ret: " + str(float(top_10 / sample_size)))
    print("Top 50 Ret: " + str(float(top_50 / sample_size)))

if __name__ == '__main__':
    main()
