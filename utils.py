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
""" SP-gra2seq data loading and processing"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import tensorflow as tf
import numpy as np
import os
import math
import cv2
import six


def get_default_hparams():
    """ Return default and initial HParams """
    hparams = tf.contrib.training.HParams(
        categories=['bee', 'bus', 'car', 'cat', 'flower', 'giraffe', 'horse', 'pig'],  # Categories used for training
        num_steps=1000001,  # Number of total steps (the process will stop automatically if the cost does not improve)
        save_every=1,  # Number of epochs before saving model
        dec_rnn_size=1024,  # Size of decoder
        dec_model='lstm',  # Decoder: lstm, layer_norm or hyper
        max_seq_len=-1,  # Max sequence length. Computed by DataLoader
        z_size=128,  # Size of latent vector z. Recommend 128.
        batch_size=256,  # Minibatch size. Recommend leaving at 128
        graph_number=21,  # Number of graph nodes for a sketch
        graph_picture_size=256,  # Cropped patch size
        num_mixture=50,  # Number of clusters
        learning_rate=0.001,  # Learning rate.
        decay_rate=0.9999,  # Learning rate decay per minibatch.
        min_learning_rate=0.00001,  # Minimum learning rate.
        grad_clip=1.,  # Gradient clipping. Recommend leaving at 1.0.
        use_recurrent_dropout=False,  # Dropout with memory loss. Recomended leaving True
        recurrent_dropout_prob=0.0,  # Probability of recurrent dropout keep
        use_input_dropout=False,  # Input dropout. Recommend leaving False.
        input_dropout_prob=0.0,  # Probability of input dropout keep
        use_output_dropout=False,  # Output droput. Recommend leaving False.
        output_dropout_prob=0.0,  # Probability of output dropout keep.
        random_scale_factor=0.0,  # Random scaling data augmention proportion.
        augment_stroke_prob=0.0,  # Point dropping augmentation proportion.
        is_training=True,  # Training mode or not
        semi_percent=0.0,  # Percentage of the labeled samples
        semi_balanced=False,  # Whether the labeled samples are balanced among categories
        num_per_category=70000  # How many training samples are taken from each category, [0, 70000]
    )
    return hparams


def copy_hparams(hparams):
  """ Return a copy of an HParams instance """
  return tf.contrib.training.HParams(**hparams.values())


def reset_graph():
    """ Close the current default session and resets the graph """
    sess = tf.get_default_session()
    if sess:
      sess.close()
    tf.reset_default_graph()


def load_data(data_dir, categories, num_per_category):
    """ Load sequence and image raw data """
    if not isinstance(categories, list):
        categories = [categories]

    train_seqs = None
    valid_seqs = None
    test_seqs = None
    train_labels = None
    valid_labels = None
    test_labels = None

    i = 0
    for ctg in categories:
        # load sequence data
        seq_path = os.path.join(data_dir, ctg + '.npz')
        if six.PY3:
            seq_data = np.load(seq_path, encoding='latin1', allow_pickle=True)
        else:
            seq_data = np.load(seq_path, allow_pickle=True)
        tf.logging.info('Loaded sequences {}/{}/{} from {}'.format(
            len(seq_data['train']), len(seq_data['valid']), len(seq_data['test']),
            ctg + '.npz'))

        if train_seqs is None:
            train_seqs = seq_data['train'][0:num_per_category]
            valid_seqs = seq_data['valid']
            test_seqs = seq_data['test']
        else:
            train_seqs = np.concatenate((train_seqs, seq_data['train'][0:num_per_category]))
            valid_seqs = np.concatenate((valid_seqs, seq_data['valid']))
            test_seqs = np.concatenate((test_seqs, seq_data['test']))

        # create labels
        if train_labels is None:
            train_labels = i * np.ones([num_per_category], dtype=np.int)
            valid_labels = i * np.ones([2500], dtype=np.int)
            test_labels = i * np.ones([2500], dtype=np.int)
        else:
            train_labels = np.concatenate([train_labels, i * np.ones([num_per_category], dtype=np.int)])
            valid_labels = np.concatenate([valid_labels, i * np.ones([2500], dtype=np.int)])
            test_labels = np.concatenate([test_labels, i * np.ones([2500], dtype=np.int)])
        i += 1

    return [train_seqs, valid_seqs, test_seqs, train_labels, valid_labels, test_labels]


def preprocess_data(raw_data, batch_size, random_scale_factor, augment_stroke_prob):
    """ Convert raw data to suitable model inputs """
    train_seqs, valid_seqs, test_seqs, train_labels, valid_labels, test_labels = raw_data
    all_strokes = np.concatenate((train_seqs, valid_seqs, test_seqs))
    max_seq_len = get_max_len(all_strokes)

    train_set = DataLoader(
        train_seqs,
        train_labels,
        batch_size,
        max_seq_length=max_seq_len,
        random_scale_factor=random_scale_factor,
        augment_stroke_prob=augment_stroke_prob)
    # seq_norm = train_set.calc_seq_norm()
    # train_set.normalize_seq(seq_norm)

    valid_set = DataLoader(
        valid_seqs,
        valid_labels,
        batch_size,
        max_seq_length=max_seq_len)
    # valid_set.normalize_seq(seq_norm)

    test_set = DataLoader(
        test_seqs,
        test_labels,
        batch_size,
        max_seq_length=max_seq_len)
    # test_set.normalize_seq(seq_norm)

    # tf.logging.info('normalizing_scale_factor %4.4f.', seq_norm)
    return train_set, valid_set, test_set, max_seq_len


def load_checkpoint(sess, checkpoint_path):
    """ Load checkpoint of saved model """
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    gmm_vars = [g for g in g_list if 'latent_' in g.name]
    var_list += bn_moving_vars
    var_list += gmm_vars
    saver = tf.train.Saver(var_list=var_list)
    # saver = tf.train.Saver(tf.global_variables())

    ckpt = tf.train.get_checkpoint_state(checkpoint_path)

    tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
    print('Loading model %s.', ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)


def save_model(sess, model_save_path, global_step):
    """ Save model """
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    gmm_vars = [g for g in g_list if 'latent_' in g.name]
    var_list += bn_moving_vars
    var_list += gmm_vars
    saver = tf.train.Saver(var_list=var_list, max_to_keep=10)
    # saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
    
    checkpoint_path = os.path.join(model_save_path, 'vector')
    tf.logging.info('saving model %s.', checkpoint_path)
    tf.logging.info('global_step %i.', global_step)
    saver.save(sess, checkpoint_path, global_step=global_step)


def summ_content(tag, val):
    """ Construct summary content """
    summ = tf.summary.Summary()
    summ.value.add(tag=tag, simple_value=float(val))
    return summ


def write_summary(summ_writer, summ_dict, step):
    """ Write summary """
    for key, val in summ_dict.iteritems():
        summ_writer.add_summary(summ_content(key, val), step)
    summ_writer.flush()


def augment_strokes(strokes, prob=0.0):
    """ Perform data augmentation by randomly dropping out strokes """
    # drop each point within a line segments with a probability of prob
    # note that the logic in the loop prevents points at the ends to be dropped.
    result = []
    prev_stroke = [0, 0, 1]
    count = 0
    stroke = [0, 0, 1]  # Added to be safe.
    for i in range(len(strokes)):
        candidate = [strokes[i][0], strokes[i][1], strokes[i][2]]
        if candidate[2] == 1 or prev_stroke[2] == 1:
            count = 0
        else:
            count += 1
        urnd = np.random.rand()  # uniform random variable
        if candidate[2] == 0 and prev_stroke[2] == 0 and count > 2 and urnd < prob:
            stroke[0] += candidate[0]
            stroke[1] += candidate[1]
        else:
            stroke = list(candidate)
            prev_stroke = list(stroke)
            result.append(stroke)
    return np.array(result)


def seq_3d_to_5d(stroke, max_len=250):
    """ Convert from 3D format (npz file) to 5D (sketch-rnn paper) """
    result = np.zeros((max_len, 5), dtype=float)
    l = len(stroke)
    assert l <= max_len
    result[0:l, 0:2] = stroke[:, 0:2]
    result[0:l, 3] = stroke[:, 2]
    result[0:l, 2] = 1 - result[0:l, 3]
    result[l:, 4] = 1
    return result


def seq_5d_to_3d(big_stroke):
    """ Convert from 5D format (sketch-rnn paper) back to 3D (npz file) """
    l = 0 # the total length of the drawing
    for i in range(len(big_stroke)):
        if big_stroke[i, 4] > 0:
            l = i
            break
        if l == 0:
            l = len(big_stroke) # restrict the max total length of drawing to be the length of big_stroke
    result = np.zeros((l, 3))
    result[:, 0:2] = big_stroke[0:l, 0:2]
    result[:, 2] = big_stroke[0:l, 3]
    return result # stroke-3


def get_max_len(strokes):
    """ Return the maximum length of an array of strokes """
    max_len = 0
    for stroke in strokes:
        ml = len(stroke)
        if ml > max_len:
            max_len = ml
    return max_len


def rescale(X, ratio=0.85):
    """ Rescale the image to a smaller size """
    h, w = X.shape

    h2 = int(h*ratio)
    w2 = int(w*ratio)

    X2 = cv2.resize(X, (w2, h2), interpolation=cv2.INTER_AREA)

    dh = int((h - h2) / 2)
    dw = int((w - w2) / 2)

    res = np.copy(X)
    res[:,:] = 1
    res[dh:(dh+h2),dw:(dw+w2)] = X2

    return res


def rotate(X, angle=15):
    """ Rotate the image """
    h, w = X.shape
    rad = np.deg2rad(angle)

    nw = ((abs(np.sin(rad)*h)) + (abs(np.cos(rad)*w)))
    nh = ((abs(np.cos(rad)*h)) + (abs(np.sin(rad)*w)))

    rot_mat = cv2.getRotationMatrix2D((nw/2,nh/2),angle,1)
    rot_move = np.dot(rot_mat,np.array([(nw-w)/2,(nh-h)/2,0]))

    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]

    res_w = int(math.ceil(nw))
    res_h = int(math.ceil(nh))

    res = cv2.warpAffine(X,rot_mat,(res_w,res_h),flags=cv2.INTER_LANCZOS4, borderValue=1)
    res = cv2.resize(res,(w,h), interpolation=cv2.INTER_AREA)

    return res


def translate(X, dx=5,dy=5):
    """ Translate the image """
    h, w = X.shape
    M = np.float32([[1,0,dx],[0,1,dy]])
    res = cv2.warpAffine(X,M,(w,h), borderValue=1)

    return res

def canvas_size_google(sketch):
    """
    读取quickDraw的画布大小及起始点
    :param sketch: google sketch, quickDraw
    :return: int list,[x, y, h, w]
    """
    # get canvas size

    vertical_sum = np.cumsum(sketch[1:], axis=0)  # 累加 排除第一笔未知的偏移量
    xmin, ymin, _ = np.min(vertical_sum, axis=0)
    xmax, ymax, _ = np.max(vertical_sum, axis=0)
    w = xmax - xmin
    h = ymax - ymin
    start_x = -xmin - sketch[0][0]  # 等效替换第一笔
    start_y = -ymin - sketch[0][1]
    # sketch[0] = sketch[0] - sketch[0]
    # 返回可能处理过的sketch
    return [int(start_x), int(start_y), int(h), int(w)]

def scale_sketch(sketch, size=(448, 448)):
    [_, _, h, w] = canvas_size_google(sketch)
    if h >= w:
        sketch_normalize = sketch / np.array([[h, h, 1]], dtype=np.float)
    else:
        sketch_normalize = sketch / np.array([[w, w, 1]], dtype=np.float)
    sketch_rescale = sketch_normalize * np.array([[size[0], size[1], 1]], dtype=np.float)
    return sketch_rescale.astype("int16")

def make_graph_(sketch, seed, seed_id, graph_num=30, graph_picture_size=128, random_color=False, mask_prob=0.0, train=True):
    tmp_img_size = 640
    thickness = int(tmp_img_size * 0.025)
    # preprocess
    sketch = scale_sketch(sketch, (tmp_img_size, tmp_img_size))  # scale the sketch.
    [start_x, start_y, h, w] = canvas_size_google(sketch=sketch)
    start_x += thickness + 1
    start_y += thickness + 1

    # graph (graph_num, 3, graph_size, graph_size)
    graphs = np.zeros((graph_num, graph_picture_size, graph_picture_size), dtype='uint8')  # must uint8

    # canvas (h, w, 3)
    canvas = np.zeros((max(h, w) + 2 * (thickness + 1), max(h, w) + 2 * (thickness + 1)), dtype='uint8')
    if random_color:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    else:
        color = (255, 255, 255)
    pen_now = np.array([start_x, start_y])
    first_zero = False

    # generate canvas.
    for index, stroke in enumerate(sketch):
        delta_x_y = stroke[0:0 + 2]
        state = stroke[2:]
        if first_zero:
            pen_now += delta_x_y
            first_zero = False
            continue
        cv2.line(canvas, tuple(pen_now), tuple(pen_now + delta_x_y), color, thickness=thickness)
        if int(state) != 0:  # next stroke
            first_zero = True
            if random_color:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            else:
                color = (255, 255, 255)
        pen_now += delta_x_y
    # canvas_first = cv2.resize(canvas, (graph_picture_size, graph_picture_size))
    # graphs[0] = canvas_first

    # generate patch pixel picture from canvas
    # make canvas larger, enlarge canvas 100 pixels boundary
    _h, _w = canvas.shape  # (h, w, c)
    boundary_size = int(graph_picture_size * 1.5)
    top_bottom = np.zeros((boundary_size, _w), dtype=canvas.dtype)
    left_right = np.zeros((boundary_size * 2 + _h, boundary_size), dtype=canvas.dtype)
    canvas = np.concatenate((top_bottom, canvas, top_bottom), axis=0)
    canvas = np.concatenate((left_right, canvas, left_right), axis=1)
    # cv2.imwrite(f"./google_large.png", canvas)
    # processing.
    pen_now = np.array([start_x + boundary_size, start_y + boundary_size])
    first_zero = False

    # Create masked canvas
    mask_id = []
    graph_count = 0
    tmp_count = 0
    _move = graph_picture_size // 2
    for index, stroke in enumerate(sketch):
        delta_x_y = stroke[0:0 + 2]
        state = stroke[2:]
        if first_zero:
            pen_now += delta_x_y
            first_zero = False
            continue
        # cv2.line(canvas, tuple(pen_now), tuple(pen_now + delta_x_y), color=(255, 0, 0), thickness=thickness)
        if tmp_count % 4 == 0:
            tmpRec = canvas[pen_now[1] - _move:pen_now[1] + _move, pen_now[0] - _move:pen_now[0] + _move]

            if graph_count + 1 > graph_num - 1:
                break

            if train:
                applied_seed = np.random.uniform(0, 1)
            else:
                applied_seed = seed[seed_id]
                seed_id += 1

            if tmpRec.shape[0] != graph_picture_size or tmpRec.shape[1] != graph_picture_size:
                # print(f'this sketch is broken: broken stroke: ', index)
                pass
            elif applied_seed < mask_prob:
                canvas[pen_now[1] - _move:pen_now[1] + _move, pen_now[0] - _move:pen_now[0] + _move] = 0
                # cv2.rectangle(canvas,
                #               tuple(pen_now - np.array([graph_picture_size // 2, graph_picture_size // 2])),
                #               tuple(pen_now + np.array([graph_picture_size // 2, graph_picture_size // 2])),
                #               color=(255, 255, 255), thickness=1)
                mask_id.append(graph_count)


            graph_count += 1
        tmp_count += 1
        if int(state) != 0:  # next stroke
            tmp_count = 0
            first_zero = True
        pen_now += delta_x_y
    # cv2.imwrite("./google_large_rec.png", 255 - canvas)
    # id = np.array(id)
    # print(id)

    canvas_first = cv2.resize(canvas[boundary_size:boundary_size + _h, boundary_size:boundary_size + _w], (graph_picture_size, graph_picture_size))
    graphs[0] = canvas_first

    # generate patches.
    # strategies:
    # 1. get box at the head of one stroke
    # 2. in a long stroke, we get box in
    pen_now = np.array([start_x + boundary_size, start_y + boundary_size])
    first_zero = False
    graph_count = 0
    tmp_count = 0
    # num_strokes = math.floor(len(sketch) / (graph_num - 1))  # number of strokes for creating a single lattice
    _move = graph_picture_size // 2
    location_of_pen = []
    for index, stroke in enumerate(sketch):
        delta_x_y = stroke[0:0 + 2]
        state = stroke[2:]
        if first_zero:
            pen_now += delta_x_y
            first_zero = False
            continue
        # cv2.line(canvas, tuple(pen_now), tuple(pen_now + delta_x_y), color=(255, 0, 0), thickness=thickness)
        if tmp_count % 4 == 0:
            tmpRec = canvas[pen_now[1] - _move:pen_now[1] + _move, pen_now[0] - _move:pen_now[0] + _move]
            if graph_count + 1 > graph_num - 1:
                break
            if tmpRec.shape[0] != graph_picture_size or tmpRec.shape[1] != graph_picture_size:
                # print(f'this sketch is broken: broken stroke: ', index)
                pass
            else:
                graphs[graph_count + 1] = tmpRec  # 第0张图是原图
                location_of_pen.append([pen_now[1], pen_now[0]])
                # cv2.rectangle(canvas,
                #               tuple(pen_now - np.array([graph_picture_size // 2, graph_picture_size // 2])),
                #               tuple(pen_now + np.array([graph_picture_size // 2, graph_picture_size // 2])),
                #               color=(255, 255, 255), thickness=1)

            graph_count += 1
            # cv2.line(canvas, tuple(pen_now), tuple(pen_now + np.array([1, 1])), color=(0, 0, 255), thickness=3)

        tmp_count += 1
        if int(state) != 0:  # next stroke
            tmp_count = 0
            first_zero = True
        pen_now += delta_x_y

    graphs_tensor = np.zeros([graph_num, graph_picture_size, graph_picture_size, 1])
    for index in range(graph_num):
        graphs_tensor[index] = np.expand_dims(graphs[index] / 255 * 2 - 1, axis=2)

    return graphs_tensor, graph_count, mask_id, seed_id

class DataLoader(object):
    """ Class for loading data from raw data (sequence and image) """

    def __init__(self,
                 strokes,
                 labels,
                 batch_size=100,
                 max_seq_length=250,
                 scale_factor=1.0,
                 random_scale_factor=0.0,
                 augment_stroke_prob=0.0,
                 limit=1000):
        self.batch_size = batch_size  # minibatch size
        self.max_seq_length = max_seq_length  # N_max in sketch-rnn paper
        # self.scale_factor = scale_factor  # divide data by this factor
        self.scale_factor = self.calculate_normalizing_scale_factor(strokes)
        self.random_scale_factor = random_scale_factor  # data augmentation method
        self.limit = limit  # removes large gaps in the data
        self.augment_stroke_prob = augment_stroke_prob  # data augmentation method
        self.start_stroke_token = [0, 0, 1, 0, 0]  # S_0 in sketch-rnn paper
        self.preprocess(strokes)
        self.labels = labels

    def calculate_normalizing_scale_factor(self, sketches):
        data = []
        for sketch in sketches:
            for stroke in sketch:
                data.append(stroke)
        return np.std(np.array(data))

    def preprocess(self, strokes):
        self.origin_strokes = []
        count_data = 0  # the number of drawing with length less than N_max
        for i in range(len(strokes)):
            data = np.copy(strokes[i])
            if len(data) <= self.max_seq_length:    # keep data with length less than N_max
                count_data += 1
                # removes large gaps from the data
                data = np.minimum(data, self.limit)     # prevent large values
                data = np.maximum(data, -self.limit)    # prevent small values
                data = np.array(data, dtype=np.float32) # change data type
                data[:, 0:2] /= self.scale_factor       # scale the first two dims of data
                self.origin_strokes.append(data)

        print("total sequences <= max_seq_len is %d" % count_data)
        self.num_batches = int(count_data / self.batch_size)

        self.strokes = np.copy(self.origin_strokes)

    def random_sample(self):
        """ Return a random sample (3D stroke, png image) """
        l = len(self.strokes)
        idx = np.random.randint(0, l)
        ori_seq = self.origin_strokes[idx]
        seq = self.strokes[idx]
        label = self.labels[idx]
        return ori_seq, seq, label


    def idx_sample(self, idx):
        """ Return one sample by idx """
        ori_strokes_3d = self.origin_strokes[idx]
        data = self.strokes[idx]
        # data = self.random_scale_seq(self.strokes[idx])
        # if self.augment_stroke_prob > 0:
        #     data = augment_strokes(data, self.augment_stroke_prob)
        strokes_3d = data
        strokes_5d = seq_3d_to_5d(strokes_3d, self.max_seq_length)

        label = self.labels[idx]
        return ori_strokes_3d, strokes_5d, label


    def random_scale_seq(self, data):
        """ Augment data by stretching x and y axis randomly [1-e, 1+e] """
        x_scale_factor = (np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        y_scale_factor = (np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        result = np.copy(data)
        result[:, 0] *= x_scale_factor
        result[:, 1] *= y_scale_factor
        return result

    def calc_seq_norm(self):
        """ Calculate the normalizing factor explained in appendix of sketch-rnn """
        data = []
        for i in range(len(self.strokes)):
            if len(self.strokes[i]) > self.max_seq_length:
                continue
            for j in range(len(self.strokes[i])):
                data.append(self.strokes[i][j, 0])
                data.append(self.strokes[i][j, 1])
        data = np.array(data)
        return np.std(data)  # standard dev of all the delta x and delta y in the datasets

    def normalize_seq(self, scale_factor=None):
        """ Normalize entire sequence dataset (delta_x, delta_y) by the scaling factor."""
        if scale_factor is None:
            scale_factor = self.calc_seq_norm()
        self.scale_factor = scale_factor
        for i in range(len(self.strokes)):
            self.strokes[i][:, 0:2] /= self.scale_factor


    def _get_batch_from_indices(self, indices):
        """Given a list of indices, return the potentially augmented batch."""
        ori_seq_batch = []
        seq_batch = []
        label_batch = []
        seq_len = []
        for idx in range(len(indices)):
            i = indices[idx]
            data = self.origin_strokes[i]
            data_copy = np.copy(data)
            ori_seq_batch.append(data_copy)
            data = self.strokes[i]
            # data = self.random_scale_seq(self.strokes[i])
            data_copy = np.copy(data)
            # if self.augment_stroke_prob > 0:
            #     data_copy = augment_strokes(data_copy, self.augment_stroke_prob)
            seq_batch.append(data_copy)
            length = len(data_copy)
            seq_len.append(length)
            label_batch.append(self.labels[i])

        seq_len = np.array(seq_len, dtype=int)
        return ori_seq_batch, self.pad_seq_batch(seq_batch, self.max_seq_length), label_batch, seq_len


    def random_batch(self):
        """Return a randomised portion of the training data."""
        idxs = np.random.permutation(list(range(0, len(self.strokes))))[0:self.batch_size]
        return self._get_batch_from_indices(idxs)


    def get_batch(self, idx):
        """Get the idx'th batch from the dataset."""
        assert idx >= 0, "idx must be non negative"
        assert idx < self.num_batches, "idx must be less than the number of batches"
        start_idx = idx * self.batch_size
        indices = list(range(start_idx, start_idx + self.batch_size))
        return self._get_batch_from_indices(indices)


    def pad_seq_batch(self, batch, max_len):
      """ Pad the batch to be 5D format, and fill the sequence to reach max_len """
      result = np.zeros((self.batch_size, max_len + 1, 5), dtype=float)
      assert len(batch) == self.batch_size
      # for i in range(self.batch_size):
      #     l = len(batch[i])
      #     assert l <= max_len
      #     result[i, 0:l, 0:2] = batch[i][:, 0:2]
      #     result[i, 0:(l-1), 3] = batch[i][:-1, 2]
      #     result[i, 0:(l-1), 2] = 1 - result[i, 0:(l-1), 3]
      #     result[i, (l-1):, 4] = 1
      #     # put in the first token, as described in sketch-rnn methodology
      #     result[i, 1:, :] = result[i, :-1, :]
      #     result[i, 0, :] = 0
      #     result[i, 0, 2] = self.start_stroke_token[2]  # setting S_0 from paper.
      #     result[i, 0, 3] = self.start_stroke_token[3]
      #     result[i, 0, 4] = self.start_stroke_token[4]
      # return result
      for i in range(self.batch_size):
          l = len(batch[i])
          assert l <= max_len
          result[i, 0:l, 0:2] = batch[i][:, 0:2]
          result[i, 0:l, 3] = batch[i][:, 2]
          result[i, 0:l, 2] = 1 - result[i, 0:l, 3]
          result[i, l:, 4] = 1
          # put in the first token, as described in sketch-rnn methodology
          result[i, 1:, :] = result[i, :-1, :]
          result[i, 0, :] = 0
          result[i, 0, 2] = self.start_stroke_token[2]  # setting S_0 from paper.
          result[i, 0, 3] = self.start_stroke_token[3]
          result[i, 0, 4] = self.start_stroke_token[4]
      return result


