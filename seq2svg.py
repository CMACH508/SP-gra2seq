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
""" Convert sequential strokes to .svg file """
import numpy as np
from six.moves import xrange
import svgwrite # conda install -c omnia svgwrite=1.1.6
import os
import six

def get_bounds(data):
  """Return bounds of data."""
  min_x = 0
  max_x = 0
  min_y = 0
  max_y = 0

  abs_x = 0
  abs_y = 0
  for i in range(len(data)):
    x = float(data[i, 0])
    y = float(data[i, 1])
    abs_x += x
    abs_y += y
    min_x = min(min_x, abs_x)
    min_y = min(min_y, abs_y)
    max_x = max(max_x, abs_x)
    max_y = max(max_y, abs_y)

  return (min_x, max_x, min_y, max_y)


def load_dataset(data_dir, dataset):
  """ fetch data from npz file """
  train_strokes = None
  valid_strokes = None
  test_strokes = None

  data_filepath = os.path.join(data_dir, dataset)

  if six.PY3:
      data = np.load(data_filepath, encoding='latin1')
  else:
      data = np.load(data_filepath)

  if train_strokes is None:
      train_strokes = data['train']
      valid_strokes = data['valid']
      test_strokes = data['test']
  else:
      train_strokes = np.concatenate((train_strokes, data['train']))
      valid_strokes = np.concatenate((valid_strokes, data['valid']))
      test_strokes = np.concatenate((test_strokes, data['test']))

  return train_strokes,valid_strokes,test_strokes


def draw_strokes(data, svg_filename = 'sample.svg', width=48, margin=1.5, color='black'):
    """ convert sequence data to svg format """
    min_x, max_x, min_y, max_y = get_bounds(data)
    if max_x - min_x > max_y - min_y:
        norm = max_x - min_x
        border_y = (norm - (max_y - min_y)) * 0.5
        border_x = 0
    else:
        norm = max_y - min_y
        border_x = (norm - (max_x - min_x)) * 0.5
        border_y = 0
  
    # normalize data
    norm = max(norm, 10e-6)
    scale = (width - 2*margin) / norm
    dx = 0 - min_x + border_x
    dy = 0 - min_y + border_y
  
    abs_x = (0 + dx) * scale + margin
    abs_y = (0 + dy) * scale + margin
  
    # start converting
    dwg = svgwrite.Drawing(svg_filename, size=(width,width))
    dwg.add(dwg.rect(insert=(0, 0), size=(width,width),fill='white'))
    lift_pen = 1
    p = "M%s,%s " % (abs_x, abs_y)
    command = "m"
    for i in xrange(len(data)):
        if (lift_pen == 1):
            command = "m"
        elif (command != "l"):
            command = "l"
        else:
            command = ""
        x = float(data[i,0]) * scale
        y = float(data[i,1]) * scale
        lift_pen = data[i, 2]
        p += command+str(x)+","+str(y)+" "
    the_color = color  # "black"
    stroke_width = 1
    dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
    dwg.save()


def main():
    dataset =['umbrella','octopus']  # Categories of target datasets
    in_dir = '/disk2/zangsicong/datasets/QuickDraw/'  # Directory of target datasets
    out_dir = '/disk2/zangsicong/datasets/QuickDraw/svg_225/'  # Directory of outputs

    for category in range(len(dataset)):
        train_strokes, valid_strokes, test_strokes = load_dataset(in_dir, dataset[category]+'.npz')

        print('finish loading files')
        out_path = os.path.join(out_dir, dataset[category])
        if os.path.exists(out_path) is False:
            os.makedirs(out_path)

        train_path = os.path.join(out_path, 'train')
        valid_path = os.path.join(out_path, 'valid')
        test_path = os.path.join(out_path, 'test')

        if os.path.exists(train_path) is False:
            os.makedirs(train_path)

        if os.path.exists(valid_path) is False:
            os.makedirs(valid_path)

        if os.path.exists(test_path) is False:
            os.makedirs(test_path)

        for i, stroke in enumerate(train_strokes):
            img_path = os.path.join(train_path, '%d.svg' % i)
            draw_strokes(stroke, img_path, width=48)  # Width: output size for .png files

            if i % 100 == 0:
                print('handled train %d' % i)

        for i, stroke in enumerate(valid_strokes):
            img_path = os.path.join(valid_path, '%d.svg' % i)
            draw_strokes(stroke, img_path, width=48)

            if i % 100 == 0:
                print('handled valid %d' % i)

        for i, stroke in enumerate(test_strokes):
            img_path = os.path.join(test_path, '%d.svg' % i)
            draw_strokes(stroke, img_path, width=48)

            if i % 100 == 0:
                print('handled test %d' % i)

if __name__ == "__main__":
    main()
