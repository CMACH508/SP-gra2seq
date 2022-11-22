# SP-gra2seq: Linking Sketch Patches by Learning Synonymous Proximity for Graphic Sketch Representation

<img src="https://github.com/SCZang/SP-gra2seq/blob/main/assets/synonym.png" width="400" alt="synonym"/>

Graphic sketch representations are effective for representing sketches. Existing methods (e.g., [SketchHealer](https://openresearch.surrey.ac.uk/esploro/outputs/conferencePresentation/SketchHealer-A-Graph-to-Sequence-Network-for-Recreating/99514536202346?institution=44SUR_INST), [SketchLattice](https://openaccess.thecvf.com/content/ICCV2021/html/Qi_SketchLattice_Latticed_Representation_for_Sketch_Manipulation_ICCV_2021_paper.html)) take the patches cropped from sketches as the graph nodes, and construct the edges based on sketch's drawing order or Euclidean distances on the canvas. However, the drawing order of a sketch may not be unique, while the patches from semantically related parts of a sketch may be far away from each other on the canvas. SP-gra2seq constructs the graph edges by linking the sketch patches with the analogous semantic contents or geometric shapes, namely the synonymous proximity. Accordingly, SP-gra2seq is an order-invariant, semantics-aware method for learning the graphic sketch representations. 

<img src="https://github.com/SCZang/SP-gra2seq/blob/main/assets/overview.png" width="800" alt="overview"/>

When training an SP-gra2seq, a sketch is cropped into patches which are embedded by the convolutional neural network (CNN) encoder. We compute the cosine similarity between every pair of the patch embeddings as the evaluation of the introduced *synonymous proximity*. Each patch is linked to the patches with the top-2 values of the cosine similarity. The constructed graph edges enable the message passing between intra-sketch patches by the graph convolutional network (GCN) encoder, and the final sketch code is sent into the recurrent neural network (RNN) decoder to reconstruct the input sketch. Furthermore, we enforce a clustering constraint over the embeddings jointly with the network learning to raise the accuracy of the computed synonymous proximity.

The corresponding article was accepted by **AAAI 2023**, and an early version will be available in the [arXiv link](). This repo will contain the TensorFlow code, the pre-trained models for SP-gra2seq in the early future.

# Training an SP-gra2seq

## Dataset

Before training an SP-gra2seq, you first need to rasterize the original sketch sequences from [QuickDraw dataset](https://quickdraw.withgoogle.com/data) to sketch images and crop sketch patches on the canvas. Our cropping method, which is the function `make_graph_` in `utils.py`, is based on the one used by SketchHealer, whose original codes can be found in [link](https://github.com/sgybupt/SketchHealer). The cropping process is automaticlly applied during the training.

## Required environments

1. Python 3.6
2. Tensorflow 1.12

## Training
```
python train.py --log_root=checkpoint_path --data_dir=dataset_path --resume_training=False --hparams="categories=[bee,bus]"
```

`checkpoint_path` and `dataset_path` denote the model saving directory and the dataset directory, respectively. For the `hparams`, we provide a list of full options for training an SP-gra2seq, along with the default settings:
```
categories=['bee', 'bus'],         # Sketch categories for training
num_steps=1000001,                 # Number of total steps (the process will stop automatically if the loss is not improved)
save_every=1,                      # Number of epochs per checkpoint creation
dec_rnn_size=1024,                 # Size of decoder
dec_model='lstm',                  # Decoder: lstm, layer_norm or hyper
max_seq_len=-1,                    # Max sequence length. Computed by DataLoader
z_size=128,                        # Dimension of latent code
batch_size=128,                    # Minibatch size
graph_number=21,                   # Number of graph nodes of a sketch
graph_picture_size=256,            # Cropped patch size
num_mixture=30,                    # Number of clusters
learning_rate=0.001,               # Learning rate
decay_rate=0.9999,                 # Learning rate decay per minibatch.
min_learning_rate=0.00001,         # Minimum learning rate
grad_clip=1.,                      # Gradient clipping
use_recurrent_dropout=False,       # Dropout with memory loss
recurrent_dropout_prob=0.0,        # Probability of recurrent dropout keep
use_input_dropout=False,           # Input dropout
input_dropout_prob=0.0,            # Probability of input dropout keep
use_output_dropout=False,          # Output droput
output_dropout_prob=0.0,           # Probability of output dropout keep
random_scale_factor=0.0,           # Random scaling data augmention proportion
augment_stroke_prob=0.0,           # Point dropping augmentation proportion
is_training=True,                  # Training mode or not
```

We also provide three pre-trained SP-pix2seq models corresponding to the three datasets used in the article, and you can get them from [link]() in the early future.

## Generating
```
python sample.py --data_dir=dataset_path --model_dir=checkpoint_path --output_dir=output_path --num_per_category=300 --prob=0.1
```

With a pre-trained model, you can generate sketches based on the input (corrupted) sketches. `output_dir`, `num_per_category` and `prob` denotes the directory for the generated sketches, the number of the generated sketches per category and the masking probability for sketch healing task (if needed).

## Evaluation

The metrics **Rec** and **Ret** are used to testify whether a method learns accurate and robust sketch representations. For calculating **Rec**, you need to train a [Sketch_a_net](https://arxiv.org/pdf/1501.07873.pdf) for each dataset as the classifier. And for **Ret**, you can run `retrieval.py` to obtain it with the generated sketches (2500 sketches per category). The following figure presents the detail calculations of both metrics for controllable sketch synthesis and sketch healing, respectively.
```
python retrieval.py --data_dir=dataset_path --model_dir=checkpoint_path --sample_dir=output_path
```
`sample_dir` indicates the directory for storing the generated sketches.

<img src="https://github.com/SCZang/SP-gra2seq/blob/main/assets/metrics.png" width="650" alt="metrics"/>

* Please make sure both the metrics are computed with the entire test set (i.e., --num_per_category=2500).

* We also provide the random seeds in `random_seed.npy` for creating the random masks for sketch healing. These seeds are the specific ones utilized in the article for the sketch healing performance evaluation. You can use them to make a fair comparision with the benchmarks in the article.

## Masking Approach

<img src="https://github.com/SCZang/SP-gra2seq/blob/main/assets/masking.png" width="800" alt="masking"/>

The figure above presents four different approaches for creating corrupted sketches for sketch healing: (a) our approach utilized in the article, (b) the approach utilized in [SketchHealer](https://github.com/sgybupt/SketchHealer), (c) the approach utilized in [SketchLattice](https://github.com/qugank/sketch-lattice.github.io) and (d) our approach adjusted for SketchLattice.

In sub-figure (a), we separate the patch cropping from the canvas masking in the pipeline. After positioning all patch centers on the canvas, we randomly select the patch centers (e.g., the patch C in the sub-figure) by a probability (10% or 30%) and remove their corresponding patches, i.e., masking. After all the selected patches are removed, we finally crop patches at the same patch centers from the corrupted canvas. The graph edges linked to the masked patches are cut off as well. Especially, the patches A and C, the patches B and C are with overlapped regions, respectively, but no additional information below the masked patch C are leaked out to neither the patch A nor B.

For the masking approach of SketchHealer shown in sub-figure (b), cropping and masking patches are applied by turns with the sketch drawing order. When two patches B and C share an overlapped region, B is cropped in front of C without being masked, but C is masked. The pixels located in the overlap leak out to the patch B, making the corrupted sketches much easier to be represented.

In sub-figure (c), SketchLattice firstly creates a lattice on the sketch canvas and obtains all the coordinates, which are the overlapping points between strokes and lattice. Then it applies a point-level masking by randomly dropping a fraction of lattice points (the gray points are dropped) to determine the finally selected coordinates for learning graphic representation. The masked region (masked points exactly) is much smaller than ours by patch-level masking.

We also adjust our masking approach for SketchLattice, shown in sub-figure (d), ensuring that the corrupted sketches fed to SketchLattice share the same corrupting level with other models. The sketch masking and coordinate selecting are separately applied by two steps. More specifically, the lattice is created after the sketch masking, and more coordinates may be dropped comparing with sub-figure (c).

# Citation
This work is proposed by Sicong Zang, Shikui Tu and Lei Xu from Shanghai Jiao Tong University. The citation is coming soon.
