# neural-sentence-matching

This repo contains Theano implementations of our original models and the models described in the following papers:

[Semi-supervised Question Retrieval with Gated Convolutions](http://arxiv.org/abs/1512.05726). NAACL 2016

These codes are based on the following codes:

[Recurrent & convolutional neural network modules](https://github.com/taolei87/rcnn)


## Similar Question Retrieval

These neural methods are used to calculate text similarity, for applications such as similar question retrieval in community-based QA forums.

##### Data
The data can be downloaded at this [repo](https://github.com/taolei87/askubuntu).

##### Dependencies
To run the code, you need the following extra packages installed:
  - [PrettyTable](https://pypi.python.org/pypi/PrettyTable)
  - Numpy and Theano

##### Usage
  1. Clone this repo
  2. Run `python -m sim_q_ranking.main.main --help` to see all running options

##### Example Comand
  - Basic Model: `python -m sim_q_ranking.main.main --corpus path/to/data/text_tokenized.txt.gz --embeddings path/to/data/vector/vectors_pruned.200.txt.gz --train path/to/data/train_random.txt --dev path/to/data/dev.txt --test path/to/data/test.txt --layer rcnn`
  - Attention Model: `python -m sim_q_ranking.main.main --corpus path/to/data/text_tokenized.txt.gz --embeddings path/to/data/vector/vectors_pruned.200.txt.gz --train path/to/data/train_random.txt --dev path/to/data/dev.txt --test path/to/data/test.txt --layer rcnn --model attention`
  - Alignment Model: `python -m sim_q_ranking.main.main --corpus path/to/data/text_tokenized.txt.gz --embeddings path/to/data/vector/vectors_pruned.200.txt.gz --train path/to/data/train_random.txt --dev path/to/data/dev.txt --test path/to/data/test.txt --layer rcnn --model alignment`

