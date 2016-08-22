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
  2. Use “export PYTHONPATH=/path/to/neural-sentence-matching-system/code” to add the neural-sentence-matching-system/code directory to Python library
  3. Run `python code/main/main.py --help` to see all running options

##### Example Comand
  - Basic RCNN Model: `python code/main/main.py --corpus path/to/data/text_tokenized.txt.gz --embeddings path/to/data/vector/vectors_pruned.200.txt.gz --train path/to/data/train_random.txt --dev path/to/data/dev.txt --test path/to/data/test.txt`
  - Attention RCNN Model: `python code/main/main.py --corpus path/to/data/text_tokenized.txt.gz --embeddings path/to/data/vector/vectors_pruned.200.txt.gz --train path/to/data/train_random.txt --dev path/to/data/dev.txt --test path/to/data/test.txt --attention 1`
