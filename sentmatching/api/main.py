import sys
import argparse

import numpy as np
import theano

import train

np.random.seed(0)
theano.config.floatX = 'float32'


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--task", type=str, default='sqr')

    ########
    # Data #
    ########
    argparser.add_argument("--data_type", type=str, default="base")
    argparser.add_argument("--corpus", type=str)
    argparser.add_argument("--train", type=str, default="")
    argparser.add_argument("--test", type=str, default="")
    argparser.add_argument("--dev", type=str, default="")
    argparser.add_argument("--embeddings", type=str, default="")

    ###################################
    # NN Architecture Hyperparameters #
    ###################################
    argparser.add_argument("--hidden_dim", "-d", type=int, default=200)
    argparser.add_argument("--learning", type=str, default="adam")
    argparser.add_argument("--learning_rate", "--lr", type=float, default=0.001)
    argparser.add_argument("--l2_reg", type=float, default=1e-5)
    argparser.add_argument("--activation", "-act", type=str, default="tanh")
    argparser.add_argument("--depth", type=int,default=1)
    argparser.add_argument("--dropout", type=float, default=0.0)
    argparser.add_argument("--normalize", type=int, default=1)
    argparser.add_argument("--average", type=int, default=0)
    argparser.add_argument("--order", type=int, default=2)
    argparser.add_argument("--layer", type=str, default="rcnn")
    argparser.add_argument("--mode", type=int, default=1)
    argparser.add_argument("--outgate", type=int, default=0)

    #########################
    # Preprocess parameters #
    #########################
    argparser.add_argument("--cut_off", type=int, default=1)
    argparser.add_argument("--max_seq_len", type=int, default=100)
    argparser.add_argument("--data_size", type=int, default=1)
    argparser.add_argument("--filter_oov", type=int, default=1)
    argparser.add_argument("--body", type=int, default=0)

    #######################
    # Training parameters #
    #######################
    argparser.add_argument("--max_epoch", type=int, default=50)
    argparser.add_argument("--batch_size", type=int, default=40)
    argparser.add_argument("--loss", type=int, default=0)
    argparser.add_argument("--reweight", type=int, default=1)
    argparser.add_argument("--save_model", type=str, default="")
    argparser.add_argument("--load_pretrain", type=str, default="")

    ###################
    # Model Selection #
    ###################
    argparser.add_argument("--model", type=str, default='basic')
    argparser.add_argument("--model_type", type=int, default=0)

    args = argparser.parse_args()
    print
    print args
    print

    train.main(args)
