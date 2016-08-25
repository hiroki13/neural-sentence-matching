import sys
import argparse

import theano

import train

theano.config.floatX = 'float32'


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument("--corpus", type=str)
    argparser.add_argument("--train", type=str, default="")
    argparser.add_argument("--test", type=str, default="")
    argparser.add_argument("--dev", type=str, default="")
    argparser.add_argument("--embeddings", type=str, default="")
    argparser.add_argument("--hidden_dim", "-d", type=int, default=200)
    argparser.add_argument("--learning", type=str, default="adam")
    argparser.add_argument("--learning_rate", type=float, default=0.001)
    argparser.add_argument("--l2_reg", type=float, default=1e-5)
    argparser.add_argument("--activation", "-act", type=str, default="tanh")
    argparser.add_argument("--batch_size", type=int, default=40)
    argparser.add_argument("--depth", type=int,default=1)
    argparser.add_argument("--dropout", type=float, default=0.0)
    argparser.add_argument("--max_epoch", type=int, default=50)
    argparser.add_argument("--cut_off", type=int, default=1)
    argparser.add_argument("--max_seq_len", type=int, default=100)
    argparser.add_argument("--normalize", type=int, default=1)
    argparser.add_argument("--reweight", type=int, default=1)
    argparser.add_argument("--order", type=int, default=2)
    argparser.add_argument("--layer", type=str, default="rcnn")
    argparser.add_argument("--mode", type=int, default=1)
    argparser.add_argument("--outgate", type=int, default=0)
    argparser.add_argument("--load_pretrain", type=str, default="")
    argparser.add_argument("--average", type=int, default=0)
    argparser.add_argument("--save_model", type=str, default="")
    argparser.add_argument("--data_size", type=int, default=1)
    argparser.add_argument("--attention", type=int, default=0)
    argparser.add_argument("--check", type=int, default=0)
    argparser.add_argument("--bi", type=int, default=0)
    argparser.add_argument("--body", type=int, default=0)
    argparser.add_argument("--double", type=int, default=0)
    argparser.add_argument("--loss", type=str, default='mm')
    argparser.add_argument("--beta", type=float, default=0.9)
    argparser.add_argument("--a_type", type=str, default='st')
    argparser.add_argument("--share_w", type=int, default=0)
    argparser.add_argument("--ranking", type=int, default=0)
    argparser.add_argument("--sim", type=int, default=0)
    argparser.add_argument("--fix", type=int, default=0)

    args = argparser.parse_args()
    print
    print args
    print

    train.main(args)
