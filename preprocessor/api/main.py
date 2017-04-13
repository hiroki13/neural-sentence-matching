import sys
import argparse
import mode


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("-mode", type=str, default='tok/pos/srl')

    argparser.add_argument("--corpus", type=str)
    argparser.add_argument("--data_size", type=int, default=1000000)

    argv = argparser.parse_args()
    print
    print argv
    print

    mode.main(argv)
