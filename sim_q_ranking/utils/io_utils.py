import sys
import gzip
import random

import numpy as np

PAD = '<padding>'
UNK = '<unk>'


def say(s, stream=sys.stdout):
    stream.write(s)
    stream.flush()


def load_embedding_iterator(path):
    file_open = gzip.open if path.endswith(".gz") else open
    with file_open(path) as fin:
        for line in fin:
            line = line.strip()
            if line:
                parts = line.split()
                word = parts[0]
                vals = np.array([float(x) for x in parts[1:]])
                yield word, vals


def load_ubuntu_corpus(path):
    empty_cnt = 0
    raw_corpus = {}
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path) as fin:
        for line in fin:
            q_id, title, body = line.split("\t")
            if len(title) == 0:
                print q_id
                empty_cnt += 1
                continue
            title = title.strip().split()
            body = body.strip().split()
            raw_corpus[q_id] = (title, body)
    say("{} empty titles ignored.\n".format(empty_cnt))
    return raw_corpus


def load_msr_corpus(path):
    corpus = []
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path) as fin:
        for line in fin:
            line = line.rstrip()
            line = line.split("\t")
            label = line[0]
            if label == '0' or label == '1':
                corpus.append([int(label), line[-2], line[-1]])
    return corpus


def load_annotations(path, n_negs=20, prune_pos_cnt=10, data_size=1):
    """
    :param path: file location
    :param n_negs: how many negative samples to be used
    :param prune_pos_cnt: threshold for positive samples
    :param data_size: threshold for sample size
    :return: lst: 1D: n_samples; elem=(qid=str, qids=[str, ..., ], qlabels=[0/1, ..., ])
    """
    lst = []
    with open(path) as fin:
        for line in fin:
            parts = line.split("\t")
            pid, pos, neg = parts[:3]  # example: [2614\t6552 298\t2222 4324 981]
            pos = pos.split()
            neg = neg.split()

            ##########################
            # Check positive samples #
            ##########################
            if len(pos) == 0 or len(pos) > prune_pos_cnt != -1:
                continue

            ############################
            # Pick up negative samples #
            ############################
            if n_negs != -1:
                random.shuffle(neg)
                neg = neg[:n_negs]

            s = set()
            qids = []
            qlabels = []

            ########################
            # Set negative samples #
            ########################
            for q in neg:
                if q not in s:
                    qids.append(q)
                    """ Set labels """
                    qlabels.append(0 if q not in pos else 1)
                    s.add(q)

            ########################
            # Set positive samples #
            ########################
            for q in pos:
                if q not in s:
                    qids.append(q)
                    """ Set labels """
                    qlabels.append(1)
                    s.add(q)

            lst.append((pid, qids, qlabels))

    if data_size > 1:
        cut_off = len(lst) / data_size
        lst = lst[:cut_off]

    print '\nSamples: %d' % len(lst)
    return lst
