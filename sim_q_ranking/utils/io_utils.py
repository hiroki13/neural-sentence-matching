import sys
import gzip
import random
from collections import Counter

import numpy as np

from ..nn.basic import EmbeddingLayer

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


def read_corpus(path):
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


def get_emb_layer(raw_corpus, n_d, embs=None, cut_off=2, unk=UNK, padding=PAD, fix_init_embs=True):
    cnt = Counter(w for q_id, pair in raw_corpus.iteritems() for x in pair for w in x)
    cnt[unk] = cut_off + 1
    cnt[padding] = cut_off + 1
    embedding_layer = EmbeddingLayer(
        n_d=n_d,
        # vocab = (w for w,c in cnt.iteritems() if c > cut_off),
        vocab=[unk, padding],
        embs=embs,
        fix_init_embs=fix_init_embs
    )
    return embedding_layer


def map_corpus(raw_corpus, embedding_layer, filter_oov, max_len=100):
    ids_corpus = {}
    for q_id, pair in raw_corpus.iteritems():
        item = (embedding_layer.map_to_ids(pair[0], filter_oov=filter_oov),
                embedding_layer.map_to_ids(pair[1], filter_oov=filter_oov)[:max_len])
        if len(item[0]) == 0:
            say("empty title after mapping to IDs. Doc No.{}\n".format(q_id))
        ids_corpus[q_id] = item
    return ids_corpus


def read_annotations(path, n_negs=20, prune_pos_cnt=10, data_size=1):
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


def create_batches(ids_corpus, data, batch_size, padding_id, data_indices=None, pad_left=True):
    """
    :param ids_corpus: {q_id: (title, body), ...}
    :param data: 1D: n_samples; elem=(qid=str, qids=[str, ..., ], qlabels=[0/1, ..., ])
    :param batch_size: int
    :param padding_id: int
    :param pad_left: boolean
    :return: lst: 1D: n_baches, 2D: n_pos_samples * batch_size; elem=(titles, bodies, triples)
                  titles (or bodies): 1D: n_words, 2D: n_cands,
                  triples: [query_index, pos_index, neg_index1, neg_index2, ...]
    """
    if data_indices is None:
        data_indices = range(len(data))
        random.shuffle(data_indices)

    n_data = len(data)
    cnt = 0
    pid2index = {}
    titles = []
    bodies = []
    triples = []
    batches = []
    for data_index in xrange(n_data):
        i = data_indices[data_index]
        query_id, cand_ids, qlabels = data[i]

        if query_id not in ids_corpus:
            continue

        cnt += 1
        for q_id in [query_id] + cand_ids:
            if q_id not in pid2index:
                if q_id not in ids_corpus:
                    continue

                """ Set index for each id """
                pid2index[q_id] = len(titles)

                t, b = ids_corpus[q_id]
                titles.append(t)
                bodies.append(b)

        query_id = pid2index[query_id]
        pos = [pid2index[q] for q, l in zip(cand_ids, qlabels) if l == 1 and q in pid2index]
        neg = [pid2index[q] for q, l in zip(cand_ids, qlabels) if l == 0 and q in pid2index]

        # 1D: batch_size, 2D: n_positive_samples; elem=[query_index, pos_index, neg1_index, neg2_index, ...]
        triples += [[query_id, x] + neg for x in pos]

        if cnt == batch_size or data_index == n_data - 1:
            titles, bodies = create_one_batch(titles, bodies, padding_id, pad_left)

            """
            triples (idps):
                Each row = [query_index, pos_index, neg_index_1, ..., neg_index_N]
                N = 20 - num_positive_questions
                [[0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                 [0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                 [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                 [21, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41],
                 [21, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41],
                 [21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]]
            """
            triples = create_hinge_batch(triples)  # padding
            batches.append((titles, bodies, triples))

            titles = []
            bodies = []
            triples = []
            pid2index = {}
            cnt = 0

    return batches


def create_eval_batches(ids_corpus, data, padding_id, pad_left):
    """
    :param ids_corpus: {q_id: (title, body), ...}
    :param data: 1D: n_samples; elem=(qid=str, qids=[str, ..., ], qlabels=[0/1, ..., ])
    :param padding_id: int
    :param pad_left: boolean
    :return: lst: 1D: n_baches: (titles, bodies, qlabels); titles (or bodies): 1D: n_words, 2D: n_cands
    """
    lst = []
    for pid, qids, qlabels in data:
        titles = []
        bodies = []
        for id in [pid] + qids:
            t, b = ids_corpus[id]
            titles.append(t)
            bodies.append(b)
        titles, bodies = create_one_batch(titles, bodies, padding_id, pad_left)
        lst.append((titles, bodies, np.array(qlabels, dtype="int32")))
    print '\nBatches: %d' % len(lst)
    return lst


def create_one_batch(titles1, bodies, padding_id, pad_left):
    max_title_len = max(1, max(len(x) for x in titles1))
    max_body_len = max(1, max(len(x) for x in bodies))
    if pad_left:
        titles = np.column_stack([np.pad(x, (max_title_len - len(x), 0), 'constant',
                                         constant_values=padding_id) for x in titles1])
        bodies = np.column_stack([np.pad(x, (max_body_len - len(x), 0), 'constant',
                                         constant_values=padding_id) for x in bodies])
    else:
        titles = np.column_stack([np.pad(x, (0, max_title_len - len(x)), 'constant',
                                         constant_values=padding_id) for x in titles1])
        bodies = np.column_stack([np.pad(x, (0, max_body_len - len(x)), 'constant',
                                         constant_values=padding_id) for x in bodies])
    return titles, bodies


def create_hinge_batch(triples):
    max_len = max(len(x) for x in triples)

    """
    [[  0   1   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20]
     [  0   2   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20]
     [  0   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20]]
    """

    triples = np.vstack([np.pad(x, (0, max_len - len(x)), 'edge')
                         for x in triples]).astype('int32')

    """
    Padded
    [[  0   1   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  20  20]
     [  0   2   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  20  20]
     [  0   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  20  20]]
    """
    return triples
