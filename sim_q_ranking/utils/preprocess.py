import random
from collections import Counter

import numpy as np

from io_utils import PAD, UNK, load_msr_corpus
from ..utils.tokenizer import tokenize
from ..nn.basic import EmbeddingLayer


def tokenize_msr_corpus(corpus):
    return [[sample[0], tokenize(sample[1]), tokenize(sample[2])] for sample in corpus]


def lower_msr_corpus(corpus):
    lowered_corpus = []
    for sample in corpus:
        sent1 = [w.lower() for w in sample[1]]
        sent2 = [w.lower() for w in sample[2]]
        lowered_corpus.append([sample[0], sent1, sent2])
    return lowered_corpus


def get_msr_corpus(path):
    corpus = load_msr_corpus(path)
    corpus = tokenize_msr_corpus(corpus)
    corpus = lower_msr_corpus(corpus)
    return corpus


def get_emb_layer(raw_corpus, n_d, embs=None, cut_off=2, unk=UNK, padding=PAD, fix_init_embs=True):
    if raw_corpus:
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
        ids_corpus[q_id] = item
    return ids_corpus


def map_msr_corpus(corpus, embedding_layer, filter_oov):
    ids_corpus = []
    for sample in corpus:
        sent1 = embedding_layer.map_to_ids(sample[1], filter_oov=filter_oov)
        sent2 = embedding_layer.map_to_ids(sample[2], filter_oov=filter_oov)
        ids_corpus.append([sample[0], sent1, sent2])
    return ids_corpus


def get_3d_batch(samples, batch_size=32, pad_id=0):
    """
    :param samples: 1D: n_samples, 2D: [label, sent1, sent2]; sent=1D np.array
    :return:
    """
    batches = []
    one_batch_x = []
    one_batch_y = []

    for sample in samples:
        one_batch_x.extend(sample[1:])
        one_batch_y.append(sample[0])

        if len(one_batch_y) == batch_size:
            batch_x = padding(one_batch_x, pad_id)
            batch_y = np.asarray(one_batch_y, dtype='float32')
            batches.append((batch_x, batch_y))
            one_batch_x = []
            one_batch_y = []

    if one_batch_x:
        batch_x = padding(one_batch_x, pad_id)
        batch_y = np.asarray(one_batch_y, dtype='float32')
        batches.append((batch_x, batch_y))

    return batches


def padding(matrix, pad_id=0):
    max_column_len = max(1, max(len(row) for row in matrix))
    return np.column_stack([np.pad(row, (max_column_len - len(row), 0), 'constant',
                                   constant_values=pad_id) for row in matrix])


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
