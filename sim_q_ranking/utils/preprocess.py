import random
from collections import Counter

import numpy as np

from io_utils import PAD, UNK
from ..nn.basic import EmbeddingLayer


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
#        if len(item[0]) == 0:
#            say("empty title after mapping to IDs. Doc No.{}\n".format(q_id))
        ids_corpus[q_id] = item
    return ids_corpus


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
