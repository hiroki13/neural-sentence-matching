from collections import Counter

from tokenizer import tokenize_msr_corpus

from ..utils.io_utils import say, PAD, UNK
from ..utils.loader import load_ubuntu_corpus, load_msr_corpus, load_embedding_iterator, load_annotations
from ..utils.feat_factory import map_word_feat_to_id, map_sem_feat_to_id, map_label_to_id, get_msr_feats, get_msr_sem_feats, get_sem_role_list, map_corpus, map_msr_corpus
from ..utils.sample_factory import get_3d_batch, get_3d_batch_samples, create_batches, create_eval_batches
from ..nn.basic import EmbeddingLayer


def get_ubuntu_samples(args):
    # raw_corpus: {q_id: (title, body), ...}
    # embs: (word, vec)
    raw_corpus = load_ubuntu_corpus(args.corpus)
    embs = load_embedding_iterator(args.embeddings) if args.embeddings else None

    ######################
    # Set the vocabulary #
    ######################
    emb_layer = get_emb_layer(raw_corpus=raw_corpus, n_d=args.hidden_dim, cut_off=args.cut_off, embs=embs)
    ids_corpus = map_corpus(raw_corpus, emb_layer, filter_oov=args.filter_oov, max_len=args.max_seq_len)
    say("vocab size={}, corpus size={}\n".format(emb_layer.n_V, len(raw_corpus)))
    padding_id = emb_layer.vocab_map[PAD]

    ################
    # Set datasets #
    ################
    if args.dev:
        dev_dataset = load_annotations(args.dev, n_negs=-1, prune_pos_cnt=-1)
        dev_dataset = create_eval_batches(ids_corpus, dev_dataset, padding_id, pad_left=not args.average)
    else:
        dev_dataset = None

    if args.test:
        test_dataset = load_annotations(args.test, n_negs=-1, prune_pos_cnt=-1)
        test_dataset = create_eval_batches(ids_corpus, test_dataset, padding_id, pad_left=not args.average)
    else:
        test_dataset = None

    if args.train:
        train_dataset = load_annotations(path=args.train, data_size=args.data_size)
        train_batches = create_batches(ids_corpus=ids_corpus, data=train_dataset, batch_size=args.batch_size,
                                       padding_id=padding_id, pad_left=not args.average)
        say("{} batches, {} tokens in total, {} triples in total\n".format(
            len(train_batches),
            sum(len(x[0].ravel()) + len(x[1].ravel()) for x in train_batches),
            sum(len(x[2].ravel()) for x in train_batches)
        ))

    return ids_corpus, dev_dataset, test_dataset, emb_layer


def get_msr_samples_old(args, emb_layer):

    train_corpus = preprocess_msr_corpus(args.train)
    test_corpus = preprocess_msr_corpus(args.test)
    train_corpus = map_msr_corpus(train_corpus, emb_layer, filter_oov=args.filter_oov)
    test_corpus = map_msr_corpus(test_corpus, emb_layer, filter_oov=args.filter_oov)
    say("vocab size={}, train corpus size={}, test corpus size={}\n".format(
        emb_layer.n_V, len(train_corpus), len(test_corpus)))
    pad_id = emb_layer.vocab_map[PAD]

    # 1D: n_batches, 2D: max_n_words, 3D: batch; word (label) id
    train_samples = get_3d_batch(train_corpus, args.batch_size, pad_id)
    test_samples = get_3d_batch(test_corpus, 1, pad_id)

    return train_samples, test_samples


def get_msr_samples(args, emb_layer):
    train_corpus = load_msr_corpus(args.train)
    test_corpus = load_msr_corpus(args.test)

    train_feats = get_msr_feats(train_corpus)
    test_feats = get_msr_feats(test_corpus)

    train_w = map_word_feat_to_id(train_feats, emb_layer, filter_oov=args.filter_oov)
    test_w = map_word_feat_to_id(test_feats, emb_layer, filter_oov=args.filter_oov)

    train_y = map_label_to_id(train_corpus)
    test_y = map_label_to_id(test_corpus)

    say("vocab size={}, train corpus size={}, test corpus size={}\n".format(
        emb_layer.n_V, len(train_corpus), len(test_corpus)))

    pad_id_w = emb_layer.vocab_map[PAD]
    pad_ids = [pad_id_w]

    # 1D: n_batches, 2D: max_n_words, 3D: batch; word (label) id
    train_samples = get_3d_batch_samples([train_w, train_y], args.batch_size, pad_ids)
    test_samples = get_3d_batch_samples([test_w, test_y], 1, pad_ids)

    return train_samples, test_samples


def get_msr_srl_samples(args, emb_layer):
    train_corpus = load_msr_corpus(args.train)
    test_corpus = load_msr_corpus(args.test)

    # 1D: (feats1, feats2); feats1: 1D: n_pairs, 2D: (words1, pos1, preds1, sem_roles1)
    train_feats = get_msr_sem_feats(train_corpus)
    test_feats = get_msr_sem_feats(test_corpus)

    train_sem_role_corpus = get_sem_role_list(train_feats)
    emb_layer_sem = get_emb_layer(raw_corpus=train_sem_role_corpus,
                                  n_d=emb_layer.n_d,
                                  embs=None,
                                  cut_off=0,
                                  fix_init_embs=False)

    # 1D: n_pairs, 2D: 2, 3D: n_words
    train_w = map_word_feat_to_id(train_feats, emb_layer, filter_oov=args.filter_oov)
    test_w = map_word_feat_to_id(test_feats, emb_layer, filter_oov=args.filter_oov)

    # 1D: n_pairs, 2D: 2, 3D: n_words, 4D: n_props
    train_s = map_sem_feat_to_id(train_feats, emb_layer_sem, filter_oov=args.filter_oov)
    test_s = map_sem_feat_to_id(test_feats, emb_layer_sem, filter_oov=args.filter_oov)

    train_y = map_label_to_id(train_corpus)
    test_y = map_label_to_id(test_corpus)

    say("vocab size={}, train corpus size={}, test corpus size={}\n".format(
        emb_layer.n_V, len(train_corpus), len(test_corpus)))

    pad_id_w = emb_layer.vocab_map[PAD]
    pad_id_s = emb_layer_sem.vocab_map[PAD]
    pad_ids = [pad_id_w, pad_id_s]

    # 1D: n_batches, 2D: max_n_words, 3D: batch; word (label) id
    train_samples = get_3d_batch_samples([train_w, train_s, train_y], args.batch_size, pad_ids)
    test_samples = get_3d_batch_samples([test_w, test_s, test_y], 1, pad_ids)

    return train_samples, test_samples, emb_layer_sem


def lower_msr_corpus(corpus):
    lowered_corpus = []
    for sample in corpus:
        sent1 = [w.lower() for w in sample[1]]
        sent2 = [w.lower() for w in sample[2]]
        lowered_corpus.append([sample[0], sent1, sent2])
    return lowered_corpus


def preprocess_msr_corpus(path):
    corpus = load_msr_corpus(path)
    corpus = tokenize_msr_corpus(corpus)
    corpus = lower_msr_corpus(corpus)
    return corpus


def get_emb_layer(raw_corpus, n_d, embs=None, cut_off=1, unk=UNK, padding=PAD, fix_init_embs=True):
    vocab = [unk, padding]
    if raw_corpus:
        cnt = Counter(w for w in raw_corpus)
        vocab = [w for w, c in cnt.iteritems() if c > cut_off]
        vocab.append(unk)
        vocab.append(padding)

    embedding_layer = EmbeddingLayer(
        n_d=n_d,
        vocab=vocab,
        embs=embs,
        fix_init_embs=fix_init_embs
    )

    return embedding_layer




