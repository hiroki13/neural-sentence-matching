import time

from ..utils.io_utils import say, PAD
from ..utils.loader import load_ubuntu_corpus, load_msr_corpus, load_embedding_iterator, load_annotations
from ..utils.preprocess import get_3d_batch_samples, map_word_feat_to_id, map_sem_feat_to_id, map_label_to_id, preprocess_msr_corpus, get_msr_sem_feats, get_sem_role_corpus, map_corpus, map_msr_corpus, map_msr_srl_corpus, get_emb_layer, get_3d_batch, create_batches, create_eval_batches
from ..model import basic_model, attention_model, alignment_model, sent_matching_model


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
        start_time = time.time()
        train_dataset = load_annotations(path=args.train, data_size=args.data_size)
        train_batches = create_batches(ids_corpus=ids_corpus, data=train_dataset, batch_size=args.batch_size,
                                       padding_id=padding_id, pad_left=not args.average)
        say("{} to create batches\n".format(time.time() - start_time))
        say("{} batches, {} tokens in total, {} triples in total\n".format(
            len(train_batches),
            sum(len(x[0].ravel()) + len(x[1].ravel()) for x in train_batches),
            sum(len(x[2].ravel()) for x in train_batches)
        ))

    return ids_corpus, dev_dataset, test_dataset, emb_layer


def get_msr_samples(args, emb_layer):

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


def get_msr_srl_samples(args, emb_layer):

    train_corpus = load_msr_corpus(args.train)
    test_corpus = load_msr_corpus(args.test)

    train_feats = get_msr_sem_feats(train_corpus)
    test_feats = get_msr_sem_feats(test_corpus)

    train_sem_role_corpus = get_sem_role_corpus(train_feats)
    emb_layer_sem = get_emb_layer(raw_corpus=train_sem_role_corpus, n_d=args.hidden_dim, embs=None, cut_off=0)

    train_w = map_word_feat_to_id(train_feats, emb_layer, filter_oov=args.filter_oov)
    test_w = map_word_feat_to_id(test_feats, emb_layer, filter_oov=args.filter_oov)

    train_s = map_sem_feat_to_id(train_feats, emb_layer_sem, filter_oov=args.filter_oov)
    test_s = map_sem_feat_to_id(test_feats, emb_layer_sem, filter_oov=args.filter_oov)

    train_y = map_label_to_id(train_corpus)
    test_y = map_label_to_id(test_corpus)

#    train_corpus = map_msr_srl_corpus(train_corpus, emb_layer, filter_oov=args.filter_oov)
#    test_corpus = map_msr_srl_corpus(test_corpus, emb_layer, filter_oov=args.filter_oov)

    say("vocab size={}, train corpus size={}, test corpus size={}\n".format(
        emb_layer.n_V, len(train_corpus), len(test_corpus)))
    pad_id = emb_layer.vocab_map[PAD]

    # 1D: n_batches, 2D: max_n_words, 3D: batch; word (label) id
    train_samples = get_3d_batch_samples([train_w, train_y], args.batch_size, pad_id)
    test_samples = get_3d_batch_samples([test_w, test_y], 1, pad_id)
#    train_samples = get_3d_batch(train_corpus, args.batch_size, pad_id)
#    test_samples = get_3d_batch(test_corpus, 1, pad_id)

    return train_samples, test_samples


def select_model(args, emb_layer):
    if args.model == 'attention':
        model = attention_model.Model(args, emb_layer)
        say('\nModel: Attention model\n')
    elif args.model == 'alignment':
        if args.model_type == 1:
            model = alignment_model.AverageModel(args, emb_layer)
            say('\nModel: Alignment Model: Average\n')
        elif args.model_type == 2:
            model = alignment_model.WeightedAverageModel(args, emb_layer)
            say('\nModel: Alignment Model: Weighted average\n')
        elif args.model_type == 3:
            model = alignment_model.AlignmentModel(args, emb_layer)
            say('\nModel: Alignment Model: Standard\n')
        else:
            model = alignment_model.AlignmentVectorModel(args, emb_layer)
            say('\nModel: Alignment Model: Attention vector\n')
    else:
        model = basic_model.Model(args, emb_layer)
        say('\nModel: Basic Model\n')
    return model


def train_pi(args):
    embs = load_embedding_iterator(args.embeddings) if args.embeddings else None
    emb_layer = get_emb_layer(raw_corpus=None, n_d=args.hidden_dim, cut_off=args.cut_off, embs=embs)

#    train_samples, test_samples = get_msr_samples(args, emb_layer)
    train_samples, test_samples = get_msr_srl_samples(args, emb_layer)

    model = sent_matching_model.Model(args, emb_layer)
    model.compile()

    model.train(train_samples, dev_samples=test_samples)


def train(args):
    ids_corpus, dev_samples, test_samples, emb_layer = get_ubuntu_samples(args)

    model = select_model(args, emb_layer)
    model.compile()

    # set parameters using pre-trained network
    if args.load_pretrain:
        say('\nLoad pretrained parameters\n')
        model.load_pretrained_parameters(args)

    model.train(
        ids_corpus,
        dev_samples if args.dev else None,
        test_samples if args.test else None
    )


def main(args):
    say('\n\tSENTENCE MATCHING START\n\n')

    if args.task == 'sqr':
        train(args)
    else:
        train_pi(args)


