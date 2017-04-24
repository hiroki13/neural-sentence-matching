from ..utils.io_utils import say, PAD
from ..utils.loader import load_msr_corpus, load_embedding_iterator
from ..utils.feat_factory import map_word_feat_to_id, map_sem_feat_to_id, map_label_to_id, get_msr_sem_feats, get_sem_role_list
from ..utils.sample_factory import get_3d_batch_samples
from ..utils.preprocess import get_emb_layer
from ..model import sent_matching_model


def main(args):
    embs = load_embedding_iterator(args.embeddings) if args.embeddings else None
    emb_layer = get_emb_layer(raw_corpus=None, n_d=args.hidden_dim, cut_off=args.cut_off, embs=embs)

    train_samples, test_samples, emb_layer_sem = _get_msr_srl_samples(args, emb_layer)

    model = sent_matching_model.SemModel(args, [emb_layer, emb_layer_sem])
    model.compile()
    train_func = _get_train_func(model)
    eval_func = model.get_eval_func()
    model.train(train_func=train_func,
                eval_func=eval_func,
                train_samples=train_samples,
                dev_samples=test_samples)


def _get_msr_srl_samples(args, emb_layer):
    train_corpus = load_msr_corpus(args.train)
    test_corpus = load_msr_corpus(args.test)

    # 1D: (feats1, feats2); feats1: 1D: n_pairs, 2D: (words1, pos1, preds1, sem_roles1)
    train_feats = get_msr_sem_feats(train_corpus)
    test_feats = get_msr_sem_feats(test_corpus)

    train_sem_role_corpus = get_sem_role_list(train_feats)
    emb_layer_sem = get_emb_layer(raw_corpus=train_sem_role_corpus, n_d=emb_layer.n_d, embs=None, cut_off=0, fix_init_embs=False)

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


def _get_train_func(model):
    return model.get_train_func()

