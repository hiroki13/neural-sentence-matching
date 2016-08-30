import time

from ..utils import say, read_corpus, load_embedding_iterator, get_emb_layer, map_corpus, read_annotations,\
    create_eval_batches, create_batches
from ..model import attention_model, basic_model, bidirectional_model, grnn_model, ranking_model, alignment_model

PAD = "<padding>"


def train(args):
    say('\n\tSENTENCE MATCHING START\n\n')

    ##############
    # Load files #
    ##############
    # raw_corpus: {q_id: (title, body), ...}
    # embs: (word, vec)
    raw_corpus = read_corpus(args.corpus)
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
        dev = read_annotations(args.dev, n_negs=-1, prune_pos_cnt=-1)
        dev = create_eval_batches(ids_corpus, dev, padding_id, pad_left=not args.average)
    if args.test:
        test = read_annotations(args.test, n_negs=-1, prune_pos_cnt=-1)
        test = create_eval_batches(ids_corpus, test, padding_id, pad_left=not args.average)

    if args.train:
        start_time = time.time()
        train = read_annotations(path=args.train, data_size=args.data_size)
        train_batches = create_batches(ids_corpus=ids_corpus, data=train, batch_size=args.batch_size,
                                                padding_id=padding_id, pad_left=not args.average)
        say("{} to create batches\n".format(time.time() - start_time))
        say("{} batches, {} tokens in total, {} triples in total\n".format(
            len(train_batches),
            sum(len(x[0].ravel()) + len(x[1].ravel()) for x in train_batches),
            sum(len(x[2].ravel()) for x in train_batches)
        ))
        train_batches = None

        ###############
        # Set a model #
        ###############
        if args.attention:
            model = attention_model.Model(args, emb_layer)
            say('\nModel: Attention model\n')
        else:
            if args.bi:
                if args.double:
                    model = bidirectional_model.DoubleModel(args, emb_layer)
                    say('\nModel: Bidirectional Double Model\n')
                else:
                    model = bidirectional_model.Model(args, emb_layer)
                    say('\nModel: Bidirectional Model\n')
            elif args.ranking:
                model = ranking_model.Model(args, emb_layer)
                say('\nModel: Ranking Model\n')
            elif args.al:
                model = alignment_model.Model(args, emb_layer)
                say('\nModel: Alignment Model\n')
            else:
                if args.layer == 'grnn':
                    model = grnn_model.Model(args, emb_layer)
                    say('\nModel: GRNN Model\n')
                else:
                    model = basic_model.Model(args, emb_layer)
                    say('\nModel: Basic Model\n')

        model.compile()

        # set parameters using pre-trained network
        if args.load_pretrain:
            say('\nLoad pretrained parameters\n')
            model.load_pretrained_parameters(args)

        #################
        # Train a model #
        #################
        model.train(
            ids_corpus,
            dev if args.dev else None,
            test if args.test else None
        )


def main(args):
    train(args)

