import time

from ..utils.io_utils import say, load_corpus, load_embedding_iterator, load_annotations, PAD
from ..utils.preprocess import map_corpus, get_emb_layer, create_batches, create_eval_batches
from ..model import basic_model, attention_model, alignment_model


def train(args):
    say('\n\tSENTENCE MATCHING START\n\n')

    ##############
    # Load files #
    ##############
    # raw_corpus: {q_id: (title, body), ...}
    # embs: (word, vec)
    raw_corpus = load_corpus(args.corpus)
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
        dev = load_annotations(args.dev, n_negs=-1, prune_pos_cnt=-1)
        dev = create_eval_batches(ids_corpus, dev, padding_id, pad_left=not args.average)
    if args.test:
        test = load_annotations(args.test, n_negs=-1, prune_pos_cnt=-1)
        test = create_eval_batches(ids_corpus, test, padding_id, pad_left=not args.average)

    if args.train:
        start_time = time.time()
        train = load_annotations(path=args.train, data_size=args.data_size)
        train_batches = create_batches(ids_corpus=ids_corpus, data=train, batch_size=args.batch_size,
                                       padding_id=padding_id, pad_left=not args.average)
        say("{} to create batches\n".format(time.time() - start_time))
        say("{} batches, {} tokens in total, {} triples in total\n".format(
            len(train_batches),
            sum(len(x[0].ravel()) + len(x[1].ravel()) for x in train_batches),
            sum(len(x[2].ravel()) for x in train_batches)
        ))
        train_batches = None

        ##################
        # Select a model #
        ##################
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

        #####################
        # Compile the model #
        #####################
        model.compile()

        # set parameters using pre-trained network
        if args.load_pretrain:
            say('\nLoad pretrained parameters\n')
            model.load_pretrained_parameters(args)

        ###################
        # Train the model #
        ###################
        model.train(
            ids_corpus,
            dev if args.dev else None,
            test if args.test else None
        )


def main(args):
    train(args)

