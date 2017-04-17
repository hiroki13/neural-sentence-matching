from ..utils.io_utils import say
from ..utils.loader import load_embedding_iterator
from ..utils.preprocess import get_ubuntu_samples, get_msr_samples, get_msr_srl_samples, get_emb_layer
from ..model import basic_model, attention_model, alignment_model, sent_matching_model


def _select_model(args, emb_layer):
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

    if args.data_type == 'base':
        train_samples, test_samples = get_msr_samples(args, emb_layer)
        model = sent_matching_model.BaseModel(args, [emb_layer])
    else:
        train_samples, test_samples, emb_layer_sem = get_msr_srl_samples(args, emb_layer)
        model = sent_matching_model.SemModel(args, [emb_layer, emb_layer_sem])

    model.compile()
    model.train(train_samples, dev_samples=test_samples)


def train(args):
    ids_corpus, dev_samples, test_samples, emb_layer = get_ubuntu_samples(args)

    model = _select_model(args, emb_layer)
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


