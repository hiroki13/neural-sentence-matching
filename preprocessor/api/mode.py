from ..utils.loader import MSRCorpusLoader
from ..utils.saver import Saver
from ..utils.converter import Converter


def main(argv):
    if argv.mode == 'tok':
        tokenize(argv)
    elif argv.mode == 'pos':
        pos_tagging(argv)
    elif argv.mode == 'srl':
        sem_role_labeling(argv)


def tokenize(argv):
    loader = MSRCorpusLoader(argv)
    saver = Saver(argv)
    converter = Converter(argv)

    corpus = loader.load_corpus(argv.corpus)
    corpus = converter.convert_raw_to_token(corpus)
    saver.save_tokenized_msr_corpus(corpus)


def pos_tagging(argv):
    loader = MSRCorpusLoader(argv)
    saver = Saver(argv)
    converter = Converter(argv)

    corpus = loader.load_corpus(argv.corpus)
    tokenized_corpus = converter.convert_raw_to_token(corpus)
    saver.save_tokenized_sents(tokenized_corpus)
    tagged_sents = converter.convert_raw_to_pos()
    saver.save_pos_msr_corpus(corpus, tagged_sents)
    saver.save_pos_corpus(tagged_sents)


def sem_role_labeling(argv):
    loader = MSRCorpusLoader(argv)
    saver = Saver(argv)
    converter = Converter(argv)

    corpus = loader.load_corpus(argv.corpus)
    tokenized_corpus = converter.convert_raw_to_token(corpus)
    saver.save_tokenized_sents(tokenized_corpus)
    tagged_sents = converter.convert_raw_to_pos()
    saver.save_pos_corpus(tagged_sents)
    srl_sents = converter.convert_pos_to_srl()
    saver.save_srl_msr_corpus(corpus, tagged_sents, srl_sents)
