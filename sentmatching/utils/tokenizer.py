from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()


def tokenize(sent):
    return tokenizer.tokenize(sent)


def tokenize_all(sents):
    return [tokenizer.tokenize(sent) for sent in sents]


def tokenize_msr_corpus(corpus):
    return [[sample[0], tokenize(sample[1]), tokenize(sample[2])] for sample in corpus]



