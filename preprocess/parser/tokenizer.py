from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()


def tokenize_all(sents):
    return [tokenizer.tokenize(sent) for sent in sents]


def tokenize(sent):
    return tokenizer.tokenize(sent)

