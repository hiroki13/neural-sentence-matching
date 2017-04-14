from abc import ABCMeta, abstractmethod

TAB = '\t'
SPACE = ' '


class CorpusLoader(object):
    __metaclass__ = ABCMeta

    def __init__(self, argv):
        self.data_size = argv.data_size

    @abstractmethod
    def load_corpus(self, path, file_encoding='utf-8'):
        raise NotImplementedError


class MSRCorpusLoader(CorpusLoader):

    def load_corpus(self, path, file_encoding='utf-8'):
        if path is None:
            return None

        corpus = []
        with open(path) as f:
            for line in f:
                line = line.decode(file_encoding)
                line = line.rstrip().split(TAB)
                corpus.append(line)

                if len(corpus)-1 == self.data_size:
                    break

        return corpus[1:]
