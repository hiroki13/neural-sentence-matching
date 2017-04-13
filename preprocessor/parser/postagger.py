import subprocess
from abc import ABCMeta, abstractmethod

TAB = '\t'
SPACE = ' '


class POSTagger(object):
    __metaclass__ = ABCMeta

    def __init__(self, argv):
        self.argv = argv

    @abstractmethod
    def tagging(self, corpus):
        raise NotImplementedError


class StanfordTagger(POSTagger):

    def tagging(self, fn='corpus.tokenized.sents.txt'):
        results = subprocess.Popen(['java', '-mx300m', '-cp', 'tagger/stanford-postagger.jar:tagger/lib/*',
                                    'edu.stanford.nlp.tagger.maxent.MaxentTagger',
                                    '-model', 'tagger/models/english-left3words-distsim.tagger',
                                    '-sentenceDelimiter', 'newline',
                                    '-tokenize', 'false',
                                    '-textFile', fn],
                                   stdout=subprocess.PIPE)
        return self._postprocess(results)

    @staticmethod
    def _postprocess(results):
        return [line[:-1].decode('utf-8') for line in results.stdout]
