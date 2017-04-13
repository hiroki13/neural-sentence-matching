from ..parser.tokenizer import tokenize
from ..parser.postagger import StanfordTagger
from ..parser.semrolelabeler import SemRoleLabeler


class Converter(object):

    def __init__(self, argv):
        self.argv = argv

    @staticmethod
    def convert_raw_to_token(corpus):
        for i, sample in enumerate(corpus):
            corpus[i][-2] = tokenize(sample[-2])
            corpus[i][-1] = tokenize(sample[-1])
        return corpus

    def convert_raw_to_pos(self):
        tagger = StanfordTagger(self.argv)
        return tagger.tagging()

    def convert_pos_to_srl(self):
        labeler = SemRoleLabeler(self.argv)
        return labeler.labeling()

