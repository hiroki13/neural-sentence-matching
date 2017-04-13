import subprocess

TAB = '\t'


class SemRoleLabeler(object):

    def __init__(self, argv):
        self.argv = argv

    def labeling(self, fn='corpus.tagged.txt'):
        results = subprocess.Popen(['python', '-m', 'srl.cons_srl.main',
                                    '-mode', 'test',
                                    '--data_type', 'pos_tagged',
                                    '--test_data', fn,
                                    '--output', '1',
                                    '--load_model', 'model.pkl.gz',
                                    '--load_word', 'word.pkl.gz',
                                    '--load_label', 'label.pkl.gz',
                                    '--load_emb', 'emb.pkl.gz'],
                                   stdout=subprocess.PIPE)
        return self._postprocess(results)

    @staticmethod
    def _postprocess(results):
        corpus = []
        sent = []
        for line in [line.decode('utf-8') for line in results.stdout][24:]:
            line = line.rstrip()
            if len(line) == 0:
                corpus.append(sent)
                sent = []
            else:
                sent.append(line.split(TAB))
        if sent:
            corpus.append(sent)
        return corpus
