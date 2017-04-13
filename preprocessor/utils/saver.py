from loader import TAB, SPACE


class Saver(object):

    def __init__(self, argv):
        self.argv = argv

    @staticmethod
    def save_tokenized_msr_corpus(corpus, fn='corpus.tokenized.txt'):
        with open(fn, 'w') as fout:
            for sample in corpus:
                sample[-2] = SPACE.join(sample[-2])
                sample[-1] = SPACE.join(sample[-1])
                text = TAB.join(sample)
                fout.writelines(text.encode('utf-8') + '\n')

    @staticmethod
    def save_tokenized_sents(tokenized_corpus, fn='corpus.tokenized.sents.txt'):
        with open(fn, 'w') as fout:
            for sample in tokenized_corpus:
                sent1 = SPACE.join(sample[-2])
                sent2 = SPACE.join(sample[-1])
                fout.writelines(sent1.encode('utf-8') + '\n')
                fout.writelines(sent2.encode('utf-8') + '\n')

    @staticmethod
    def save_raw_msr_sents(corpus, fn='corpus.raw.sents.txt'):
        with open(fn, 'w') as fout:
            for sample in corpus:
                fout.writelines(sample[-2].encode('utf-8') + '\n')
                fout.writelines(sample[-1].encode('utf-8') + '\n')

    @staticmethod
    def save_pos_msr_corpus(corpus, tagged_sents, fn='corpus.msr.tagged.txt'):
        assert len(corpus) == len(tagged_sents) / 2, '%d %d' % (len(corpus), len(tagged_sents))
        with open(fn, 'w') as fout:
            for i, sample in enumerate(corpus):
                sample[-2] = tagged_sents[i * 2]
                sample[-1] = tagged_sents[i * 2 + 1]
                text = TAB.join(sample)
                fout.writelines(text.encode('utf-8') + '\n')

    @staticmethod
    def save_pos_corpus(tagged_sents, fn='corpus.tagged.txt'):
        with open(fn, 'w') as fout:
            for sent in tagged_sents:
                fout.writelines(sent.encode('utf-8') + '\n')

    @staticmethod
    def save_srl_msr_corpus(corpus, pos_sents, srl_sents, fn='corpus.msr.srl.txt'):
        def _concat_srl_info(pos_sent, srl_sent):
            sent = pos_sent.split()
            srl_sent = ['_'.join(word[1:]) for word in srl_sent]
            assert len(sent) == len(srl_sent), '%s %s' % (str(sent), str(srl_sent))
            return SPACE.join([s1 + '_' + s2 for s1, s2 in zip(sent, srl_sent)])

        assert len(corpus) == len(srl_sents) / 2, '%d %d' % (len(corpus), len(srl_sents))
        assert len(pos_sents) == len(srl_sents)
        with open(fn, 'w') as fout:
            for i, sample in enumerate(corpus):
                sample[-2] = _concat_srl_info(pos_sent=pos_sents[i * 2], srl_sent=srl_sents[i * 2])
                sample[-1] = _concat_srl_info(pos_sent=pos_sents[i * 2 + 1], srl_sent=srl_sents[i * 2 + 1])
                text = TAB.join(sample)
                fout.writelines(text.encode('utf-8') + '\n')

