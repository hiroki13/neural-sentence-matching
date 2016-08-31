import unittest

import numpy as np
import theano
import theano.tensor as T

from ..model.attention_model import AttentionLayer

query = T.ftensor3('query')
cands = T.ftensor4('cands')

#q = np.ones(shape=(1, 2, 3), dtype='float32')
#c = np.ones(shape=(1, 2, 2, 3), dtype='float32')

q = np.asarray([[[2, 2, 2], [-1, -1, -1]]], dtype='float32')
c = np.asarray([[[[1, 2, 3], [3, 1, -2]], [[1, 3, 1], [2, 1, 1]]]], dtype='float32')


class TestAttention(unittest.TestCase):

    def setUp(self):
        self.layer = AttentionLayer(1, 3, T.tanh)
        for i in xrange(len(self.layer.params)):
            w = self.layer.params[i].get_value(borrow=True)
            self.layer.params[i].set_value(np.ones(shape=w.shape, dtype='float32'))

    def test_alignment_matrix(self):
        # 1D: n_queries, 2D: n_cands-1, 3D: n_words, 4D: n_words
        y = self.layer.alignment_matrix(query, cands)
        self.f = theano.function(inputs=[query, cands], outputs=[y], on_unused_input='ignore')
        print
        print self.f(q, c)

    def test_vector_composition(self):
        # 1D: n_queries, 2D: n_cands-1, 3D: n_words, 4D: n_words
        y = self.layer.vector_composition(query, cands)
        self.f = theano.function(inputs=[query, cands], outputs=[y], on_unused_input='ignore')
        print
        print self.f(q, c)

if __name__ == '__main__':
    unittest.main()
