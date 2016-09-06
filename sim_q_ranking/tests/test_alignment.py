import unittest

import numpy as np
import theano
import theano.tensor as T

from ..nn.advanced import AlignmentLayer

query = T.ftensor3('query')
cands = T.ftensor4('cands')

# 1D: batch, 2D: n_words, 3n_e
#q = np.ones(shape=(1, 2, 3), dtype='float32')
q = np.asarray([[[2, 2, 2], [-1, -1, -1]]], dtype='float32')

# 1D: batch, 2D: n_cands, 3D: n_words, 4D: n_e
#c = np.ones(shape=(1, 2, 2, 3), dtype='float32')
c = np.asarray([[[[1, -5, 3], [3, 1, -2], [3, 1, 5]], [[1, 3, 1], [2, 1, 1], [2, 3, 1]]]], dtype='float32')


class TestAlignment(unittest.TestCase):

    def setUp(self):
        self.layer = AlignmentLayer(3, 2, T.tanh)
        for i in xrange(len(self.layer.params)):
            w = self.layer.params[i].get_value(borrow=True)
            self.layer.params[i].set_value(np.ones(shape=w.shape, dtype='float32'))

    def test_alignment_matrix(self):
        # 1D: n_queries, 2D: n_cands-1, 3D: n_words (cands), 4D: n_words (query)
        y = self.layer.alignment_matrix(query, cands)
        self.f = theano.function(inputs=[query, cands], outputs=[y], on_unused_input='ignore')
        print
        print self.f(q, c)

    def test_get_max_aligned_scores(self):
        # 1D: n_queries, 2D: n_cands-1, 3D: n_words (cands), 4D: n_words (query)
        y = self.layer.alignment_matrix(query, cands)
        # 1D: n_queries, 2D: n_cands-1, 3D: n_words (cands)
        y = T.max(y, axis=3)
        y = T.sum(y, axis=2)
        self.f = theano.function(inputs=[query, cands], outputs=[y], on_unused_input='ignore')
        print
        print self.f(q, c)

    """
    def test_get_max_aligned_word_index(self):
        # 1D: n_queries, 2D: n_cands-1, 3D: n_words (cands), 4D: n_words (query)
        y = self.layer.alignment_matrix(query, cands)

        # 1D: n_queries, 2D: n_cands-1, 3D: n_words (query)
        y = T.argmax(y, axis=2)

        # 1D: n_cands-1
        v = T.arange(y.shape[1]) * y.shape[2]
        v = v.dimshuffle((0, 'x'))
        # 1D: n_cands-1, 2D: n_words (query)
        v = T.repeat(v, y.shape[2], axis=1)

        # 1D: n_queries, 2D: n_cands-1, 3D: n_words (query)
        y = y + v
        # 1D: n_queries, 2D: n_cands-1 * n_words (query)
        y = y.reshape((y.shape[0], y.shape[1] * y.shape[2]))

        M = cands.reshape((cands.shape[0], cands.shape[1] * cands.shape[2], -1))
        y = M[:, y]
        y = y.reshape((y.shape[0], v.shape[0], v.shape[1], -1))

        self.f = theano.function(inputs=[query, cands], outputs=[y], on_unused_input='ignore')
        print
        print self.f(q, c)

    def test_get_max_aligned_word(self):
        # cands: 1D: n_queries, 2D: n_cands-1, 3D: n_words, 4D: n_e

        # 1D: n_queries, 2D: n_cands-1, 3D: n_words (cands), 4D: n_words (query)
        y = self.layer.alignment_matrix(query, cands)

        # 1D: n_queries, 2D: n_cands-1, 3D: n_words (query)
        y = T.argmax(y, axis=2)

        y = cands[T.arange(y.shape[0]), T.arange(y.shape[1]), y.dimshuffle((0, 2, 1))]
        y = y.dimshuffle((0, 2, 1, 3))

        self.f = theano.function(inputs=[query, cands], outputs=[y], on_unused_input='ignore')
        print
        print self.f(q, c)

    def test_vector_composition(self):
        # 1D: n_queries, 2D: n_cands-1, 3D: n_words, 4D: n_words
        y = self.layer.vector_composition(query, cands)
        self.f = theano.function(inputs=[query, cands], outputs=[y], on_unused_input='ignore')
        print
        print self.f(q, c)
    """

if __name__ == '__main__':
    unittest.main()

