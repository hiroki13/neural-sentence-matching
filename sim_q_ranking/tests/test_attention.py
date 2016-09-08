import numpy as np
import theano
import theano.tensor as T

query = T.ftensor3('query')
cands = T.ftensor4('cands')

# 1D: batch, 2D: n_words, 3n_e
q = np.asarray([[[2, 2, 2], [-1, -1, -1]]], dtype='float32')

# 1D: batch, 2D: n_cands, 3D: n_words, 4D: n_e
c = np.asarray([[[[1, 5, 3], [3, 1, 2], [3, 1, 5]], [[1, 3, 1], [2, 1, 1], [2, 3, 1]]]], dtype='float32')


def main():
    get_attention_vectors()


def get_attention_vectors():
    A = _softmax_4d(cands, 3)
    y = _get_attention_vectors(cands, A)

    f = theano.function(inputs=[cands], outputs=[y, A, cands], on_unused_input='ignore')
    a, b, d = f(c)
    print a
    print b
    print d


def _get_attention_vectors(x, A):
    """
    :param x: 1D: n_queries, 2D: n_cands-1/1, 3D: n_words (query/cands), 4D: dim_h
    :param A: 1D: n_queries, 2D: n_cands-1, 3D: n_words (cands/query), 4D: n_words (query/cands)
    :return: 1D: n_queries, 2D: n_cands-1, 3D: n_words (cands/query), 4D: dim_h
    """
    # 1D: n_queries, 2D: n_cands-1, 3D: 1, 4D: n_words (query/cands), 5D: dim_h
    x = x.dimshuffle((0, 1, 'x', 2, 3))
    # 1D: n_queries, 2D: n_cands-1, 3D: n_words (cands/query), 4D: n_words (query/cands), 5D: 1
    A = A.dimshuffle((0, 1, 2, 3, 'x'))
    return T.sum(A * x, axis=3)


def softmax_4d():
    y = _softmax_4d(cands, 3)
    f = theano.function(inputs=[cands], outputs=[y], on_unused_input='ignore')
    print f(c)


def _softmax_4d(x, axis, eps=1e-8):
    """
    :param x: 1D: n_queries, 2D: n_cands-1, 3D: n_words (cands), 4D: n_words (query)
    :return: 1D: n_queries, 2D: n_cands-1, 3D: n_words (cands), 4D: n_words (query)
    """
#    x = T.exp(x)
    z = T.sum(x, axis=axis, keepdims=True)
    return x / (z + eps)


if __name__ == '__main__':
    main()


