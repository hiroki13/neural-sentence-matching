import theano
import theano.tensor as T


def hinge_loss(pos_scores, neg_scores):
    """
    :param pos_scores: 1D: n_queries
    :param neg_scores: 1D: n_queries, 2D: n_cands
    :return: avg hinge_loss: float
    """
    loss = 1.0 + T.max(neg_scores, axis=1) - pos_scores
    return T.mean((loss > 0) * loss)


def cross_entropy_loss(scores):
    """
    :param scores: 1D: n_queries, 2D: n_cands; column0=pos, column1-=neg
    :return: avg hinge_loss: float
    """
    probs = T.nnet.softmax(scores)
    return - T.mean(T.log(probs[:, 0]))


def binary_cross_entropy(y, y_prob):
    return - (T.sum(y * T.log(y_prob) + (1. - y) * T.log(1. - y_prob)))


def normalize_2d(x, eps=1e-8):
    # x is batch*d
    # l2 is batch*1
    l2 = x.norm(2, axis=1).dimshuffle((0, 'x'))
    return x / (l2 + eps)


def normalize_3d(x, eps=1e-8):
    # x is len*batch*d
    # l2 is len*batch*1
    l2 = x.norm(2, axis=2).dimshuffle((0, 1, 'x'))
    return x / (l2 + eps)


def average_without_padding(x, ids, padding_id, eps=1e-8):
    """
    :param x: 1D: n_words, 2D: batch, 3D: n_d
    :param ids: 1D: n_words, 2D: batch, 3D: n_d
    :return: 1D: batch, 2D: n_d
    """
    # 1D: n_words, 2D: batch, 3D: 1
    mask = T.neq(ids, padding_id).dimshuffle((0, 1, 'x'))
    mask = T.cast(mask, theano.config.floatX)
    # 1D: batch, 2D: n_d
    return T.sum(x * mask, axis=0) / (T.sum(mask, axis=0) + eps)


def k_max_pooling(x, k, axis):
    x = T.sort(x, axis=axis)
    return x[:, -k:]


def euclid_dist(x, y, axis):
    return T.exp(- T.sum((x - y) ** 2, axis=axis))


def cosine_sim(x, y, axis):
    return T.sum(x * y, axis-1) / (x.norm(2, axis) * y.norm(2, axis))


def softmax(x, axis, eps=1e-8):
    """
    :param x: 1D: n_queries, 2D: n_cands-1, 3D: n_words (cands), 4D: n_words (query)
    :return: 1D: n_queries, 2D: n_cands-1, 3D: n_words (cands), 4D: n_words (query)
    """
    x = T.exp(x)
    z = T.sum(x, axis=axis, keepdims=True)
    return x / (z + eps)


def get_mask(ids, idps, padding_id):
    # 1D: n_queries * n_cands, 2D: n_words
    ids = ids.dimshuffle((1, 0))
    # 1D: n_queries * n_cands, 2D: n_words
    ids = ids[idps.ravel()]
    # 1D: n_queries, 2D: n_cands, 3D: n_words
    ids = ids.reshape((idps.shape[0], idps.shape[1], -1))
    mask = T.neq(ids, padding_id)
    mask = T.cast(mask, theano.config.floatX)
    return mask


def average_without_padding_3d(x, mask, eps=1e-8):
    """
    :param x: 1D: n_queries, 2D: n_cands-1, 3D: n_words
    :param mask: 1D: n_queries, 2D: n_cands, 3D: n_words
    :return: 1D: batch, 2D: n_d
    """
    mask = mask[:, 1:]
    # 1D: n_queries, 2D: n_cands
    return T.sum(x * mask, axis=2) / (T.sum(mask, axis=2) + eps)
