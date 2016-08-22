import sys
import numpy as np
import theano
import theano.tensor as T

from initialization import default_srng, USE_XAVIER_INIT
from initialization import random_init, create_shared
from initialization import ReLU, sigmoid, tanh, softmax, linear


def say(s, stream=sys.stdout):
    stream.write(s)
    stream.flush()


class Dropout(object):

    def __init__(self, dropout_prob, srng=None, v2=False):
        """
        :param dropout_prob: theano shared variable that stores the dropout probability
        :param srng: theano random stream or None (default rng will be used)
        :param v2: which dropout version to use
        """
        self.dropout_prob = dropout_prob
        self.srng = srng if srng is not None else default_srng
        self.v2 = v2

    def forward(self, x):
        d = (1-self.dropout_prob) if not self.v2 else (1-self.dropout_prob)**0.5
        mask = self.srng.binomial(n=1, p=1-self.dropout_prob, size=x.shape, dtype=theano.config.floatX)
        return x * mask / d


def apply_dropout(x, dropout_prob, v2=False):
    return Dropout(dropout_prob, v2=v2).forward(x)


class EmbeddingLayer(object):
    """
        Embedding layer that
            (1) maps string tokens into integer IDs
            (2) maps integer IDs into embedding vectors (as matrix)
    """

    def __init__(self, n_d, vocab, oov="<unk>", embs=None, fix_init_embs=True):
        """
        :param n_d: dimension of word embeddings; may be over-written if embs is specified
        :param vocab: an iterator of string tokens; the layer will allocate an ID and a vector for each token in it
        :param oov: out-of-vocabulary token
        :param embs: an iterator of (word, vector) pairs; these will be added to the layer
        :param fix_init_embs: whether to fix the initial word vectors loaded from embs
        """

        if embs is not None:
            lst_words = []
            vocab_map = {}
            emb_vals = []
            for word, vector in embs:
                assert word not in vocab_map, "Duplicate words in initial embeddings"
                vocab_map[word] = len(vocab_map)
                emb_vals.append(vector)
                lst_words.append(word)

            self.init_end = len(emb_vals) if fix_init_embs else -1
            if n_d != len(emb_vals[0]):
                say("WARNING: n_d ({}) != init word vector size ({}). Use {} instead.\n".format(
                        n_d, len(emb_vals[0]), len(emb_vals[0])
                    ))
                n_d = len(emb_vals[0])

            say("{} pre-trained embeddings loaded.\n".format(len(emb_vals)))

            for word in vocab:
                if word not in vocab_map:
                    vocab_map[word] = len(vocab_map)
                    emb_vals.append(random_init((n_d,)) * (0.001 if word != oov else 0.0))
                    lst_words.append(word)

            emb_vals = np.vstack(emb_vals).astype(theano.config.floatX)
            self.vocab_map = vocab_map
            self.lst_words = lst_words
        else:
            lst_words = []
            vocab_map = {}
            for word in vocab:
                if word not in vocab_map:
                    vocab_map[word] = len(vocab_map)
                    lst_words.append(word)

            self.lst_words = lst_words
            self.vocab_map = vocab_map
            emb_vals = random_init((len(self.vocab_map), n_d))
            self.init_end = -1

        if oov is not None and oov is not False:
            assert oov in self.vocab_map, "oov {} not in vocab".format(oov)
            self.oov_tok = oov
            self.oov_id = self.vocab_map[oov]
        else:
            self.oov_tok = None
            self.oov_id = -1

        self.embeddings = create_shared(emb_vals)
        if self.init_end > -1:
            self.embeddings_trainable = self.embeddings[self.init_end:]
        else:
            self.embeddings_trainable = self.embeddings

        self.n_V = len(self.vocab_map)
        self.n_d = n_d

    def map_to_words(self, ids):
        n_V, lst_words = self.n_V, self.lst_words
        return [lst_words[i] if i < n_V else "<err>" for i in ids]

    def map_to_ids(self, words, filter_oov=False):
        """
        :param words: the list of string tokens
        :param filter_oov: whether to remove oov tokens in the returned array
        :return: the numpy array of word IDs
        """
        vocab_map = self.vocab_map
        oov_id = self.oov_id
        if filter_oov:
            not_oov = lambda x: x != oov_id
            return np.array(filter(not_oov, [vocab_map.get(x, oov_id) for x in words]), dtype="int32")
        else:
            return np.array([vocab_map.get(x, oov_id) for x in words], dtype="int32")

    def forward(self, x):
        """
        :param x: a theano array of integer IDs
        :return: a theano matrix of word embeddings
        """
        return self.embeddings[x]

    @property
    def params(self):
        return [self.embeddings_trainable]

    @params.setter
    def params(self, param_list):
        self.embeddings.set_value(param_list[0].get_value())


class Layer(object):

    def __init__(self, n_in, n_out, activation, clip_gradients=False, has_bias=True):
        """
        :param n_in: input dimension
        :param n_out: output dimension
        :param activation: the non-linear activation function to apply
        :param clip_gradients:
        :param has_bias: whether to include the bias term b in the computation
        """
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.clip_gradients = clip_gradients
        self.has_bias = has_bias

        self.create_parameters()

        # not implemented yet
        if clip_gradients is True:
            raise Exception("gradient clip not implemented")

    def create_parameters(self):
        n_in, n_out, activation = self.n_in, self.n_out, self.activation
        self.initialize_params(n_in, n_out, activation)

    def initialize_params(self, n_in, n_out, activation):
        if USE_XAVIER_INIT:
            if activation == ReLU:
                scale = np.sqrt(4.0/(n_in+n_out), dtype=theano.config.floatX)
                b_vals = np.ones(n_out, dtype=theano.config.floatX) * 0.01
            elif activation == softmax:
                scale = np.float64(0.001).astype(theano.config.floatX)
                b_vals = np.zeros(n_out, dtype=theano.config.floatX)
            else:
                scale = np.sqrt(2.0/(n_in+n_out), dtype=theano.config.floatX)
                b_vals = np.zeros(n_out, dtype=theano.config.floatX)
            W_vals = random_init((n_in,n_out), rng_type="normal") * scale
        else:
            W_vals = random_init((n_in,n_out))
            if activation == softmax:
                W_vals *= 0.00
            if activation == ReLU:
                b_vals = np.ones(n_out, dtype=theano.config.floatX) * 0.01
            else:
                b_vals = random_init((n_out,))
        self.W = create_shared(W_vals, name="W")
        if self.has_bias: self.b = create_shared(b_vals, name="b")

    def forward(self, x):
        if self.has_bias:
            return self.activation(T.dot(x, self.W) + self.b)
        else:
            return self.activation(T.dot(x, self.W))

    @property
    def params(self):
        if self.has_bias:
            return [self.W, self.b]
        else:
            return [self.W]

    @params.setter
    def params(self, param_list):
        self.W.set_value(param_list[0].get_value())
        if self.has_bias:
            self.b.set_value(param_list[1].get_value())


class RecurrentLayer(Layer):

    def __init__(self, n_in, n_out, activation, clip_gradients=False):
        super(RecurrentLayer, self).__init__(n_in, n_out, activation, clip_gradients=clip_gradients)

    def create_parameters(self):
        n_in, n_out, activation = self.n_in, self.n_out, self.activation

        # re-use the code in super-class Layer
        self.initialize_params(n_in + n_out, n_out, activation)

    def forward(self, x, h):
        n_in, n_out, activation = self.n_in, self.n_out, self.activation
        return activation(T.dot(x, self.W[:n_in]) + T.dot(h, self.W[n_in:]) + self.b)

    def forward_all(self, x, h0=None):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out,), dtype=theano.config.floatX)
        h, _ = theano.scan(fn=self.forward, sequences=x, outputs_info=[h0])
        return h


class LSTM(Layer):

    def __init__(self, n_in, n_out, activation=tanh, clip_gradients=False):

        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.clip_gradients = clip_gradients

        self.in_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        self.forget_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        self.out_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        self.input_layer = RecurrentLayer(n_in, n_out, activation, clip_gradients)

        self.internal_layers = [self.input_layer, self.in_gate, self.forget_gate, self.out_gate]

    def forward(self, x, hc):
        """
        :param x: the input vector or matrix
        :param hc: the vector/matrix of [ c_tm1, h_tm1 ], i.e. hidden state and visible state concatenated together
        :return: [ c_t, h_t ] as a single concatenated vector/matrix
        """
        n_in, n_out, activation = self.n_in, self.n_out, self.activation

        if hc.ndim > 1:
            c_tm1 = hc[:, :n_out]
            h_tm1 = hc[:, n_out:]
        else:
            c_tm1 = hc[:n_out]
            h_tm1 = hc[n_out:]

        in_t = self.in_gate.forward(x, h_tm1)
        forget_t = self.forget_gate.forward(x, h_tm1)
        out_t = self.out_gate.forward(x, h_tm1)

        c_t = forget_t * c_tm1 + in_t * self.input_layer.forward(x, h_tm1)
        h_t = out_t * T.tanh(c_t)

        if hc.ndim > 1:
            return T.concatenate([c_t, h_t], axis=1)
        else:
            return T.concatenate([c_t, h_t])

    def forward_all(self, x, h0=None, return_c=False):
        """
        :param x: input as a matrix (n*d) or a tensor (n*batch*d)
        :param h0: the initial states [ c_0, h_0 ] including both hidden and visible states
        :param return_c: whether to return hidden state {c1, ..., c_n}
        :return: if return_c is False, return {h_1, ..., h_n}, otherwise return
                 { [c_1,h_1], ... , [c_n,h_n] }. Both represented as a matrix or tensor.
        """
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out*2), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out*2,), dtype=theano.config.floatX)
        h, _ = theano.scan(fn=self.forward, sequences=x, outputs_info=[h0])
        if return_c:
            return h
        elif x.ndim > 1:
            return h[:, :, self.n_out:]
        else:
            return h[:, self.n_out:]

    @property
    def params(self):
        return [x for layer in self.internal_layers for x in layer.params]

    @params.setter
    def params(self, param_list):
        start = 0
        for layer in self.internal_layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
            start = end


class GRU(Layer):

    def __init__(self, n_in, n_out, activation=tanh, clip_gradients=False):

        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.clip_gradients = clip_gradients

        self.reset_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        self.update_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        self.input_layer = RecurrentLayer(n_in, n_out, activation, clip_gradients)

        self.internal_layers = [self.reset_gate, self.update_gate, self.input_layer]

    def forward(self, x, h):
        reset_t = self.reset_gate.forward(x, h)
        update_t = self.update_gate.forward(x, h)
        h_reset = reset_t * h

        h_new = self.input_layer.forward(x, h_reset)
        h_out = update_t*h_new + (1.0-update_t)*h
        return h_out

    def forward_all(self, x, h0=None, return_c=True):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out,), dtype=theano.config.floatX)
        h, _ = theano.scan(fn=self.forward, sequences=x, outputs_info=[h0])
        return h

    @property
    def params(self):
        return [x for layer in self.internal_layers for x in layer.params]

    @params.setter
    def params(self, param_list):
        start = 0
        for layer in self.internal_layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
            start = end


class CNN(Layer):

    def __init__(self, n_in, n_out, activation=tanh, order=1, clip_gradients=False):

        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.order = order
        self.clip_gradients = clip_gradients

        internal_layers = self.internal_layers = []
        for i in range(order):
            input_layer = Layer(n_in, n_out, linear, has_bias=False, clip_gradients=clip_gradients)
            internal_layers.append(input_layer)

        self.bias = create_shared(random_init((n_out,)), name="bias")

    def forward(self, x, hc):
        order, n_in, n_out, activation = self.order, self.n_in, self.n_out, self.activation
        layers = self.internal_layers
        if hc.ndim > 1:
            h_tm1 = hc[:, n_out*order:]
        else:
            h_tm1 = hc[n_out*order:]

        lst = []
        for i in range(order):
            if hc.ndim > 1:
                c_i_tm1 = hc[:, n_out*i:n_out*i+n_out]
            else:
                c_i_tm1 = hc[n_out*i:n_out*i+n_out]
            if i == 0:
                c_i_t = layers[i].forward(x)
            else:
                c_i_t = c_im1_tm1 + layers[i].forward(x)
            lst.append(c_i_t)
            c_im1_tm1 = c_i_tm1

        h_t = activation(c_i_t + self.bias)
        lst.append(h_t)

        if hc.ndim > 1:
            return T.concatenate(lst, axis=1)
        else:
            return T.concatenate(lst)

    def forward_all(self, x, h0=None, return_c=False):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out*(self.order+1)), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out*(self.order+1),), dtype=theano.config.floatX)
        h, _ = theano.scan(fn=self.forward, sequences=x, outputs_info=[h0])
        if return_c:
            return h
        elif x.ndim > 1:
            return h[:, :, self.n_out*self.order:]
        else:
            return h[:, self.n_out*self.order:]

    @property
    def params(self):
        return [x for layer in self.internal_layers for x in layer.params] + [self.bias]

    @params.setter
    def params(self, param_list):
        start = 0
        for layer in self.internal_layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
            start = end
        self.bias.set_value(param_list[-1].get_value())


