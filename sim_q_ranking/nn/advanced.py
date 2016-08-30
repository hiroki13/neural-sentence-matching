import numpy as np
import theano
import theano.tensor as T

from initialization import random_init, create_shared
from initialization import tanh, linear, sigmoid, ReLU
from basic import Layer, RecurrentLayer


class GRNN(Layer):
    def __init__(self, n_in, n_out, activation=tanh, order=1, has_outgate=False, mode=1, clip_gradients=False):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.order = order
        self.clip_gradients = clip_gradients
        self.has_outgate = has_outgate
        self.has_bias = False
        self.mode = mode
        self.create_parameters()

    def create_parameters(self):
        self.W_e = create_shared(random_init((self.n_in, self.n_out)), name="W_e")
        self.W = create_shared(random_init((self.n_out * 2, self.n_out)), name="W")
        self.U = theano.shared(random_init((self.n_out * 3, self.n_out * 3)), name="U")
        self.G = theano.shared(random_init((self.n_out * 2, self.n_out * 2)), name="G")
        self.lst_params = [self.W_e, self.W, self.G, self.U]

    def forward(self, x, eps=1e-8):
        x = tanh(T.dot(x, self.W_e))
        zero = T.zeros(shape=(x.shape[0], x.shape[1], self.n_out), dtype=theano.config.floatX)

        def recursive(t, m):
            def step(_t, _m):
                h_l = _m[:, _t]
                h_r = _m[:, _t + 1]

                # 1D: batch, 2D: 2 * n_d
                r = sigmoid(T.dot(T.concatenate([h_l, h_r], axis=1), self.G))
                half = r.shape[1] / 2
                # 1D: batch, 2D: n_d
                r_l = r[:, :half]
                r_r = r[:, half:]

                # 1D: batch, 2D: n_d
                h_hat = tanh(T.dot(T.concatenate([r_l * h_l, r_r * h_r], axis=1), self.W))

                # 1D: batch, 2D: 3 * n_d
                z_hat = T.exp(T.dot(T.concatenate([h_hat, h_l, h_r], axis=1), self.U))
                # 1D: batch, 2D: 3 (h_hat, h_l, h_r), 3D: n_d
                z_hat = z_hat.reshape((z_hat.shape[0], 3, z_hat.shape[1] / 3))

                # 1D: batch, 2D: n_d
                Z = T.sum(z_hat, axis=1)
                # 1D: batch, 2D: 3, 3D: n_d
                Z = T.repeat(Z, repeats=3, axis=1).reshape((Z.shape[0], Z.shape[1], 3)).dimshuffle((0, 2, 1))

                z = z_hat / Z + eps

                h = h_hat * z[:, 0] + h_l * z[:, 1] + h_r * z[:, 2]
                seq_sub_tensor = h

                return _m, seq_sub_tensor

            [_, u], _ = theano.scan(fn=step,
                                    sequences=T.arange(m.shape[1] - t - 1),
                                    outputs_info=[m, None])

            return T.set_subtensor(zero[:, :m.shape[1] - t - 1], u.dimshuffle((1, 0, 2)))

        # x: 1D: batch, 2D: n_words, 3D: n_d
        # y: 1D: n_words -1, 2D: batch, 3D: n_words, 4D: n_d
        y, _ = theano.scan(fn=recursive,
                           sequences=T.arange(x.shape[1] - 1),
                           outputs_info=x)

        return y[-1][:, 0]

    @property
    def params(self):
        return [param for param in self.lst_params]


class AlignmentLayer(Layer):

    def __init__(self, n_e, n_d, activation):
        self.n_e = n_e
        self.n_d = n_d
        self.activation = activation
        self.create_parameters()

    def create_parameters(self):
        n_e = self.n_e
        n_d = self.n_d
#        self.W1_c = create_shared(random_init((n_e, n_d)), name="W1_c")
        self.W1_q = create_shared(random_init((n_e, n_d)), name="W1_h")
#        self.lst_params = [self.W1_q, self.W1_c]
        self.lst_params = [self.W1_q]

    def alignment(self, query, cands, mask=None):
        """
        :param query: 1D: n_queries, 2D: n_words, 3D: dim_h
        :param cands: 1D: n_queries, 2D: n_cands-1, 3D: n_words, 4D: dim_h
        :param mask: 1D: n_queries, 2D: n_cands, 3D: n_words
        :return: h_after: 1D: n_queries, 2D: n_cands-1, 3D: dim_h
        """

        # 1D: n_queries, 2D: n_cands-1, 3D: n_words, 4D: dim_h
        M = T.sum(query.dimshuffle(0, 'x', 1, 2) * cands, axis=3)

        if mask is not None:
            if mask.dtype != theano.config.floatX:
                mask = T.cast(mask, theano.config.floatX)
            M = M * mask.dimshuffle((0, 1, 2, 'x'))

        return M

#    def linear(self, x, y):
#        return T.tanh(T.dot(x, self.W1_q) + T.dot(y, self.W1_c))

    def linear(self, x):
        return T.tanh(T.dot(x, self.W1_q))

    def bilinear(self, x, y):
        return T.tanh(T.sum(T.dot(x, self.W1_q) * y, axis=2))

    @property
    def params(self):
        return self.lst_params

    @params.setter
    def params(self, param_list):
        assert len(param_list) == len(self.lst_params)
        for p, q in zip(self.lst_params, param_list):
            p.set_value(q.get_value())


class AttentionLayer(Layer):
    def __init__(self, a_type, n_d, activation):
        self.a_type = a_type
        self.n_d = n_d
        self.activation = activation
        self.forward = None
        self.create_parameters()

    def create_parameters(self):
        n_d = self.n_d
        self.W1_c = create_shared(random_init((n_d, n_d)), name="W1_c")
        self.W1_q = create_shared(random_init((n_d, n_d)), name="W1_h")

        if self.a_type > 1:
            self.W2_r_a = create_shared(random_init((n_d, n_d)), name="W2_r")
            self.W2_r_b = create_shared(random_init((n_d, n_d)), name="W2_r")
            self.w_a = create_shared(random_init((n_d,)), name="w")
            self.w_b = create_shared(random_init((n_d,)), name="w_b")
            self.lst_params = [self.W1_q, self.W1_c, self.W2_r_a, self.W2_r_b, self.w_a, self.w_b]
            self.forward = self.forward_second_order
        else:
            self.W2_r = create_shared(random_init((n_d, n_d)), name="W2_r")
            self.w = create_shared(random_init((n_d,)), name="w")
            self.lst_params = [self.W1_q, self.W1_c, self.W2_r, self.w]
            self.forward = self.forward_standard

    def forward_second_order(self, Y, h, mask=None, eps=1e-8):
        """
        :param Y: 1D: n_queries, 2D: n_words, 3D: dim_h
        :param h: 1D: n_queries, 2D: n_cands-1, 3D: dim_h
        :param mask: 1D: n_queries, 2D: n_cands, 3D: n_words
        :param eps: float
        :return: h_after: 1D: n_queries, 2D: n_cands-1, 3D: dim_h
        """

        n_queries = Y.shape[0]
        n_words = Y.shape[1]
        n_cands = h.shape[1]
        dim_h = h.shape[2]

        # 1D: n_queries, 2D: n_cands, 3D: n_words, 4D: dim_h
        M = T.tanh(T.dot(Y, self.W1_q).dimshuffle(0, 'x', 1, 2) + T.dot(h, self.W1_c).dimshuffle(0, 1, 'x', 2))

        # 1D: n_queries, 2D: n_cands, 3D: n_words
        M_a = T.dot(M, self.w_a)
        M_a = normalize_3d(M_a)

        # 1D: n_queries, 2D: n_cands, 3D: n_words, 4D: 1
        alpha = T.nnet.softmax(M_a.reshape((n_queries * n_cands, n_words)))
        alpha = alpha.reshape((n_queries, n_cands, n_words, 1))

        if mask is not None:
            if mask.dtype != theano.config.floatX:
                mask = T.cast(mask, theano.config.floatX)
            alpha = alpha * mask.dimshuffle((0, 1, 2, 'x'))
            alpha /= T.sum(alpha, axis=2, keepdims=True) + eps

        # 1D: n_queries, 2D: 1, 3D: n_words, 4D: dim_h
        Y = Y.dimshuffle((0, 'x', 1, 2))

        # 1D: n_queries, 2D: n_cands, 3D: dim_h
        r_a = T.sum(Y * alpha, axis=2)

        # 1D: n_queries, 2D: n_cands, 3D: n_words, 4D: dim_h
        M_a = M_a.dimshuffle((0, 1, 2, 'x'))
        M_a = T.repeat(M_a, dim_h, axis=3)

        # 1D: n_words, 2D: dim_h
        w_a = self.w_a.dimshuffle('x', 0)
        w_a = T.repeat(w_a, M.shape[3], axis=0)
        w_a = w_a.dimshuffle((1, 0))

        # 1D: n_queries, 2D: n_cands-1, 3D: n_words, 4D: dim_h
        M_b = M - T.dot(M_a, w_a) / T.sum(self.w_a ** 2)

        # 1D: n_queries, 2D: n_cands-1, 3D: n_words
        M_b = T.dot(M_b, self.w_b)
        M_b = normalize_3d(M_b)

        # 1D: n_queries, 2D: n_cands-1, 3D: n_words, 4D: 1
        beta = T.nnet.softmax(M_b.reshape((n_queries * n_cands, n_words)))
        beta = beta.reshape((n_queries, n_cands, n_words, 1))

        if mask is not None:
            beta = beta * mask.dimshuffle((0, 1, 2, 'x'))
            beta /= T.sum(beta, axis=2, keepdims=True) + eps

        # 1D: n_queries, 2D: n_cands-1, 3D: dim_h
        r_b = T.sum(Y * beta, axis=2)

        # 1D: n_queries, 2D: n_cands-1, 3D: dim_h
        h_after = T.tanh(T.dot(r_a, self.W2_r_a) + T.dot(r_b, self.W2_r_b))

        return h_after

    def forward_standard(self, query, cands, mask=None, eps=1e-8):
        """
        :param query: 1D: n_queries, 2D: n_words, 3D: dim_h
        :param cands: 1D: n_queries, 2D: n_cands-1, 3D: dim_h
        :param mask: 1D: n_queries, 2D: n_cands, 3D: n_words
        :param eps: float
        :return: h_after: 1D: n_queries, 2D: n_cands-1, 3D: dim_h
        """

        # 1D: n_queries, 2D: n_cands-1, 3D: n_words, 4D: dim_h
        M = T.tanh(T.dot(query, self.W1_q).dimshuffle(0, 'x', 1, 2) + T.dot(cands, self.W1_c).dimshuffle(0, 1, 'x', 2))

        # 1D: n_queries, 2D: n_cands-1, 3D: n_words
        u = T.dot(M, self.w)
        u = normalize_3d(u)

        # 1D: n_queries, 2D: n_cands-1, 3D: n_words, 4D: 1
        alpha = T.nnet.softmax(u.reshape((cands.shape[0] * cands.shape[1], query.shape[1])))
        alpha = alpha.reshape((cands.shape[0], cands.shape[1], query.shape[1], 1))

        if mask is not None:
            if mask.dtype != theano.config.floatX:
                mask = T.cast(mask, theano.config.floatX)
            alpha = alpha * mask.dimshuffle((0, 1, 2, 'x'))
            alpha /= T.sum(alpha, axis=2, keepdims=True) + eps

        # 1D: n_queries, 2D: 1, 3D: n_words, 4D: dim_h
        query = query.dimshuffle((0, 'x', 1, 2))
        # 1D: n_queries, 2D: n_cands-1, 3D: dim_h
        r = T.sum(query * alpha, axis=2)

        # 1D: n_queries, 2D: n_cands-1, 3D: dim_h
        h_after = T.tanh(T.dot(r, self.W2_r))

        return h_after

    @property
    def params(self):
        return self.lst_params

    @params.setter
    def params(self, param_list):
        assert len(param_list) == len(self.lst_params)
        for p, q in zip(self.lst_params, param_list):
            p.set_value(q.get_value())


def normalize_3d(x, eps=1e-8):
    # x is len*batch*d
    # l2 is len*batch*1
    l2 = x.norm(2, axis=2).dimshuffle((0, 1, 'x'))
    return x / (l2 + eps)


class RCNN(Layer):
    def __init__(self, n_in, n_out, activation=tanh, order=1, has_outgate=False, mode=1, clip_gradients=False):
        """
        :param n_in:
        :param n_out:
        :param activation:
        :param order: CNN feature width
        :param has_outgate: whether to add a output gate as in LSTM; this can be useful for language modeling
        :param mode: 0 if non-linear filter; 1 if linear filter (default)
        :param clip_gradients:
        """

        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.order = order
        self.clip_gradients = clip_gradients
        self.has_outgate = has_outgate
        self.mode = mode

        internal_layers = self.internal_layers = []
        for i in range(order):
            input_layer = Layer(n_in, n_out, linear, has_bias=False, clip_gradients=clip_gradients)
            internal_layers.append(input_layer)

        forget_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        internal_layers.append(forget_gate)

        self.bias = create_shared(random_init((n_out,)), name="bias")

        if has_outgate:
            self.out_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
            self.internal_layers += [self.out_gate]

    def forward(self, x, hc):
        """
        :param x: input token at current time/position t
        :param hc: hidden/visible states at time/position t-1
        :return: hidden/visible states at time/position t
        """
        order, n_in, n_out, activation = self.order, self.n_in, self.n_out, self.activation
        layers = self.internal_layers
        if hc.ndim > 1:
            h_tm1 = hc[:, n_out * order:]
        else:
            h_tm1 = hc[n_out * order:]

        forget_t = layers[order].forward(x, h_tm1)
        lst = []
        for i in range(order):
            if hc.ndim > 1:
                c_i_tm1 = hc[:, n_out * i:n_out * i + n_out]
            else:
                c_i_tm1 = hc[n_out * i:n_out * i + n_out]
            in_i_t = layers[i].forward(x)
            if i == 0:
                c_i_t = forget_t * c_i_tm1 + (1 - forget_t) * in_i_t
            elif self.mode == 0:
                c_i_t = forget_t * c_i_tm1 + (1 - forget_t) * (in_i_t * c_im1_t)
            else:
                c_i_t = forget_t * c_i_tm1 + (1 - forget_t) * (in_i_t + c_im1_tm1)
            lst.append(c_i_t)
            c_im1_tm1 = c_i_tm1
            c_im1_t = c_i_t

        if not self.has_outgate:
            h_t = activation(c_i_t + self.bias)
        else:
            out_t = self.out_gate.forward(x, h_tm1)
            h_t = out_t * activation(c_i_t + self.bias)
        lst.append(h_t)

        if hc.ndim > 1:
            return T.concatenate(lst, axis=1)
        else:
            return T.concatenate(lst)

    def forward_all(self, x, h0=None, return_c=False):
        """
        :param x: input tokens x_1, ... , x_n
        :param h0: initial states
        :param return_c: whether to return hidden states in addition to visible state
        :return: visible states (and hidden states) of all positions/time
        """
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out * (self.order + 1)), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out * (self.order + 1),), dtype=theano.config.floatX)

        h, _ = theano.scan(fn=self.forward, sequences=x, outputs_info=[h0])

        if return_c:
            return h
        elif x.ndim > 1:
            return h[:, :, self.n_out * self.order:]
        else:
            return h[:, self.n_out * self.order:]

    def forward2(self, x, hc, f_tm1):
        order, n_in, n_out, activation = self.order, self.n_in, self.n_out, self.activation
        layers = self.internal_layers
        if hc.ndim > 1:
            h_tm1 = hc[:, n_out * order:]
        else:
            h_tm1 = hc[n_out * order:]

        forget_t = layers[order].forward(x, h_tm1)
        lst = []
        for i in range(order):
            if hc.ndim > 1:
                c_i_tm1 = hc[:, n_out * i:n_out * i + n_out]
            else:
                c_i_tm1 = hc[n_out * i:n_out * i + n_out]
            in_i_t = layers[i].forward(x)
            if i == 0:
                c_i_t = forget_t * c_i_tm1 + (1 - forget_t) * in_i_t
            elif self.mode == 0:
                c_i_t = forget_t * c_i_tm1 + (1 - forget_t) * (in_i_t * c_im1_t)
            else:
                c_i_t = forget_t * c_i_tm1 + (1 - forget_t) * (in_i_t + c_im1_tm1)
            lst.append(c_i_t)
            c_im1_tm1 = c_i_tm1
            c_im1_t = c_i_t

        if not self.has_outgate:
            h_t = activation(c_i_t + self.bias)
        else:
            out_t = self.out_gate.forward(x, h_tm1)
            h_t = out_t * activation(c_i_t + self.bias)
        lst.append(h_t)

        if hc.ndim > 1:
            return T.concatenate(lst, axis=1), forget_t
        else:
            return T.concatenate(lst), forget_t

    def get_input_gate(self, x, h0=None):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out * (self.order + 1)), dtype=theano.config.floatX)
                f0 = T.zeros((x.shape[1], self.n_out), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out * (self.order + 1),), dtype=theano.config.floatX)
                f0 = T.zeros((self.n_out,), dtype=theano.config.floatX)

        [h, f], _ = theano.scan(fn=self.forward2, sequences=x, outputs_info=[h0, f0])
        return 1.0 - f

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


class StrCNN(Layer):

    def __init__(self, n_in, n_out, activation=None, decay=0.0, order=2, use_all_grams=True):
        self.n_in = n_in
        self.n_out = n_out
        self.order = order
        self.use_all_grams = use_all_grams
        self.decay = theano.shared(np.float64(decay).astype(theano.config.floatX))
        if activation is None:
            self.activation = lambda x: x
        else:
            self.activation = activation

        self.create_parameters()

    def create_parameters(self):
        n_in, n_out = self.n_in, self.n_out
        rng_type = "uniform"
        scale = 1.0 / self.n_out ** 0.5
        # rng_type = None
        # scale = 1.0
        self.P = create_shared(random_init((n_in, n_out), rng_type=rng_type) * scale, name="P")
        self.Q = create_shared(random_init((n_in, n_out), rng_type=rng_type) * scale, name="Q")
        self.R = create_shared(random_init((n_in, n_out), rng_type=rng_type) * scale, name="R")
        self.O = create_shared(random_init((n_out, n_out), rng_type=rng_type) * scale, name="O")
        if self.activation == ReLU:
            self.b = create_shared(np.ones(n_out, dtype=theano.config.floatX) * 0.01, name="b")
        else:
            self.b = create_shared(random_init((n_out,)), name="b")

    def forward(self, x_t, f1_tm1, s1_tm1, f2_tm1, s2_tm1, f3_tm1):
        P, Q, R, decay = self.P, self.Q, self.R, self.decay
        f1_t = T.dot(x_t, P)
        s1_t = s1_tm1 * decay + f1_t
        f2_t = T.dot(x_t, Q) * s1_tm1
        s2_t = s2_tm1 * decay + f2_t
        f3_t = T.dot(x_t, R) * s2_tm1
        return f1_t, s1_t, f2_t, s2_t, f3_t

    def forward_all(self, x, v0=None):
        if v0 is None:
            if x.ndim > 1:
                v0 = T.zeros((x.shape[1], self.n_out), dtype=theano.config.floatX)
            else:
                v0 = T.zeros((self.n_out,), dtype=theano.config.floatX)
        ([f1, s1, f2, s2, f3], updates) = theano.scan(
            fn=self.forward,
            sequences=x,
            outputs_info=[v0, v0, v0, v0, v0]
        )
        if self.order == 3:
            h = f1 + f2 + f3 if self.use_all_grams else f3
        elif self.order == 2:
            h = f1 + f2 if self.use_all_grams else f2
        elif self.order == 1:
            h = f1
        else:
            raise ValueError(
                "Unsupported order: {}".format(self.order)
            )
        return self.activation(T.dot(h, self.O) + self.b)

    @property
    def params(self):
        if self.order == 3:
            return [self.b, self.O, self.P, self.Q, self.R]
        elif self.order == 2:
            return [self.b, self.O, self.P, self.Q]
        elif self.order == 1:
            return [self.b, self.O, self.P]
        else:
            raise ValueError(
                "Unsupported order: {}".format(self.order)
            )

    @params.setter
    def params(self, param_list):
        for p, q in zip(self.params, param_list):
            p.set_value(q.get_value())
