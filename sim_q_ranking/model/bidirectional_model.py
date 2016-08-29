import numpy as np
import theano
import theano.tensor as T

from ..model import basic_model
from ..nn.initialization import get_activation_by_name
from ..nn.basic import LSTM, GRU
from ..nn.advanced import RCNN
from ..nn.basic import apply_dropout


class Model(basic_model.Model):

    def __init__(self, args, emb_layer):
        super(Model, self).__init__(args, emb_layer)

    def compile(self):
        self.set_input_format()
        self.set_layers(args=self.args, n_d=self.n_d, n_e=self.n_e)

        self.set_input_layer(ids=self.idts, idbs=self.idbs, embedding_layer=self.emb_layer,
                             n_e=self.n_e, dropout=self.dropout)
        self.set_mid_layer(args=self.args, prev_ht_l=self.xt, prev_hb=self.xb, layers=self.layers, n_d=self.n_d)
        self.set_output_layer(args=self.args, ht=self.ht, hb=self.hb, dropout=self.dropout)

        self.set_params(layers=self.layers)
        self.set_loss(n_d=self.n_d, idps=self.idps, h_o=self.h_final)
        self.set_cost(args=self.args, params=self.params, loss=self.loss)

        self.get_predict_scores(h=self.h_final)

    def set_layers(self, args, n_d, n_e):
        activation = get_activation_by_name(args.activation)

        if args.layer.lower() == "rcnn":
            layer_type = RCNN
        elif args.layer.lower() == "lstm":
            layer_type = LSTM
        elif args.layer.lower() == "gru":
            layer_type = GRU

        if args.share_w:
            depth = args.depth
        else:
            depth = args.depth * 2

        for i in range(depth):
            if layer_type != RCNN:
                feature_layer = layer_type(
                    n_in=n_e,
                    n_out=n_d,
                    activation=activation
                )
            else:
                feature_layer = layer_type(
                    n_in=n_e,
                    n_out=n_d,
                    activation=activation,
                    order=args.order,
                    mode=args.mode,
                    has_outgate=args.outgate
                )
            self.layers.append(feature_layer)

    def set_mid_layer(self, args, prev_ht_l, prev_hb, layers, n_d):
        # 1D: n_words, 2D: batch, 3D: n_d
        prev_ht_r = prev_ht_l[::-1]

        for i in range(args.depth):
            # 1D: n_words, 2D: batch_size * n_cands, 3D: n_d
            if args.share_w:
                ht_l = layers[i].forward_all(prev_ht_l)
                ht_r = layers[i].forward_all(prev_ht_r)
            else:
                ht_l = layers[i * 2].forward_all(prev_ht_l)
                ht_r = layers[i * 2 + 1].forward_all(prev_ht_r)
            prev_ht_l = ht_l
            prev_ht_r = ht_r

        if args.normalize:
            ht_l = self.normalize_3d(ht_l)
            ht_r = self.normalize_3d(ht_r)

        ht = self.conv_without_padding(ht_l, ht_r, self.idts)
        ht = self.normalize_2d(ht)

        self.ht = ht

    def conv_without_padding(self, h_l, h_r, ids, eps=1e-8):
        """
        :param h_l: 1D: n_words, 2D: batch, 3D: n_d
        :param h_r: 1D: n_words, 2D: batch, 3D: n_d
        :param ids: 1D: n_words, 2D: batch
        :return: 1D: batch, 2D: n_d
        """
        # 1D: n_words, 2D: batch, 3D: 1
        mask = T.neq(ids, self.padding_id).dimshuffle((0, 1, 'x'))
        mask = T.cast(mask, theano.config.floatX)

        # 1D: batch, 2D: n_words, 3D: n_d
        h_l = h_l * mask
        h_l = h_l.dimshuffle((1, 0, 2))
        h_r = h_r[::-1] * mask
        h_r = h_r.dimshuffle((1, 0, 2))

        # 1D: batch, 2D: 2 * n_words, 3D: n_d
        h = T.concatenate([h_l, h_r], axis=2)
        h = T.max(h + eps, axis=1)
        return h

    def set_loss(self, n_d, idps, h_o):
        # 1D: n_queries, 2D: n_cands-1, 3D: 2 * dim_h
        xp = h_o[idps.ravel()]
        xp = xp.reshape((idps.shape[0], idps.shape[1], 2 * n_d))

        if self.args.loss == 'ce':
            self.cross_entropy(xp)
        elif self.args.loss == 'sbs':
            self.soft_bootstrapping(xp, self.args.beta)
        elif self.args.loss == 'hbs':
            self.hard_bootstrapping(xp, self.args.beta)
        else:
            self.hinge_loss(xp)


class DoubleModel(basic_model.Model):

    def __init__(self, args, emb_layer):
        super(DoubleModel, self).__init__(args, emb_layer)
        self.ht_b = None

    def compile(self):
        self.set_input_format()
        self.set_layers(args=self.args, n_d=self.n_d, n_e=self.n_e)

        self.set_input_layer(ids=self.idts, idbs=self.idbs, embedding_layer=self.emb_layer,
                             n_e=self.n_e, dropout=self.dropout)
        self.set_mid_layer(args=self.args, prev_h=self.xt, prev_hb=self.xb, layers=self.layers, n_d=self.n_d)
        self.set_output_layer(args=self.args, ht=self.ht, ht_b=self.ht_b, dropout=self.dropout)

        self.set_params(layers=self.layers)
        self.set_loss(n_d=self.n_d, idps=self.idps, h_o=self.h_final)
        self.set_cost(args=self.args, params=self.params, loss=self.loss)

        self.get_predict_scores(h=self.h_final)

    def set_layers(self, args, n_d, n_e):
        activation = get_activation_by_name(args.activation)

        if args.layer.lower() == "rcnn":
            layer_type = RCNN
        elif args.layer.lower() == "lstm":
            layer_type = LSTM
        elif args.layer.lower() == "gru":
            layer_type = GRU

        for i in range(args.depth * 2):
            if layer_type != RCNN:
                feature_layer = layer_type(
                    n_in=n_e,
                    n_out=n_d,
                    activation=activation
                )
            else:
                feature_layer = layer_type(
                    n_in=n_e,
                    n_out=n_d,
                    activation=activation,
                    order=args.order,
                    mode=args.mode,
                    has_outgate=args.outgate
                )
            self.layers.append(feature_layer)

    def set_mid_layer(self, args, prev_h, prev_hb, layers, n_d):
        prev_ht_b = prev_h[::-1]

        for i in range(args.depth):
            # 1D: n_words, 2D: batch_size * n_cands, 3D: n_d
            ht = layers[i * 2].forward_all(prev_h)
            ht_b = layers[i * 2 + 1].forward_all(prev_ht_b)
            prev_h = ht
            prev_ht_b = ht_b

        if args.normalize:
            ht = self.normalize_3d(ht)
            ht_b = self.normalize_3d(ht_b)

        self.ht = ht[-1]
        self.ht_b = ht_b[-1]

    def set_output_layer(self, args, ht, ht_b, dropout):
        alpha = theano.shared(np.asarray(np.random.uniform(low=-1., high=1.), dtype=theano.config.floatX))
        self.params.append(alpha)

        # 1D: n_queries * n_cands, 2D: dim_h
        h_final = apply_dropout(ht, dropout)
        h_final_b = apply_dropout(ht_b, dropout)
        h_final = h_final + alpha * h_final_b
        self.h_final = self.normalize_2d(h_final)
