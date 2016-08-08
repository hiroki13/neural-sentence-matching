import numpy as np
import theano
import theano.tensor as T

from model import basic_model
from nn.initialization import get_activation_by_name
from nn.basic import LSTM, GRU
from nn.advanced import RCNN
from nn.initialization import random_init, create_shared
from nn.basic import apply_dropout


class Model(basic_model.Model):

    def __init__(self, args, emb_layer):
        super(Model, self).__init__(args, emb_layer)

    def compile(self):
        self.set_input_format()
        self.set_layers(args=self.args, n_d=self.n_d, n_e=self.n_e)

        self.set_input_layer(idts=self.idts, idbs=self.idbs, embedding_layer=self.emb_layer,
                             n_e=self.n_e, dropout=self.dropout)
        self.set_intermediate_layer(args=self.args, prev_ht=self.xt, prev_hb=self.xb, layers=self.layers, n_d=self.n_d)
        self.set_output_layer(args=self.args, ht=self.ht, hb=self.hb, dropout=self.dropout)

        self.set_params(layers=self.layers)
        self.set_loss(n_d=self.n_d, idps=self.idps, h_final=self.h_final)
        self.set_cost(args=self.args, params=self.params, loss=self.loss)

        self.set_scores(h_final=self.h_final)

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

    def set_intermediate_layer(self, args, prev_ht, prev_hb, layers, n_d):
        prev_ht_b = prev_ht[::-1]

        for i in range(args.depth):
            # 1D: n_words, 2D: batch_size * n_cands, 3D: n_d
            ht = layers[i * 2].forward_all(prev_ht)
            ht_b = layers[i * 2 + 1].forward_all(prev_ht_b)
            prev_ht = ht
            prev_ht_b = ht_b

        if args.normalize:
            ht = self.normalize_3d(ht)
            ht_b = self.normalize_3d(ht_b)

#        W = create_shared(random_init((n_d*2, n_d)))
#        ht = T.tanh(T.dot(T.concatenate([ht[-1], ht_b[-1]], axis=1), W))
        ht = ht[-1] * ht_b[-1]
        ht = self.normalize_2d(ht)

        self.ht = ht
#        self.params.append(W)


class DoubleModel(basic_model.Model):

    def __init__(self, args, emb_layer):
        super(DoubleModel, self).__init__(args, emb_layer)
        self.ht_b = None

    def compile(self):
        self.set_input_format()
        self.set_layers(args=self.args, n_d=self.n_d, n_e=self.n_e)

        self.set_input_layer(idts=self.idts, idbs=self.idbs, embedding_layer=self.emb_layer,
                             n_e=self.n_e, dropout=self.dropout)
        self.set_intermediate_layer(args=self.args, prev_ht=self.xt, prev_hb=self.xb, layers=self.layers, n_d=self.n_d)
        self.set_output_layer(args=self.args, ht=self.ht, ht_b=self.ht_b, dropout=self.dropout)

        self.set_params(layers=self.layers)
        self.set_loss(n_d=self.n_d, idps=self.idps, h_final=self.h_final)
        self.set_cost(args=self.args, params=self.params, loss=self.loss)

        self.set_scores(h_final=self.h_final)

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

    def set_intermediate_layer(self, args, prev_ht, prev_hb, layers, n_d):
        prev_ht_b = prev_ht[::-1]

        for i in range(args.depth):
            # 1D: n_words, 2D: batch_size * n_cands, 3D: n_d
            ht = layers[i * 2].forward_all(prev_ht)
            ht_b = layers[i * 2 + 1].forward_all(prev_ht_b)
            prev_ht = ht
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
