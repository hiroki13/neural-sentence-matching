import theano.tensor as T

from model import basic_model
from nn.advanced import AttentionLayer
from nn.basic import apply_dropout


class Model(basic_model.Model):

    def __init__(self, args, emb_layer):
        super(Model, self).__init__(args, emb_layer)
        self.a_ht = None
        self.a_hb = None

    def compile(self):
        self.set_input_format()
        self.set_layers(args=self.args, n_d=self.n_d, n_e=self.n_e)

        self.set_input_layer(idts=self.idts, idbs=self.idbs, embedding_layer=self.emb_layer,
                             n_e=self.n_e, dropout=self.dropout)
        self.set_intermediate_layer(args=self.args, prev_ht=self.xt, prev_hb=self.xb, layers=self.layers)
        self.set_attention_layer(args=self.args, activation=self.activation, idts=self.idts, idbs=self.idbs,
                                 idps=self.idps, ht=self.ht, hb=self.hb, n_d=self.n_d, dropout=self.dropout)
        self.set_output_layer(args=self.args, ht=self.ht[-1], hb=self.hb[-1], dropout=self.dropout)

        self.set_params(layers=self.layers)
        self.set_loss(args=self.args, n_d=self.n_d, idps=self.idps, h_final=self.h_final,
                      a_ht=self.a_ht, a_hb=self.a_hb)
        self.set_cost(args=self.args, params=self.params, loss=self.loss)

        self.set_scores(args=self.args, h_final=self.h_final, a_ht=self.a_ht, a_hb=self.a_hb)

    def set_intermediate_layer(self, args, prev_ht, prev_hb, layers):
        for i in range(len(layers)):
            # 1D: n_words, 2D: n_queries * n_cands, 3D: n_d
            ht = layers[i].forward_all(prev_ht)
            hb = layers[i].forward_all(prev_hb)
            prev_ht = ht
            prev_hb = hb

        if args.normalize:
            ht = self.normalize_3d(ht)
            hb = self.normalize_3d(hb)

        # 1D: n_words, 2D: n_queries * n_cands, 3D: n_d
        self.ht = ht
        self.hb = hb

    def set_attention_layer(self, args, activation, idts, idbs, idps, ht, hb, n_d, dropout):
        attention_layer = AttentionLayer(n_d=n_d, activation=activation)
        self.layers.append(attention_layer)

        # 1D: n_queries, 2D: n_cands-1, 3D: dim_h
        a_ht = self.attention(layer=attention_layer, h=ht, idps=idps, n_d=n_d, ids=idts)
        self.a_ht = apply_dropout(a_ht, dropout)

        if args.body:
            a_hb = self.attention(layer=attention_layer, h=hb, idps=idps, n_d=n_d, ids=idbs)
            self.a_hb = apply_dropout(a_hb, dropout)

    def set_output_layer(self, args, ht, hb, dropout):
        # 1D: n_queries * n_cands, 2D: dim_h
        if args.body:
            h_final = (ht + hb) * 0.5
        else:
            h_final = ht

        h_final = apply_dropout(h_final, dropout)
        self.h_final = self.normalize_2d(h_final)

    def set_loss(self, args, n_d, idps, h_final, a_ht, a_hb):
        # 1D: n_queries, 2D: n_cands-1, 3D: dim_h
        xp = h_final[idps.ravel()]
        xp = xp.reshape((idps.shape[0], idps.shape[1], n_d))

        query_vecs = xp[:, 0, :]

        if args.body:
            a_ht = (a_ht + a_hb) * 0.5
        cand_vecs = (xp[:, 1:, :] + a_ht) * 0.5

        # 1D: n_queries, 2D: 1, 3D: n_d
        pos_scores = T.sum(query_vecs * cand_vecs[:, 0, :], axis=1)  # 1D: n_queries

        # 1D: n_queries, 2D: n_cands-2
        neg_scores = T.sum(query_vecs.dimshuffle((0, 'x', 1)) * cand_vecs[:, 1:, :], axis=2)
        neg_scores = T.max(neg_scores, axis=1)  # 1D: n_queries

        diff = neg_scores - pos_scores + 1.0
        self.loss = T.mean((diff > 0) * diff)

    def set_scores(self, args, h_final, a_ht, a_hb):
        a_ht = a_ht.reshape((a_ht.shape[0] * a_ht.shape[1], a_ht.shape[2]))[:h_final.shape[0]-1]

        if args.body:
            a_hb = a_hb.reshape((a_hb.shape[0] * a_hb.shape[1], a_hb.shape[2]))[:h_final.shape[0]-1]
            a_ht = (a_ht + a_hb) * 0.5

        cand_vecs = (h_final[1:] + a_ht) * 0.5
        self.scores = T.dot(cand_vecs, h_final[0])

    def attention(self, layer, h, idps, n_d, ids):
        """
        :param layer:
        :param h: 1D: n_words, 2D: n_queries * n_cands, 3D: dim_h
        :param idps: 1D: n_queries, 2D: n_cands (zero-padded)
        :param n_d: float32
        :param ids: 1D: n_words, 2D: batch_size * n_cands
        :return: 1D: 1D: n_queries, 2D: n_cands, 3D: dim_h
        """

        #####################
        # Contexts and Mask #
        #####################
        # cands, ids: 1D: n_queries * n_cands, 3D: dim_h
        cands = h[-1, idps.ravel()]
        ids = ids[:, idps.ravel()]

        # C, ids: 1D: n_queries, 2D: n_cands-1, 3D: n_d
        C = cands.reshape((idps.shape[0], idps.shape[1], n_d))[:, 1:]

        # 1D: n_queries, 2D: n_cands-1, 3D: n_words
        ids = ids.reshape((ids.shape[0], idps.shape[0], idps.shape[1])).dimshuffle(1, 2, 0)[:, 0]
        mask = T.neq(ids, self.padding_id)
        mask = mask.dimshuffle((0, 'x', 1))

        #################
        # Query vectors #
        #################
        # query: 1D: n_queries, 2D: n_words, 3D: n_d
        q = h[:, idps.ravel()]
        query = q.reshape((q.shape[0], idps.shape[0], idps.shape[1], n_d)).dimshuffle((1, 2, 0, 3))[:, 0, :, :]

        return layer.forward(query, C, mask)
