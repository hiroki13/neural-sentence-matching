import theano.tensor as T

import basic_model
from ..nn.advanced import AttentionLayer
from ..nn.basic import apply_dropout
from ..nn.nn_utils import normalize_2d, normalize_3d, average_without_padding


class Model(basic_model.Model):

    def __init__(self, args, emb_layer):
        super(Model, self).__init__(args, emb_layer)

    def compile(self):
        args = self.args

        self.set_layers(args=self.args, n_d=self.n_d, n_e=self.n_e)
        self.set_attention_layer()

        ###########
        # Network #
        ###########
        xt = self.input_layer(ids=self.idts)
        ht, ht_all = self.mid_layer(prev_h=xt, ids=self.idts, padding_id=self.padding_id)
        ht_a = self.attention(h=ht_all, ids=self.idts, idps=self.idps)

        if args.body:
            xb = self.input_layer(ids=self.idbs)
            hb, hb_all = self.mid_layer(prev_h=xb, ids=self.idbs, padding_id=self.padding_id)
            hb_a = self.attention(h=hb_all, ids=self.idbs, idps=self.idps)
        else:
            hb = None
            hb_a = None

        query_vecs, cand_vecs = self.output_layer(ht, ht_a, hb, hb_a, self.idps)

        #########
        # Score #
        #########
        # The first one in each batch is a positive sample, the rest are negative ones
        scores = self.get_scores(query_vecs, cand_vecs)
        pos_scores = scores[:, 0]
        neg_scores = scores[:, 1:]
        self.train_scores = T.argmax(scores, axis=1)
        self.predict_scores = scores[0]

        ##########################
        # Set training objective #
        ##########################
        self.set_params(layers=self.layers)
        self.set_loss(scores, pos_scores, neg_scores)
        self.set_cost(args=self.args, params=self.params, loss=self.loss)

    def mid_layer(self, prev_h, ids, padding_id):
        args = self.args

        for i in range(args.depth):
            # 1D: n_words, 2D: n_queries * n_cands, 3D: n_d
            ht = self.layers[i].forward_all(prev_h)
            prev_h = ht

        if args.normalize:
            ht = normalize_3d(ht)

        # 1D: batch, 2D: n_d
        if args.average or args.layer == "cnn" or args.layer == "str_cnn":
            h = average_without_padding(ht, ids, padding_id)
        else:
            h = ht[-1]

        # 1D: n_words, 2D: n_queries * n_cands, 3D: n_d
        return h, ht

    def set_attention_layer(self):
        attention_layer = AttentionLayer(a_type=self.args.attention, n_d=self.n_d, activation=self.activation)
        self.layers.append(attention_layer)

    def attention(self, h, idps, ids):
        """
        :param h: 1D: n_words, 2D: n_queries * n_cands, 3D: dim_h
        :param idps: 1D: n_queries, 2D: n_cands (zero-padded)
        :param ids: 1D: n_words, 2D: batch_size * n_cands
        :return: 1D: 1D: n_queries, 2D: n_cands, 3D: dim_h
        """

        #################
        # Query vectors #
        #################
        # query: 1D: n_queries, 2D: n_words, 3D: n_d
        q = h[:, idps.ravel()]
        query = q.reshape((q.shape[0], idps.shape[0], idps.shape[1], -1)).dimshuffle((1, 2, 0, 3))[:, 0, :, :]

        ################
        # Cand vectors #
        ################
        # cands, ids: 1D: n_queries * n_cands, 3D: dim_h
        cands = h[-1, idps.ravel()]
        # C, ids: 1D: n_queries, 2D: n_cands-1, 3D: n_d
        C = cands.reshape((idps.shape[0], idps.shape[1], -1))[:, 1:]

        ########
        # Mask #
        ########
        # 1D: n_queries * n_cands, 2D: n_words
        ids = ids.dimshuffle((1, 0))
        # 1D: n_queries * n_cands, 2D: n_words
        ids = ids[idps.ravel()]
        # 1D: n_queries, 2D: n_cands-1, 3D: n_words
        ids = ids.reshape((idps.shape[0], idps.shape[1], -1))[:, 0]
        mask = T.neq(ids, self.padding_id)
        mask = mask.dimshuffle((0, 'x', 1))

        return self.layers[-1].forward(query, C, mask)

    def output_layer(self, ht, ht_a, hb, hb_a, idps):
        ht = ht[idps.ravel()]
        ht = ht.reshape((idps.shape[0], idps.shape[1], -1))
        # 1D: n_queries, 2D: 1, 3D: n_d
        query_vecs = ht[:, 0, :]
        # 1D: n_queries, 2D: n_cands, 3D: n_d
        cand_vecs = (ht[:, 1:, :] + ht_a) * 0.5

        if self.args.body:
            hb = hb[idps.ravel()]
            hb = hb.reshape((idps.shape[0], idps.shape[1], -1))

            query_vecs_b = hb[:, 0, :]
            cand_vecs_b = (hb[:, 1:, :] + hb_a) * 0.5

            query_vecs = (query_vecs + query_vecs_b) * 0.5
            cand_vecs = (cand_vecs + cand_vecs_b) * 0.5

        query_vecs = apply_dropout(query_vecs, self.dropout)
        query_vecs = normalize_2d(query_vecs)
        cand_vecs = apply_dropout(cand_vecs, self.dropout)
        cand_vecs = normalize_3d(cand_vecs)

        return query_vecs, cand_vecs
