from prettytable import PrettyTable
import theano
import theano.tensor as T

import basic_model
from ..nn.initialization import get_activation_by_name
from ..nn.basic import Layer
from ..utils.io_utils import say
from ..nn.nn_utils import normalize_3d, average_without_padding, average_without_padding_3d, get_mask, softmax


class AverageModel(basic_model.Model):

    def __init__(self, args, emb_layer):
        super(AverageModel, self).__init__(args, emb_layer)

    def compile(self):
        args = self.args

        ###########
        # Network #
        ###########
        xt = self.input_layer(ids=self.idts)
        ht = self.mid_layer(h=xt, ids=self.idts, padding_id=self.padding_id)

        if args.body:
            xb = self.input_layer(ids=self.idbs)
            hb = self.mid_layer(h=xb, ids=self.idbs, padding_id=self.padding_id)
        else:
            hb = None

        query_vecs, cand_vecs = self.output_layer(ht, hb, self.idps)

        #########
        # Score #
        #########
        # The first one in each batch is a positive sample, the rest are negative ones
        scores = self.get_scores(query_vecs, cand_vecs)
        self.predict_scores = scores[0]

    def mid_layer(self, h, ids, padding_id):
        return average_without_padding(h, ids, padding_id)

    def train(self, ids_corpus, dev=None, test=None):
        say('\nBuilding functions...\n\n')

        #####################
        # Set the functions #
        #####################
        eval_func = theano.function(
            inputs=[self.idts, self.idbs, self.idps],
            outputs=self.predict_scores,
            on_unused_input='ignore'
        )

        result_table = PrettyTable(["Epoch", "dev MAP", "dev MRR", "dev P@1", "dev P@5"] +
                                   ["tst MAP", "tst MRR", "tst P@1", "tst P@5"])

        dev_MAP = dev_MRR = dev_P1 = dev_P5 = 0
        test_MAP = test_MRR = test_P1 = test_P5 = 0

        if dev is not None:
            dev_MAP, dev_MRR, dev_P1, dev_P5 = self.evaluate(dev, eval_func)
        if test is not None:
            test_MAP, test_MRR, test_P1, test_P5 = self.evaluate(test, eval_func)

        result_table.add_row(
            [0] +
            ["%.2f" % x for x in [dev_MAP, dev_MRR, dev_P1, dev_P5] + [test_MAP, test_MRR, test_P1, test_P5]]
        )

        say("\r\n\n")
        say("{}".format(result_table))
        say("\n")


class WeightedAverageModel(basic_model.Model):

    def __init__(self, args, emb_layer):
        super(WeightedAverageModel, self).__init__(args, emb_layer)

    def compile(self):
        args = self.args

        self.set_layers(args=self.args, n_d=self.n_d, n_e=self.n_e)

        ###########
        # Network #
        ###########
        xt = self.input_layer(ids=self.idts)
        ht = self.mid_layer(h=xt, ids=self.idts, padding_id=self.padding_id)

        if args.body:
            xb = self.input_layer(ids=self.idbs)
            hb = self.mid_layer(h=xb, ids=self.idbs, padding_id=self.padding_id)
        else:
            hb = None

        query_vecs, cand_vecs = self.output_layer(ht, hb, self.idps)

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

    def set_layers(self, args, n_d, n_e):
        activation = get_activation_by_name(args.activation)
        feature_layer = Layer(n_in=n_e, n_out=n_e, activation=activation)
        self.layers.append(feature_layer)

    def mid_layer(self, h, ids, padding_id):
        return average_without_padding(h, ids, padding_id)

    def get_scores(self, query_vecs, cand_vecs):
        # 1D: n_queries, 2D: n_cands
        q = self.layers[-1].forward(query_vecs).dimshuffle(0, 'x', 1)
        return T.sum(q * cand_vecs, axis=2)


class AlignmentModel(basic_model.Model):

    def __init__(self, args, emb_layer):
        super(AlignmentModel, self).__init__(args, emb_layer)

    def compile(self):
        args = self.args

        self.set_layers(args=self.args, n_d=self.n_d, n_e=self.n_e)

        ###########
        # Network #
        ###########
        mask_t = get_mask(self.idts, self.idps, self.padding_id)
        xt = self.input_layer(ids=self.idts)
        Ht = self.mid_layer(prev_h=xt)
        At = self.alignment(Ht, self.idps, mask_t)
        a_scores_t = get_alignment_scores(At, mask_t)

        if args.body:
            mask_b = get_mask(self.idbs, self.idps, self.padding_id)
            xb = self.input_layer(ids=self.idbs)
            Hb = self.mid_layer(prev_h=xb)
            Ab = self.alignment(Hb, self.idps, mask_b)
            a_scores_b = get_alignment_scores(Ab, mask_b)
            a_scores_t = (a_scores_t + a_scores_b) * 0.5

        #########
        # Score #
        #########
        # The first one in each batch is a positive sample, the rest are negative ones
        # 1D: n_queries, 2D: n_cands
        scores = a_scores_t
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

    def mid_layer(self, prev_h):
        """
        :param prev_h: 1D: n_words, 2D: batch, 3D: n_d
        :return: 1D: n_words, 2D: batch, 3D: n_d
        """
        args = self.args

        for i in range(args.depth):
            # 1D: n_words, 2D: batch, 3D: n_d
            h = self.layers[i].forward_all(prev_h)
            prev_h = h

        if args.normalize:
            h = normalize_3d(h)

        return h

    def alignment(self, x, idps, mask):
        # 1D: batch * n_cands, 2D: n_words, 3D: n_e
        x = x.dimshuffle((1, 0, 2))

        # 1D: batch * n_cands, 2D: n_words, 3D: n_e
        h = x[idps.ravel()]
        # 1D: batch, 2D: n_cands, 3D: n_words, 4D: n_e
        h = h.reshape((idps.shape[0], idps.shape[1], h.shape[1], h.shape[2]))

        ###########
        # Masking #
        ###########
        h = h * mask.dimshuffle((0, 1, 2, 'x'))

        #################
        # Query & Cands #
        #################
        # 1D: batch, 2D: n_words, 3D: n_e
        query = h[:, 0]
        # 1D: batch, 2D: n_cands-1, 3D: n_words, 4D: n_e
        cands = h[:, 1:]

        return get_alignment_matrix(query, cands)


class AlignmentVectorModel(AlignmentModel):

    def __init__(self, args, emb_layer):
        super(AlignmentVectorModel, self).__init__(args, emb_layer)

    def compile(self):
        args = self.args

        self.set_layers(args=self.args, n_d=self.n_d, n_e=self.n_e)

        ###########
        # Network #
        ###########
        mask_t = get_mask(self.idts, self.idps, self.padding_id)
        xt = self.input_layer(ids=self.idts)
        Ht = self.mid_layer(prev_h=xt)

        # Q: 1D: n_queries, 2D: n_words, 3D: n_d
        # C: 1D: n_queries, 2D: n_cands-1, 3D: n_words, 4D: n_d
        # A: 1D: n_queries, 2D: n_cands-1, 3D: n_words (cands), 4D: n_words (query)
        Qt, Ct, At = self.alignment(Ht, self.idps, mask_t)

        # A_q, 1D: n_queries, 2D: n_cands-1, 3D: n_words (query), 4D: n_words (cands)
        # A_c, 1D: n_queries, 2D: n_cands-1, 3D: n_words (cands), 4D: n_words (query)
        Aq_t, Ac_t = get_attention_matrix(At)

        # 1D: n_queries, 2D: n_cands-1, 3D: n_words (cands/query), 4D: dim_h
        Qv_t = get_attention_vectors(Ct, Aq_t)
        Cv_t = get_attention_vectors(Qt.dimshuffle((0, 'x', 1, 2)), Ac_t)

        # 1D: n_queries, 2D: n_cands-1, 3D: dim_h
        Qv_t = T.mean(Qv_t, axis=2)
        Cv_t = T.mean(Cv_t, axis=2)

        # 1D: n_queries, 2D: n_cands-1
        a_scores_t = T.sum(Qv_t * Cv_t, axis=2)

        if args.body:
            mask_b = get_mask(self.idbs, self.idps, self.padding_id)
            xb = self.input_layer(ids=self.idbs)
            Hb = self.mid_layer(prev_h=xb)
            Ab = self.alignment(Hb, self.idps, mask_b)
            a_scores_b = get_alignment_scores(Ab, mask_b)
            a_scores_t = (a_scores_t + a_scores_b) * 0.5

        #########
        # Score #
        #########
        # The first one in each batch is a positive sample, the rest are negative ones
        # 1D: n_queries, 2D: n_cands
        scores = a_scores_t
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

    def alignment(self, x, idps, mask):
        # 1D: batch * n_cands, 2D: n_words, 3D: n_d
        x = x.dimshuffle((1, 0, 2))

        # 1D: batch * n_cands, 2D: n_words, 3D: n_d
        h = x[idps.ravel()]
        # 1D: batch, 2D: n_cands, 3D: n_words, 4D: n_d
        h = h.reshape((idps.shape[0], idps.shape[1], h.shape[1], h.shape[2]))

        ###########
        # Masking #
        ###########
        h = h * mask.dimshuffle((0, 1, 2, 'x'))

        #################
        # Query & Cands #
        #################
        # 1D: batch, 2D: n_words, 3D: n_d
        query = h[:, 0]
        # 1D: batch, 2D: n_cands-1, 3D: n_words, 4D: n_d
        cands = h[:, 1:]

        return query, cands, get_alignment_matrix(query, cands)


def get_alignment_matrix(query, cands):
    """
    :param query: 1D: n_queries, 2D: n_words, 3D: dim_h
    :param cands: 1D: n_queries, 2D: n_cands-1, 3D: n_words, 4D: dim_h
    :return: 1D: n_queries, 2D: n_cands-1, 3D: n_words (cands), 4D: n_words (query)
    """
    q = query.dimshuffle(0, 'x', 'x', 1, 2)
    c = cands.dimshuffle(0, 1, 2, 'x', 3)
    return T.sum(q * c, axis=4)


def get_attention_matrix(A):
    """
    :param A: 1D: n_queries, 2D: n_cands-1, 3D: n_words (cands), 4D: n_words (query)
    :return: A_q, 1D: n_queries, 2D: n_cands-1, 3D: n_words (query), 4D: n_words (cands)
    :return: A_c, 1D: n_queries, 2D: n_cands-1, 3D: n_words (cands), 4D: n_words (query)
    """
    # 1D: n_queries, 2D: n_cands-1, 3D: n_words (query), 4D: n_words (cands)
    A_q = softmax(x=A.dimshuffle((0, 1, 3, 2)), axis=3)
    # 1D: n_queries, 2D: n_cands-1, 3D: n_words (cands), 4D: n_words (query)
    A_c = softmax(x=A, axis=3)
    return A_q, A_c


def get_attention_vectors(x, A):
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


def get_alignment_scores(A, mask):
    """
    :param A: 1D: n_queries, 2D: n_cands-1, 3D: n_words (cands), 4D: n_words (query)
    :return: 1D: n_queries, 2D: n_cands-1
    """
    return average_without_padding_3d(T.max(A, axis=3), mask)
