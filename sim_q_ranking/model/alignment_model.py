import gzip
import time
import math
import cPickle as pickle

from prettytable import PrettyTable
import numpy as np
import theano
import theano.tensor as T

import basic_model
from ..nn.advanced import AttentionLayer, AlignmentLayer
from ..nn.initialization import get_activation_by_name
from ..nn.optimization import create_optimization_updates
from ..nn.basic import LSTM, GRU, CNN, apply_dropout, Layer
from ..nn.advanced import RCNN, GRNN, StrCNN
from ..utils.io_utils import say, read_annotations, create_batches
from ..utils.eval import Evaluation

PAD = "<padding>"


class Model(basic_model.Model):

    def __init__(self, args, emb_layer):
        super(Model, self).__init__(args, emb_layer)

    def compile(self):
        args = self.args

        self.set_input_format()

        xt = self.input_layer(ids=self.idts)
#        query, ht1, ht2 = self.set_mid_layer(prev_h=xt, idps=self.idps)
        query, ht1 = self.mid_layer(prev_h=xt, idps=self.idps)
#        h_o = self.set_output_layer(ht1, ht2)
#        h_o = self.set_output_layer(ht1)

#        scores = self.set_scores(query, h_o)
        scores = self.get_predict_scores(query, ht1)
        self.set_params(layers=self.layers)
        self.set_loss(scores)
        self.set_cost(args=args, params=self.params, loss=self.loss)

    def mid_layer(self, prev_h, idps):
        # 1D: batch * n_cands, 2D: n_words, 3D: n_e
        prev_h = prev_h.dimshuffle((1, 0, 2))

        # 1D: batch * n_cands, 2D: n_words, 3D: n_e
        h = prev_h[idps.ravel()]
        # 1D: batch, 2D: n_cands, 3D: n_words, 4D: n_e
        h = h.reshape((idps.shape[0], idps.shape[1], h.shape[1], h.shape[2]))

        # 1D: batch, 2D: n_words, 3D: n_e
        query = h[:, 0]
        # 1D: batch, 2D: n_cands-1, 3D: n_words, 4D: n_e
        cands = h[:, 1:]

        return query, cands

    """
    def set_mid_layer(self, prev_h, idps):
        # 1D: batch * n_cands, 2D: n_words, 3D: n_e
        prev_h = prev_h.dimshuffle((1, 0, 2))

        # 1D: batch * n_cands, 2D: n_words, 3D: n_e
        h = prev_h[idps.ravel()]
        # 1D: batch, 2D: n_cands, 3D: n_words, 4D: n_e
        h = h.reshape((idps.shape[0], idps.shape[1], h.shape[1], h.shape[2]))

        # 1D: batch, 2D: n_words, 3D: n_e
        query = h[:, 0]
        # 1D: batch, 2D: n_cands-1, 3D: n_words, 4D: n_e
        cands = h[:, 1:]

        layer = AlignmentLayer(n_e=self.n_e, n_d=self.n_d, activation=self.activation)
        self.layers.append(layer)

        # 1D: batch, 2D: n_cands-1, 3D: n_words
        A = self.alignment(layer, query, cands)
        A = A / T.sum(cands ** 2, axis=3)

        # 1D: batch, 2D: n_cands-1, 3D: n_words, 4D: n_e
#        vec1, vec2 = self.decompose(A, cands)
        vec = self.sim_weighted(A, cands)

        return query, vec
    """

    def output_layer(self, h1):
        # 1D: batch, 2D: n_cands-1, 3D: n_d
#        h_o = T.max(self.layers[-1].linear(h1, h2), axis=2)
        h_o = T.max(self.layers[-1].linear(h1), axis=2)
        h_o = apply_dropout(h_o, self.dropout)
        return self.normalize_3d(h_o)

    """
    def set_scores(self, query_vecs, cand_vecs):
        layer = Layer(self.n_e, self.n_d, T.tanh)
        self.layers.append(layer)
        query_vecs = T.mean(layer.forward(query_vecs), axis=1, keepdims=True)
        scores = T.sum(query_vecs * cand_vecs, axis=2)
        self.scores = scores
        return scores
    """

    def get_predict_scores(self, query_vecs, cand_vecs):
        layer = Layer(self.n_e, self.n_d, T.tanh)
        self.layers.append(layer)
        query_vecs = T.mean(layer.forward(query_vecs), axis=1, keepdims=True)
        cand_vecs = T.mean(layer.forward(cand_vecs), axis=2)
        scores = T.sum(query_vecs * cand_vecs, axis=2)
        self.predict_scores = scores
        return scores

    def set_loss(self, scores):
        pos_scores = scores[:, 0]
        neg_scores = scores[:, 1:]
        neg_scores = T.max(neg_scores, axis=1)

        diff = neg_scores - pos_scores + 1.0
        self.loss = T.mean((diff > 0) * diff)
        self.train_scores = T.argmax(scores, axis=1)

    def alignment(self, layer, query, cands):
        return layer.alignment(query, cands)

    def sim_weighted(self, A, cands):
        # 1D: batch, 2D: n_cands-1, 3D: n_words, 4D: n_e
        return A.dimshuffle((0, 1, 2, 'x')) * cands

    def decompose(self, A, cands):
        # 1D: batch, 2D: n_cands-1, 3D: n_words, 4D: n_e
        vec1 = A.dimshuffle((0, 1, 2, 'x')) * cands
        vec2 = cands - vec1
        return vec1, vec2

    def evaluate(self, data, eval_func):
        res = []
        for idts, idbs, labels in data:
            # idts, idbs: 1D: n_words, 2D: n_cands
            idps = np.asarray([[i for i in xrange(idts.shape[1])]], dtype='int32')
            scores = eval_func(idts, idbs, idps)[0]

            assert len(scores) == len(labels)
            ranks = (-scores).argsort()
            ranked_labels = labels[ranks]
            res.append(ranked_labels)

        e = Evaluation(res)
        MAP = e.MAP() * 100
        MRR = e.MRR() * 100
        P1 = e.Precision(1) * 100
        P5 = e.Precision(5) * 100

        return MAP, MRR, P1, P5

    def train(self, ids_corpus, dev=None, test=None):
        say('\nBuilding functions...\n\n')

        args = self.args

        batch_size = args.batch_size
        padding_id = self.padding_id

        ############################
        # Set the update procedure #
        ############################
        updates, lr, gnorm = create_optimization_updates(
            cost=self.cost,
            params=self.params,
            lr=args.learning_rate,
            method=args.learning
        )[:3]

        #####################
        # Set the functions #
        #####################
        train_func = theano.function(
            inputs=[self.idts, self.idbs, self.idps],
            outputs=[self.cost, self.loss, gnorm, self.train_scores],
            updates=updates,
            on_unused_input='ignore'
        )

        eval_func = theano.function(
            inputs=[self.idts, self.idbs, self.idps],
            outputs=self.predict_scores,
            on_unused_input='ignore'
        )

        say("\tp_norm: {}\n".format(self.get_pnorm_stat()))

        result_table = PrettyTable(["Epoch", "dev MAP", "dev MRR", "dev P@1", "dev P@5"] +
                                   ["tst MAP", "tst MRR", "tst P@1", "tst P@5"])

        unchanged = 0
        best_dev = -1

        dev_MAP = dev_MRR = dev_P1 = dev_P5 = 0
        test_MAP = test_MRR = test_P1 = test_P5 = 0
        max_epoch = args.max_epoch

        for epoch in xrange(max_epoch):
            unchanged += 1
            if unchanged > 15:
                break

            start_time = time.time()

            train = read_annotations(args.train, data_size=args.data_size)
            train_batches = create_batches(ids_corpus, train, batch_size, padding_id, pad_left=not args.average)
            n_train_batches = len(train_batches)

            train_loss = 0.0
            train_cost = 0.0
            crr = 0.
            ttl = 0.

            for i in xrange(n_train_batches):
                # get current batch
                idts, idbs, idps = train_batches[i]

                cur_cost, cur_loss, grad_norm, preds = train_func(idts, idbs, idps)

                if math.isnan(cur_loss):
                    say('\n\nNAN: Index: %d\n' % i)
                    exit()

                train_loss += cur_loss
                train_cost += cur_cost

                crr += len([s for s in preds if s == 0])
                ttl += len(preds)

                if i % 10 == 0:
                    say("\r{}/{}".format(i, n_train_batches))

            ##############
            # Validating #
            ##############
            # Set the dropout prob for validating
            self.dropout.set_value(0.0)

            if dev is not None:
                dev_MAP, dev_MRR, dev_P1, dev_P5 = self.evaluate(dev, eval_func)
            if test is not None:
                test_MAP, test_MRR, test_P1, test_P5 = self.evaluate(test, eval_func)

            if dev_MRR > best_dev:
                unchanged = 0
                best_dev = dev_MRR
                result_table.add_row(
                    [epoch] +
                    ["%.2f" % x for x in [dev_MAP, dev_MRR, dev_P1, dev_P5] +
                     [test_MAP, test_MRR, test_P1, test_P5]]
                )
                if args.save_model:
                    self.save_model(args.save_model)

            # Set the dropout prob for training
            dropout_p = np.float32(args.dropout).astype(theano.config.floatX)
            self.dropout.set_value(dropout_p)

            say("\r\n\n")
            say(("Epoch {}\tcost={:.3f}\tloss={:.3f}" + "\tMRR={:.2f},{:.2f}\t|g|={:.8f}\t[{:.3f}m]\n").format(
                epoch,
                train_cost / (i + 1),
                train_loss / (i + 1),
                dev_MRR,
                best_dev,
                float(grad_norm),
                (time.time() - start_time) / 60.0))
            say("\tTrain Accuracy: %f (%d/%d)\n" % (crr / ttl, crr, ttl))
            say("\tp_norm: {}\n".format(self.get_pnorm_stat()))

            say("\n")
            say("{}".format(result_table))
            say("\n")
