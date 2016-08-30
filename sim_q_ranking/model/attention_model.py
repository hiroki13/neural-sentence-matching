import time
import math

from prettytable import PrettyTable
import numpy as np
import theano
import theano.tensor as T

import basic_model
from ..nn.advanced import AttentionLayer
from ..nn.optimization import create_optimization_updates
from ..nn.basic import apply_dropout
from ..nn.nn_utils import normalize_3d, average_without_padding
from ..utils.io_utils import say, read_annotations, create_batches
from ..utils.eval import Evaluation


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
        xt = self.set_input_layer(ids=self.idts)
        ht, ht_all = self.set_mid_layer(prev_h=xt, ids=self.idts, padding_id=self.padding_id)
        ht_a = self.attention(h=ht_all, ids=self.idts, idps=self.idps)

        if args.body:
            xb = self.set_input_layer(ids=self.idbs)
            hb, hb_all = self.set_mid_layer(prev_h=xb, ids=self.idbs, padding_id=self.padding_id)
            hb_a = self.attention(h=hb_all, ids=self.idbs, idps=self.idps)

            ht = (ht + hb) * 0.5
            ht_a = (ht_a + hb_a) * 0.5

        h = self.set_output_layer(h=ht)

        #########
        # Score #
        #########
        # The first one in each batch is a query, the rest are candidate questions
        self.predict_scores = self.get_predict_scores(h, ht_a)

        # The first one in each batch is a positive sample, the rest are negative ones
        scores = self.get_scores(h, ht_a, self.idps)
        pos_scores = scores[:, 0]
        neg_scores = scores[:, 1:]
        self.train_scores = T.argmax(scores, axis=1)

        ##########################
        # Set training objective #
        ##########################
        self.set_params(layers=self.layers)
        self.set_loss(scores, pos_scores, neg_scores)
        self.set_cost(args=self.args, params=self.params, loss=self.loss)

    def set_mid_layer(self, prev_h, ids, padding_id):
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

        #####################
        # Contexts and Mask #
        #####################
        # cands, ids: 1D: n_queries * n_cands, 3D: dim_h
        cands = h[-1, idps.ravel()]
#        ids = ids[:, idps.ravel()]

        # 1D: n_queries * n_cands, 2D: n_words
        ids = ids.dimshuffle((1, 0))
        # 1D: n_queries * n_cands, 2D: n_words
        ids = ids[idps.ravel()]

        # C, ids: 1D: n_queries, 2D: n_cands-1, 3D: n_d
        C = cands.reshape((idps.shape[0], idps.shape[1], -1))[:, 1:]

        # 1D: n_queries, 2D: n_cands-1, 3D: n_words
#        ids = ids.reshape((ids.shape[0], idps.shape[0], idps.shape[1])).dimshuffle(1, 2, 0)[:, 0]
        ids = ids.reshape((idps.shape[0], idps.shape[1], -1))[:, 0]
        mask = T.neq(ids, self.padding_id)
        self.mask = mask
        mask = mask.dimshuffle((0, 'x', 1))

        #################
        # Query vectors #
        #################
        # query: 1D: n_queries, 2D: n_words, 3D: n_d
        q = h[:, idps.ravel()]
        query = q.reshape((q.shape[0], idps.shape[0], idps.shape[1], -1)).dimshuffle((1, 2, 0, 3))[:, 0, :, :]

        h_a = self.layers[-1].forward(query, C, mask)
        h_a = apply_dropout(h_a, self.dropout)
        return normalize_3d(h_a)

    def get_scores(self, h, h_a, idps):
        h = h[idps.ravel()]
        h = h.reshape((idps.shape[0], idps.shape[1], -1))

        # 1D: n_queries, 2D: 1, 3D: n_d
        query_vecs = h[:, 0, :].dimshuffle((0, 'x', 1))
        # 1D: n_queries, 2D: n_cands, 3D: n_d
        cand_vecs = (h[:, 1:, :] + h_a) * 0.5
        # 1D: n_queries, 2D: n_cands
        scores = T.sum(query_vecs * cand_vecs, axis=2)
        return scores

    def get_predict_scores(self, h, h_a):
        # h: 1D: n_cands, 2D: dim_h
        h_a = h_a.reshape((h_a.shape[0] * h_a.shape[1], h_a.shape[2]))
        cand_vecs = (h[1:] + h_a) * 0.5
        return T.dot(cand_vecs, h[0])

    def evaluate(self, data, eval_func):
        res = []
        for idts, idbs, labels in data:
            # idts, idbs: 1D: n_words, 2D: n_cands
            idps = np.asarray([[i for i in xrange(idts.shape[1])]], dtype='int32')
            scores = eval_func(idts, idbs, idps)

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

                if i == n_train_batches - 1:
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

            say("\n")
            say("{}".format(result_table))
            say("\n")

    def train2(self, ids_corpus, dev=None, test=None):
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
            outputs=[self.cost, self.loss, gnorm, self.train_scores, self.mask],
            updates=updates,
            on_unused_input='ignore'
        )

        say("\tp_norm: {}\n".format(self.get_pnorm_stat()))

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
                if i < 180:
                    continue

                # get current batch
                idts, idbs, idps = train_batches[i]
                cur_cost, cur_loss, grad_norm, preds, mask = train_func(idts, idbs, idps)

                if math.isnan(cur_loss):
                    say('\n\nNAN: Index: %d\n' % i)
                    """
                    for p in idts.T:
                        print p[-1]
                    print
                    for m in mask:
                        print m
                    print
                    """
                    say('Cost: %f  Loss: %f  G_norm: %f\n' % (cur_cost, cur_loss, float(grad_norm)))
                    exit()

                """
                for p in idts.T:
                    print p[-1]
                print
                for m in mask:
                    print m
                print
                say('Cost: %f  Loss: %f  G_norm: %f\n' % (cur_cost, cur_loss, float(grad_norm)))
                """

                train_loss += cur_loss
                train_cost += cur_cost

                crr += len([s for s in preds if s == 0])
                ttl += len(preds)

                if i % 10 == 0:
                    say("\r{}/{}".format(i, n_train_batches))

                if i == n_train_batches - 1:
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
