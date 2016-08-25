import time

from prettytable import PrettyTable

import numpy as np
import theano
import theano.tensor as T

from ..nn.optimization import create_optimization_updates
from ..nn.initialization import create_shared, random_init
from ..utils.io_utils import say, read_annotations, create_batches
from ..utils.eval import Evaluation
import basic_model


class Model(basic_model.Model):

    def __init__(self, args, emb_layer):
        super(Model, self).__init__(args, emb_layer)

    def compile(self):
        self.set_input_format()
        self.set_layers(args=self.args, n_d=self.n_d, n_e=self.n_e)

        self.set_input_layer(ids=self.idts, idbs=self.idbs, embedding_layer=self.emb_layer,
                             n_e=self.n_e, dropout=self.dropout)
        self.set_mid_layer(args=self.args, prev_h=self.xt, prev_hb=self.xb, layers=self.layers)

        self.set_output_layer(args=self.args, ht=self.ht, hb=self.hb, dropout=self.dropout)

        self.set_scores(h_o=self.h_final, idps=self.idps, n_d=self.n_d)

        self.set_params(layers=self.layers)
        self.set_loss(scores=self.scores)
        self.set_cost(args=self.args, params=self.params, loss=self.loss)

    def set_scores(self, h_o, idps, n_d):
        sim_option = self.args.sim
        if sim_option == 1:
            score_f = self.set_softmax
        elif sim_option == 2:
            score_f = self.set_softmax_conc
        else:
            score_f = self.set_sigmoid

        score_f(h_o, idps, n_d)

    def set_softmax(self, h_final, idps, n_d):
        W = self.initialize_params(n_d, n_d, self.activation)
        self.params += [W]

        # 1D: n_queries, 2D: n_cands, 3D: n_d
        xp = h_final[idps.ravel()]
        xp = xp.reshape((idps.shape[0], idps.shape[1], n_d))

        # 1D: n_queries, 2D: n_d
        query_vecs = xp[:, 0, :]  # 3D -> 2D
        cand_vecs = xp[:, 1:, :]

        self.scores = T.nnet.softmax(T.sum(T.dot(query_vecs, W).dimshuffle((0, 'x', 1)) * cand_vecs, axis=2))

    def set_softmax_conc(self, h_final, idps, n_d):
        W = create_shared(random_init((2 * n_d,)), name="w")
        self.params += [W]

        # 1D: n_queries, 2D: n_cands, 3D: n_d
        xp = h_final[idps.ravel()]
        xp = xp.reshape((idps.shape[0], idps.shape[1], n_d))

        # 1D: n_queries, 2D: n_d
        query_vecs = xp[:, 0, :]  # 3D -> 2D
        cand_vecs = xp[:, 1:, :]

        # 1D: n_queries, 2D: n_cands
        query_vecs = T.repeat(query_vecs.dimshuffle((0, 'x', 1)), cand_vecs.shape[1], axis=1)
        vecs = T.concatenate([query_vecs, cand_vecs], axis=2)

        self.scores = T.nnet.softmax(T.dot(vecs, W))

    def set_sigmoid(self, h_final, idps, n_d):
        W = self.initialize_params(n_d, n_d, self.activation)
        self.params += [W]

        # 1D: n_queries, 2D: n_cands, 3D: n_d
        xp = h_final[idps.ravel()]
        xp = xp.reshape((idps.shape[0], idps.shape[1], n_d))

        # 1D: n_queries, 2D: n_d
        query_vecs = xp[:, 0, :]  # 3D -> 2D
        cand_vecs = xp[:, 1:, :]

        self.scores = T.nnet.sigmoid(T.sum(T.dot(query_vecs, W).dimshuffle((0, 'x', 1)) * cand_vecs, axis=2))

    def set_loss(self, scores):
        self.train_scores = T.argmax(scores, axis=1)

        sim_option = self.args.sim
        if sim_option > 0:
            self.loss = - T.mean(T.log(scores[:, 0]))
        else:
            y_p = scores[:, 0]
            n_p = T.max(scores[:, 1:], axis=1)
            self.loss = - (T.sum(T.log(y_p)) + T.sum(T.log(1. - n_p)))

    def evaluate(self, data, eval_func):
        res = []
        for idts, idbs, labels in data:
            # idts, idbs: 1D: n_words, 2D: n_cands
            idps = np.asarray([[i for i in xrange(idts.shape[1])]], dtype='int32')
            scores = eval_func(idts, idbs, idps)
            scores = scores[0]
            assert len(scores) == len(labels)
            ranks = (-scores).argsort()
            ranked_labels = labels[ranks]
            res.append(ranked_labels)

            """
            print 'Score: %s' % str(scores)
            print 'Labels: %s' % str(labels)
            print 'Ranks: %s' % str(ranks)
            print 'Ranked Labels: %s' % str(ranked_labels)
            print
            """

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

        updates, lr, gnorm = create_optimization_updates(
            cost=self.cost,
            params=self.params,
            lr=args.learning_rate,
            method=args.learning
        )[:3]

        train_func = theano.function(
            inputs=[self.idts, self.idbs, self.idps],
            outputs=[self.cost, self.loss, gnorm, self.train_scores],
            updates=updates,
            on_unused_input='ignore'
        )

        eval_func = theano.function(
            inputs=[self.idts, self.idbs, self.idps],
            outputs=self.scores,
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

                train_loss += cur_loss
                train_cost += cur_cost

                crr += len([s for s in preds if s == 0])
                ttl += len(preds)

                if i % 10 == 0:
                    say("\r{}/{}".format(i, n_train_batches))

                if i == n_train_batches - 1:
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
                    say(("Epoch {}\tcost={:.3f}\tloss={:.3f}" + "\tMRR={:.2f},{:.2f}\t|g|={:.3f}\t[{:.3f}m]\n").format(
                        epoch,
                        train_cost / (i + 1),
                        train_loss / (i + 1),
                        dev_MRR,
                        best_dev,
                        float(grad_norm),
                        (time.time() - start_time) / 60.0
                    ))
                    say("\tTrain Accuracy: %f (%d/%d)\n" % (crr / ttl, crr, ttl))
                    say("\tp_norm: {}\n".format(self.get_pnorm_stat()))
                    say("\n")
                    say("{}".format(result_table))
                    say("\n")


