import gzip
import time
import math
import gc
import cPickle as pickle

from prettytable import PrettyTable
import numpy as np
import theano
import theano.tensor as T

import basic_model
from ..nn.advanced import AttentionLayer
from ..nn.optimization import create_optimization_updates
from ..nn.basic import apply_dropout
from ..utils.io_utils import say, read_annotations, create_batches
from ..utils.eval import Evaluation


class Model(basic_model.Model):

    def __init__(self, args, emb_layer):
        super(Model, self).__init__(args, emb_layer)

    def compile(self):
        self.set_input_format()
        self.set_layers(args=self.args, n_d=self.n_d, n_e=self.n_e)
        if not self.args.fix:
            self.set_params(layers=self.layers)

        xt = self.set_input_layer(ids=self.idts)
        ht = self.set_mid_layer(prev_h=xt)
        h_a = self.set_attention_layer(idts=self.idts, idps=self.idps, ht=ht)
        h_o = self.set_output_layer(h=ht[-1])

        self.set_loss(ids=self.idps, h_o=h_o, h_a=h_a)
        self.set_cost(args=self.args, params=self.params, loss=self.loss)
        self.set_scores(h_o=h_o, h_a=h_a)

    def set_mid_layer(self, prev_h):
        args = self.args

        for i in range(args.depth):
            # 1D: n_words, 2D: n_queries * n_cands, 3D: n_d
            ht = self.layers[i].forward_all(prev_h)
            prev_h = ht

        if args.normalize:
            ht = self.normalize_3d(ht)

        # 1D: n_words, 2D: n_queries * n_cands, 3D: n_d
        return ht

    def set_attention_layer(self, idts, idps, ht):
        args = self.args

        attention_layer = AttentionLayer(a_type=args.a_type, n_d=self.n_d, activation=self.activation)
#        self.layers.append(attention_layer)
        self.params += attention_layer.params

        if args.a_type == 'c':
            attention = self.attention_c
        else:
            attention = self.attention

        # 1D: n_queries, 2D: n_cands-1, 3D: dim_h
        h_a = attention(layer=attention_layer, h=ht, idps=idps, n_d=self.n_d, ids=idts)
#        h_a, self.alpha = attention(layer=attention_layer, h=ht, idps=idps, n_d=self.n_d, ids=idts)
#        h_a = apply_dropout(h_a, self.dropout)
        return self.normalize_3d(h_a)

    def set_output_layer(self, h):
        # 1D: n_queries * n_cands, 2D: dim_h
        h_o = apply_dropout(h, self.dropout)
        return self.normalize_2d(h_o)

    def set_loss(self, ids, h_o, h_a):
        # 1D: n_queries, 2D: n_cands-1, 3D: dim_h
        xp = h_o[ids.ravel()]
        xp = xp.reshape((ids.shape[0], ids.shape[1], -1))

        query_vecs = xp[:, 0, :]
#        cand_vecs = (xp[:, 1:, :] + h_a) * 0.5
        cand_vecs = xp[:, 1:, :] + h_a

        scores = T.sum(query_vecs.dimshuffle((0, 'x', 1)) * cand_vecs, axis=2)
        pos_scores = scores[:, 0]
        neg_scores = scores[:, 1:]
        neg_scores = T.max(neg_scores, axis=1)

        diff = neg_scores - pos_scores + 1.0
        self.loss = T.mean((diff > 0) * diff)
        self.train_scores = T.argmax(scores, axis=1)

    def set_scores(self, h_o, h_a):
        # h_o: 1D: n_cands, 2D: dim_h
        h_a = h_a.reshape((h_a.shape[0] * h_a.shape[1], h_a.shape[2]))
#        cand_vecs = (h_o[1:] + h_a) * 0.5
        cand_vecs = h_o[1:] + h_a
        self.scores = T.dot(cand_vecs, h_o[0])

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
#        return layer.forward(query, C)

    def attention_c(self, layer, h, idps, n_d, ids):
        """
        :param layer:
        :param h: 1D: n_words, 2D: n_queries * n_cands, 3D: dim_h
        :param idps: 1D: n_queries, 2D: n_cands (zero-padded)
        :param n_d: float32
        :param ids: 1D: n_words, 2D: n_queries * n_cands
        :return: 1D: 1D: n_queries, 2D: n_cands, 3D: dim_h
        """

        #####################
        # Contexts and Mask #
        #####################
        # 1D: n_words, 1D: n_queries * n_cands, 3D: dim_h
        c = h[:, idps.ravel()]
        # C, ids: 1D: n_queries, 2D: n_cands-1, 3D: n_words, 4D: n_d
        C = c.reshape((ids.shape[0], idps.shape[0], idps.shape[1], n_d)).dimshuffle((1, 2, 0, 3))[:, 1:]

        #################
        # Query vectors #
        #################
        # 1D: n_queries * n_cands, 2D: n_d
        q = h[-1, idps.ravel()]
        # 1D: n_queries, 2D: n_d
        query = q.reshape((idps.shape[0], idps.shape[1], n_d))[:, 0]

        ########
        # Mask #
        ########
        # 1D: n_words, 2D: n_queries * n_cands
        ids = ids[:, idps.ravel()]
        # 1D: n_queries, 3D: n_cands, 3D: n_words
        ids = ids.reshape((ids.shape[0], idps.shape[0], idps.shape[1])).dimshuffle(1, 2, 0)[:, 1:]
        mask = T.neq(ids, self.padding_id)

        return layer.forward(query, C, mask)

    def evaluate(self, data, eval_func):
        res = []
        for idts, idbs, labels in data:
            # idts, idbs: 1D: n_words, 2D: n_cands
            if self.args.attention:
                idps = np.asarray([[i for i in xrange(idts.shape[1])]], dtype='int32')
                scores = eval_func(idts, idbs, idps)
            else:
                scores = eval_func(idts, idbs)

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
#            outputs=[self.cost, self.loss, gnorm, self.train_scores, self.alpha],
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
                if epoch == 0:
                    break

                # get current batch
                idts, idbs, idps = train_batches[i]

                cur_cost, cur_loss, grad_norm, preds = train_func(idts, idbs, idps)
#                cur_cost, cur_loss, grad_norm, preds, alpha = train_func(idts, idbs, idps)

#                if i == 0:
#                    print alpha

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

            gc.collect()

    def load_pretrained_parameters(self, args):
        with gzip.open(args.load_pretrain) as fin:
            data = pickle.load(fin)
            assert args.hidden_dim == data["d"]
            for i, p in enumerate(data["params"]):
                l = self.layers[i]
                l.params = p

