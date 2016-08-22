import time
import math
import gzip
import cPickle as pickle

from prettytable import PrettyTable
import numpy as np
import theano
import theano.tensor as T

from nn.initialization import get_activation_by_name
from nn.optimization import create_optimization_updates
from nn.basic import LSTM, GRU, CNN, apply_dropout
from nn.advanced import RCNN, GRNN, StrCNN
from utils import io_utils
from utils.io_utils import say
from utils.eval import Evaluation

PAD = "<padding>"


class Model(object):

    def __init__(self, args, emb_layer):
        self.args = args

        ##########
        # Layers #
        ##########
        self.emb_layer = emb_layer
        self.layers = []
        self.params = []

        ###################
        # Network options #
        ###################
        self.activation = args.activation
        self.n_d = args.hidden_dim
        self.n_e = emb_layer.n_d
        self.padding_id = emb_layer.vocab_map[PAD]
        self.dropout = theano.shared(np.float32(args.dropout).astype(theano.config.floatX))

        #########################
        # Input variable format #
        #########################
        self.idts = None
        self.idbs = None
        self.idps = None

        ###########################################################################################
        # Example idps: C0=Query_id, C1=Positive q_id, C2-20=Negative q_ids                       #
        # [[  0   1   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  20  20]  #
        #  [  0   2   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  20  20]  #
        #  [  0   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  20  20]  #
        #  ...                                                                                    #
        #  [ 42  43  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  62]  #
        #  [ 42  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  62]  #
        #  ...                                                                                    #
        #  [105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125]] #
        ###########################################################################################

        #########################
        # Hidden representation #
        #########################
        self.xt = None
        self.xb = None
        self.ht = None
        self.hb = None
        self.h_final = None

        ######################
        # Training objective #
        ######################
        self.loss = None
        self.cost = None

        ##################
        # Testing scores #
        ##################
        self.scores = None
        self.alpha = None

    def compile(self):
        self.set_input_format()
        self.set_layers(args=self.args, n_d=self.n_d, n_e=self.n_e)

        self.set_input_layer(idts=self.idts, idbs=self.idbs, embedding_layer=self.emb_layer,
                             n_e=self.n_e, dropout=self.dropout)
        self.set_intermediate_layer(args=self.args, prev_ht=self.xt, prev_hb=self.xb, layers=self.layers)
        self.set_output_layer(args=self.args, ht=self.ht, hb=self.hb, dropout=self.dropout)

        self.set_params(layers=self.layers)
        self.set_loss(n_d=self.n_d, idps=self.idps, h_final=self.h_final)
        self.set_cost(args=self.args, params=self.params, loss=self.loss)

        self.set_scores(h_final=self.h_final)

    def set_input_format(self):
        # len(title) * batch
        self.idts = T.imatrix()

        # len(body) * batch
        self.idbs = T.imatrix()

        # num queries * candidate size
        self.idps = T.imatrix()

    def set_layers(self, args, n_d, n_e):
        activation = get_activation_by_name(args.activation)

        ##################
        # Set layer type #
        ##################
        if args.layer.lower() == "rcnn":
            layer_type = RCNN
        elif args.layer.lower() == "lstm":
            layer_type = LSTM
        elif args.layer.lower() == "gru":
            layer_type = GRU
        elif args.layer.lower() == "grnn":
            layer_type = GRNN
        elif args.layer.lower() == "cnn":
            layer_type = CNN
        elif args.layer.lower() == "str_cnn":
            layer_type = StrCNN

        ##############
        # Set layers #
        ##############
        for i in range(args.depth):
            if layer_type != RCNN:
                feature_layer = layer_type(
                    n_in=n_e if i == 0 else n_d,
                    n_out=n_d,
                    activation=activation
                )
            else:
                feature_layer = layer_type(
                    n_in=n_e if i == 0 else n_d,
                    n_out=n_d,
                    activation=activation,
                    order=args.order,
                    mode=args.mode,
                    has_outgate=args.outgate
                )
            self.layers.append(feature_layer)

    def set_input_layer(self, idts, idbs, embedding_layer, n_e, dropout):
        # (len*batch)*n_e
        xt = embedding_layer.forward(idts.ravel())
        xb = embedding_layer.forward(idbs.ravel())

        # len*batch*n_e
        xt = xt.reshape((idts.shape[0], idts.shape[1], n_e))
        xb = xb.reshape((idbs.shape[0], idbs.shape[1], n_e))
        self.xt = apply_dropout(xt, dropout)
        self.xb = apply_dropout(xb, dropout)

    def set_intermediate_layer(self, args, prev_ht, prev_hb, layers):
        for i in range(args.depth):
            # 1D: n_words, 2D: batch, 3D: n_d
            ht = layers[i].forward_all(prev_ht)
            hb = layers[i].forward_all(prev_hb)
            prev_ht = ht
            prev_hb = hb

        # normalize vectors
        if args.normalize:
            ht = self.normalize_3d(ht)
            hb = self.normalize_3d(hb)

        # average over length, ignore paddings
        # 1D: batch, 2D: n_d
        if self.args.average:
            ht = self.average_without_padding(ht, self.idts)
            hb = self.average_without_padding(hb, self.idbs)
        elif self.args.layer == 'cnn' or self.args.layer == 'str_cnn':
            ht = self.max_without_padding(ht, self.idts)
            hb = self.max_without_padding(hb, self.idbs)
        else:
            ht = ht[-1]
            hb = hb[-1]

        self.ht = ht
        self.hb = hb

    def set_output_layer(self, args, ht, hb, dropout):
        # 1D: n_queries * n_cands, 2D: dim_h
        if args.body:
            h_final = (ht + hb) * 0.5
        else:
            h_final = ht

        h_final = apply_dropout(h_final, dropout)
        self.h_final = self.normalize_2d(h_final)

    def set_params(self, layers):
        for l in layers:
            self.params += l.params
        say("num of parameters: {}\n".format(sum(len(x.get_value(borrow=True).ravel()) for x in self.params)))

    def set_loss(self, n_d, idps, h_final):
        # 1D: n_queries, 2D: n_cands-1, 3D: dim_h
        xp = h_final[idps.ravel()]
        xp = xp.reshape((idps.shape[0], idps.shape[1], n_d))

        if self.args.loss == 'ce':
            self.cross_entropy(xp)
        elif self.args.loss == 'sbs':
            self.soft_bootstrapping(xp, self.args.beta)
        elif self.args.loss == 'hbs':
            self.hard_bootstrapping(xp, self.args.beta)
        else:
            self.max_margin(xp)

    def cross_entropy(self, xp):
        # num query * n_d
        query_vecs = xp[:, 0, :]  # 3D -> 2D
        # 1D: n_queries, 2D: n_cands
        scores = T.sum(query_vecs.dimshuffle((0, 'x', 1)) * xp[:, 1:, :], axis=2)
        probs = T.nnet.softmax(scores)

        self.train_scores = T.argmax(scores, axis=1)
        self.loss = - T.mean(T.log(probs[:, 0]))

    def soft_bootstrapping(self, xp, beta=0.9):
        # num query * n_d
        query_vecs = xp[:, 0, :]  # 3D -> 2D
        # 1D: n_queries, 2D: n_cands
        scores = T.sum(query_vecs.dimshuffle((0, 'x', 1)) * xp[:, 1:, :], axis=2)
        probs = T.nnet.softmax(scores)
        zeros = T.zeros(shape=(probs.shape[0], probs.shape[1]-1), dtype=theano.config.floatX)
        ones = T.ones(shape=(probs.shape[0], 1), dtype=theano.config.floatX)
        target = T.concatenate([ones, zeros], axis=1)

        self.train_scores = T.argmax(scores, axis=1)
        self.loss = - T.mean((beta * target + (1. - beta) * probs) * T.log(probs))

    def hard_bootstrapping(self, xp, beta=0.9):
        # num query * n_d
        query_vecs = xp[:, 0, :]  # 3D -> 2D
        # 1D: n_queries, 2D: n_cands
        scores = T.sum(query_vecs.dimshuffle((0, 'x', 1)) * xp[:, 1:, :], axis=2)
        probs = T.nnet.softmax(scores)
        z = T.argmax(probs, axis=1)

        pos_probs = probs[:, 0]
        max_probs = probs[T.arange(z.shape[0]), z]
        pos_loss = (beta + (1. - beta) * pos_probs) * T.log(pos_probs)
        max_loss = (beta + (1. - beta) * max_probs) * T.log(max_probs)

        self.train_scores = T.argmax(scores, axis=1)
        self.loss = - T.mean(pos_loss + max_loss)

    def max_margin(self, xp):
        # 1D: n_queries, 2D: n_d
        query_vecs = xp[:, 0, :]  # 3D -> 2D

        # 1D: n_queries, 2D: n_cands
        scores = T.sum(query_vecs.dimshuffle((0, 'x', 1)) * xp[:, 1:, :], axis=2)
        pos_scores = scores[:, 0]
        neg_scores = scores[:, 1:]
        neg_scores = T.max(neg_scores, axis=1)

        diff = neg_scores - pos_scores + 1.0
        self.loss = T.mean((diff > 0) * diff)
        self.train_scores = T.argmax(scores, axis=1)

    def set_cost(self, args, params, loss):
        l2_reg = None
        for p in params:
            if l2_reg is None:
                l2_reg = p.norm(2)
            else:
                l2_reg += p.norm(2)
        self.cost = loss + l2_reg * args.l2_reg

    def set_scores(self, h_final):
        # first one in batch is query, the rest are candidate questions
        self.scores = T.dot(h_final[1:], h_final[0])

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
#            outputs=[self.cost, self.loss, gnorm, self.train_scores, self.alpha],
            updates=updates,
            on_unused_input='ignore'
        )

        if self.args.attention:
            eval_func = theano.function(
                inputs=[self.idts, self.idbs, self.idps],
                outputs=self.scores,
                on_unused_input='ignore'
            )
        else:
            eval_func = theano.function(
                inputs=[self.idts, self.idbs],
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

            train = io_utils.read_annotations(args.train, data_size=args.data_size)
            train_batches = io_utils.create_batches(ids_corpus, train, batch_size, padding_id, pad_left=not args.average)
            n_train_batches = len(train_batches)

            train_loss = 0.0
            train_cost = 0.0
            crr = 0.
            ttl = 0.

            for i in xrange(n_train_batches):
                # get current batch
                idts, idbs, idps = train_batches[i]

                cur_cost, cur_loss, grad_norm, preds = train_func(idts, idbs, idps)
#                cur_cost, cur_loss, grad_norm, preds, alpha = train_func(idts, idbs, idps)

                if i == 100000000000:
                    say('NAN: Index %d\n' % i)
                    say('grad_norm\n%s\n' % str(grad_norm))
                    say("\tp_norm: {}\n".format(self.get_pnorm_stat()))
                    say('idts\n%s\n' % str(idts))
                    say('idps\n%s\n' % str(idps))
#                    say('alpha\n%s\n' % str(alpha))
                    exit()

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

    def get_pnorm_stat(self):
        lst_norms = []
        for p in self.params:
            vals = p.get_value(borrow=True)
            l2 = np.linalg.norm(vals)
            lst_norms.append("{:.3f}".format(l2))
        return lst_norms

    def normalize_2d(self, x, eps=1e-8):
        # x is batch*d
        # l2 is batch*1
        l2 = x.norm(2, axis=1).dimshuffle((0, 'x'))
        return x / (l2 + eps)

    def normalize_3d(self, x, eps=1e-8):
        # x is len*batch*d
        # l2 is len*batch*1
        l2 = x.norm(2, axis=2).dimshuffle((0, 1, 'x'))
        return x / (l2 + eps)

    def average_without_padding(self, x, ids, eps=1e-8):
        """
        :param x: 1D: n_words, 2D: batch, 3D: n_d
        :param ids: 1D: n_words, 2D: batch, 3D: n_d
        :return: 1D: batch, 2D: n_d
        """
        # 1D: n_words, 2D: batch, 3D: 1
        mask = T.neq(ids, self.padding_id).dimshuffle((0, 1, 'x'))
        mask = T.cast(mask, theano.config.floatX)
        mask = mask.dimshuffle((1, 0, 2))
        # 1D: batch, 2D: n_d
        x = x.dimshuffle((1, 0, 2))
        s = T.sum(x * mask + eps, axis=1) / (T.sum(mask + eps, axis=1) + eps)
        return s

    def max_without_padding(self, x, ids, eps=1e-8):
        """
        :param x: 1D: n_words, 2D: batch, 3D: n_d
        :param ids: 1D: n_words, 2D: batch, 3D: n_d
        :return: 1D: batch, 2D: n_d
        """
        # 1D: n_words, 2D: batch, 3D: 1
        mask = T.neq(ids, self.padding_id).dimshuffle((0, 1, 'x'))
        mask = T.cast(mask, theano.config.floatX)
        mask = mask.dimshuffle((1, 0, 2))
        # 1D: batch, 2D: n_d
        x = x.dimshuffle((1, 0, 2))
        s = T.max(x * mask + eps, axis=1)
        return s

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

    def load_pretrained_parameters(self, args):
        with gzip.open(args.load_pretrain) as fin:
            data = pickle.load(fin)
            assert args.hidden_dim == data["d"]
            # assert args.layer == data["layer_type"]
            for l, p in zip(self.layers, data["params"]):
                l.params = p

    def save_model(self, path):
        if not path.endswith(".pkl.gz"):
            path += ".gz" if path.endswith(".pkl") else ".pkl.gz"

        args = self.args
        params = [x.params for x in self.layers]
        with gzip.open(path, "w") as fout:
            pickle.dump(
                {
                    "args": args,
                    "d": args.hidden_dim,
                    "params": params,
                },
                fout,
                protocol=pickle.HIGHEST_PROTOCOL
            )

