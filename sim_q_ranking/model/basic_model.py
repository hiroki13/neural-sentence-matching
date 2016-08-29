import gzip
import time
import math
import cPickle as pickle

from prettytable import PrettyTable
import numpy as np
import theano
import theano.tensor as T

from ..nn.initialization import get_activation_by_name
from ..nn.optimization import create_optimization_updates
from ..nn.basic import LSTM, GRU, CNN, apply_dropout
from ..nn.advanced import RCNN, GRNN, StrCNN
from ..nn.nn_utils import hinge_loss, cross_entropy_loss, normalize_2d, normalize_3d, average_without_padding
from ..utils.io_utils import say, read_annotations, create_batches
from ..utils.eval import Evaluation

PAD = "<padding>"


class Model(object):

    def __init__(self, args, emb_layer):
        self.args = args

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

        ###################
        # Input variables #
        ###################
        # 1D: n_words, 2D: batch * n_cands
        self.idts = T.imatrix()
        self.idbs = T.imatrix()
        # 1D: batch, 2D: n_cands
        self.idps = T.imatrix()

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

        ######################
        # Training objective #
        ######################
        self.loss = None
        self.cost = None

        ##################
        # Testing scores #
        ##################
        self.train_scores = None
        self.predict_scores = None

    def compile(self):
        args = self.args

        self.set_layers(args=self.args, n_d=self.n_d, n_e=self.n_e)

        ###########
        # Network #
        ###########
        xt = self.set_input_layer(ids=self.idts)
        ht = self.set_mid_layer(prev_h=xt, ids=self.idts, padding_id=self.padding_id)

        if args.body:
            xb = self.set_input_layer(ids=self.idbs)
            hb = self.set_mid_layer(prev_h=xb, ids=self.idbs, padding_id=self.padding_id)
            ht = (ht + hb) * 0.5

        h = self.set_output_layer(h=ht)

        #########
        # Score #
        #########
        # The first one in each batch is a query, the rest are candidate questions
        self.predict_scores = self.get_predict_scores(h)

        # The first one in each batch is a positive sample, the rest are negative ones
        scores = self.get_scores(h, self.idps)
        pos_scores = scores[:, 0]
        neg_scores = scores[:, 1:]
        self.train_scores = T.argmax(scores, axis=1)

        ##########################
        # Set training objective #
        ##########################
        self.set_params(layers=self.layers)
        self.set_loss(scores, pos_scores, neg_scores)
        self.set_cost(args=self.args, params=self.params, loss=self.loss)

    def set_layers(self, args, n_d, n_e):
        activation = get_activation_by_name(args.activation)

        ##################
        # Set layer type #
        ##################
        if args.layer.lower() == "lstm":
            layer_type = LSTM
        elif args.layer.lower() == "gru":
            layer_type = GRU
        elif args.layer.lower() == "grnn":
            layer_type = GRNN
        elif args.layer.lower() == "cnn":
            layer_type = CNN
        elif args.layer.lower() == "str_cnn":
            layer_type = StrCNN
        else:
            layer_type = RCNN

        ##############
        # Set layers #
        ##############
        for i in range(args.depth):
            if layer_type == CNN or layer_type == StrCNN:
                feature_layer = layer_type(
                    n_in=n_e if i == 0 else n_d,
                    n_out=n_d,
                    activation=activation,
                    order=args.order
                )
            elif layer_type != RCNN:
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

    def set_input_layer(self, ids):
        # 1D: n_words * n_queries * batch, 2D: n_e
        xt = self.emb_layer.forward(ids.ravel())
        # 1D: n_words, 2D: n_queries * batch, 3D: n_e
        xt = xt.reshape((ids.shape[0], ids.shape[1], -1))
        return apply_dropout(xt, self.dropout)

    def set_mid_layer(self, prev_h, ids, padding_id):
        args = self.args

        for i in range(args.depth):
            # 1D: n_words, 2D: batch, 3D: n_d
            h = self.layers[i].forward_all(prev_h)
            prev_h = h

        if args.normalize:
            h = normalize_3d(h)

        # 1D: batch, 2D: n_d
        if args.average or args.layer == "cnn" or args.layer == "str_cnn":
            h = average_without_padding(h, ids, padding_id)
        else:
            h = h[-1]

        return h

    def set_output_layer(self, h):
        # 1D: n_queries * n_cands, 2D: dim_h
        h = apply_dropout(h, self.dropout)
        return normalize_2d(h)

    def set_params(self, layers):
        for l in layers:
            self.params += l.params
        say("num of parameters: {}\n".format(sum(len(x.get_value(borrow=True).ravel()) for x in self.params)))

    def get_scores(self, h, idps):
        h = h[idps.ravel()]
        h = h.reshape((idps.shape[0], idps.shape[1], -1))

        # 1D: n_queries, 2D: 1, 3D: n_d
        query_vecs = h[:, 0, :].dimshuffle((0, 'x', 1))
        # 1D: n_queries, 2D: n_cands, 3D: n_d
        cand_vecs = h[:, 1:, :]
        # 1D: n_queries, 2D: n_cands
        scores = T.sum(query_vecs * cand_vecs, axis=2)
        return scores

    def get_predict_scores(self, h):
        # first one in batch is query, the rest are candidate questions
        return T.dot(h[1:], h[0])

    def set_loss(self, scores, pos_scores, neg_scores):
        if self.args.loss == 1:
            self.loss = cross_entropy_loss(scores)
        else:
            self.loss = hinge_loss(pos_scores, neg_scores)

    def set_cost(self, args, params, loss):
        l2_reg = None
        for p in params:
            if l2_reg is None:
                l2_reg = p.norm(2)
            else:
                l2_reg += p.norm(2)
        self.cost = loss + l2_reg * args.l2_reg

    def get_pnorm_stat(self):
        lst_norms = []
        for p in self.params:
            vals = p.get_value(borrow=True)
            l2 = np.linalg.norm(vals)
            lst_norms.append("{:.3f}".format(l2))
        return lst_norms

    def load_pretrained_parameters(self, args):
        with gzip.open(args.load_pretrain) as fin:
            data = pickle.load(fin)
            assert args.hidden_dim == data["d"]
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
            updates=updates,
            on_unused_input='ignore'
        )

        eval_func = theano.function(
            inputs=[self.idts, self.idbs],
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
