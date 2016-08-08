from model import basic_model


class Model(basic_model.Model):

    def __init__(self, args, emb_layer):
        super(Model, self).__init__(args, emb_layer)

    def set_intermediate_layer(self, args, prev_ht, prev_hb, layers):
        prev_ht = prev_ht.dimshuffle((1, 0, 2))
        prev_hb = prev_hb.dimshuffle((1, 0, 2))

        for i in range(args.depth):
            # len*batch*n_d
            ht = layers[i].forward(prev_ht)
            hb = layers[i].forward(prev_hb)
            prev_ht = ht
            prev_hb = hb

        # normalize vectors
        if args.normalize:
            ht = self.normalize_2d(ht)
            hb = self.normalize_2d(hb)

        self.ht = ht
        self.hb = hb
