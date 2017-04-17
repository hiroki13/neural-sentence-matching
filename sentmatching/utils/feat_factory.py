from io_utils import UNDER_BAR


def get_msr_sem_feats(corpus):
    feats1 = []
    feats2 = []
    for sample in corpus:
        words1, pos1, preds1, sem_roles1 = _separate_elems(sample[1].split())
        words2, pos2, preds2, sem_roles2 = _separate_elems(sample[2].split())
        feats1.append((words1, pos1, preds1, sem_roles1))
        feats2.append((words2, pos2, preds2, sem_roles2))
    return feats1, feats2


def _separate_elems(sent):
    """
    :param sent: word_POS_PRED_SR_SR_..._SR word_...
    :return:
    """
    words = []
    pos = []
    preds = []
    sem_roles = []
    for word in sent:
        elems = word.split(UNDER_BAR)
        words.append(elems[0])
        pos.append(elems[1])
        preds.append(elems[2])
        if len(elems) > 3:
            sem_roles.append(elems[3:])
        else:
            sem_roles.append([])
    assert len(words) == len(pos) == len(preds)
    return words, pos, preds, sem_roles


def map_corpus(raw_corpus, embedding_layer, filter_oov, max_len=100):
    ids_corpus = {}
    for q_id, pair in raw_corpus.iteritems():
        item = (embedding_layer.map_to_ids(pair[0], filter_oov=filter_oov),
                embedding_layer.map_to_ids(pair[1], filter_oov=filter_oov)[:max_len])
        ids_corpus[q_id] = item
    return ids_corpus


def map_msr_corpus(corpus, embedding_layer, filter_oov):
    ids_corpus = []
    for sample in corpus:
        sent1 = embedding_layer.map_to_ids(sample[1], filter_oov=filter_oov)
        sent2 = embedding_layer.map_to_ids(sample[2], filter_oov=filter_oov)
        ids_corpus.append([sample[0], sent1, sent2])
    return ids_corpus


def map_word_feat_to_id(feats, emb_layer, filter_oov):
    ids = []
    for feat1, feat2 in zip(feats[0], feats[1]):
        words1 = emb_layer.map_to_ids(feat1[0], filter_oov=filter_oov)
        words2 = emb_layer.map_to_ids(feat2[0], filter_oov=filter_oov)
        ids.append([words1, words2])
    return ids


def map_sem_feat_to_id(feats, emb_layer, filter_oov):
    ids = []
    for feat1, feat2 in zip(feats[0], feats[1]):
        sem1 = [emb_layer.map_to_ids(sem, filter_oov=filter_oov) for sem in feat1[-1]]
        sem2 = [emb_layer.map_to_ids(sem, filter_oov=filter_oov) for sem in feat2[-1]]
        ids.append([sem1, sem2])
    return ids


def map_label_to_id(corpus):
    return [sample[0] for sample in corpus]


def get_sem_role_list(feats):
    """
    :param feats: (feats1, feats2)
    :return: 1D: [sem_role, ...]
    """
    sem_roles = []
    for feats1, feats2 in zip(feats[0], feats[1]):
        for sent_sr1, sent_sr2 in zip(feats1[-1], feats2[-1]):
            for word_sr in sent_sr1:
                sr = word_sr.split(UNDER_BAR)
                sem_roles.extend(sr)
            for word_sr in sent_sr2:
                sr = word_sr.split(UNDER_BAR)
                sem_roles.extend(sr)
    return sem_roles


