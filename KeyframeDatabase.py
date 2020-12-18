import numpy as np


class KeyframeDatabase:
    def __init__(self):
        self.inverted_idx = dict()  # Dict containing inverted index of keyframes that contain a word {word_id : {KFs}}

    def insert(self, keyframe):
        for key in keyframe.bow_ind:
            try:
                self.inverted_idx[key].add(keyframe)
            except KeyError:
                self.inverted_idx[key] = {keyframe}

    def get_candidates(self, keyframe):
        # intersect images that contain words from keyframe
        candidates = self.inverted_idx[keyframe.bow_ind[0]]
        for key in keyframe.bow_ind:
            candidates = candidates & self.inverted_idx[key]
            if len(candidates) < 10:
                break
        candidates.discard(keyframe)
        for kf in keyframe.neighbors:
            candidates.discard(kf[0])
        return candidates

    def get(self):
        return self.inverted_idx

    def keys(self):
        return self.inverted_idx.keys()

    def score_l2(self, bow1, bow2, normalize=False):
        if normalize:
            bow1 = self.normalize(bow1)
            bow2 = self.normalize(bow2)
        key1 = sorted(bow1.keys())
        key2 = sorted(bow2.keys())
        iter1 = 0
        iter2 = 0
        print("lengths: ", len(key1), len(key2))

        score = 0.
        counter = 0
        while iter1 < len(key1) and iter2 < len(key2):
            if key1[iter1] == key2[iter2]:
                score += bow1.__getitem__(key1[iter1]) * bow2.__getitem__(key2[iter2])
                iter1 += 1
                iter2 += 1
                # print(iter1, iter2, "YAAAASSS!")
                counter += 1
            elif key1[iter1] > key2[iter2]:
                while iter2 < len(key2) and key1[iter1] > key2[iter2]:
                    iter2 += 1
            else:
                while iter1 < len(key1) and key1[iter1] < key2[iter2]:
                    iter1 += 1

        print("Initial score: ", score, "Matched bags: ", counter)
        if score >= 1.:  # rounding errors
            score = 1.
        else:
            score = 1. - np.sqrt(1. - score)
        return score

    def score_l1(self, bow1, bow2, normalize=False):
        if normalize:
            bow1 = self.normalize(bow1)
            bow2 = self.normalize(bow2)
        key1 = sorted(bow1.keys())
        key2 = sorted(bow2.keys())
        iter1 = 0
        iter2 = 0

        score = 0.
        counter = 0
        while iter1 < len(key1) and iter2 < len(key2):
            if key1[iter1] == key2[iter2]:
                score += np.abs(bow1.__getitem__(key1[iter1]) - bow2.__getitem__(key2[iter2]))
                iter1 += 1
                iter2 += 1
                # print(iter1, iter2, "YAAAASSS!")
                counter += 1
            elif key1[iter1] > key2[iter2]:
                while iter2 < len(key2) and key1[iter1] > key2[iter2]:
                    score += np.abs(bow2.__getitem__(key2[iter2]))
                    iter2 += 1
            else:
                while iter1 < len(key1) and key1[iter1] < key2[iter2]:
                    score += np.abs(bow1.__getitem__(key1[iter1]))
                    iter1 += 1
        score = 1 - 0.5 * score
        return score

    @classmethod
    def normalize(cls, bow):
        bow_new = {}
        keys = bow.keys()
        norm = 0
        for key in keys:
            norm += bow[key]**2
        for key in keys:
            bow_new[key] = bow[key]/norm
        return bow_new
