import numpy as np

class EvalATESP():

    @classmethod
    def match_ts(self, gold_ts_sequence, pred_ts_sequence):
        """
        calculate the number of correctly predicted targeted sentiment [(0, 1, 'NEG')]
        :param gold_ts_sequence: gold standard targeted sentiment sequence
        :param pred_ts_sequence: predicted targeted sentiment sequence
        :return:
        """

        # positive, negative and neutral
        tag2tagid = {'POS': 0, 'NEG': 1, 'NEU': 2}
        hit_count, gold_count, pred_count = np.zeros(3), np.zeros(3), np.zeros(3)
        # 分别代表击中POS，NEG，NEU的个数、总个数、预测个数

        for t in gold_ts_sequence:
            #print(t)
            ts_tag = t[2]
            tid = tag2tagid[ts_tag]
            gold_count[tid] += 1

        for t in pred_ts_sequence:
            ts_tag = t[2]
            tid = tag2tagid[ts_tag]
            if t in gold_ts_sequence:
                hit_count[tid] += 1
            pred_count[tid] += 1
        return hit_count, gold_count, pred_count


    @classmethod
    def tag2ts(self, ts_tag_sequence):
        """
        transform ts tag sequence to targeted sentiment
        :param ts_tag_sequence: tag sequence for ts task
        :return:
        """
        n_tags = len(ts_tag_sequence)
        ts_sequence, sentiments = [], []
        beg, end = -1, -1
        for i in range(n_tags):
            ts_tag = ts_tag_sequence[i]
            # current position and sentiment
            eles = ts_tag.split('-')
            if len(eles) == 2:
                pos, sentiment = eles
            else:
                pos, sentiment = 'O', 'O'
            if sentiment != 'O':
                # current word is a subjective word
                sentiments.append(sentiment)
            if pos == 'S':
                # singleton
                ts_sequence.append((i, i, sentiments[0]))
                sentiments = []
                beg, end = -1, -1
            elif pos == 'B':
                beg = i
            elif pos == 'E':
                end = i
                # schema1: only the consistent sentiment tags are accepted
                # that is, all of the sentiment tags are the same
                if end > beg > -1 and len(set(sentiments)) == 1:
                    ts_sequence.append((beg, end, sentiment))
                    sentiments = []
                    beg, end = -1, -1
        return ts_sequence

    @classmethod
    def evaluateBatchOte(self, batchTokensTrueLabel, batchTokensPredLabel):
        return self.evaluate_ts(batchTokensTrueLabel, batchTokensPredLabel)

    @classmethod
    def evaluate_ts(self, gold_ts, pred_ts):
        """
        evaluate the model performance for the ts task
        :param gold_ts: gold standard ts tags, [['B-POS', 'E-POS'], ['S']]
        :param pred_ts: predicted ts tags [['O', 'O'], ['S']]
        :return:
        """
        length = [len(tags) for tags in gold_ts]
        assert len(gold_ts) == len(pred_ts)
        n_samples = len(gold_ts)
        # number of true postive, gold standard, predicted targeted sentiment
        n_tp_ts, n_gold_ts, n_pred_ts = np.zeros(3), np.zeros(3), np.zeros(3)
        ts_precision, ts_recall, ts_f1 = np.zeros(3), np.zeros(3), np.zeros(3)

        for i in range(n_samples):
            g_ts = gold_ts[i]
            p_ts = pred_ts[i]
            length_i = length[i]
            g_ts = g_ts[:length_i]
            p_ts = p_ts[:length_i]
            g_ts_sequence, p_ts_sequence = self.tag2ts(ts_tag_sequence=g_ts), self.tag2ts(ts_tag_sequence=p_ts)
            hit_ts_count, gold_ts_count, pred_ts_count = self.match_ts(gold_ts_sequence=g_ts_sequence, pred_ts_sequence=p_ts_sequence)
            n_tp_ts += hit_ts_count
            n_gold_ts += gold_ts_count
            n_pred_ts += pred_ts_count
            # calculate macro-average scores for ts task

        for i in range(3):
            n_ts = n_tp_ts[i]
            n_g_ts = n_gold_ts[i]
            n_p_ts = n_pred_ts[i]
            ts_precision[i] = float(n_ts) / float(n_p_ts + 1e-8)
            ts_recall[i] = float(n_ts) / float(n_g_ts + 1e-8)
            ts_f1[i] = 2 * ts_precision[i] * ts_recall[i] / (ts_precision[i] + ts_recall[i] + 1e-8)

        ts_macro_f1 = ts_f1.mean()

        # calculate micro-average scores for ts task
        n_tp_total = sum(n_tp_ts)
        # total sum of TP and FN
        n_g_total = sum(n_gold_ts)
        # total sum of TP and FP
        n_p_total = sum(n_pred_ts)

        ts_micro_p = float(n_tp_total) / (n_p_total + 1e-8)
        ts_micro_r = float(n_tp_total) / (n_g_total + 1e-8)
        ts_micro_f1 = 2 * ts_micro_p * ts_micro_r / (ts_micro_p + ts_micro_r + 1e-8)
        ts_scores = (ts_macro_f1, ts_micro_p, ts_micro_r, ts_micro_f1)
        ts_macro_precision = ts_precision.mean()
        ts_macro_recall = ts_recall.mean()

        """
        ot_precision = float(n_tp_ts) / float(n_pred_ts + 1e-5)
        ot_recall = float(n_tp_ts) / float(n_gold_ts + 1e-5)
        ot_f1 = 2 * ot_precision * ot_recall / (ot_precision + ot_recall + 1e-5)
        """

        # 由于有3类情绪，使用微f1值
        
        evalResultDict = {
            'TPExtractTargetsNum': n_tp_ts,
            'PredExtractTargetsNum': n_pred_ts,
            'TrueExtractTargetsNum': n_gold_ts,
            'MacroF1': round(ts_macro_f1, 5),
            'MicroF1': round(ts_micro_f1, 5),
            'Precision': round(ts_micro_p, 5),
            'Recall': round(ts_micro_r, 5)
        }
        return evalResultDict

    @classmethod
    def strEvalResultDict(cls, evalResultDict):
        f1 = evalResultDict['MicroF1']
        pre = evalResultDict['Precision']
        recall = evalResultDict['Recall']
        tn = evalResultDict['TPExtractTargetsNum']
        an = evalResultDict['PredExtractTargetsNum']
        en = evalResultDict['TrueExtractTargetsNum']
        # 使用微平均
        

        return f"F1:{f1} Pre:{pre} Recall:{recall} - {tn}/{an}/{en} = trueNum/allNum/extractNum"

