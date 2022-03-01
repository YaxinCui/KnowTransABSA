import numpy as np

class EvalATE():
    @classmethod
    def match_ot(self, gold_ote_sequence, pred_ote_sequence):
        """
        calculate the number of correctly predicted opinion target
        :param gold_ote_sequence: gold standard opinion target sequence
        :param pred_ote_sequence: predicted opinion target sequence
        :return: matched number
        """
        n_hit = 0
        for t in pred_ote_sequence:
            if t in gold_ote_sequence:
                n_hit += 1
        return n_hit

    @classmethod
    def tag2ot(self, ote_tag_sequence):
        """
        transform ote tag sequence to a sequence of opinion target
        :param ote_tag_sequence: tag sequence for ote task
        :return:
        """
        n_tags = len(ote_tag_sequence)
        ot_sequence = []
        beg, end = -1, -1
        for i in range(n_tags):
            tag = ote_tag_sequence[i]
            if tag == 'S':
                ot_sequence.append((i, i))
                beg, end = -1, -1
            elif tag == 'B':
                beg = i
            elif tag == 'E':
                end = i
                if end > beg > -1:
                    ot_sequence.append((beg, end))
                    beg, end = -1, -1
            elif tag == 'O':
                beg, end = -1, -1
                
        return ot_sequence

    @classmethod
    def evaluateBatchOte(self, batchTokensTrueLabel, batchTokensPredLabel):
        return self.evaluate_ote(batchTokensTrueLabel, batchTokensPredLabel)
    
    @classmethod
    def evaluate_ote(self, gold_ot, pred_ot):
        """
        evaluate the model performce for the ote task
        :param gold_ot: gold standard ote tags
        :param pred_ot: predicted ote tags
        :return:
        """
        assert len(gold_ot) == len(pred_ot)
        length = [len(tags) for tags in gold_ot]
        n_samples = len(gold_ot)
        # number of true positive, gold standard, predicted opinion targets
        n_tp_ot, n_gold_ot, n_pred_ot = 0, 0, 0
        for i in range(n_samples):
            g_ot = gold_ot[i]
            p_ot = pred_ot[i]
            length_i = length[i]

            g_ot = g_ot[:length_i]
            p_ot = p_ot[:length_i]

            g_ot_sequence, p_ot_sequence = self.tag2ot(ote_tag_sequence=g_ot), self.tag2ot(ote_tag_sequence=p_ot)

            n_hit_ot = self.match_ot(gold_ote_sequence=g_ot_sequence, pred_ote_sequence=p_ot_sequence)
            n_tp_ot += n_hit_ot
            n_gold_ot += len(g_ot_sequence)
            n_pred_ot += len(p_ot_sequence)
        
        ot_precision = float(n_tp_ot) / float(n_pred_ot + 1e-8)
        ot_recall = float(n_tp_ot) / float(n_gold_ot + 1e-8)
        ot_f1 = 2 * ot_precision * ot_recall / (ot_precision + ot_recall + 1e-8)
        # 这里计算的是宏f1值
        
        evalResultDict = {
            'TPExtractTargetsNum': n_tp_ot,
            'PredExtractTargetsNum': n_pred_ot,
            'TrueExtractTargetsNum': n_gold_ot,
            'MacroF1': round(ot_f1, 5),
            'Precision': round(ot_precision, 5),
            'Recall': round(ot_recall, 5)
        }
        return evalResultDict
        
    @classmethod
    def strEvalResultDict(cls, evalResultDict):
        f1 = evalResultDict['MacroF1']
        pre = evalResultDict['Precision']
        recall = evalResultDict['Recall']
        tn = evalResultDict['TPExtractTargetsNum']
        an = evalResultDict['PredExtractTargetsNum']
        en = evalResultDict['TrueExtractTargetsNum']

        return f"F1:{f1} Pre:{pre} Recall:{recall} - {tn}/{an}/{en} = trueNum/allNum/extractNum"