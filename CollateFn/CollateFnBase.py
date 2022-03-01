from abc import abstractmethod
import time

class CollateFnBase:
    BIEOS = "BIEOS"
    ATElabel2id = {}
    ATEid2label = {}
    for id, label in enumerate(BIEOS):
        ATElabel2id[label] = id
        ATEid2label[id] = label

    ATESPlabel2id = {}
    ATESPid2label = {}
    for id, label in enumerate("BIES"):
        for sen_index, sen in enumerate(['-POS', '-NEG', '-NEU']):
            ATESPlabel2id[label+sen] = id * 3 + sen_index
            ATESPid2label[id * 3 + sen_index] = label+sen
    ATESPlabel2id['O'] = 12
    ATESPid2label[12] = 'O'

    @classmethod
    def getRawData(cls, batchDataSet):
        # 记录下原始数据
        batchDataSet =list(zip(*batchDataSet))
        # 原始数据顺序为 text, UniOpinions, phrase, language, domain, reviewId, sentenceId
        batchRawData = {}
        rawBatchDataKeys = ["batchText", "batchOpinions", "batchPhrase", "batchLanguage", 
                                            "batchDomain", "batchReviewId", "batchSentenceId"]
        for i, key in enumerate(rawBatchDataKeys):
            batchRawData[key] = batchDataSet[i]
        batchRawData['batchText'] = list(batchRawData['batchText'])
        return batchRawData

    @abstractmethod
    def collate_fn(self, batchDataSet):
        pass


class CollateFnTransformer(CollateFnBase):
    
    @classmethod
    def collate_fn(self):
        pass

    @classmethod
    def getBatchBIEOSLabels(self, batchTokensOffset, batchOpinons, taskMode="ATESP"):
        # offset是前闭后开
        batchLabels = []
        batchOpinionsOffsets = []
        for (offsets, opionions) in (zip(batchTokensOffset, batchOpinons)):
            labels = []
            opinionsOffsets = []
            message = []
            opionions = list(opionions)
            # 从offset列表从后往前推理对应的标签
            while 0 < len(opionions):
                while opionions[-1].end <= offsets[-1][0]:
                    labels.append('O')
                    message.append(str(offsets[-1]) + " is " + labels[-1])
                    offsets.pop()
                if offsets[-1][0] <= opionions[-1].begin:
                    labels.append("S")
                    if taskMode=="ATESP":
                        labels[-1] = labels[-1] + '-' + (opionions[-1].polarity)[:3].upper()
                    message.append(str(offsets[-1]) + " is " + labels[-1])
                    opinionsOffsets.append(offsets[-1])
                    offsets.pop()
                    opionions.pop()
                else:
                    # 说明是B I E
                    labels.append("E")
                    if taskMode=="ATESP":
                        labels[-1] = labels[-1] + '-' + (opionions[-1].polarity)[:3].upper()

                    message.append(str(offsets[-1]) + " is " + labels[-1])
                    opinionOffsetEnd = offsets[-1][1]
                    offsets.pop()
                    while opionions[-1].begin < offsets[-1][0]:
                        labels.append("I")
                        if taskMode=="ATESP":
                            labels[-1] = labels[-1] + '-' + (opionions[-1].polarity)[:3].upper()

                        message.append(str(offsets[-1]) + " is " + labels[-1])
                        offsets.pop()
                    labels.append("B")
                    if taskMode=="ATESP":
                        labels[-1] = labels[-1] + '-' + (opionions[-1].polarity)[:3].upper()
                    message.append(str(offsets[-1]) + " is " + labels[-1])
                    opinionOffsetBegin = offsets[-1][0]
                    opinionsOffsets.append((opinionOffsetBegin, opinionOffsetEnd))
                    offsets.pop()
                    opionions.pop()
            labels.extend(['O'] * len(offsets))
            labels.reverse()
            batchLabels.append(labels)
            opinionsOffsets.reverse()
            batchOpinionsOffsets.append(opinionsOffsets)
            message.reverse()

        return batchLabels
    
    @classmethod
    def getBatchBIEOSSPLabels(cls, batchTokensOffset, batchOpinons):
        # offset是前闭后开
        batchLabels = []
        batchOpinionsOffsets = []
        for (offsets, opionions) in (zip(batchTokensOffset, batchOpinons)):
            labels = []
            opinionsOffsets = []
            message = []
            opionions = list(opionions)
            # 从offset列表从后往前推理对应的标签
            while 0 < len(opionions):
                while opionions[-1].end <= offsets[-1][0]:
                    labels.append('O')
                    message.append(str(offsets[-1]) + " is " + labels[-1])
                    offsets.pop()
                if offsets[-1][0] <= opionions[-1].begin:
                    labels.append('S' + '-' + (opionions[-1].polarity)[:3].upper())
                    message.append(str(offsets[-1]) + " is " + labels[-1])
                    opinionsOffsets.append(offsets[-1])
                    offsets.pop()
                    opionions.pop()
                else:
                    # 说明是B I E
                    labels.append("E" + '-' + (opionions[-1].polarity)[:3].upper())
                    message.append(str(offsets[-1]) + " is " + labels[-1])
                    opinionOffsetEnd = offsets[-1][1]
                    offsets.pop()
                    while opionions[-1].begin < offsets[-1][0]:
                        labels.append("I" + '-' + (opionions[-1].polarity)[:3].upper())
                        message.append(str(offsets[-1]) + " is " + labels[-1])
                        offsets.pop()
                    labels.append("B" + '-' + (opionions[-1].polarity)[:3].upper())
                    message.append(str(offsets[-1]) + " is " + labels[-1])
                    opinionOffsetBegin = offsets[-1][0]
                    opinionsOffsets.append((opinionOffsetBegin, opinionOffsetEnd))
                    offsets.pop()
                    opionions.pop()
            labels.extend(['O'] * len(offsets))
            labels.reverse()
            batchLabels.append(labels)
            opinionsOffsets.reverse()
            batchOpinionsOffsets.append(opinionsOffsets)
            message.reverse()

        return batchLabels

    @classmethod
    def getXLMRTokenizerEncode(cls):
        pass

    @classmethod
    def getBatchLabelsEncode(cls):
        pass

def getFileName(dataParams):
    # fileName = f".{dataParams.PretrainModel.replace('/', '_')}/"
    timestr = time.asctime(time.localtime(time.time())).replace(' ','|')
    fileName = dataParams.Source+'|'+dataParams.Target+'|'+timestr+".Record"
    return fileName