import time
import pickle
import numpy as np

from CollateFn.CollateFnBase import CollateFnBase, getFileName

from ModelSummary.EvalATESP import EvalATESP
from ModelSummary.EvalATE import EvalATE

from os import makedirs, path

        # 'tokenLogSoftmax', 'batchTextTokensLength', 'batchLabelsEncode', 'batchRawData'
        #                                           -- 'batchBIEOSLabels'
        #                                           -- 'batchBIEOSIds'
        #                                           -- 'batchFlattenLabels'
        #                                           -- 'batchFlattenIds'
        # 结构 Epochmodeloutputs phraseEpochModelOutputs --> epochModelOutputs --> batchModelOutputs
        # batchModelOutputs = {'tokenLogSoftmax': , 'batchLoss':, 'batchLengths':, 'batchDataEncode':}
        # epochmodelOutputs = [batchModelOutputs, batchModelOutputs ……]
        # recordDict['phrase'] = [epochmodelOutputs, epochmodelOutputs, ……]

class ModelOutputsRecord():
    def __init__(self, dataParams, phrases=['train', 'trial', 'test']) -> None:
    
        self.dataParams = dataParams
        self.timestr = time.asctime(time.localtime(time.time())).replace(' ','|')
        self.phrases = phrases

        self.logDir = self.dataParams.RecordsDir

        self.fileName=getFileName(dataParams)

        if self.logDir[-1] != '/':
            self.logDir += '/'
        self.makeLogDir()

        self.recordDict = {phrase:[] for phrase in phrases}

        if self.dataParams.Task == "ATESP":
            self.strEvalResultDict = EvalATESP.strEvalResultDict
        elif self.dataParams.Task == "ATE":
            self.strEvalResultDict = EvalATE.strEvalResultDict
        
    def makeLogDir(self):
        if not path.exists(self.logDir):
            makedirs(self.logDir)

    def dump(self):
        with open(self.logDir + self.fileName, 'wb') as f:
            pickle.dump(self, f)

    def addEpochModelOutputs(self, epochModelOutputs, phrase):
        assert phrase in self.phrases
        self.recordDict[phrase].append(epochModelOutputs)
        evalResultDict = self.analyseEpochModelOutputs(epochModelOutputs)
        return evalResultDict
    
    @classmethod
    def unsqueezeBatchLabel(cls, batchFlattenLabels, batchLabelsLength):
        batchTokensLabel = []
        batchLabelsBeginOffset = [sum(batchLabelsLength[:i]) for i in range(len(batchLabelsLength))]
        batchLabelsEndOffset = [sum(batchLabelsLength[:i+1]) for i in range(len(batchLabelsLength))]
        for beginOffset, endOffset in zip(batchLabelsBeginOffset, batchLabelsEndOffset):
            batchTokensLabel.append(batchFlattenLabels[beginOffset: endOffset])
        return batchTokensLabel

    def analyseEpochModelOutputs(self, epochModelOutputs, id2label=CollateFnBase.id2label):
        # batchModelOutputs = {'tokenLogSoftmax': , 'batchLoss':, 'batchLengths':, 'batchDataEncode':}
        epochPredLabels = []
        epochTrueLabels = []
        epochLoss = []
        for batchModelOutputs in epochModelOutputs:
            batchTrueLabels = batchModelOutputs['batchLabelsEncode']['batchBIEOSLabels']
            batchTextTokensLength = batchModelOutputs['batchTextTokensLength']

            batchPredFlattenLabelsId = np.argmax(batchModelOutputs['tokenLogSoftmax'], axis=1).tolist()
            batchPredFlattenLabels = [id2label[lableId] for lableId in batchPredFlattenLabelsId]
            batchPredLabels = self.unsqueezeBatchLabel(batchPredFlattenLabels, batchTextTokensLength)
            batchLoss = batchModelOutputs['batchLoss']
            epochPredLabels.extend(batchPredLabels)
            epochTrueLabels.extend(batchTrueLabels)
            epochLoss.append(batchLoss)
        # print("loss:", np.mean(epochLoss))
        if self.dataParams.Task == "ATESP":
            evalResultDict  = EvalATESP.evaluateBatchOte(epochTrueLabels, epochPredLabels)
        elif self.dataParams.Task == "ATE":
            evalResultDict  = EvalATE.evaluateBatchOte(epochTrueLabels, epochPredLabels)

        return evalResultDict

    def analyseModel(self):
        # 返回训练到现在，模型的最佳效果
        assert len(self.recordDict['train'])==len(self.recordDict['trial']) and len(self.recordDict['test'])==len(self.recordDict['trial'])
        best_epoch_iter = None
        bestTrialEvalResultDict = None
        for epoch_iter, trialEpochModelOutputs in enumerate(self.recordDict['trial']):
            trialEvalResultDict = self.analyseEpochModelOutputs(trialEpochModelOutputs)
            if self.dataParams.Task == "ATESP":
                key = 'MicroF1'
            else:
                key = 'MacroF1'
            if (best_epoch_iter is None) or bestTrialEvalResultDict[key] < trialEvalResultDict[key]:
                best_epoch_iter = epoch_iter
                bestTrialEvalResultDict = trialEvalResultDict
        # 已经获得了最好的那次训练的数据
        bestEvalResultDict = {
            'iter': best_epoch_iter,
            'train': self.analyseEpochModelOutputs(self.recordDict['train'][best_epoch_iter]),
            'trial': self.analyseEpochModelOutputs(self.recordDict['trial'][best_epoch_iter]),
            'test': self.analyseEpochModelOutputs(self.recordDict['test'][best_epoch_iter]),
        }
        return bestEvalResultDict

    @classmethod
    def load(cls, logDirPath):
        with open(logDirPath, 'rb') as f:
            modelOutputsRecord = pickle.load(f)
        return modelOutputsRecord

    def getMetrics(self):
        # 获得性能参数，分为两种，一种是计算BIEOS各个标签的总数、预测正确数、F1值、另一种是以target为单位
        pass

    def AnalyseError(self):
        pass

    def getTargets(self):
        # 分析真实的target，抽取错误的target
        pass
