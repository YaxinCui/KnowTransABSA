# coding: utf-8
# jupyter nbconvert --to script BaseModel.ipynb

import torch
import torch.utils.data as data

from ArgumentParser.BaseArgParser import BaseArgumentParser
from Dataset.SemEval16Task5Dataset import SemEvalXMLDataset

argument = BaseArgumentParser()
dataParams = argument.parse_args()

# Dataset
trainSourceDataset = SemEvalXMLDataset(phrase="Train", language=dataParams.Source)

trainDataset, trialDataset = data.random_split(trainSourceDataset, [int(len(trainSourceDataset)*0.9), len(trainSourceDataset)-int(len(trainSourceDataset)*0.9)])

"""
trialDataset = SemEvalXMLDataset(phrase="Trial", language=dataParams.Source) + trainSourceDataset[:200]
trainDataset = trainSourceDataset[200:]
"""

testDataset = SemEvalXMLDataset(phrase="Test", language=dataParams.Target)

# DataLoader
from CollateFn.CollateFnBERTology import CollateFnBERTology
collateFn = CollateFnBERTology(pretrained_model_name_or_path=dataParams.PretrainModel, dataParams=dataParams)

from torch.utils.data import DataLoader

trainDataLoader = DataLoader(trainDataset, batch_size=dataParams.Batchsize, collate_fn=collateFn.collate_fn, shuffle=False, drop_last=False)
trialDataLoader = DataLoader(trialDataset, batch_size=dataParams.Batchsize, collate_fn=collateFn.collate_fn, shuffle=False, drop_last=False)
testDataLoader = DataLoader(testDataset, batch_size=dataParams.Batchsize, collate_fn=collateFn.collate_fn, shuffle=False, drop_last=False)

dataLoader = {
    'train': trainDataLoader,
    'trial': trialDataLoader,
    'test': testDataLoader
}

from Model.ModelBERTology import BERTology
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTology(pretrained_model=dataParams.PretrainModel).to(DEVICE)

# 优化层
from torch import optim
from transformers import get_linear_schedule_with_warmup

optimizer = optim.AdamW(model.parameters(), lr=dataParams.LearningRate)
warm_up_ratio = 0.1 # 定义要预热的step
total_steps = (len(trainDataset) // dataParams.Batchsize) * dataParams.TrainEpochs

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * warm_up_ratio), num_training_steps=total_steps)

from torch.nn import NLLLoss
criterion = NLLLoss()

from ModelSummary.ModelOutputsRecord import ModelOutputsRecord

modelOutputsRecord = ModelOutputsRecord(dataParams = dataParams, phrases=['train', 'trial', 'test'])

from Model.ModelRun import ModelRun
for epoch in range(dataParams.TrainEpochs):
    print('*'*40 + ' '*10 + str(epoch) + ' '*10 + "*"*40)
    
    for phrase in ['train', 'trial', 'test']:
        print("\n"+"+"*20+' '*20 + phrase + ' '*20 + '+'*20 + '\n')
        epochModelOutputs = ModelRun.runEpochModel(model, criterion, dataLoader[phrase], optimizer, scheduler, isTrain=(phrase=='train'), DEVICE=DEVICE)
        evalResultDict = modelOutputsRecord.addEpochModelOutputs(epochModelOutputs, phrase=phrase)
        print(modelOutputsRecord.strEvalResultDict(evalResultDict))
    
    bestEvalResultDict = modelOutputsRecord.analyseModel()
    print("best iter is ", bestEvalResultDict['iter'])
    print('train: ', modelOutputsRecord.strEvalResultDict(bestEvalResultDict['train']))
    print('trial: ', modelOutputsRecord.strEvalResultDict(bestEvalResultDict['trial']))
    print('test : ', modelOutputsRecord.strEvalResultDict(bestEvalResultDict['test' ]))

modelOutputsRecord.dump()