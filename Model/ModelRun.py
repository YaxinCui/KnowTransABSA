from tqdm import tqdm
import torch

class ModelRun():
    @classmethod
    def runEpochModel(cls, model, criterion, dataLoader, optimizer, scheduler=None, isTrain=False, DEVICE="cpu"):
        # epochModelOutputs
        epochModelOutputs = []
        if isTrain:
            model.train()
        else:
            model.eval()

        epochLoss = 0
        for batchDataEncode in tqdm(dataLoader, mininterval=20):
            batchModelOutputs = cls.runBatchModel(batchDataEncode, model, criterion, DEVICE=DEVICE)
            batchLoss = batchModelOutputs.pop('batchLoss')
            batchModelOutputs['batchLoss'] = batchLoss.detach().item()
            epochLoss += batchLoss.detach().item()
            if isTrain:
                optimizer.zero_grad()
                batchLoss.backward()
                optimizer.step()
            
            
            epochModelOutputs.append(batchModelOutputs)

            torch.cuda.empty_cache()

            if isTrain and (scheduler is not None):
                scheduler.step()
        print("epochLoss\n", epochLoss)
        return epochModelOutputs

    

    @classmethod
    def runBatchModel(cls, batchDataEncode, model, criterion, DEVICE="cpu"):
        # 返回loss, PredLabelList, TrueLabelList
        # batchDataEncode['batchTokenizerEncode']['batchTextEncodePlus'] = batchDataEncode['batchTokenizerEncode']['batchTextEncodePlus'].to(DEVICE)
        batchDataEncode = model.dataTo(batchDataEncode, DEVICE)

        batchFlattenIds = batchDataEncode['batchLabelsEncode']['batchFlattenIds']
        batchModelOutputs = model(**batchDataEncode)
        batchTrueLabelsTensor = torch.LongTensor(batchFlattenIds).to(DEVICE)
        batchLoss = criterion(batchModelOutputs['tokenLogSoftmax'], batchTrueLabelsTensor)
        batchModelOutputs['batchLoss'] = batchLoss

        tokenLogSoftmax = batchModelOutputs.pop('tokenLogSoftmax')

        batchModelOutputs['tokenLogSoftmax'] = tokenLogSoftmax.detach().cpu().numpy()
        batchModelOutputs['batchLabelsEncode'] = batchDataEncode['batchLabelsEncode']
        batchModelOutputs['batchTextTokensLength'] = batchDataEncode['batchTokenizerEncode']['batchTextTokensLength']
        batchModelOutputs['batchRawData'] = batchDataEncode['batchRawData']

        return batchModelOutputs