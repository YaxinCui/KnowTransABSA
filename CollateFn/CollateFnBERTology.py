from CollateFn.CollateFnBase import CollateFnTransformer
from transformers import AutoTokenizer
import copy

class CollateFnBERTology(CollateFnTransformer):
    
    def __init__(self, pretrained_model_name_or_path, dataParams=None) -> None:
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.dataParams = dataParams
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, verbose=False, use_fast=True)
        print(f"load {pretrained_model_name_or_path} tokenizer success")

    def collate_fn(self, batchDataSet):
        # 获取原始未处理数据
        batchRawData = self.getRawData(batchDataSet)
        # 获取经过tokenizer不同方式处理后的数据
        batchTokenizerEncode = self.getTokenizerEncode(batchRawData["batchText"])
        # 获取标签相关的数据，这一部分跟具体任务有关，因为不同任务有不同的标签
        batchLabelsEncode = self.getBatchLabelsEncode(copy.deepcopy(batchTokenizerEncode['batchTokensOffset']), batchRawData)
        
        batchDataEncode = {
            'batchRawData': batchRawData,
            'batchTokenizerEncode': batchTokenizerEncode,
            'batchLabelsEncode': batchLabelsEncode
        }
        return batchDataEncode
        
    @classmethod
    def to(cls, batchDataEncode, DEVICE):
        batchDataEncode['batchTokenizerEncode']['batchTextEncodePlus'] = batchDataEncode['batchTokenizerEncode']['batchTextEncodePlus'].to(DEVICE)
        return batchDataEncode

    def getTokenizerEncode(self, batchText):
        batchTextEncodePlus = self.tokenizer.batch_encode_plus(batchText, return_tensors='pt', 
                                    return_offsets_mapping=True, padding=True)
        batchTextTokens = [self.tokenizer.tokenize(text) for text in batchText]
        batchTextTokensLength = [len(tokens) for tokens in batchTextTokens]
        batchOffsetMapping = batchTextEncodePlus.pop('offset_mapping').tolist()
        batchTokensOffset = [tokensOffset[1: 1+length] for tokensOffset, length in zip(batchOffsetMapping, batchTextTokensLength)]
        batchTokenizerEncode = {
            'batchTextEncodePlus': batchTextEncodePlus,
            'batchTextTokens': batchTextTokens,
            'batchTextTokensLength': batchTextTokensLength,
            'batchTokensOffset': batchTokensOffset
        }
        return batchTokenizerEncode
        
    def getBatchBIEOSLabels(self, batchTokensOffset, batchOpinons):
        batchLabels = super().getBatchBIEOSLabels(batchTokensOffset, batchOpinons, taskMode=self.dataParams.Task)
        return batchLabels
        
    def getBatchLabelsEncode(self, batchTokensOffset, batchRawData):
        # 返回数据初始处理后，得到的数据，供反向传播时使用，这一步跟具体任务有关

        batchBIEOSLabels = self.getBatchBIEOSLabels(batchTokensOffset, batchRawData['batchOpinions'])
        batchBIEOSIds = [[self.label2id[label] for label in labels] for labels in batchBIEOSLabels]
        batchFlattenLabels = sum(batchBIEOSLabels, [])
        batchFlattenIds = sum(batchBIEOSIds, [])
        batchLabelsEncode = {
            'batchBIEOSLabels': batchBIEOSLabels,
            'batchBIEOSIds': batchBIEOSIds,
            'batchFlattenLabels': batchFlattenLabels,
            'batchFlattenIds': batchFlattenIds
        }
        return batchLabelsEncode