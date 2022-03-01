from torch import nn
from transformers import AutoModel, AutoConfig
import torch
import torch.nn.functional as F
from CollateFn.CollateFnBERTology import CollateFnBERTology

## 适用范围
# mBERT
# xlm-roberta

class BERTology(nn.Module):
    def __init__(self, pretrained_model, label_num=len(CollateFnBERTology.id2label)):
        super(BERTology, self).__init__()
        self.label_num = label_num
        self.model = AutoModel.from_pretrained(pretrained_model, return_dict=True, output_hidden_states=True)
        self.config = AutoConfig.from_pretrained(pretrained_model)
        self.classifier = nn.Linear(self.config.hidden_size, label_num)
        
    def forward(self, *args, **kwargs):

        batchLengths = kwargs['batchTokenizerEncode']['batchTextTokensLength']
        batchTextEncodePlus = kwargs['batchTokenizerEncode']['batchTextEncodePlus']
        model_outputs = self.model(**batchTextEncodePlus)
        
        token_embeddings = torch.cat([(model_outputs.last_hidden_state)[i][1:1+length] for i, length in enumerate(batchLengths)], dim=0)
        
        tokenLogSoftmax = F.log_softmax(self.classifier(F.relu(token_embeddings)), dim=1)
        
        modelOutDir = {'tokenLogSoftmax': tokenLogSoftmax}
        return modelOutDir

    def dataTo(self, batchDataEncode, DEVICE):
        batchDataEncode = CollateFnBERTology.to(batchDataEncode, DEVICE)
        return batchDataEncode