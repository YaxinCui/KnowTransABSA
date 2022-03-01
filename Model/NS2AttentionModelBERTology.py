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
        self.NerModel = AutoModel.from_pretrained("Davlan/xlm-roberta-base-ner-hrl", return_dict=True, output_hidden_states=True)
        self.SentimentModel = AutoModel.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment", return_dict=True, output_hidden_states=True)

        self.config = AutoConfig.from_pretrained(pretrained_model)
        self.classifier = nn.Linear(self.config.hidden_size, label_num)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.config.hidden_size, num_heads=8, batch_first=True)

    def forward(self, *args, **kwargs):

        batchLengths = kwargs['batchTokenizerEncode']['batchTextTokensLength']
        batchTextEncodePlus = kwargs['batchTokenizerEncode']['batchTextEncodePlus']
        model_outputs = self.model(**batchTextEncodePlus)
        ner_model_outputs = self.NerModel(**batchTextEncodePlus)
        sentiment_model_outputs = self.SentimentModel(**batchTextEncodePlus)

        token_embeddings = torch.cat([(model_outputs.last_hidden_state)[i][1:1+length] for i, length in enumerate(batchLengths)], dim=0)
        ner_token_embeddings = torch.cat([(ner_model_outputs.last_hidden_state)[i][1:1+length] for i, length in enumerate(batchLengths)], dim=0)
        sentiment_token_embeddings = torch.cat([(sentiment_model_outputs.last_hidden_state)[i][1:1+length] for i, length in enumerate(batchLengths)], dim=0)

        attention_token = torch.concat([token_embeddings.unsqueeze(1), ner_token_embeddings.unsqueeze(1), sentiment_token_embeddings.unsqueeze(1)], dim=1)
        
        attn_output, attn_output_weights = self.multihead_attn(attention_token, attention_token, attention_token)

        tokenLogSoftmax = F.log_softmax(self.classifier(F.relu(torch.mean(attn_output, dim=1))), dim=1)
        
        modelOutDir = {'tokenLogSoftmax': tokenLogSoftmax}
        return modelOutDir

    def dataTo(self, batchDataEncode, DEVICE):
        batchDataEncode = CollateFnBERTology.to(batchDataEncode, DEVICE)
        return batchDataEncode