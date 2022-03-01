
# 输入：xlm文件的文件路径
# 输出：一个DataSet，每个样例包含[reviewid, sentenceId, text, UniOpinions]
#      每个样例包含的Opinion，是一个列表，包含的是单个Opinion的详情

from torch.utils.data import Dataset


class Opinion:
    def __init__(self, target, category, polarity, from_, to) -> None:
        self.target = target
        self.category = category
        self.polarity = polarity
        self.begin = int(from_)
        self.end = int(to)
    
    def __str__(self) -> str:
        return f"Target:{self.target} | C:{self.category} | P:{self.polarity} | Begin:{self.begin} | End:{self.end}"

    def __repr__(self) -> str:
        return f"Target:{self.target} | C:{self.category} | P:{self.polarity} | Begin:{self.begin} | End:{self.end}"

class SemEvalXMLDataset(Dataset):
    # data process
    
    def __init__(self, phrase='Train', language='english', domain="Restaurant", tag=""):
        # 获得SentenceWithOpinions，一个List包含(reviewId, sentenceId, text, Opinions)

        self.tag = tag
        self.SentenceWithOpinions = []
        self.file_name = f"SemEval_2016_Task5/{domain}_{phrase}_Subtask1/{domain.lower()}s_{phrase.lower()}_{language}_subtask1.xml"
        self.xml_path = __file__.replace("SemEval16Task5Dataset.py", self.file_name)

        from xml.dom.minidom import parse
        self.sentenceXmlList = parse(self.xml_path).getElementsByTagName('sentence')

        for sentenceXml in self.sentenceXmlList:
            reviewId = sentenceXml.getAttribute("id").split(':')[0]
            sentenceId = sentenceXml.getAttribute("id")
            if len(sentenceXml.getElementsByTagName("text")[0].childNodes) < 1:
                # skip no reviews part
                continue
            text = sentenceXml.getElementsByTagName("text")[0].childNodes[0].nodeValue
            OpinionXmlList = sentenceXml.getElementsByTagName("Opinion")
            Opinions = []
            for opinionXml in OpinionXmlList:
                # some text maybe have no opinion
                target = opinionXml.getAttribute("target")
                category = opinionXml.getAttribute("category")
                polarity = opinionXml.getAttribute("polarity")
                from_ = opinionXml.getAttribute("from")
                to = opinionXml.getAttribute("to")
                # skill NULL
                if target.lower() == 'null' or target == '' or from_ == to:
                    continue
                Opinions.append(Opinion(target, category, polarity, from_, to))
            # delete repeate Opinions
            Opinions.sort(key=lambda x: int(x.begin)) # 从小到大排序
            UniOpinions = []
            for opinion in Opinions:
                if len(UniOpinions) < 1:
                    UniOpinions.append(opinion)
                else:
                    if opinion.begin != UniOpinions[-1].begin and opinion.end != UniOpinions[-1].end:
                        UniOpinions.append(opinion)
            self.SentenceWithOpinions.append((text, UniOpinions, phrase, language, domain, reviewId, sentenceId))

        self.length = len(self.SentenceWithOpinions)
        self.showDataLength()
    
    def showDataLength(self):
        print()
        print(f"Get {self.length} sentence from {self.xml_path} ")
        print(f"The first data is {self.SentenceWithOpinions[0]}")
        
    def __getitem__(self, index):
        return self.SentenceWithOpinions[index]
    
    def __len__(self):
        return self.length