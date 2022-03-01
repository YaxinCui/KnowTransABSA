import imp
import os
from argparse import ArgumentParser
from CollateFn.CollateFnBase import CollateFnBase
import sys

class BaseArgumentParser():
    # 这是管理参数的类
    absolute_path = os.getcwd() + "/"
    def __init__(self) -> None:
        self.parser = ArgumentParser()
        self.addTrainArgument()
        self.addDataArgument()
    def addTrainArgument(self):
        self.parser.add_argument('--LearningRate',type=float, default=4e-5, help='learning rate')
        self.parser.add_argument('--TrainEpochs', type=int, default=20, help='Epochs')

    def addDataArgument(self):
        # 数据层参数
        self.parser.add_argument('--Task', type=str, default='ATESP') # ATE和ATESP任务
        self.parser.add_argument('--Batchsize',type=int, default=16, help='Batchsize')
        self.parser.add_argument('--Source', type=str, default='english')
        self.parser.add_argument('--Target', type=str, default='spanish')
        self.parser.add_argument('--GPU', type=int, default=0)
        self.parser.add_argument('--PretrainModel', type=str)
        self.parser.add_argument('--RecordsDir', type=str, default="./Records/")

    @classmethod
    def getEnv(self) -> None:
        env = sys.argv[0].split('/')[-1]
        if env == "ipykernel_launcher.py":
            env = "notebook"
        else:
            env = "terminal"
            # 这个时候，需要通过判断调用用户，来判断是nohup里，还是py xx.py方式引用
        print()
        print("environment ", env)
        return env

    def parse_args(self):
        if self.getEnv()=="notebook":
            dataParams = self.parser.parse_args([])
        else:
            dataParams = self.parser.parse_args()
        if dataParams.Task == "ATESP":
            CollateFnBase.id2label = CollateFnBase.ATESPid2label
            CollateFnBase.label2id = CollateFnBase.ATESPlabel2id
        elif dataParams.Task == "ATE":
            CollateFnBase.id2label = CollateFnBase.ATEid2label
            CollateFnBase.label2id = CollateFnBase.ATElabel2id
        print(dataParams)
        return dataParams
