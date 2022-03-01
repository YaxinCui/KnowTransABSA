#!/bin/bash

mkdir ./RecordXLMRBase

for Source in english
    do
    for Target in spanish french
        do
        for seed in 1 2 3 4 5
            do
            nohup python -u BERTologyModel.py --PretrainModel xlm-roberta-base --Source ${Source} --Target ${Target} --RecordsDir RecordXLMRBase > ./RecordXLMRBase/${Source}${Target}${seed}.txt 2>&1
            done
        done
    done

mkdir ./RecordTwitterSentiment

for Source in english
    do
    for Target in spanish french
        do
        for seed in 1 2 3 4 5
            do
            nohup python -u BERTologyModel.py --PretrainModel cardiffnlp/twitter-xlm-roberta-base-sentiment --Source ${Source} --Target ${Target} --RecordsDir RecordTwitterSentiment > ./RecordTwitterSentiment/${Source}${Target}${seed}.txt 2>&1
            done
        done
    done

mkdir ./RecordNER

for Source in english
    do
    for Target in spanish french
        do
        for seed in 1 2 3 4 5
            do
            nohup python -u BERTologyModel.py --PretrainModel Davlan/xlm-roberta-base-ner-hrl --Source ${Source} --Target ${Target} --RecordsDir RecordNER > ./RecordNER/${Source}${Target}${seed}.txt 2>&1
            done
        done
    done