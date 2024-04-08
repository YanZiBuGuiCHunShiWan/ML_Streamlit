import os
import pandas as pd
import numpy as np
import requests
import json,copy
from collections import Counter
from flask import Flask
app = Flask(__name__)

def sample(categories,probs):
    sample = np.random.choice(categories, p=probs)
    return sample


df = pd.read_excel("chapters/datasets/分类问答语料库样例20231006.xlsx",sheet_name="分类问答语料库")
df_detail = df.iloc[:,4:7]
single_turn_metadata=[]
single_turn_dialogs=[]
dict_info={}
for index,question in enumerate(df_detail["问题描述"]):
    if type(question)==str:
        if dict_info!={}:
            single_turn_metadata.append(dict_info)
            dict_info={}
        dict_info["question"] = question
        dict_info["answerlist"] = []
        dict_info["labels"] = []
        dict_info["answerlist"].append(df_detail.iloc[index,1])
        dict_info["labels"].append(df_detail.iloc[index,2])
        
    else:
        dict_info["answerlist"].append(df_detail.iloc[index,1])
        dict_info["labels"].append(df_detail.iloc[index,2])
        
single_turn_dialogs=[]
for dict_info in single_turn_metadata:
    question = dict_info["question"]
    dialog="求助者："+question+"支持者："
    for response,label in zip(dict_info["answerlist"],dict_info["labels"]):
        if response[-1]!="。"  and response[-1]!="？":
            response=response+"。"
        dialog+="({})".format(label)+response[3:]
    single_turn_dialogs.append(dialog)

df2= pd.read_excel("chapters/datasets/分类问答语料库样例20231006.xlsx",sheet_name="分类问答标注类别")
strategy_list=[]
for index,strategy in enumerate(df2.iloc[:,0]):
    strategy_list.append(strategy+"："+str(df2.iloc[index,1]))
    existed_stratygies=[]
for dict_info in single_turn_metadata:
    existed_stratygies.append(dict_info["labels"])

initialized_distribution=[]
for strategy_group in existed_stratygies:
    after_strategies=[]
    for strategy in strategy_group:
        strategy=strategy.replace("开发","开放")
        len_st=len(strategy.split("、"))
        if len_st>1:
            cleared_strategy=strategy.split("、")
            for strategy2 in cleared_strategy:
                if strategy2 in ["获取信息","启发思考","鼓励表达"]:
                    continue
                cleared_strategy2=strategy2.split("-")[0]
                after_strategies.append(cleared_strategy2)
        elif len(strategy.split("-"))>1:
            after_strategies.append(strategy.split("-")[0][:5])
        else:
            after_strategies.append(strategy[:5])
    initialized_distribution.append(after_strategies)
    
senlen_distribution=[]
for strategy_group in existed_stratygies:
    senlen_distribution.append(len(strategy_group))
len_dict=dict(Counter(senlen_distribution))
sorted_dict = {k: v for k, v in sorted(len_dict.items(), key=lambda item: item[0]) if k<8}
len_pseudo_probabilities=(np.array(list(sorted_dict.values()))/50)
prob_sum=np.sum(len_pseudo_probabilities)

len_probabilities=len_pseudo_probabilities/prob_sum
len_categories=list(sorted_dict.keys())

strategy_set=set()
for s in initialized_distribution:
    for final_strategy in s:
        strategy_set.add(final_strategy)
        
strategy2id_dict={}
for index,s in enumerate(list(strategy_set)):
    strategy2id_dict[s]=index
strategy2id_dict

strategy_transfer_matrix=np.zeros((len(strategy_set),len(strategy_set)))
for strategy_group in initialized_distribution:
    for index in range(0,len(strategy_group)-1):
        row_num=strategy2id_dict[strategy_group[index]]
        colum_num=strategy2id_dict[strategy_group[index+1]]
        if row_num==colum_num:
            pass
        else:
            strategy_transfer_matrix[row_num][colum_num]+=1
row_sums=strategy_transfer_matrix.sum(axis=1)
normalized_matrix=strategy_transfer_matrix/row_sums[:,np.newaxis]

init_strategy_dist={}
for key in strategy2id_dict.keys():
    init_strategy_dist[key]=0
for strategy_group in initialized_distribution:
    init_strategy_dist[strategy_group[0]]+=1
init_distribution=np.array(list(init_strategy_dist.values()))/50
init_strategy_categories=list(init_strategy_dist.keys())

@app.route('/post', methods=['POST'])
def generate_one_turn_strategy():
    stimu_strategies=[]
    strategy=sample(init_strategy_categories,init_distribution)
    current_length=sample(len_categories,len_probabilities)
    stimu_strategies.append(strategy)
    for step in range(current_length-1):
        strategy_index=strategy2id_dict[strategy]
        current_distribution=normalized_matrix[strategy_index,:].copy()
        if step>0:
            #遍历已经采样到的策略列表 stimu_strategis
            for stgy in stimu_strategies:
                stgy_idx=strategy2id_dict[stgy]
                current_distribution[stgy_idx]=0
            if np.sum(current_distribution)>0:
                current_sum=np.sum(current_distribution)
                normalized_current_distribution=current_distribution/current_sum #####
                strategy=sample(init_strategy_categories,normalized_current_distribution)########
            else:
                break
        else:
            strategy=sample(init_strategy_categories,current_distribution)
        stimu_strategies.append(strategy)
    return "->".join(stimu_strategies)

if __name__ == '__main__':
    app.run()