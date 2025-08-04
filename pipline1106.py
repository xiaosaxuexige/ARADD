import pandas as pd
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from model import CLBert
from dataloader import QueryDataset
from tools import *

# 读取JSON文件
file_path = './single_trdata_withfunc.json'  # (8089, 4) 是已经去掉了 ['ask_for_LLM','start_inquiry','start_comfort']这三个函数的data
df = pd.read_json(file_path,lines=True)
# print(df.shape)

#基于function list 得到该轮初始的ACC
function_list = ['search_for_poi', 'qa_for_poi','qa_for_navi','search_for_navi','ride_hailing']         #这个是基础的function_list
#将只含有function list 中函数的 分离出来，为good case，剩余的为bad case
good_case_Qi,bad_case_Qi=filter_fun_and_separate(df, function_list)
#保留prompt 和answer列 ，作为 sft的data
# delete_columns_and_save_json(good_case_Qi, 'iter0_sft.json')            # 只保留 prompt 和answer列，作为sft的data
ACC= good_case_Qi.shape[0]/df.shape[0]
print('iteration 0 的query answer准确率为:',ACC)
# 把queries 也提取出来， 得到【'prompt', 'answer', '函数个数', '函数名称', 'query'】的dataframe
bad_case_Qi=extract_queries(bad_case_Qi)         # 得到【'prompt', 'answer', '函数个数', '函数名称', 'query'】的dataframe
print('第i轮badcaseQi的columns是:',bad_case_Qi.columns) # 得到【'prompt', 'answer', '函数个数', '函数名称', 'query'】的dataframe
print(bad_case_Qi.shape)   # (1157, 5)
# bad_case_Qi.to_json('./iter0_bad.json', orient='records', lines=True, force_ascii=False)
#从这些bad case中sample 出100个 喂给gpt4 进行初步意图总结
iter0_sample4gpt = sample_from_dataframe(bad_case_Qi, 100)
print(iter0_sample4gpt.shape)
iter0_sample4gpt[['query']].to_json('./iter1/iter1_bad_query4gpt.json', orient='records', lines=True, force_ascii=False)
# biu
intent_queries = ['602公交车在哪个位置','182最早一班是几点发车','4号线还有几分钟才能到灵境胡同站',
'能不能找一个适合闺蜜约会的餐厅，到融泽嘉园、长安九里、竹溪园都差不多方便？','帮推荐几个离我公司、望京、亦庄、中关村和西二旗差不多远，适合我们大伙儿一起聚餐的地方','有什么能看到星空的免费公园么，离西二旗、清河、国贸这些地都近的']
# bert_semantic_embedding(intent_queries, iter0_sample4gpt, k=10)
qwen_semantic_embedding(intent_queries, bad_case_Qi, k=10)




