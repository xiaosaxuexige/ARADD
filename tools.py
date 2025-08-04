'''
This file contains different functions used for processing our data.
'''
import json
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

from model import CLBert
from dataloader import QueryDataset

def extract_functions_from_json(json_file_path):
    '''
    检测函数名，输入json日志数据，输出总的函数个数和函数名列表
    '''
    # 定义正则表达式模式来匹配函数名和参数
    # 示例使用方法
    # json_file_path = './single_traindata.json'  # 替换为实际的json文件路径
    # function_count, function_list = extract_functions_from_json(json_file_path)
    # print(f"函数个数: {function_count}")
    # print(f"函数名称: {function_list}")
    pattern = r'([a-zA-Z_][a-zA-Z0-9_]*_[a-zA-Z0-9_]*)\s*\(([^)]*)\)'
    
    function_names = []
    
    # 打开文件逐行读取每个 JSON 对象
    with open(json_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 解析每行的 JSON 对象
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                continue  # 忽略无法解析的行
            
            # 提取“answer”列中的内容
            answer_text = entry.get('answer', '')
            
            # 使用正则表达式查找所有匹配的函数名称
            matches = re.findall(pattern, answer_text)
            for match in matches:
                function_name = match[0]
                if function_name not in function_names:
                    function_names.append(function_name)
    
    # 返回函数个数和函数名称的列表
    return len(function_names), function_names

def json2excel():
    '''
    json转 Excel
    '''
    file_path = './center_gen.json'  # 替换为你的JSON文件路径
    df = pd.read_json(file_path,lines=True)
    output_file = 'center_gen.xlsx'  # 定义输出文件名
    df.to_excel(output_file, index=False)  # index=False 表示不保存行索引

def merge_json():
    '''
    self-instruction生成的
    '''
    df1=pd.read_json('./search_qa_poi.json',lines=True)
    df2=pd.read_json('./center_gen.json',lines=True)
    df_combined = pd.concat([df1, df2], ignore_index=True)
    # 按行打乱顺序 (shuffle)
    df_shuffled = df_combined.sample(frac=1).reset_index(drop=True)
    output_file = '3funcs_tr.json'
    df_shuffled.to_json(output_file, orient='records', lines=True, force_ascii=False)

def sample_from_dataframe(df, sample_size):
    """
    从给定的 DataFrame 中随机抽取指定数量的样本。

    参数：
    df : pandas.DataFrame
        原始的 DataFrame。
    sample_size : int
        抽样的数量。

    返回：
    sample_df : pandas.DataFrame
        抽样得到的 DataFrame。
    """
    # 使用 sample() 方法随机抽样
    sample_df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    return sample_df

def extract_queries(df):
    """
    从 prompt 列中提取匹配的 query，并保存到 query 列。

    参数：
    df : pandas.DataFrame
        输入的 DataFrame，包含 'prompt', 'answer', '函数个数', '函数名称' 列。

    返回：
    df : pandas.DataFrame
        添加了 query 列的 DataFrame。
    """
    # 定义正则表达式
    pattern = r'动作序列。\n(.*?)\n\nassistant'

    # 使用 apply 和 loc 操作，避免 SettingWithCopyWarning
    df = df.copy()  # 创建 DataFrame 的副本，避免修改视图
    df.loc[:, 'query'] = df['prompt'].apply(
        lambda x: re.search(pattern, x, re.S).group(1) if re.search(pattern, x, re.S) else None
    )

    return df

def delete_funcdata(df, func_list):
    """
    根据函数名称列表删除 DataFrame 中的行。

    参数：
    df : pandas.DataFrame
        包含 'prompt', 'answer', '函数个数', '函数名称' 列的 DataFrame。
    func_list : list
        需要删除的函数名称列表。

    返回：
    initial_df : pandas.DataFrame
        删除指定行后的新 DataFrame。
    """
    # 过滤掉 "函数名称" 列中包含任一 func_list 中函数的行
    filtered_df = df[~df['函数名称'].apply(lambda x: any(func in x for func in func_list))]
    
    # 将结果保存为 initial_df
    initial_df = filtered_df.reset_index(drop=True)
    
    return initial_df

def delete_columns_and_save_json(df, output_file):
    """
    删除 DataFrame 中的第三列和第四列，并将结果保存为 JSON 文件。

    参数：
    df : pandas.DataFrame
        原始的 DataFrame。
    output_file : str
        保存的 JSON 文件路径。

    返回：
    new_df : pandas.DataFrame
        删除列后的新 DataFrame。
    """
    # 删除第三列和第四列（列的索引是从0开始的）
    new_df = df.drop(df.columns[[2, 3]], axis=1)
    
    # 将新 DataFrame 保存为 JSON 文件
    new_df.to_json(output_file, orient='records', lines=True, force_ascii=False)

    # return new_df

def filter_fun_and_separate(result_df, function_list):
    # 将输入的函数名列表转换为集合，方便子集判断
    function_set = set(function_list)
    
    # 使用apply和lambda函数进行筛选，确保函数名称列是function_list的子集
    iter_df = result_df[result_df['函数名称'].apply(
        lambda x: set(x).issubset(function_set)
    )]
    
    # bad_case_df 包含除了 iter_df 之外的部分
    bad_case_df = result_df[~result_df['函数名称'].apply(
        lambda x: set(x).issubset(function_set)
    )]
    
    return iter_df, bad_case_df

def extract_functions_from_dataframe(df):
    '''
    该函数熟入为 [promt,answer] 列的 dataframe
    输出为[prompt, answer, 函数个数, 函数名称] 列的dataframe
    其中，函数个数和函数名称都去重了。
    '''
    # 定义正则表达式模式来匹配函数名和参数
    pattern = r'([a-zA-Z_][a-zA-Z0-9_]*_[a-zA-Z0-9_]*)\s*\(([^)]*)\)'

    # 创建新的列用于存储函数个数和函数名称
    result_data = {
        'prompt': [],
        'answer': [],
        '函数个数': [],
        '函数名称': []
    }

    # 遍历DataFrame的每一行
    for index, row in df.iterrows():
        answer_text = row['answer']
        # 使用正则表达式查找所有匹配的函数名
        matches = re.findall(pattern, answer_text)
        # 提取唯一的函数名称
        unique_functions = list(set([match[0] for match in matches]))
        function_count = len(unique_functions)

        # 填充结果数据
        result_data['prompt'].append(row['prompt'])
        result_data['answer'].append(row['answer'])
        result_data['函数个数'].append(function_count)
        result_data['函数名称'].append(unique_functions)

    # 创建新的DataFrame
    result_df = pd.DataFrame(result_data)

    return result_df

def bert_semantic_embedding(intent_queries, df, k=10):
    """
    输入多个 query 字符串，返回每个 query 与其相似度最高的前 k 个 query 字符串。
    
    参数：
    - intent_queries: list, 包含多个要查找相似 query 的输入字符串。
    - df: pandas.DataFrame，包含原始 query 和 embedding 列。
    - k: int，返回相似度最高的前 k 个 query，默认值为 10。
    
    返回：
    - list of lists: 每个 query 对应的相似度最高的 query 列表（第一个元素为输入 query，接下来是相似度最高的 k 个 query）。
    """
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 初始化模型和 tokenizer，并将模型移动到正确设备
    model = CLBert('/root/paddlejob/workspace/env_run/qwen/model/saved_models/joint-intent-bert-base-uncased-bank77', device=device).to(device)
    tokenizer = AutoTokenizer.from_pretrained('/root/paddlejob/workspace/env_run/qwen/model/saved_models/joint-intent-bert-base-uncased-bank77')

    # 3. 获取 queries 列表，并创建 DataLoader
    queries = df['query'].tolist()
    dataset = QueryDataset(queries)
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 4. 存储所有嵌入的列表
    all_embeddings = []
    results = []

    # 5. 遍历 DataLoader 进行批量推理
    with torch.no_grad():
        for batch in dataloader:
            # 对当前批次进行编码并将其移动到设备
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)

            # 调用模型的 forward 方法
            outputs = model(inputs)

            # 获取 features
            embeddings = outputs["features"].cpu()  # 将嵌入移到 CPU

            # 将当前批次的 embeddings 添加到 all_embeddings 列表
            all_embeddings.append(embeddings)
            
    # 6. 将 all_embeddings 列表中的张量拼接成一个二维张量
    all_embeddings = torch.cat(all_embeddings, dim=0)
    print(all_embeddings.shape)
    # 7. 遍历 intent_queries，计算其相似度最高的前 k 个 query
    for query in intent_queries:
        with torch.no_grad():
            # 获取 query 的 BERT 嵌入
            inputs = tokenizer(query, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(inputs)
            query_embedding = outputs["features"].cpu().numpy()
            print(query_embedding.shape)

        # 计算当前 query embedding 与 DataFrame 中所有嵌入的余弦相似度
        similarity_scores = cosine_similarity(query_embedding, all_embeddings.numpy())[0]

        # 获取相似度最高的前 k 个索引
        top_k_indices = similarity_scores.argsort()[-k:][::-1]

        # 根据索引从 DataFrame 中提取相应的 query
        top_k_queries = df.iloc[top_k_indices]['query'].tolist()

        # 将输入 query 添加到该 query 的相似结果列表的开头
        result_list = [query] + top_k_queries
        results.append(result_list)

    # 8. 将结果保存到 Excel 文件
    columns = ['query'] + [f'similar_query_{i+1}' for i in range(k)]
    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_excel("similar_queries.xlsx", index=False)

    return 0
    # return results

def qwen_semantic_embedding(intent_queries, df, k=10):
    """
    输入多个 query 字符串，返回每个 query 与其相似度最高的前 k 个 query 字符串。
    
    参数：
    - intent_queries: list, 包含多个要查找相似 query 的输入字符串。
    - df: pandas.DataFrame，包含原始 query 和 embedding 列。
    - k: int，返回相似度最高的前 k 个 query，默认值为 10。
    
    返回：
    - list of lists: 每个 query 对应的相似度最高的 query 列表（第一个元素为输入 query，接下来是相似度最高的 k 个 query）。
    """
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 初始化 Qwen 模型和 tokenizer
    # model_path = '/root/paddlejob/workspace/env_run/qwen/model/Qwen2-1.5B'
    model_path = '/root/paddlejob/workspace/env_run/qwen/model/qwen_1.5b_iter1'
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # 确保使用 float32 避免 BFloat16 的不兼容问题
        device_map='auto'  # 自动分配设备
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 3. 获取 queries 列表，并创建 DataLoader
    queries = df['query'].tolist()
    batch_size = 32

    # 4. 存储所有嵌入的列表
    all_embeddings = []
    results = []
    '''
    (batch_size, sequence_length, hidden_size)：这种形状的张量表示模型的隐藏层输出。
	•	batch_size 是批次大小，例如在你的例子中为 32 或 1。
	•	sequence_length 是输入序列（token）的长度，取决于输入文本的长度。
	•	hidden_size 是每个 token 的嵌入维度（例如 1536），它由模型的配置决定，是固定的。
    '''

    # 5. 遍历 queries 列表，生成嵌入
    with torch.no_grad():
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i+batch_size]
            inputs = tokenizer(batch_queries, padding=True, truncation=True, return_tensors="pt").to(device)

            # 获取所有隐藏层的输出
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # 所有隐藏层的输出

            # 提取第一个和倒数第二层的隐藏层输出，计算平均嵌入
            first_layer_hid = hidden_states[0].cpu().numpy()
            if i==0:
                print(first_layer_hid.dtype) # float32
                print(first_layer_hid.shape) # (32, 24, 1536)
            last_layer_hid = hidden_states[-2].cpu().numpy()
            if i==0:
                print('last_layer_hid的形状是:', last_layer_hid.shape) #(32, 24, 1536)
            avg_embeddings = np.mean(first_layer_hid + last_layer_hid, axis=1)
            all_embeddings.append(avg_embeddings)

    # 6. 将所有批次的嵌入拼接成一个二维数组
    all_embeddings = np.vstack(all_embeddings)  # shape: (num_queries, embedding_dim)
    print('all_embeddings的形状是:', all_embeddings.shape)  #all_embeddings的形状是: (50, 1536)
    # 7. 遍历 intent_queries，计算其相似度最高的前 k 个 query
    for query in intent_queries:
        with torch.no_grad():
            # 获取单个 query 的 Qwen 嵌入
            inputs = tokenizer(query, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            # 提取第一个和倒数第二层的隐藏层输出，计算平均嵌入
            first_layer_hid = hidden_states[0].cpu().numpy()
            # print(first_layer_hid.shape) # (32, 24, 1536)
            last_layer_hid = hidden_states[-2].cpu().numpy()
            # print('last_layer_hid的形状是:', last_layer_hid.shape)
            query_embedding = np.mean(first_layer_hid + last_layer_hid, axis=1)
            # print(query_embedding.shape)

        # 计算当前 query embedding 与 DataFrame 中所有嵌入的余弦相似度
        similarity_scores = cosine_similarity(query_embedding.reshape(1, -1), all_embeddings)[0]

        # 获取相似度最高的前 k 个索引
        top_k_indices = similarity_scores.argsort()[-k:][::-1]

        # 根据索引从 DataFrame 中提取相应的 query
        top_k_queries = df.iloc[top_k_indices]['query'].tolist()

        # 将输入 query 添加到该 query 的相似结果列表的开头
        result_list = [query] + top_k_queries
        # print(result_list)
        results.append(result_list)
    # print(results)

    # 展平嵌套列表
    flat_list = [item for sublist in results for item in sublist]
    # 转换为指定格式
    # formatted_list=[]
    formatted_list = [{"query": query} for query in flat_list]
    # 保存为 JSON 文件
    with open("./iter1/iter1_for_intent.json", "w", encoding="utf-8") as f:
        for item in formatted_list:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    # print("所有 queries 已保存为 JSON 文件：iter1_for_intent.json")
    # return results
    return 0