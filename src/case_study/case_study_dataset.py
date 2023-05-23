import argparse
from src.dataset_classes import Subgraph
import os.path as osp
import os
import pandas as pd
import shutil
import numpy as np
import networkx as nx
import gc
from node2vec import Node2Vec
def parse_args():
    parser = argparse.ArgumentParser(description="generate_dataset.")
    parser.add_argument('--dimensions', type=int, default=64,
                        help='Number of dimensions. Default is 64.')
    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')
    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')
    parser.add_argument('--window-size', type=int, default=5,
                        help='Context size for optimization. Default is 5.')
    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')
    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')
    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')
    return parser.parse_args()
def read_interaction(path):
    interaction = set()
    with open(path, 'r') as f:
        for line in f.readlines():
            arr = line.strip().split(',')
            interaction.add((arr[0], arr[1]))
    return interaction
def read_rpin(path):
    edge_list = pd.read_csv(path, header=None,delimiter=',').reset_index(drop=True)
    edge_list = np.array(edge_list)
    node_names = set()
    G = nx.Graph()
    for edge in edge_list:
        node_names.add(edge[0])
        node_names.add(edge[1])
        G.add_edge(edge[0], edge[1])
    G = G.to_undirected()
    return G, edge_list,node_names
def generate_node_vec(dict_n2v):
    node_feature_path = f'./node_feature'
    # 处理rna节点特征
    if not osp.exists(node_feature_path):
        os.makedirs(node_feature_path)
    with open(f'{node_feature_path}/features.txt', mode='w') as f:
        for i, value in dict_n2v.items():
            node_vec = [str(i) for i in value]
            #print(len(node_vec))
            f.write(i + ',' + ','.join(node_vec) + '\n')

    # 处理protein节点特征
    with open(f'{node_feature_path}/features.txt', mode='a') as f:
        for i, value in dict_n2v.items():
            node_vec=[str(i) for i in value]
            #print(len(node_vec))
            f.write(i + ',' + ','.join(node_vec) + '\n')
def generate_n2v(G, path, dimensions, walk_length, num_walks, p, q, workers):
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, p=p, q=q,
                        workers=workers)
    model = node2vec.fit()
    # 创建保存node2vec结果的文件夹
    if not osp.exists(path):
        os.makedirs(path)
        print(f'创建了文件夹: {path} 用于保存node2vec')
    model.wv.save_word2vec_format(path + '/result.emb')
def read_node2vec_file(path):
    node_name_vec_dict = {}
    node2vec_file = open(path, mode='r')
    lines = node2vec_file.readlines()
    count = 1
    # 读取node2vec文件
    for i in range(len(lines)):
        if count ==1:
            count = 0
            continue
        line = lines[i].strip().split(' ')
        node_name_vec_dict[line[0]] = line[1:]
    node2vec_file.close()
    return node_name_vec_dict
if __name__ == "__main__":
    args = parse_args()
    processed_data_path = f'./processed_data'
    case_study_path= './'
    #生成节点特征
    print('start generate node feature vector\n')
    G, _,_ = read_rpin(f'{processed_data_path}/train/pos_edges') #读取全图
    # 添加结点
    node_names = []
    with open(f'{processed_data_path}/all_node_name') as f:
        lines = f.readlines()
        for line in lines:
            node_names.append(line.strip())
    G.add_nodes_from(node_names)
    generate_n2v(G, f'{case_study_path}/node2vec', args.dimensions, args.walk_length, args.num_walks, args.p, args.q, args.workers)
    dict_n2v = read_node2vec_file(f'{case_study_path}/node2vec/result.emb')
    generate_node_vec( dict_n2v)
    print('generate node feature vector end\n')

    #生成数据集
    print('start generate case study dataset\n')
    node_feature_path = f'./node_feature'
    node_vecs = pd.read_csv(f'{node_feature_path}/features.txt', header=None).reset_index(drop=True)
    dict_node_name_vec = dict(zip(node_vecs.values[:,0],node_vecs.values[:,1:]) )#根据第0列排序
    # 生成case study的训练数据集
    case_study_train_path=f'{case_study_path}/train_dataset'
    if not osp.exists(case_study_train_path):
        print(f'创建了文件夹：{case_study_train_path}')
        os.makedirs(case_study_train_path)
    else:
        shutil.rmtree(case_study_train_path,True)
        os.makedirs(case_study_train_path)
    path_pos_train = f'{processed_data_path}/train/pos_edges'
    path_neg_train = f'{processed_data_path}/train/neg_edges'
    pos_train = read_interaction(path_pos_train)
    neg_train = read_interaction(path_neg_train)
    train_interactions=[]
    train_interactions.extend(pos_train)
    num_pos_train = len(train_interactions)
    train_interactions.extend(neg_train)
    num_neg_train = len(train_interactions) - num_pos_train
    y = np.ones(num_pos_train).tolist()
    y.extend(np.zeros(num_neg_train).tolist())
    train_dataset = Subgraph(case_study_train_path,G,dict_node_name_vec,train_interactions,y)

    # 生成case study的验证数据集
    case_study_val_path=f'{case_study_path}/val_dataset'
    if not osp.exists(case_study_val_path):
        print(f'创建了文件夹：{case_study_val_path}')
        os.makedirs(case_study_val_path)
    else:
        shutil.rmtree(case_study_val_path,True)
        os.makedirs(case_study_val_path)
    path_pos_val = f'{processed_data_path}/val/pos_edges'
    path_neg_val = f'{processed_data_path}/val/neg_edges'
    pos_val = read_interaction(path_pos_val)
    neg_val = read_interaction(path_neg_val)
    val_interactions=[]
    val_interactions.extend(pos_val)
    num_pos_val = len(val_interactions)
    val_interactions.extend(neg_val)
    num_neg_val = len(val_interactions) - num_pos_val
    y = np.ones(num_pos_val).tolist()
    y.extend(np.zeros(num_neg_val).tolist())
    val_dataset = Subgraph(case_study_val_path,G,dict_node_name_vec,val_interactions,y)

    # 生成case study edges的分析数据集
    case_study_edges_path=f'{case_study_path}/case_study_dataset'
    if not osp.exists(case_study_edges_path):
        print(f'创建了文件夹：{case_study_edges_path}')
        os.makedirs(case_study_edges_path)
    else:
        shutil.rmtree(case_study_edges_path,True)
        os.makedirs(case_study_edges_path)
    path_case_study_edges = f'{processed_data_path}/case_study_edges'
    case_study_edges = read_interaction(path_case_study_edges)
    case_study_interactions=[]
    case_study_interactions.extend(case_study_edges)
    num_case_study_edges = len(case_study_interactions)
    y = np.zeros(num_case_study_edges).tolist()
    case_study_dataset = Subgraph(case_study_edges_path,G,dict_node_name_vec,case_study_interactions,y)


    print('generate case study dataset end\n')