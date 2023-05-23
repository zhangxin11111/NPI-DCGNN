import argparse
import os.path as osp
import os
import pandas as pd
import shutil
import numpy as np
import networkx as nx
from src.dataset_classes import Subgraph
def parse_args():
    parser = argparse.ArgumentParser(description="generate_dataset.")
    # NPInter2 RPI369
    parser.add_argument('--dataset', default="RPI369", help='dataset name')

    parser.add_argument('--fold',type=int, default=0,help='which fold is this')
    return parser.parse_args()
def read_interaction(path):
    interaction = set()
    with open(path, 'r') as f:
        for line in f.readlines():
            arr = line.strip().split(',')
            interaction.add((arr[0], arr[1]))
    return interaction
def read_rpin(path):
    edge_list = pd.read_csv(path, header=None).reset_index(drop=True)
    edge_list = np.array(edge_list)
    node_names = set()
    G = nx.Graph()
    for edge in edge_list:
        node_names.add(edge[0])
        node_names.add(edge[1])
        G.add_edge(edge[0], edge[1])
    G = G.to_undirected()
    return G, node_names
if __name__ == "__main__":
    print('start generate pytorch dataset\n')
    args=parse_args()
    path = f'../data/{args.dataset}'

    path_cross_valid = f'{path}/path_cross_valid'
    # grn网络
    graph_path = f'{path_cross_valid}/dataset_{args.fold}/pos_train_edges'
    rpin, node_names = read_rpin(graph_path)
    #读取节点特征文件
    node_feature_path = f'{path}/node_feature/{args.fold}'

    node_vecs = pd.read_csv(f'{node_feature_path}/features.txt', header=None).reset_index(drop=True)
    print(node_vecs.shape)
    dict_node_name_vec = dict(zip(node_vecs.values[:,0],node_vecs.values[:,1:]) )#根据第0列排序
    # 生成训练集
    dataset_train_path=f'{path}/dataset/dataset_{args.fold}/train'
    if not osp.exists(dataset_train_path):
        print(f'创建了文件夹：{dataset_train_path}')
        os.makedirs(dataset_train_path)
    else:
        shutil.rmtree(dataset_train_path,True)
        os.makedirs(dataset_train_path)
    path_pos_train =  f'{path_cross_valid}/dataset_{args.fold}/pos_train_edges'
    path_neg_train =f'{path_cross_valid}/dataset_{args.fold}/neg_train_edges'
    pos_train = read_interaction(path_pos_train)
    neg_train = read_interaction(path_neg_train)
    train_interactions=[]
    train_interactions.extend(pos_train)
    num_pos_train = len(train_interactions)
    train_interactions.extend(neg_train)
    num_neg_train = len(train_interactions) - num_pos_train
    y = np.ones(num_pos_train).tolist()
    y.extend(np.zeros(num_neg_train).tolist())
    train_dataset = Subgraph(dataset_train_path,rpin,dict_node_name_vec,train_interactions,y)
    # 生成验证集
    dataset_val_path=f'{path}/dataset/dataset_{args.fold}/val'
    if not osp.exists(dataset_val_path):
        print(f'创建了文件夹：{dataset_val_path}')
        os.makedirs(dataset_val_path)
    else:
        shutil.rmtree(dataset_val_path,True)
        os.makedirs(dataset_val_path)
    path_pos_val =  f'{path_cross_valid}/dataset_{args.fold}/pos_val_edges'
    path_neg_val =f'{path_cross_valid}/dataset_{args.fold}/neg_val_edges'
    pos_val = read_interaction(path_pos_val)
    neg_val = read_interaction(path_neg_val)
    val_interactions=[]
    val_interactions.extend(pos_val)
    num_pos_val = len(val_interactions)
    val_interactions.extend(neg_val)
    num_neg_val = len(val_interactions) - num_pos_val
    y = np.ones(num_pos_val).tolist()
    y.extend(np.zeros(num_neg_val).tolist())
    val_dataset = Subgraph(dataset_val_path,rpin,dict_node_name_vec,val_interactions,y)
    #生成测试集
    dataset_test_path = f'{path}/dataset/dataset_{args.fold}/test'
    if not osp.exists(dataset_test_path):
        print(f'创建了文件夹：{dataset_test_path}')
        os.makedirs(dataset_test_path)
    else:
        shutil.rmtree(dataset_test_path,True)
        os.makedirs(dataset_test_path)
    path_neg_test =f'{path_cross_valid}/dataset_{args.fold}/neg_test_edges'
    path_pos_test =  f'{path_cross_valid}/dataset_{args.fold}/pos_test_edges'
    pos_test = read_interaction(path_pos_test)
    neg_test= read_interaction(path_neg_test)
    test_interactions=[]
    test_interactions.extend(pos_test)
    num_pos_test=len(test_interactions)
    test_interactions.extend(neg_test)
    num_neg_test = len(test_interactions)-num_pos_test
    y=np.ones(num_pos_test).tolist()
    y.extend(np.zeros(num_neg_test).tolist())
    test_dataset = Subgraph(dataset_test_path, rpin,dict_node_name_vec,test_interactions,y)
    print('generate pytorch dataset end\n')