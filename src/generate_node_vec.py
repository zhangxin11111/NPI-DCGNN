# coding:utf-8
import networkx as nx
import argparse
import gc
import pandas as pd
import numpy as np
from node2vec import Node2Vec
import os.path as osp
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Run node2vec.")
    # NPInter2 RPI369
    parser.add_argument('--dataset', default="RPI369", help='dataset name')
    # 1：代表rna的k取4，蛋白质的k取3 ；  2: 表示不使用序列信息    # 2代表：代表rna的k取3，蛋白质的k取2  ;
    parser.add_argument('--node_vec_type', type=int, default=2, help='dataset name')
    parser.add_argument('--fold', default=0, type=int, help='which fold is this')
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
    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)
    return parser.parse_args()


def read_grn(path):
    print('读取Network文件')
    edge_list = pd.read_csv(path, header=None).reset_index(drop=True)
    edge_list = np.array(edge_list)
    gene_names = set()
    G = nx.Graph()
    for edge in edge_list:
        gene_names.add(edge[0])
        gene_names.add(edge[1])
        G.add_edge(edge[0], edge[1])
    G = G.to_undirected()
    return G, gene_names


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
def generate_node_vec(rna_Dim,proteinDim):
    # 处理rna节点特征
    rna_zero= []
    for i in range(proteinDim):
        rna_zero.append(0)
    features = pd.read_csv(f'{processed_data_path}/rna_sequence_features.csv')
    dict_rna_sequence_feature = dict(zip(features.iloc[:, 0].values.tolist(), features.iloc[:, 1:].values.tolist()))
    node_feature_path=f'{path}/node_feature/{args.fold}'
    if not osp.exists(node_feature_path):
        os.makedirs(node_feature_path)
    with open(f'{node_feature_path}/features.txt', mode='w') as f:
        for i, value in dict_rna_sequence_feature.items():
            node_vec = [str(i) for i in value]
            if i in dict_n2v.keys():
                node_vec.extend([str(i) for i in rna_zero])
                node_vec.extend([str(i) for i in dict_n2v[i]])
            else:
                print(f'{i} 没有原始特征')
                for i in range(len(value)):
                    node_vec.append('0')
            #print(len(node_vec))
            f.write(i + ',' + ','.join(node_vec) + '\n')

    # 处理protein节点特征
    rna_zero= []
    for i in range(rna_Dim):
        rna_zero.append(0)
    features = pd.read_csv(f'{processed_data_path}/protein_sequence_features.csv')
    dict_protein_sequence_feature = dict(zip(features.iloc[:, 0].values.tolist(), features.iloc[:, 1:].values.tolist()))
    node_feature_path=f'{path}/node_feature/{args.fold}'
    if not osp.exists(node_feature_path):
        os.makedirs(node_feature_path)
    with open(f'{node_feature_path}/features.txt', mode='a') as f:
        for i, value in dict_protein_sequence_feature.items():
            node_vec = [str(i) for i in value]
            if i in dict_n2v.keys():
                node_vec.extend([str(i) for i in rna_zero])
                node_vec.extend([str(i) for i in dict_n2v[i]])
            else:
                print(f'{i} 没有原始特征')
                for i in range(len(value)):
                    node_vec.append('0')
            #print(len(node_vec))
            f.write(i + ',' + ','.join(node_vec) + '\n')
def generate_node_vec1():
    # 处理rna节点特征
    features = pd.read_csv(f'{processed_data_path}/rna_sequence_features.csv')
    dict_rna_sequence_feature = dict(zip(features.iloc[:, 0].values.tolist(), features.iloc[:, 1:].values.tolist()))
    node_feature_path=f'{path}/node_feature/{args.fold}'
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
if __name__ == '__main__':
    print('start generate node feature vector\n')
    args = parse_args()
    path = f'../data/{args.dataset}'

    processed_data_path = f'{path}/processed_data'
    path_cross_valid = f'{path}/path_cross_valid'
    # 读取所有节点
    node_names = []
    with open(f'{processed_data_path}/all_node_name') as f:
        lines = f.readlines()
        for line in lines:
            node_names.append(line.strip())
    # grn网络
    graph_path = f'{path_cross_valid}/dataset_{args.fold}/pos_train_edges'
    G, gene_names = read_grn(graph_path)
    # 添加结点
    G.add_nodes_from(node_names)

    path_n2v = f'{path}/node2vec/{args.fold}'
    if not osp.exists(path_n2v):
        os.makedirs(path_n2v)
    generate_n2v(G, path_n2v, args.dimensions, args.walk_length, args.num_walks, args.p, args.q, args.workers)
    dict_n2v = read_node2vec_file(f'{path_n2v}/result.emb')
    if args.node_vec_type == 1:
        generate_node_vec(256,343)
    # elif args.node_vec_type == 2:
    #     generate_node_vec(64,49)
    else:
        generate_node_vec1()
    del G
    gc.collect()
    print('generate node feature vector end\n')