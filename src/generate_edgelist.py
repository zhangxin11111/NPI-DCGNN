# coding:utf-8
import argparse
import os.path as osp
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold,train_test_split
import pandas as pd
import random
def parse_args():
    parser = argparse.ArgumentParser(description="generate_dataset.")
    # NPInter2 RPI369 RPI3265
    parser.add_argument('--dataset', default="RPI369", help='project name')

    parser.add_argument('--num_fold', default=5,type=int, help='how num of fold is this')

    parser.add_argument('--output', default=1,type=int,  help='output dataset or not')
    return parser.parse_args()
def write_interactor(interactor,output_path):
    with open(output_path,'w') as f:
        for triplet in interactor:
            f.write(triplet[0]+','+triplet[1]+'\n')
def generate_training_and_testing(set_interaction, set_negativeInteraction,path_cross_valid,num_fold):
    # 把set_interactionKey和set_negativeInteractionKey分5份
    list_set_interactionKey=[]
    list_set_negativeInteractionKey=[]
    for i in range(num_fold):
        list_set_interactionKey.append(set())
        list_set_negativeInteractionKey.append(set())
    count = 0
    while len(set_interaction) > 0:
        list_set_interactionKey[count % num_fold].add(set_interaction.pop())
        count += 1
    count = 0
    while len(set_negativeInteraction) > 0:
        list_set_negativeInteractionKey[count % num_fold].add(set_negativeInteraction.pop())
        count += 1
    # 每次那四份组成训练集，另一份是测试集
    for i in range(num_fold):
        pos_train_edges = set()
        neg_train_edges = set()
        pos_test_edges = set()
        neg_test_edges = set()
        for j in range(num_fold):
            if i == j:
                pos_test_edges.update(list_set_interactionKey[j])
                neg_test_edges.update(list_set_negativeInteractionKey[j])
            else:
                pos_train_edges.update(list_set_interactionKey[j])
                neg_train_edges.update(list_set_negativeInteractionKey[j])
        if args.output == 1:
            if not osp.exists(path_cross_valid +f'/dataset_{i}'):
                os.makedirs(path_cross_valid + f'/dataset_{i}')
            write_interactor(pos_test_edges,path_cross_valid + f'/dataset_{i}/pos_test_edges')
            write_interactor(neg_test_edges,path_cross_valid + f'/dataset_{i}/neg_test_edges')
            # 从训练集中抽出一部分作为验证集
            pos_train_edges = np.array(list(pos_train_edges))
            pos_train_edges, pos_val_edges = train_test_split(pos_train_edges, test_size=0.1, random_state=1337)
            neg_train_edges = np.array(list(neg_train_edges))
            neg_train_edges, neg_val_edges = train_test_split(neg_train_edges, test_size=0.1, random_state=1337)
            write_interactor(pos_train_edges,path_cross_valid + f'/dataset_{i}/pos_train_edges')
            write_interactor(neg_train_edges,path_cross_valid + f'/dataset_{i}/neg_train_edges')
            write_interactor(pos_val_edges, path_cross_valid + f'/dataset_{i}/pos_val_edges')
            write_interactor(neg_val_edges, path_cross_valid + f'/dataset_{i}/neg_val_edges')

def random_negative_sampling(set_interaction,rna_name_set,protein_name_set):
    set_negativeInteraction = set()
    #num_of_interaction = len(set_interaction)
    num_of_lncRNA = len(rna_name_set)
    rna_name_list=list(rna_name_set)
    protein_name_list = list(protein_name_set)
    num_of_protein = len(protein_name_set)
    negative_interaction_count = 0
    while(negative_interaction_count < len(set_interaction)):
        random_index_lncRNA = random.randint(0, num_of_lncRNA - 1)
        random_index_protein = random.randint(0, num_of_protein - 1)
        temp_lncRNA = rna_name_list[random_index_lncRNA]
        temp_protein = protein_name_list[random_index_protein]
        # 检查随机选出的lncRNA和protein是不是有已知相互作用
        negativeInteraction = (temp_lncRNA, temp_protein)
        if negativeInteraction in set_interaction:
            continue
        if negativeInteraction in set_negativeInteraction:
            continue
        # 经过检查，随机选出的lncRNA和protein是可以作为负样本的
        set_negativeInteraction.add(negativeInteraction)
        negative_interaction_count = negative_interaction_count + 1
    return set_negativeInteraction

def get_k_fold_data(k, data):
    X, y = data[:, :], data[:, -1]
    #sfolder = StratifiedKFold(n_splits = k, shuffle=True,random_state=1)
    sfolder = StratifiedKFold(n_splits=k, shuffle=True)

    train_data = []
    test_data = []
    train_label = []
    test_label = []

    for train, test in sfolder.split(X, y):
        train_data.append(X[train])
        test_data.append(X[test])
        train_label.append(y[train])
        test_label.append(y[test])
    return train_data, test_data
def output_dict_file(dict,path):
    output_file = open(path, mode='w')
    for item in dict.items():
        output_file.write(item[0] + '\n')
        #output_file.write(item[0]+' '+str(item[1])+'\n')
    output_file.close()
def read_rpi(rpi_path):
    print('读取RPI文件')
    npi_df = pd.read_excel(rpi_path)
    npi_df = npi_df.rename(columns={'RNA names': 'source', 'Protein names': 'target', 'Labels': 'type'})
    npi_df['type'] = 0
    rna_name_set=set(npi_df['source'].tolist())
    protein_name_set=set(npi_df['target'].tolist())
    npi_df = npi_df[['source', 'target']]
    # 打乱样本顺序
    #rppi_df = rppi_df.sample(frac=1, replace=False, random_state=1)
    from sklearn.utils import shuffle
    npi_df = shuffle(npi_df)
    print('RPI文件读取完成')
    return npi_df,rna_name_set,protein_name_set
if __name__ == '__main__':
    print('start partition dataset\n')
    # 参数
    args = parse_args()

    path = f'../data/{args.dataset}'
    processed_data_path = f'{path}/processed_data'
    #读取network文件
    positive_samples, rna_name_set, protein_name_set = read_rpi(f'{path}/source_data/{args.dataset}.xlsx')
    # 处理节点
    rna_features = pd.read_csv(f'{path}/k_mer/rna/kmer_frequency.txt', header=None)
    rna_features = rna_features.loc[rna_features[0].isin(rna_name_set)]

    protein_features = pd.read_csv(f'{path}/k_mer/protein/kmer_frequency.txt', header=None)
    protein_features = protein_features.loc[protein_features[0].isin(protein_name_set)]
    # 保存网络中所有节点的特征向量
    if not osp.exists(processed_data_path):
        os.makedirs(processed_data_path)
    rna_features.to_csv(f'{processed_data_path}/rna_sequence_features.csv',index=False)
    protein_features.to_csv(f'{processed_data_path}/protein_sequence_features.csv',index=False)
    # 保存网络中所有的基因名称
    node_name_set = rna_name_set.union(protein_name_set)
    with open(processed_data_path+f'/all_node_name',mode='w') as f:
        for item in node_name_set:
            f.write(item+'\n')
    set_interaction = [(triplet[0], triplet[1]) for triplet in positive_samples.values]
    set_interaction = set(set_interaction)
    set_negativeInteraction = random_negative_sampling(set_interaction,rna_name_set,protein_name_set)

    # 保存所有正样本的边
    write_interactor(set_interaction, processed_data_path + '/all_postitive_edges')
    # 保存所有负样本的边
    write_interactor(set_negativeInteraction, processed_data_path + '/all_negative_edges')
    # 生成训练集-测试集
    path_cross_valid=f'{path}/path_cross_valid'
    if not osp.exists(path_cross_valid):
        os.makedirs(path_cross_valid)
    set_interaction = [(triplet[0], triplet[1]) for triplet in set_interaction]
    generate_training_and_testing(set_interaction,set_negativeInteraction,path_cross_valid,args.num_fold)
    print('partition dataset end\n')