from torch_geometric.data import Data,HeteroData
from torch_geometric.data import InMemoryDataset
import torch
import networkx as nx

class Subgraph(InMemoryDataset):
    '''
    分别生成source和target的一阶子图
    '''
    def __init__(self, root,rpin=None, dict_node_name_vec=None,interaction_list=None,y=None,transform=None, pre_transform=None):
        self.rpin=rpin
        self.dict_node_name_vec=dict_node_name_vec
        self.interaction_list = interaction_list
        self.y=y
        self.source_node = 0.0
        self.target_node = 0.0
        super(Subgraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.subscript = 1

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        if self.interaction_list != None:
            num_data = len(self.interaction_list)
            print(f'the number of samples:{num_data}')
            data_list = []
            count = 0
            for interaction in self.interaction_list:
                data = self.local_subgraph_generation(interaction, self.y[count])
                data_list.append(data)
                count = count + 1
                if count % 100 == 0:
                    print(f'{count}/{num_data}')
                    print(f'average node number of source = {self.source_node / count}')
                    print(f'average node number of target = {self.target_node / count}')
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]
            data, slices = self.collate(data_list)
            print(f'average node number of source = {self.source_node / count}')
            print(f'average node number of target = {self.target_node / count}')
            torch.save((data, slices), self.processed_paths[0])

    def local_subgraph_generation(self, interaction,y):
        '''
        查找source在rpin的一阶邻居
        '''
        source = []
        source_edge_index = [[], []]
        try:
            source_neig = set(nx.neighbors(self.rpin, interaction[0]))
        except:
            #print(f'{interaction[0]} not exist {y}')
            source_neig=set()
        # 构造source的edge_list
        dict_source={}
        source_subgraph_serial_number=0
        source_subgraphSerialNumber=source_subgraph_serial_number
        dict_source[interaction[0]] = source_subgraphSerialNumber
        for rna in source_neig:
            if rna==interaction[1]:
                continue
            if rna in dict_source.keys():
                node2_subgraphSerialNumber=dict_source[rna]
            else:
                source_subgraph_serial_number += 1
                node2_subgraphSerialNumber = source_subgraph_serial_number
                dict_source[rna] = node2_subgraphSerialNumber
            source_edge_index[0].append(source_subgraphSerialNumber)
            source_edge_index[1].append(node2_subgraphSerialNumber)
            source_edge_index[0].append(node2_subgraphSerialNumber)
            source_edge_index[1].append(source_subgraphSerialNumber)
        #构造source的x
        dict_subgraphNodeSerialNumber_nodeName=dict(zip(dict_source.values(),dict_source.keys()))
        for i in range(len(dict_subgraphNodeSerialNumber_nodeName)):
            vector = []
            vector.append(0)
            vector.extend(self.dict_node_name_vec[dict_subgraphNodeSerialNumber_nodeName[i]])
            source.append(vector)
        '''
        查找traget在rpin的一阶邻居
        '''
        target = []
        target_edge_index = [[], []]
        try:
            target_neig=set(nx.neighbors(self.rpin,interaction[1]))
        except:
            target_neig=set()
        # 构造target的edge_list
        dict_target = {}
        target_subgraph_serial_number = 0
        target_subgraphSerialNumber = target_subgraph_serial_number
        dict_target[interaction[1]] = target_subgraphSerialNumber
        for protein in target_neig:
            if protein==interaction[0]:
                continue
            if protein in dict_target.keys():
                node2_subgraphSerialNumber=dict_target[protein]
            else:
                target_subgraph_serial_number += 1
                node2_subgraphSerialNumber = target_subgraph_serial_number
                dict_target[protein] = node2_subgraphSerialNumber
            target_edge_index[0].append(target_subgraphSerialNumber)
            target_edge_index[1].append(node2_subgraphSerialNumber)
            target_edge_index[0].append(node2_subgraphSerialNumber)
            target_edge_index[1].append(target_subgraphSerialNumber)

        #构造source的x
        dict_subgraphNodeSerialNumber_nodeName=dict(zip(dict_target.values(),dict_target.keys()))
        for i in range(len(dict_subgraphNodeSerialNumber_nodeName)):
            vector = []
            vector.append(0)
            vector.extend(self.dict_node_name_vec[dict_subgraphNodeSerialNumber_nodeName[i]])
            target.append(vector)
        # y记录这个interaction的真假
        if y == 1:
            y = [1]
        else:
            y = [0]
        self.source_node += len(source)
        self.target_node += len(target)
        source = torch.tensor(source, dtype=torch.float)
        target = torch.tensor(target, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        source_edge_index = torch.tensor(source_edge_index, dtype=torch.long)
        target_edge_index = torch.tensor(target_edge_index, dtype=torch.long)
        #data = Data(x=source,edge_index=source_edge_index,target=target,y=y,target_link=interaction)
        data = HeteroData()
        data['source'].x = source
        data['source'].edge_index = source_edge_index
        data['target'].x = target
        data['target'].edge_index = target_edge_index
        data.y = y
        data.target_link=interaction
        return data


