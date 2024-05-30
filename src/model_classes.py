from torch_geometric.nn import TopKPooling, SAGEConv,GCNConv,GATConv,SAGPooling,ARGA,GATv2Conv
import torch
from torch_geometric.nn import global_max_pool,global_mean_pool,Set2Set,global_sort_pool,BatchNorm
import torch.nn.functional as F

# max || mean GATv2 1层
class Model_1(torch.nn.Module):
    def __init__(self, num_node_features, num_of_classes=2):
        super(Model_1, self).__init__()
        #用于source的子图
        self.conv1_1 = GATv2Conv(num_node_features, 128)
        self.pool1_1 = TopKPooling(128, ratio=0.5)

        #elf.conv1_3 = SAGEConv(128, 128)
        #self.pool1_3 = TopKPooling(128, ratio=0.5)
        #用于target的子图
        self.conv2_1 = GATv2Conv(num_node_features, 128)
        self.pool2_1 = TopKPooling(128, ratio=0.5)

        #self.conv2_3 = SAGEConv(128, 128)
        #self.pool2_3 = TopKPooling(128, ratio=0.5)
        #全连接层分类器
        self.lin1 = torch.nn.Linear(512, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, num_of_classes)

    def forward(self, data):
        source, source_edge_index,source_batch,target,target_edge_index,target_batch = data['source'].x, data['source'].edge_index,data['source'].batch.to(dtype=torch.int64),\
                                                             data['target'].x, data['target'].edge_index,data['target'].batch.to(dtype=torch.int64)
        # 提取source子图的表征向量
        source = F.leaky_relu(self.conv1_1(source, source_edge_index))
        source, source_edge_index, _, source_batch, _, _ = self.pool1_1(source, source_edge_index, None,source_batch)
        source1 = torch.cat([global_max_pool(source, source_batch), global_mean_pool(source, source_batch)], dim=1)


        #source = F.leaky_relu(self.conv1_3(source, source_edge_index))
        #source, source_edge_index, _, source_batch, _, _ = self.pool1_3(source, source_edge_index, None, source_batch)
        #source3 = torch.cat([global_max_pool(source, source_batch), global_mean_pool(source, source_batch)], dim=1)
        #source = source1 + source2 + source3

        source = source1
        # 提取target子图的表征向量
        target = F.leaky_relu(self.conv2_1(target, target_edge_index))
        target, target_edge_index, _, target_batch, _, _ = self.pool2_1(target, target_edge_index, None,target_batch)
        target1 = torch.cat([global_max_pool(target, target_batch), global_mean_pool(target, target_batch)], dim=1)

        #target = F.leaky_relu(self.conv2_3(target, target_edge_index))
        #target, target_edge_index, _, target_batch, _, _ = self.pool2_3(target, target_edge_index, None, target_batch)
        #target3 = torch.cat([global_max_pool(target, target_batch), global_mean_pool(target, target_batch)], dim=1)
        #target = target1 + target2 + target3

        target = target1

        x = torch.cat([source,target], dim=1)
        x = F.leaky_relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.leaky_relu(self.lin2(x))
        x = self.lin3(x)
        x = F.log_softmax(x, dim=-1)
        return x

# max || mean GATv2 2层
class Model_2(torch.nn.Module):
    def __init__(self, num_node_features, num_of_classes=2):
        super(Model_2, self).__init__()
        #用于source的子图
        self.conv1_1 = GATv2Conv(num_node_features, 128)
        self.pool1_1 = TopKPooling(128, ratio=0.5)
        self.conv1_2 = GATv2Conv(128, 128)
        self.pool1_2 = TopKPooling(128, ratio=0.5)
        #elf.conv1_3 = SAGEConv(128, 128)
        #self.pool1_3 = TopKPooling(128, ratio=0.5)
        #用于target的子图
        self.conv2_1 = GATv2Conv(num_node_features, 128)
        self.pool2_1 = TopKPooling(128, ratio=0.5)
        self.conv2_2 = GATv2Conv(128, 128)
        self.pool2_2 = TopKPooling(128, ratio=0.5)
        #self.conv2_3 = SAGEConv(128, 128)
        #self.pool2_3 = TopKPooling(128, ratio=0.5)
        #全连接层分类器
        self.lin1 = torch.nn.Linear(512, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, num_of_classes)

    def forward(self, data):
        source, source_edge_index,source_batch,target,target_edge_index,target_batch = data['source'].x, data['source'].edge_index,data['source'].batch.to(dtype=torch.int64),\
                                                             data['target'].x, data['target'].edge_index,data['target'].batch.to(dtype=torch.int64)
        # 提取source子图的表征向量
        source = F.leaky_relu(self.conv1_1(source, source_edge_index))
        source, source_edge_index, _, source_batch, _, _ = self.pool1_1(source, source_edge_index, None,source_batch)
        source1 = torch.cat([global_max_pool(source, source_batch), global_mean_pool(source, source_batch)], dim=1)
        source = F.leaky_relu(self.conv1_2(source, source_edge_index))
        source, source_edge_index, _, source_batch, _, _ = self.pool1_2(source, source_edge_index, None, source_batch)
        source2 = torch.cat([global_max_pool(source, source_batch), global_mean_pool(source, source_batch)], dim=1)

        #source = F.leaky_relu(self.conv1_3(source, source_edge_index))
        #source, source_edge_index, _, source_batch, _, _ = self.pool1_3(source, source_edge_index, None, source_batch)
        #source3 = torch.cat([global_max_pool(source, source_batch), global_mean_pool(source, source_batch)], dim=1)
        #source = source1 + source2 + source3

        source = source1 + source2
        # 提取target子图的表征向量
        target = F.leaky_relu(self.conv2_1(target, target_edge_index))
        target, target_edge_index, _, target_batch, _, _ = self.pool2_1(target, target_edge_index, None,target_batch)
        target1 = torch.cat([global_max_pool(target, target_batch), global_mean_pool(target, target_batch)], dim=1)
        target = F.leaky_relu(self.conv2_2(target, target_edge_index))
        target, target_edge_index, _, target_batch, _, _ = self.pool2_2(target, target_edge_index, None, target_batch)
        target2 = torch.cat([global_max_pool(target, target_batch), global_mean_pool(target, target_batch)], dim=1)


        #target = F.leaky_relu(self.conv2_3(target, target_edge_index))
        #target, target_edge_index, _, target_batch, _, _ = self.pool2_3(target, target_edge_index, None, target_batch)
        #target3 = torch.cat([global_max_pool(target, target_batch), global_mean_pool(target, target_batch)], dim=1)
        #target = target1 + target2 + target3

        target = target1 + target2

        x = torch.cat([source,target], dim=1)
        x = F.leaky_relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.leaky_relu(self.lin2(x))
        x = self.lin3(x)
        x = F.log_softmax(x, dim=-1)
        return x

# max || mean GATv2 3层
class Model_3(torch.nn.Module):
    def __init__(self, num_node_features, num_of_classes=2):
        super(Model_3, self).__init__()
        #用于source的子图
        self.conv1_1 = GATv2Conv(num_node_features, 128)
        self.pool1_1 = TopKPooling(128, ratio=0.5)
        self.conv1_2 = GATv2Conv(128, 128)
        self.pool1_2 = TopKPooling(128, ratio=0.5)
        self.conv1_3 = SAGEConv(128, 128)
        self.pool1_3 = TopKPooling(128, ratio=0.5)
        #用于target的子图
        self.conv2_1 = GATv2Conv(num_node_features, 128)
        self.pool2_1 = TopKPooling(128, ratio=0.5)
        self.conv2_2 = GATv2Conv(128, 128)
        self.pool2_2 = TopKPooling(128, ratio=0.5)
        self.conv2_3 = SAGEConv(128, 128)
        self.pool2_3 = TopKPooling(128, ratio=0.5)
        #全连接层分类器
        self.lin1 = torch.nn.Linear(512, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, num_of_classes)

    def forward(self, data):
        source, source_edge_index,source_batch,target,target_edge_index,target_batch = data['source'].x, data['source'].edge_index,data['source'].batch.to(dtype=torch.int64),\
                                                             data['target'].x, data['target'].edge_index,data['target'].batch.to(dtype=torch.int64)
        # 提取source子图的表征向量
        source = F.leaky_relu(self.conv1_1(source, source_edge_index))
        source, source_edge_index, _, source_batch, _, _ = self.pool1_1(source, source_edge_index, None,source_batch)
        source1 = torch.cat([global_max_pool(source, source_batch), global_mean_pool(source, source_batch)], dim=1)
        source = F.leaky_relu(self.conv1_2(source, source_edge_index))
        source, source_edge_index, _, source_batch, _, _ = self.pool1_2(source, source_edge_index, None, source_batch)
        source2 = torch.cat([global_max_pool(source, source_batch), global_mean_pool(source, source_batch)], dim=1)
        source = F.leaky_relu(self.conv1_3(source, source_edge_index))
        source, source_edge_index, _, source_batch, _, _ = self.pool1_3(source, source_edge_index, None, source_batch)
        source3 = torch.cat([global_max_pool(source, source_batch), global_mean_pool(source, source_batch)], dim=1)
        source = source1 + source2 + source3

        # 提取target子图的表征向量
        target = F.leaky_relu(self.conv2_1(target, target_edge_index))
        target, target_edge_index, _, target_batch, _, _ = self.pool2_1(target, target_edge_index, None,target_batch)
        target1 = torch.cat([global_max_pool(target, target_batch), global_mean_pool(target, target_batch)], dim=1)
        target = F.leaky_relu(self.conv2_2(target, target_edge_index))
        target, target_edge_index, _, target_batch, _, _ = self.pool2_2(target, target_edge_index, None, target_batch)
        target2 = torch.cat([global_max_pool(target, target_batch), global_mean_pool(target, target_batch)], dim=1)
        target = F.leaky_relu(self.conv2_3(target, target_edge_index))
        target, target_edge_index, _, target_batch, _, _ = self.pool2_3(target, target_edge_index, None, target_batch)
        target3 = torch.cat([global_max_pool(target, target_batch), global_mean_pool(target, target_batch)], dim=1)
        target = target1 + target2 + target3
        x = torch.cat([source,target], dim=1)
        x = F.leaky_relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.leaky_relu(self.lin2(x))
        x = self.lin3(x)
        x = F.log_softmax(x, dim=-1)
        return x
