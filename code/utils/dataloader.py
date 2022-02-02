import torch
from torch_geometric.data import InMemoryDataset, Data
import numpy as np
import os
import pandas as pd
import random


class Dataset(InMemoryDataset):
    def __init__(self, root, dataset, transform=None, pre_transform=None):

        self.path = root
        self.dataset = dataset

        super(Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.statistical_info = torch.load(self.processed_paths[1])
        self.node_num = self.statistical_info['node_num']
        self.data_num = self.statistical_info['data_num']

    @property
    def raw_file_names(self):
        return '{}{}/{}.data'.format(self.path, self.dataset, self.dataset)

    @property
    def processed_file_names(self):
        return ['{}/{}.dataset'.format(self.dataset, self.dataset), \
                '{}/{}.info'.format(self.dataset, self.dataset)]


    def download(self):
        # Download to `self.raw_dir`.
        pass

    def read_data(self):
        # handle node and class 
        node_list_pos = []
        node_list_neg = []
        edge_pos = []
        edge_neg = []
        label_pos = []
        label_neg = []
        max_node_index = 0
        data_num = 0

        with open(self.datafile, 'r') as f:
            for line in f:
                data_num += 1
                data = line.split()
                # the first element is the label of the class
                y = float(data[0])
                #the rest of the elements are the nodes
                int_list = [int(data[i]) for i in range(len(data))[1:]]
                #edge_index = self.construct_full_edge_list(int_list)
                edge_index =[] 
                if y  > 0:
                    label_pos.append(float(data[0]))
                    edge_pos.append(edge_index)
                    node_list_pos.append(int_list)
                else:
                    label_neg.append(float(data[0]))
                    edge_neg.append(edge_index)
                    node_list_neg.append(int_list)

                if max_node_index < max(int_list):
                    max_node_index = max(int_list)


        return (node_list_pos, node_list_neg), (edge_pos, edge_neg), (label_pos, label_neg), max_node_index + 1, data_num


    def construct_full_edge_list(self, nodes):
        num_node = len(nodes)
        edge_list = [[],[]]         #first for sender, second for receiver
        sender_receiver_list = []
        for i in range(num_node):
            for j in range(num_node)[i:]:
                edge_list[0].append(i)
                edge_list[1].append(j)

        #return edge_list, sender_receiver_list
        return edge_list

    def process(self):
        self.datafile = self.raw_file_names
        self.node, edge, label, node_num, data_num = self.read_data()

        data_list = []
        processed_graphs = 0
        num_graphs = data_num 
        one_per = int(num_graphs/1000)
        percent = 0.0
        pos_num = len(self.node[0])
        neg_num = len(self.node[1])
        print(f"Pos num: {pos_num}, Neg num: {neg_num}")
        for k in range(2):
            for i in range(len(self.node[k])):
                if processed_graphs % one_per == 0:
                    print(f"Processing [{self.dataset}]: {percent/10.0}%, {processed_graphs}/{num_graphs}", end="\r")
                    percent += 1
                processed_graphs += 1 
                node_features = torch.LongTensor(self.node[k][i]).unsqueeze(1)
                edge_index = torch.LongTensor(edge[k][i])
                x = node_features
                y = torch.FloatTensor([label[k][i]])
                data = Data(x=x, edge_index=edge_index, y=y)
                data_list.append(data)


        #check whether foler path exist
        if not os.path.exists(f"{self.path}processed/{self.dataset}"):
            os.mkdir(f"{self.path}processed/{self.dataset}")
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        statistical_info = {'data_num': data_num, 'node_num': node_num, "pos_neg": (pos_num, neg_num)}
        torch.save(statistical_info, self.processed_paths[1])


    def node_M(self):
        return self.node_num
    
    def data_N(self):
        return self.data_num
