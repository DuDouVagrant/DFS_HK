import torch
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from random import shuffle
import random
from prompt_graph.utils import mkdir, graph_views
from prompt_graph.data import load4graph, NodePretrain
from torch.optim import Adam
import time
import numpy as np
import warnings
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import os

from .base import PreTrain

class GraphCL(PreTrain):
    def __init__(self, *args, hid_dim = 16, **kwargs):    # hid_dim=16
        super().__init__(*args, **kwargs)
        self.hid_dim = hid_dim 
        self.load_graph_data()
        self.initialize_gnn(self.input_dim, hid_dim)  
        self.projection_head = torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.hid_dim),
                                                   torch.nn.ReLU(inplace=True),
                                                   torch.nn.Linear(self.hid_dim, self.hid_dim)).to(self.device)
    def load_graph_data(self):
        if self.dataset_name in ['PubMed', 'Citeseer', 'Cora', 'Cora_ml','Computers', 'Photo', 'Reddit', 'WikiCS', 'Flickr', 'Wisconsin','ogbn-arxiv']:
            self.graph_list, self.input_dim = NodePretrain(dataname = self.dataset_name, preprocess_method = self.preprocess_method, num_parts = 200)
        else:
            self.input_dim, self.out_dim, self.graph_list= load4graph(self.dataset_name, pretrained=True)
    
    def get_loader(self, graph_list, batch_size,aug1=None, aug2=None, aug_ratio=None):
        if len(graph_list) % batch_size == 1:
            raise KeyError(
                "batch_size {} makes the last batch only contain 1 graph, \n which will trigger a zero bug in GraphCL!")
        
        shuffle(graph_list)
        if aug1 is None:
            aug1 = random.sample(['dropN', 'permE', 'maskN'], k=1)
        if aug2 is None:
            aug2 = random.sample(['dropN', 'permE', 'maskN'], k=1)
        if aug_ratio is None:
            aug_ratio = random.randint(1, 3) * 1.0 / 10  

        print("===graph views: {} and {} with aug_ratio: {}".format(aug1, aug2, aug_ratio))

        view_list_1 = []
        view_list_2 = []
        for g in graph_list:
            view_g = graph_views(data=g, aug=aug1, aug_ratio=aug_ratio)
            view_g = Data(x=view_g.x, edge_index=view_g.edge_index)
            view_list_1.append(view_g)
            view_g = graph_views(data=g, aug=aug2, aug_ratio=aug_ratio)
            view_g = Data(x=view_g.x, edge_index=view_g.edge_index)
            view_list_2.append(view_g)

        loader1 = DataLoader(view_list_1, batch_size=batch_size, shuffle=False, num_workers=1) 
        loader2 = DataLoader(view_list_2, batch_size=batch_size, shuffle=False, num_workers=1) 

        return loader1, loader2
    
    def forward_cl(self, x, edge_index, batch):
        x = self.gnn(x, edge_index, batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss = - torch.log(pos_sim / (sim_matrix.sum(dim=1) + 1e-4)).mean()
        return loss

    def train_graphcl(self, loader1, loader2, optimizer):
        self.train()
        train_loss_accum = 0
        total_step = 0
        for step, batch in enumerate(zip(loader1, loader2)):
            batch1, batch2 = batch
            optimizer.zero_grad()
            x1 = self.forward_cl(batch1.x.to(self.device), batch1.edge_index.to(self.device), batch1.batch.to(self.device))
            x2 = self.forward_cl(batch2.x.to(self.device), batch2.edge_index.to(self.device), batch2.batch.to(self.device))
            loss = self.loss_cl(x1, x2)

            loss.backward()
            optimizer.step()

            train_loss_accum += float(loss.detach().cpu().item())
            total_step = total_step + 1

        return train_loss_accum / total_step

    # --- 核心修复：直接读取真实验证数据，避开无监督预训练中被屏蔽的标签 ---
    def eval_downstream_task(self):
        self.eval() 
        
        # 1. 安全获取带有真实 Label 的全量图数据
        data = None
        data_path = f'./data_{self.dataset_name}.pt'
        
        # 优先读取本地你处理好的文件
        if os.path.exists(data_path):
            try:
                loaded = torch.load(data_path, map_location='cpu')
                data = loaded[0] if isinstance(loaded, tuple) else loaded
            except:
                pass
        
        # 本地找不到或异常，兜底使用 PyG 官方数据集重新下载加载
        if data is None or getattr(data, 'y', None) is None:
            try:
                from torch_geometric.datasets import Planetoid, Amazon, Coauthor
                if self.dataset_name in ['Cora', 'Citeseer', 'PubMed']:
                    data = Planetoid(root='./data', name=self.dataset_name)[0]
                elif self.dataset_name in ['Computers', 'Photo']:
                    data = Amazon(root='./data', name=self.dataset_name)[0]
                elif self.dataset_name in ['CS', 'Physics']:
                    data = Coauthor(root='./data', name=self.dataset_name)[0]
                else:
                    data = self.graph_list[0] # 最后的挣扎
            except:
                data = self.graph_list[0]

        data = data.to(self.device)

        # 2. 提取全量节点特征
        with torch.no_grad():
            try:
                emb = self.gnn(data.x, data.edge_index)
            except TypeError:
                emb = self.gnn(data.x, data.edge_index, None)
            
            if isinstance(emb, tuple):
                emb = emb[0]
            embs = emb.cpu().numpy()

        # 获取标签，如果实在没有则报 0
        lbls = data.y.cpu().numpy() if (hasattr(data, 'y') and data.y is not None) else np.zeros(data.x.size(0))
        if lbls.ndim > 1:
            lbls = lbls.flatten()

        # 防御机制：如果标签全是0，说明读图失败，优雅跳过避免报错
        if lbls.max() == 0:
            print("⚠️ 未找到有效节点标签 (y)，无法进行下游分类准确率评估，返回 0...")
            return 0.0

        # 适配离散与连续型标签
        if lbls.dtype in [np.float32, np.float64] or type(lbls[0]) in [float, np.float64, np.float32] or lbls.max() % 1 != 0:
            bins = np.percentile(lbls, [20, 40, 60, 80])
            lbls = np.digitize(lbls, bins)
        else:
            lbls = lbls.astype(int)

        num_classes = int(lbls.max() + 1)
        
        # 3. K-shot 划分逻辑
        shot_num = 5 
        train_idx = []
        for c in range(num_classes):
            idx_c = np.where(lbls == c)[0]
            if len(idx_c) == 0: continue
            np.random.shuffle(idx_c)
            train_idx.extend(idx_c[:shot_num])

        remaining_idx = np.setdiff1d(np.arange(len(lbls)), train_idx)
        np.random.shuffle(remaining_idx)

        val_size = min(500, len(remaining_idx) // 3)
        test_size = min(1000, len(remaining_idx) - val_size)
        test_idx = remaining_idx[val_size:val_size + test_size]

        x_train, y_train = embs[train_idx], lbls[train_idx]
        x_test, y_test = embs[test_idx], lbls[test_idx]

        # 4. 下游 SVC 线性评估（加上动态 CV 安全网）
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            
            # 计算 y_train 中最少类的样本数，防止 StratifiedKFold 崩溃
            counts = np.bincount(y_train)
            counts = counts[counts > 0]
            min_class_count = counts.min() if len(counts) > 0 else 0
            
            # 只有当每个类至少有2个样本时，才敢放心做交叉验证
            if min_class_count >= 2:
                try:
                    params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
                    cv_folds = min(5, min_class_count)
                    classifier = GridSearchCV(SVC(), params, cv=cv_folds, scoring='accuracy', verbose=0)
                    classifier.fit(x_train, y_train)
                except Exception:
                    # 如果还是发生意外，降维打击：直接放弃网格搜索，使用坚挺的泛化参数 C=10
                    classifier = SVC(C=10)
                    classifier.fit(x_train, y_train)
            else:
                # 极端少样本情况（1-shot），坚决不做 CV
                classifier = SVC(C=10)
                classifier.fit(x_train, y_train)

            acc_test = accuracy_score(y_test, classifier.predict(x_test))

        return acc_test

    def pretrain(self, batch_size=10, aug1='dropN', aug2="permE", aug_ratio=None, lr=0.01, decay=0.0001):
        epochs = self.epochs
        self.to(self.device)
        if self.dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY', 'ogbg-ppa', 'DD']:
            batch_size = 512
            
        # 保护机制：如果加载的图数量过小（比如全量 Cora 单图），强制让 batch_size 自适应
        if len(self.graph_list) < batch_size:
            batch_size = max(1, len(self.graph_list))

        # GraphCL 是无监督的，直接使用全部无标签图数据进行对比学习
        loader1, loader2 = self.get_loader(self.graph_list, batch_size, aug1=aug1, aug2=aug2)

        print('start training {} | {} | {}...'.format(self.dataset_name, 'GraphCL', self.gnn_type))

        optimizer = Adam(self.parameters(), lr=lr, weight_decay=decay)

        train_loss_min = 1000000
        patience = 10000000
        cnt_wait = 0
        
        for epoch in range(1, epochs + 1):  
            train_loss = self.train_graphcl(loader1, loader2, optimizer)

            # 每 50 个 epoch 输出一次结果
            if epoch % 50 == 0:
                # 评估时，内部会自动安全提取节点并切分 5-shot
                test_acc = self.eval_downstream_task()
                print("***epoch: {}/{} | train_loss: {:.8}".format(epoch, epochs, train_loss))
                print("   ---> [Downstream] epoch: {}/{} | SVC test_acc: {:.4f}".format(epoch, epochs, test_acc))

            if train_loss_min > train_loss:
                train_loss_min = train_loss
                cnt_wait = 0
            else:
                cnt_wait += 1
                if cnt_wait == patience:
                    print('-' * 100)
                    print('Early stopping at '+str(epoch) +' eopch!')
                    break
        
        file_suffix = "{}.{}.{}.{}_hidden_dim.aug1_{}.aug2_{}.lr_{}.pth".format(
                    self.dataset_name, 'GraphCL', self.gnn_type, str(self.hid_dim), aug1, aug2, str(lr)
                )
        
        torch.save(self.gnn.state_dict(), "./pre_trained_model/" + file_suffix)
        print("+++model saved ! " + file_suffix)