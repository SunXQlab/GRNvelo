import os
import ast
import torch
import pickle
import random
import warnings
from datetime import datetime, timedelta
import matplotlib.cm as cm
import numpy as np
import TFvelo as TFv
import scanpy as sc
import scvelo as scv
import anndata as ad
from collections import OrderedDict
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# CUDA support
# device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
warnings.filterwarnings("ignore")

seed = random.randint(1, 8888)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)


def loaddata(args):
    adata_filname = os.path.join(args.data_dir, args.dataset_data, f"{args.dataset_data}.h5ad")
    adata = sc.read_h5ad(adata_filname)

    datafilname = os.path.join(args.data_dir, args.dataset_data, f"{args.dataset_data}.npy")
    data = np.load(datafilname, allow_pickle=True)
    data_train = []
    for i in range(len(data)):
        data_train.append(torch.from_numpy(data[i]).type(torch.float32).to(device))
    return adata, data_train


def root_cell(data, cluster):
    time_points = len(data)
    cell_num = [data[i].size()[0] for i in range(time_points)]
    distance = torch.zeros((time_points - 1, cell_num[0])).to(device)

    for i in range(time_points - 1):
        for k in range(cell_num[0]):
            distance[i, k] = torch.dist(data[0][k], cluster[i + 1]).item()
    # The distances are normalized by row data
    transfer = MinMaxScaler(feature_range=(0, 1))  # The range can be changed
    nor_distance = torch.tensor(transfer.fit_transform(distance.cpu())).float().to(device)
    distance_all = torch.sum(nor_distance, dim=0)
    root_cell_index = torch.argmax(distance_all, 0).item()
    root_cell_exp = data[0][root_cell_index]

    return root_cell_index, root_cell_exp


# +++++++++++++++++++++++++++++++++++++++++++++++++++++ the deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.Tanh  # Tanh ReLU

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x_and_t):
        out = self.layers(x_and_t)
        return out


class Latent_time():
    def __init__(self, data, intial_lt, cell_num, rootcell_index, rootcell_exp, timepoints,
                 layers):

        # data
        self.intial_lt = intial_lt
        self.rootcell_exp = rootcell_exp.float().to(device)
        self.rootcell_index = torch.tensor(rootcell_index).to(device)
        self.data = torch.tensor(data).float().to(device)
        self.t = torch.linspace(0, 1, 2000).unsqueeze(1).requires_grad_(True).to(device)
        self.N_gene = data.size()[1]
        self.num_cell = torch.tensor(cell_num).long().to(device)
        self.timepoints = torch.tensor(timepoints).long().to(device)

        # settings
        self.K1 = torch.empty((self.N_gene, self.N_gene), dtype=torch.float32).uniform_(0.1, 10).float().requires_grad_(
            True).to(device)  # 0.5, 10
        self.K2 = torch.empty((self.N_gene, self.N_gene), dtype=torch.float32).uniform_(0.1, 10).float().requires_grad_(
            True).to(device)  # 0.5, 10
        # self.h1 = torch.ones(N_gene, N_gene).float().requires_grad_(True).to(device)
        # self.h2 = torch.ones(N_gene, N_gene).float().requires_grad_(True).to(device)
        self.a = torch.empty((self.N_gene, self.N_gene), dtype=torch.float32).uniform_(0, 1).float().requires_grad_(
            True).to(device)  # 0.1,5
        self.b = torch.empty((self.N_gene, self.N_gene), dtype=torch.float32).uniform_(0, 1).float().requires_grad_(
            True).to(device)  # 0.1,5
        self.d = torch.empty((1, self.N_gene), dtype=torch.float32).uniform_(0, 1).float().requires_grad_(True).to(
            device)  # 0.1,1 | 0, 0.5

        self.K1 = torch.nn.Parameter(self.K1)
        self.K2 = torch.nn.Parameter(self.K2)
        self.a = torch.nn.Parameter(self.a)
        self.b = torch.nn.Parameter(self.b)
        self.d = torch.nn.Parameter(self.d)

        # deep neural networks
        self.dnn = DNN(layers).to(device)
        self.dnn.register_parameter('K1', self.K1)
        self.dnn.register_parameter('K2', self.K2)
        self.dnn.register_parameter('a', self.a)
        self.dnn.register_parameter('b', self.b)
        self.dnn.register_parameter('d', self.d)

        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters(), lr=0.005)  # lr=0.001
        self.iter = 0

    def net_x(self):
        t = self.t
        x0 = self.rootcell_exp.repeat(t.size(0), 1)
        x_and_t = torch.cat([x0, t], dim=1)
        x_dnn = self.dnn(x_and_t)
        return x_dnn

    def pos(self, x, a, K, h):
        positive = a * (x ** h) / (K ** h + x ** h)
        return positive

    def neg(self, x, b, K, h):
        negative = b * (K ** h) / (K ** h + x ** h)
        return negative

    def net_f2(self):
        """ The pytorch autograd version of calculating residual """
        K1 = self.K1
        K2 = self.K2
        # h1 = self.h1
        # h2 = self.h2
        a = self.a
        b = self.b
        d = self.d
        t = self.t
        N_gene = self.N_gene
        x_dnn = self.net_x()
        for i in range(N_gene):
            x_t_pre = torch.autograd.grad(
                x_dnn[:, i], t,
                grad_outputs=torch.ones_like(x_dnn[:, i]),
                retain_graph=True,
                create_graph=True
            )[0]
            if i == 0:
                dx_dt = x_t_pre
            else:
                dx_dt = torch.cat((dx_dt, x_t_pre), 1)

        x_dnn = torch.where(x_dnn > 0, x_dnn, torch.full_like(x_dnn, 0))

        ## 之前的程序
        # 指数为2
        zero_ = torch.zeros(N_gene, N_gene).float().to(device)
        zero_x = torch.zeros(x_dnn.size(0), N_gene).float().to(device)
        pos_x = torch.where(x_dnn > 0, x_dnn, zero_x)
        x_repeat = pos_x.repeat(1, N_gene).reshape(t.size(0), N_gene, N_gene)

        pos = torch.where(a > 0, a, zero_)
        K1_repeat = K1.repeat(t.size(0), 1).reshape(t.size(0), N_gene, N_gene)
        pos_repeat = pos.repeat(t.size(0), 1).reshape(t.size(0), N_gene, N_gene)
        pos_pre = torch.pow(input=torch.div(K1_repeat, x_repeat + (1e-3)), exponent=2)
        pos_ = torch.sum(torch.mul(pos_repeat, torch.div(torch.ones(N_gene, N_gene).to(device), 1 + pos_pre)),
                         1)  # 按行求和

        neg = torch.where(b > 0, b, zero_)
        K2_repeat = K2.repeat(t.size(0), 1).reshape(t.size(0), N_gene, N_gene)
        neg_repeat = neg.repeat(t.size(0), 1).reshape(t.size(0), N_gene, N_gene)
        neg_pre = torch.pow(input=torch.div(x_repeat, K2_repeat + (1e-3)), exponent=2)
        neg_ = torch.sum(torch.mul(neg_repeat, torch.div(torch.ones(N_gene, N_gene).to(device), 1 + neg_pre)), 1)

        d_repeat = d.repeat(t.size(0), 1).reshape(t.size(0), N_gene)

        vv = pos_ + neg_ - torch.mul(d_repeat, x_dnn)
        f = vv - dx_dt

        return f, dx_dt

    def discrete_timepoint_latenttime(self):
        x_pred = self.net_x()
        x_pred = torch.where(x_pred > 0, x_pred, torch.full_like(x_pred, 0))
        z_pred = x_pred
        z_obs = self.data
        loss_cell_to_t = torch.sum((z_pred.unsqueeze(1) - z_obs.unsqueeze(0)) ** 2, dim=2)
        cell_latenttime = torch.argmin(loss_cell_to_t, dim=0)
        return cell_latenttime

    def cell_velo(self):
        cell_num = torch.sum(self.num_cell)
        cell_latenttime = self.discrete_timepoint_latenttime()
        velo = torch.zeros_like(self.data).to(device)
        v_all_t = self.net_f2()[1]
        for i in range(cell_num):
            velo[i] = v_all_t[cell_latenttime[i]]
        return velo

    def train(self, nIter, nIter_thres):
        self.dnn.train()
        loss_adam = []
        iteration_adam = []
        a = 0

        for epoch in range(nIter):
            x_pred = self.net_x()
            x_pred = torch.where(x_pred > 0, x_pred, torch.full_like(x_pred, 0))
            f_pred = self.net_f2()[0]
            loss2 = torch.mean(f_pred ** 2)
            if epoch < nIter_thres:
                cell_pred_exp = x_pred[self.intial_lt]
                loss1 = torch.mean((self.data - cell_pred_exp) ** 2)
                loss3 = 0

            else:
                cell_latenttime = self.discrete_timepoint_latenttime()
                # loss of latent time
                latenttime_mode = torch.zeros((1, self.timepoints)).long().to(device)
                nu = 0
                for i in range(self.timepoints):
                    latnettime_i = cell_latenttime[nu: nu + self.num_cell[i]].cpu()
                    latenttime_mode[0, i] = torch.tensor(np.argmax(np.bincount(latnettime_i))).to(device)
                    nu += self.num_cell[i].cpu()
                latenttime_diff = torch.diff(latenttime_mode)
                count_negative = torch.sum(latenttime_diff < 0).item()
                cell_pred_exp = x_pred[self.discrete_timepoint_latenttime()]
                loss1 = torch.mean((self.data - cell_pred_exp) ** 2)
                loss3 = count_negative

            loss = 1 * loss1 + 0.1 * loss2 + 1 * loss3
            # Backward and optimize
            self.optimizer_Adam.zero_grad()
            loss.backward()
            self.optimizer_Adam.step()
            iteration_adam.append(a)
            a += 1
            loss_adam.append(loss.item())
            if epoch % 100 == 0:
                # print('loss1: %.3e, loss2: %.3e' % (loss1.item(), loss2.item()))
                print('It: %d, Loss: %.3e' % (epoch, loss.item()))
        return iteration_adam, loss_adam


def cluster(data, num_cluster, time_points, umap_alldata, time_alldata, filname):
    '''
    The data at all time points are clustered by trajectory into num_cluster trajectories
    :param data: Data by point in time
    :param num_cluster: The data needs to be clustered into a number of clusters
    :param time_points: All the time
    :return: The result of clustering the clusters
    '''
    umap_alldata = torch.tensor(umap_alldata).to(device)
    time_alldata = torch.tensor(time_alldata).to(device)

    data_cluster_var = {}
    cluster_center_var = {}
    umap_cluster_var = {}
    for mm in range(num_cluster):
        umap_cluster_var[f'umap_cluster{mm + 1}'] = []
        data_cluster_var[f'data_cluster{mm + 1}'] = []
        cluster_center_var[f'cluster_center{mm + 1}'] = []

    cluster_ids_all = []
    ident = torch.zeros((time_points, num_cluster)).to(device)  # 4类
    all_clu_id = torch.tensor([k for k in range(num_cluster)]).to(device)
    for i in range(time_points):
        data_i = data[i]
        umap_data_i = umap_alldata[time_alldata == i]

        # # kmeans
        # data_size, dims, num_clusters = data_i.size(0), data_i.size(1), num_cluster  # 设定：数据集数量，数据集维数，聚类的类别数
        # cluster_ids_x, cluster_centers = kmeans(
        #     X=data_i, num_clusters=num_clusters, distance='euclidean', device=device)  # euclidean or cosine
        # cluster_ids_x = cluster_ids_x.to(device)
        # cluster_centers = cluster_centers.to(device)

        # Gaussian mixture distribution clustering
        model_umap = GaussianMixture(n_components=num_cluster, random_state=6)
        model_umap.fit(data_i.cpu().numpy())
        cluster_centers = torch.tensor(model_umap.means_).to(device)
        cluster_ids_x = torch.tensor(model_umap.predict(data_i.cpu().numpy())).to(device)

        cluster_ids_all.append(cluster_ids_x)
        if i == 0:
            for k in range(num_cluster):
                ident[i, k] = k
                cluster_center_var[f'cluster_center{k + 1}'].append(cluster_centers[k])
                data_cluster_var[f'data_cluster{k + 1}'].append(data_i[cluster_ids_x == k])
                umap_cluster_var[f'umap_cluster{k + 1}'].append(umap_data_i[cluster_ids_x == k])
        else:
            # num_cluster cells
            dis_all = torch.zeros((num_cluster, num_cluster)).to(device)
            for j in range(num_cluster):
                for k in range(num_cluster):
                    dis_all[k, j] = torch.dist(cluster_centers[j], cluster_center_var[f'cluster_center{k + 1}'][-1],
                                               p=2)
            a = dis_all.argmin(dim=0)
            has_duplicates = len(a) != len(set(a.tolist()))
            while has_duplicates:  # There are repeated elements in a
                unique_clu, dup_count = torch.unique(a, return_counts=True)
                duplicates = unique_clu[dup_count > 1]
                no_appear_clu = all_clu_id[~torch.isin(all_clu_id, a)].int()
                for ii in duplicates:  # Loop over existing categories
                    index_rep = torch.where(a == ii)[0]  # index of category ii in a
                    select_dis_all = dis_all[no_appear_clu][:, index_rep]
                    min_clus = torch.where(select_dis_all == select_dis_all.min())

                    # row: The newly matched class corresponds to the previous class col: i the class of time
                    row, col = no_appear_clu[min_clus[0].item()], index_rep[min_clus[1].item()]
                    a[col] = row
                    no_appear_clu = no_appear_clu[no_appear_clu != row]
                has_duplicates = len(a) != len(set(a.tolist()))

            for k in range(num_cluster):
                ident[i, k] = torch.where(a == k)[0].item()  # The zeros in ident[i, 0] correspond to cluster_center1
                cluster_center_var[f'cluster_center{k + 1}'].append(cluster_centers[torch.where(a == k)[0].item()])
                data_cluster_var[f'data_cluster{k + 1}'].append(data_i[cluster_ids_x == torch.where(a == k)[0].item()])
                umap_cluster_var[f'umap_cluster{k + 1}'].append(
                    umap_data_i[cluster_ids_x == torch.where(a == k)[0].item()])
        plot_cluster_timei(umap_alldata, umap_data_i, cluster_ids_x, i, filname)
    plot_cluster(umap_cluster_var, time_points, filname)
    return data_cluster_var, umap_cluster_var, cluster_center_var, cluster_ids_all, ident


def plot_gene(gene, cluster_num, data_cat_i, intial_lt, num_cell, filname):
    folder_name = os.path.join(filname, "gene_dynamics_cluster%d" % cluster_num)
    os.makedirs(folder_name, exist_ok=True)
    gene = gene.cpu().detach().numpy()
    data_cat_i = data_cat_i.cpu()
    plt.figure(figsize=(10, 6))
    color_scatter = [np.array([250, 187, 110]) / 255,
                     np.array([173, 219, 136]) / 255,
                     np.array([250, 199, 179]) / 255,
                     np.array([238, 68, 49]) / 255,
                     np.array([206, 223, 239]) / 255,
                     np.array([3, 149, 198]) / 255,
                     np.array([180, 180, 213]) / 255,
                     np.array([178, 143, 237]) / 255]

    for i in range(data_cat_i.size(1)):
        plt.plot(gene[:, i], color='black')
        plt.scatter(intial_lt, data_cat_i[:, i], s=10, alpha=0.3)
        plt.title('gene expression embedding %d' % (i + 1))
        plt.savefig(folder_name + "/gene_dynamics_cluster%d_gene_embedding%d.png" % (cluster_num, i + 1),
                    bbox_inches='tight', dpi=400)
        plt.close()

    for i in range(data_cat_i.size(1)):
        plt.figure(figsize=(10, 6))
        plt.plot(gene[:, i], c='gray')

        nu = 0
        labels = []
        labels_ins = []
        for j in range(len(num_cell)):
            lab = plt.scatter(intial_lt[nu: nu + num_cell[j]], data_cat_i[nu: nu + num_cell[j], i],
                              color=color_scatter[j],
                              s=10, alpha=0.3)
            labels_ins.append(lab)
            labels.append("t%s" % (j))
            nu += num_cell[j]

        plt.title('Gene expression embedding %d dynamics cluster%s' % (i + 1, cluster_num))
        plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0, handles=labels_ins, labels=labels)
        plt.savefig(folder_name + "/gene_dynamics_timepoints_cluster%d_gene_embedding%d.png" % (cluster_num, i + 1),
                    bbox_inches='tight',
                    dpi=400)
        plt.close()


def plot_velo(data, velocity, max_vv, filname):
    time_points = len(data)
    v_scale = 1 * max_vv.cpu().detach().numpy()
    aa = 1  # 3
    widths = 0.2  # arrow width
    fig, ax = plt.subplots()
    color_wanted = [np.array([250, 187, 110]) / 255,
                    np.array([173, 219, 136]) / 255,
                    np.array([250, 199, 179]) / 255,
                    np.array([238, 68, 49]) / 255,
                    np.array([206, 223, 239]) / 255,
                    np.array([3, 149, 198]) / 255,
                    np.array([180, 180, 213]) / 255,
                    np.array([178, 143, 237]) / 255]

    # add inferrred trajecotry
    for i in range(time_points):
        ax.scatter(data[i][:, 0], data[i][:, 1], s=aa * 10, linewidth=0, color=color_wanted[i], zorder=3)
        ax.quiver(data[i][:, 0], data[i][:, 1],
                  velocity[i].detach().numpy()[:, 0] / v_scale,
                  velocity[i].detach().numpy()[:, 1] / v_scale,
                  color='k', alpha=0.5, linewidths=widths, zorder=4)
    path = os.path.join(filname, 'Discrete cell velocity.png')
    plt.savefig(path, bbox_inches='tight', dpi=600)
    plt.close()


def plot_velo_umap(data_umap, velo_umap, time, filname):
    velo_umap = velo_umap * 100
    fig, ax = plt.subplots()
    widths = 0.2  # arrow width
    ax.scatter(data_umap[:, 0], data_umap[:, 1], s=10, linewidth=0, c=time, zorder=3, alpha=0.5)
    ax.quiver(data_umap[:, 0], data_umap[:, 1], velo_umap[:, 0], velo_umap[:, 1],
              color='k', alpha=0.5, linewidths=widths)
    path = os.path.join(filname, 'Discrete cell velocity_umap.png')
    plt.savefig(path, bbox_inches='tight', dpi=600)
    plt.close()


def plot_cluster(umap_cluster, timepoints, filname):
    # All categories together
    fig, ax = plt.subplots()
    colors = cm.get_cmap('viridis', 4)
    col_list = [colors(i / (4 - 1)) for i in range(4)]

    for umap_traj_i, clu in zip(umap_cluster, range(len(umap_cluster))):
        for i in range(timepoints):
            ax.scatter(umap_cluster[umap_traj_i][i].cpu()[:, 0], umap_cluster[umap_traj_i][i].cpu()[:, 1],
                       color=col_list[clu], s=10, marker='o',
                       alpha=0.1)
    plt.tight_layout()
    path = os.path.join(filname, "Discrete cell clusters.png")
    plt.savefig(path, bbox_inches='tight', dpi=600)
    plt.close()

    # Draw the cells for each trajectory separately
    for j in range(len(umap_cluster)):
        fig, ax = plt.subplots()
        for umap_traj_i, clu in zip(umap_cluster, range(len(umap_cluster))):
            if clu == j:
                for i in range(timepoints):
                    ax.scatter(umap_cluster[umap_traj_i][i].cpu()[:, 0], umap_cluster[umap_traj_i][i].cpu()[:, 1],
                               color=col_list[clu], s=10, marker='o',alpha=0.1)
            else:
                for i in range(timepoints):
                    ax.scatter(umap_cluster[umap_traj_i][i].cpu()[:, 0], umap_cluster[umap_traj_i][i].cpu()[:, 1],
                               color='gainsboro', s=10, marker='o',alpha=0.1)
        plt.tight_layout()
        path = os.path.join(filname, "DiscreteCellCluster%d.png" % (j+1))
        plt.savefig(path, bbox_inches='tight', dpi=600)
        plt.close()


def plot_cluster_timei(umap_allcell, umap_timei, cluster_ids_x, timepoint, filname):
    umap_allcell = umap_allcell.cpu()
    umap_timei = umap_timei.cpu()
    cluster_ids_x = cluster_ids_x.cpu()
    fig, ax = plt.subplots()

    # add inferrred trajecotry
    ax.scatter(umap_allcell[:, 0], umap_allcell[:, 1], color='gainsboro', s=10, marker='o',
               facecolors='none', edgecolors='gainsboro', alpha=0.5)
    ax.scatter(umap_timei[:, 0], umap_timei[:, 1], c=cluster_ids_x, s=10, edgecolors='none', cmap='viridis',
               alpha=0.5)
    plt.tight_layout()
    path = os.path.join(filname, "Discrete cell cluster_time%s.png" % (timepoint))
    plt.savefig(path, bbox_inches='tight', dpi=600)
    plt.close()


def plot_latenttime(data, latenttime, num_cluster, filename):
    time_points = len(data)
    xx_scale = np.linspace(0, 2000, 11)
    xx = np.linspace(100, 1900, 10)
    fig, axs = plt.subplots(time_points, 1, figsize=(12, 8))
    for i in range(time_points):
        yy = np.zeros_like(xx)
        aa = latenttime[i].numpy()
        for j in range(np.size(xx)):
            yy[j] = np.sum((aa >= xx_scale[j]) & (aa < xx_scale[j + 1]))
        axs[i].bar(xx, yy, width=0.8)
        axs[i].set_title('Latent Time of at time t_%s' % i)
    plt.tight_layout()
    path = os.path.join(filename, "Latent_time_cluster%s.png" % (num_cluster))
    plt.savefig(path, bbox_inches='tight', dpi=600)
    plt.close()


def itial_latenttime(data, cell_num, rootcell_index):
    # For each time point, the mean of the latent time of all cells is assigned
    cell_feat_range = torch.linspace(0, 1, steps=len(cell_num) + 1).to(device)
    start = cell_feat_range[0]
    end = cell_feat_range[-1]
    mid = (cell_feat_range[:-1] + cell_feat_range[1:]) / 2
    cell_feat_nor = torch.cat((start.unsqueeze(0), mid, end.unsqueeze(0)))

    intial_lt = torch.zeros((data.size(0), 1)).to(device)
    nu = 0
    for i in range(len(cell_num)):
        random_values = torch.normal(cell_feat_nor[i + 1], 0.05, size=(cell_num[i], 1)).to(device)
        intial_lt[nu:nu + cell_num[i]] = torch.clamp(random_values, min=cell_feat_nor[0], max=cell_feat_nor[-1])
        nu += cell_num[i]
    intial_lt[rootcell_index] = 0  # The time of the Root cell is set to 0
    intial_lt = torch.round(intial_lt * (2000 - 1)).long().t().squeeze()
    return intial_lt


def plot_latenttime_allcell(adata, data, num_cluster, filname):
    time_alldata = adata.obs['time']
    umap_alldata = torch.tensor(adata.obsm['umap_10to2']).to(device)
    timepoints = len(data)

    for i in range(num_cluster):
        file_path_lt = os.path.join(filname, 'latent_time%d.pkl' % (i+1))
        with open(file_path_lt, 'rb') as file:
            locals()[f'latent_time{int(i + 1)}'] = pickle.load(file)

    file_path_lt = os.path.join(filname, 'cluster_ids_allcell.pkl')
    with open(file_path_lt, 'rb') as file:
        cluster_ids_allcell = pickle.load(file)

    file_path_lt = os.path.join(filname, 'cluster_sign_ids.pkl')
    with open(file_path_lt, 'rb') as file:
        cluster_sign_ids = pickle.load(file).cpu().numpy()

    timedetails_pred = []
    for i in range(timepoints):
        time_detail_pre_i = np.zeros_like(data[i][:, 0].cpu())
        for j in range(num_cluster):
            time_detail_pre_i[cluster_ids_allcell[i].cpu().numpy() == cluster_sign_ids[i][j]] = \
            locals()[f'latent_time{int(j + 1)}'][i]
        timedetails_pred.append(time_detail_pre_i)
        time_detail_real_i = np.ones_like(time_detail_pre_i) * i

    data_numpy = list_turn_numpy(data)[0]
    lt_np = np.concatenate(timedetails_pred)
    path = os.path.join(filname, 'latenttime_np.npy')
    np.save(path, lt_np)
    # plot latent time
    plot_latenttime3(umap_alldata, time_alldata, timedetails_pred, timepoints, filname)


def list_turn_numpy(data):  # Combine all the data in the list into a single numpy file
    numcell_accum = np.zeros((1, len(data) + 1))
    data_all = data[0].cpu().detach().numpy()
    numcell_accum[0, 0] = 0
    numcell_accum[0, 1] = np.size(data[0], 0)
    for k in range(len(data) - 1):
        data_next = data[k + 1].cpu().detach().numpy()
        numcell_accum[0, k + 2] = numcell_accum[0, k + 1] + np.size(data_next, 0)
        data_all = np.concatenate((data_all, data_next), axis=0)
    return data_all, numcell_accum

def plot_latenttime3(data_umap, time_allcell, latenttime, timepoints, filname):
    fig, ax = plt.subplots()
    data_umap = data_umap.cpu().numpy()
    aa = 1
    plt.tight_layout()
    plt.axis('on')
    for i in range(timepoints):
        sc = ax.scatter(data_umap[time_allcell==i][:, 0], data_umap[time_allcell==i][:, 1], s=aa * 10,
                   linewidth=0, c=latenttime[i], cmap='gnuplot', zorder=3, alpha=0.5)

    cbar = fig.colorbar(
        sc,
        shrink=0.3, pad=0.02, location='right', anchor=(0.0, 0.2))
    cbar.outline.set_edgecolor('none')
    cbar.set_ticks([0, 2000])
    cbar.set_ticklabels(['0', '1'])
    cbar.ax.tick_params(labelsize=12)

    # Hides the default axes
    ax.set(xlabel=None, ylabel=None)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    arrow_length = 0.25
    ax.annotate('',
                xy=(arrow_length, 0),
                xytext=(-0.005, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1),
                xycoords='axes fraction')
    ax.text(arrow_length / 2, -0.03, 'UMAP 1',
            size=16,
            transform=ax.transAxes,
            ha='center',
            va='center')
    ax.annotate('',
                xy=(0, arrow_length),
                xytext=(0, -0.005),
                arrowprops=dict(arrowstyle='->', color='black', lw=1),
                xycoords='axes fraction')
    ax.text(-0.03, arrow_length / 2, 'UMAP 2',
            size=16,
            transform=ax.transAxes,
            ha='center',
            va='center',
            rotation=90)

    plt.tight_layout()
    path = os.path.join(filname, 'latenttime_initialdata.png')
    plt.savefig(path, bbox_inches='tight', dpi=600)
    plt.close()

def plot_LossAdam(loss_adam, filname):
    ## loss
    path = os.path.join(filname, 'Loss_adam.pt')
    torch.save(loss_adam, path)
    plt.figure(figsize=(10, 4))
    plt.plot(loss_adam)  # linestyle为线条的风格（共五种）,color为线条颜色
    plt.title('Loss of Adam')
    path = os.path.join(filname, 'Loss_adam.png')
    plt.savefig(path, bbox_inches='tight', dpi=600)
    plt.close()


def main(args):
    seed = args.VelocityGeneration_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set the seed for Python's built-in random number generator
    random.seed(seed)
    # Set the Numpy random seed
    np.random.seed(seed)

    output_velocity_filname = os.path.join(args.out_dir, args.dataset_data, 'DiscreteVelocity')

    if not os.path.exists(output_velocity_filname):
        os.makedirs(output_velocity_filname)

    start = datetime.now()  # 记录时间 4-1
    # load dataset
    adata, data = loaddata(args)
    time_points = len(data)
    time_alldata = adata.obs['time']
    umap_alldata = adata.obsm['umap_10to2']

    umap_alldata_list = []
    for l in range(time_points):
        umap_alldata_list.append(umap_alldata[time_alldata == l])

    num_cluster = args.VelocityGeneration_num_cluster
    iters = args.VelocityGeneration_iters  # iters must >= iters_thres
    iters_thres = args.VelocityGeneration_iters_thres
    data_cluster_var, umap_cluster_var, cluster_center_var, cluster_ids_all, ident = (
        cluster(data, num_cluster, time_points, umap_alldata=umap_alldata, time_alldata=time_alldata, filname=output_velocity_filname))

    file_path_id = os.path.join(output_velocity_filname, 'cluster_ids_allcell.pkl')
    with open(file_path_id, 'wb') as file:
        pickle.dump(cluster_ids_all, file)
    file_path_sign = os.path.join(output_velocity_filname, 'cluster_sign_ids.pkl')
    with open(file_path_sign, 'wb') as file:
        pickle.dump(ident, file)

    # Sets the initial value of velocity
    v_allcell = []
    for i in range(len(data)):
        v_allcell.append(torch.zeros_like(data[i]).cpu())

    # Discrete velocity is trained by trajectory
    for tt in range(num_cluster):
        print('\nTraining the latent time of Trajectory %d/%d...' % (tt + 1, num_cluster))
        gene_num = data_cluster_var[f'data_cluster{tt + 1}'][0].size()[1]
        cell_num = [data_cluster_var[f'data_cluster{tt + 1}'][i].size()[0] for i in range(time_points)]
        rootcell_index, root_cell_exp = root_cell(data_cluster_var[f'data_cluster{tt + 1}'],
                                                  cluster_center_var[f'cluster_center{tt + 1}'])

        hidden_layers = ast.literal_eval(args.VelocityGeneration_hidden_layers)
        layers = [gene_num + 1] + hidden_layers + [gene_num]
        t = torch.linspace(0, 1, 2000).unsqueeze(1).to(device)

        data_cat_i = data_cluster_var[f'data_cluster{tt + 1}'][0]
        for i in range(time_points - 1):
            data_cat_i = torch.cat([data_cat_i, data_cluster_var[f'data_cluster{tt + 1}'][i + 1]], dim=0)
        # Initialize the latent time
        intial_lt = itial_latenttime(data_cat_i, cell_num, rootcell_index)

        model = Latent_time(data_cat_i, intial_lt, cell_num, rootcell_index, root_cell_exp, time_points, layers)
        iteration_adam, loss_adam = model.train(iters, iters_thres)
        path_model = os.path.join(output_velocity_filname, 'model_dis_velo.pth')
        torch.save(model, path_model)
        # model = torch.load("model_dis_velo.pth") # Import the saved model and parameters
        plot_LossAdam(loss_adam, output_velocity_filname)
        gene_pre = model.net_x()
        gene_pre = torch.where(gene_pre > 0, gene_pre, torch.full_like(gene_pre, 0)).detach()  # .numpy()
        cluster_num = tt + 1

        latenttime_cluster2 = []
        nu = 0
        velo_tem = model.cell_velo().cpu()
        if iters <= iters_thres:
            lattim_tem = model.intial_lt.cpu()
        else:
            lattim_tem = model.discrete_timepoint_latenttime().cpu()
        for i in range(time_points):
            v_allcell[i][cluster_ids_all[i] == ident[i, tt]] = velo_tem[nu: nu + cell_num[i]]  # 第tt类数据
            latenttime_cluster2.append(lattim_tem[nu: nu + cell_num[i]])
            nu += cell_num[i]

        plot_gene(gene_pre, tt + 1, data_cat_i, lattim_tem, cell_num, output_velocity_filname)  # 画基因的动力学图
        plot_latenttime(data, latenttime_cluster2, cluster_num, output_velocity_filname)

        file_path_v = os.path.join(output_velocity_filname, 'velocity_allcell.pkl')
        with open(file_path_v, 'wb') as file:
            pickle.dump(v_allcell, file)
        file_path_l = os.path.join(output_velocity_filname, 'latent_time%d.pkl'%(tt+1))
        with open(file_path_l, 'wb') as file:
            pickle.dump(latenttime_cluster2, file)

        max_v = []
        for i in range(len(v_allcell)):
            max_v.append(torch.max(v_allcell[i]))
        max_vv = max(max_v)

    plot_velo(umap_alldata_list, v_allcell, max_vv, output_velocity_filname)


    plot_latenttime_allcell(adata, data, num_cluster, output_velocity_filname)

    end = datetime.now()
    time_ = end - start
    print('')
    print('Run Time:', time_)
    path = os.path.join(output_velocity_filname, 'seed.txt')
    with open(path, 'w') as file:
        file.write(str(seed))

if __name__ == "__main__":
    main(input())
