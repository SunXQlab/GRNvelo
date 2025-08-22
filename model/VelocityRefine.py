import os
import ast
import torch
import joblib
import pickle
import random
import warnings
import scanpy as sc
import numpy as np
import anndata as ad
import seaborn as sns
import scipy.stats as stats
from functools import partial
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from collections import OrderedDict
from TorchDiffEqPack import odesolve
from datetime import datetime
from matplotlib.ticker import FormatStrFormatter
from torchquad import MonteCarlo
from matplotlib.ticker import MaxNLocator
import sys
sys.path.append('Path to AE folder')

# CUDA support
# device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
warnings.filterwarnings("ignore")


def loaddata(args):
    adata_filname = os.path.join(args.data_dir, args.dataset_data, f"{args.dataset_data}.h5ad")
    adata = sc.read_h5ad(adata_filname)

    datafilname = os.path.join(args.data_dir, args.dataset_data, f"{args.dataset_data}.npy")
    data = np.load(datafilname, allow_pickle=True)
    data_train=[]
    for i in range(len(data)):
        data_train.append(torch.from_numpy(data[i]).type(torch.float32).to(device))
    return adata, data_train


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


class DNNrho(torch.nn.Module):
    def __init__(self, layers):
        super(DNNrho, self).__init__()

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

    def forward(self, x_and_delta_t):
        out = self.layers(x_and_delta_t)
        return out


class DNNg(torch.nn.Module):
    def __init__(self, layers):
        super(DNNg, self).__init__()

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

    def forward(self, x_and_delta_t):
        out = self.layers(x_and_delta_t)
        return out


class PhysicsInformedNN():
    def __init__(self, t, samples_x, samples_v, samples_rho, layers1, layers2):
        # data ——x; t
        self.t = torch.tensor(t).float().requires_grad_(True).to(device)
        self.x = torch.tensor(samples_x).float().requires_grad_(True).to(device)
        self.samples_v = torch.tensor(samples_v).float().to(device)
        self.samples_rho = torch.tensor(samples_rho).float().to(device)
        self.timepoints = self.t.size()[0]
        self.genenum = self.x.size(-1)

        # settings
        self.D = torch.empty((1, 1), dtype=torch.float32).uniform_(1.5, 2).float().requires_grad_(True).to(device)
        self.alpha = torch.empty((1, 1), dtype=torch.float32).uniform_(0, 1).float().requires_grad_(True).to(device)
        self.M = torch.empty((1, 1), dtype=torch.float32).uniform_(5000, 15000).float().requires_grad_(True).to(device)

        self.D = torch.nn.Parameter(self.D)
        self.alpha = torch.nn.Parameter(self.alpha)
        self.M = torch.nn.Parameter(self.M)
        # deep neural networks
        self.dnn = DNN(layers1).to(device)
        self.dnnrho = DNNrho(layers2).to(device)
        self.dnng = DNNg(layers2).to(device)
        self.dnn.register_parameter('D', self.D)
        self.dnn.register_parameter('alpha', self.alpha)
        self.dnn.register_parameter('M', self.M)
        self.dnnrho.register_parameter('D', self.D)
        self.dnnrho.register_parameter('alpha', self.alpha)
        self.dnnrho.register_parameter('M', self.M)
        self.dnng.register_parameter('D', self.D)
        self.dnng.register_parameter('alpha', self.alpha)
        self.dnng.register_parameter('M', self.M)

        # optimizers: using the same settings
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=0.05,
            max_iter=50000,  # 50000
            max_eval=50000,  # 50000
            history_size=50,
            tolerance_grad=1e-5,  # 1e-5
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"  # can be "strong_wolfe"
        )

        # optimizers: using the same settings
        self.optimizerrho = torch.optim.LBFGS(
            self.dnnrho.parameters(),
            lr=0.05,
            max_iter=50000,  # 50000
            max_eval=50000,  # 50000
            history_size=50,
            tolerance_grad=1e-5,  # 1e-5
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"  # can be "strong_wolfe"
        )

        # optimizers: using the same settings
        self.optimizerg = torch.optim.LBFGS(
            self.dnng.parameters(),
            lr=0.05,
            max_iter=50000,  # 50000
            max_eval=50000,  # 50000
            history_size=50,
            tolerance_grad=1e-5,  # 1e-5
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"  # can be "strong_wolfe"
        )

        self.optimizer_Adamv = torch.optim.Adam(self.dnn.parameters(), lr=0.001)  # Default lr=1e-3
        self.optimizer_Adamrho = torch.optim.Adam(self.dnnrho.parameters(), lr=0.001)
        self.optimizer_Adamg = torch.optim.Adam(self.dnng.parameters(), lr=0.001)
        self.iter = 0

        self.loss_LBFGS = []
        self.iteration_LBFGS = []

    def net_v(self, t, x):
        t0 = t.repeat(x.size(0), 1)  # size=[100,1]
        x_and_t = torch.cat([x, t0], dim=1)
        v = self.dnn(x_and_t)
        return v

    def net_rho(self, t, x):
        t0 = t.repeat(x.size(0), 1)  # size=[100,1]
        if x.device != device:
            x = x.to(device)
        x_and_t = torch.cat([x, t0], dim=1)
        rho = self.dnnrho(x_and_t)
        return rho

    def net_g(self, t, x):
        t0 = t.repeat(x.size(0), 1)  # size=[100,1]
        x_and_t = torch.cat([x, t0], dim=1)
        g = self.dnng(x_and_t)
        return g

    def net_f(self, x_last, t):
        """ The pytorch autograd version of calculating residual """
        v = self.net_v(t, x_last)
        rho = self.net_rho(t, x_last)
        g = self.net_g(t, x_last)
        func_rho = partial(self.net_rho, t)
        mc = MonteCarlo()
        # Compute the function integral by sampling 10000 points over domain
        integral_rho = mc.integrate(
            func_rho,
            dim=x_last.size(1),
            N=10000,
            integration_domain=[[0, 1] for i in range(self.genenum)],  # 这个跟基因的个数有关系
            backend="torch", )

        rho_t = torch.autograd.grad(rho, t, grad_outputs=torch.ones_like(rho),
                                    retain_graph=True, create_graph=True)[0]
        rho_x = torch.autograd.grad(rho, x_last, grad_outputs=torch.ones_like(rho),
                                    retain_graph=True, create_graph=True)[0]
        rho_xx = torch.autograd.grad(outputs=rho_x, inputs=x_last, grad_outputs=torch.ones_like(rho_x),
                                     create_graph=True, retain_graph=True)[0].sum(dim=1)
        v_x = torch.autograd.grad(outputs=v, inputs=x_last, grad_outputs=torch.ones_like(v),
                                  create_graph=True, retain_graph=True)[0].sum(dim=1)
        rho_x_v = (rho_x * v).sum(dim=1)

        inte_rho = integral_rho.repeat(rho.size(0), 1)
        f = self.D * rho_xx - rho * v_x - rho_x_v + g * rho * (1 - inte_rho / self.M) - rho_t

        return f, g * (1 - inte_rho / self.M)

    def exp_g(self, x0, t1):
        func_g = partial(self.net_g, x=x0)
        y00 = torch.zeros_like(x0)[:, 0]
        exp_g_result = odeint(func_g, y0=y00, t=torch.tensor([t1]), atol=1e-5, rtol=1e-5, method='euler')
        return exp_g_result

    def train(self, nIter):
        self.dnn.train()
        self.dnnrho.train()
        self.dnng.train()
        loss_adam = []
        iteration_adam = []
        a = 0
        # configure training options
        options = {}
        options.update({'method': 'Dopri5'})
        options.update({'h': None})
        options.update({'rtol': 1e-3})
        options.update({'atol': 1e-5})
        options.update({'print_neval': False})
        options.update({'neval_max': 1000000})
        options.update({'safety': None})
        func_v = self.net_v

        for epoch in range(nIter):
            # if epoch % 100 == 0:
                # print("--------------------------------------------")
            for num_time in range(self.timepoints - 1):  # self.timepoints
                pred_v = self.net_v(self.t[num_time], self.x[num_time])
                pred_rho = self.net_rho(self.t[num_time], self.x[num_time])
                pred_f = self.net_f(self.x[num_time], self.t[num_time])[0]
                loss_rho = torch.mean((self.samples_rho[num_time] - pred_rho) ** 2)
                cos_similarity = torch.nn.functional.cosine_similarity(self.samples_v[num_time], pred_v, dim=1,
                                                                       eps=1e-8)
                loss_v_size = torch.mean((self.samples_v[num_time] - 10 * pred_v) ** 2)
                loss_v_diret = torch.mean(1 - cos_similarity)
                loss_f = torch.mean(pred_f ** 2)
                loss = 1 * loss_rho + 1 * loss_f + 1 * loss_v_size + 1 * loss_v_diret  # + loss_WFR

                # Backward and optimize
                self.optimizer_Adamv.zero_grad()
                self.optimizer_Adamrho.zero_grad()
                self.optimizer_Adamg.zero_grad()
                loss.backward()
                self.optimizer_Adamv.step()
                self.optimizer_Adamrho.step()
                self.optimizer_Adamg.step()
                iteration_adam.append(a)
                a += 1
                loss_adam.append(loss.item())
                # print('It: %d, Loss: %.3e' % (epoch, loss.item()))
                if epoch % 100 == 0:
                    print('It: %d, Loss: %.3e' % (epoch, loss.item()))
                #     print('loss_rho: %.3e, loss_v_size: %.3e, loss_f: %.3e, loss_v_direct: %.3e'
                #           % (loss_rho.item(), loss_v_size.item(), loss_f.item(), loss_v_diret.item()))

        # Backward and optimize
        # self.optimizer.step(self.loss_func)
        return iteration_adam, loss_adam

    def predict(self, time_scale, x, delta):
        t = torch.tensor(time_scale).float().requires_grad_(True).to(device)
        x = torch.tensor(x).float().to(device)
        pre_v = torch.zeros(x.size(0), x.size(1), x.size(2)).to(device)
        pre_rho = torch.zeros(x.size(0), x.size(1)).unsqueeze(-1).to(device)
        pre_g = torch.zeros(x.size(0), x.size(1)).unsqueeze(-1).to(device)

        for i in range(len(time_scale)):
            pre_v[i] = self.net_v(t[i], x[i])
            pre_rho[i] = self.net_rho(t[i], x[i])
            pre_g[i] = self.net_g(t[i], x[i])

        pre_x = torch.zeros((len(time_scale) + (len(time_scale) - 1) * delta, x.size(1), x.size(2))).to(device)

        # configure training options
        options = {}
        options.update({'method': 'Dopri5'})
        options.update({'h': None})
        options.update({'rtol': 1e-3})
        options.update({'atol': 1e-5})
        options.update({'print_neval': False})
        options.update({'neval_max': 1000000})
        options.update({'safety': None})
        func_v = self.net_v

        pre_t = torch.linspace(t[0].detach(), t[-1].detach(), len(time_scale) + delta * (len(time_scale) - 1)).to(
            device)
        pre_x[0] = x[0]
        for i in range(pre_t.size(0) - 1):
            options.update({'t0': pre_t[i]})
            options.update({'t1': pre_t[i + 1]})
            pre_x[i + 1] = odesolve(func_v, y0=pre_x[i], options=options)
        pre_x[pre_x < 0] = 0

        pre_x_t = torch.zeros((len(time_scale), x.size(1), x.size(2))).float().to(device)
        pre_x_t_v = torch.zeros((len(time_scale), x.size(1), x.size(2))).float().to(device)
        for i in range(len(time_scale)):
            pre_x_t[i] = pre_x[i * (delta + 1)]
            pre_x_t_v[i] = self.net_v(t[i], pre_x_t[i])

        return pre_rho, pre_g, pre_v, pre_x, pre_x_t, pre_t, pre_x_t_v

    def predict_traj(self, time_scale, x, delta):
        t = torch.tensor(time_scale).float().requires_grad_(True).to(device)
        x = torch.tensor(x).float().to(device)
        pre_v = torch.zeros(x.size(0), x.size(1), x.size(2)).to(device)

        pre_v[0] = self.net_v(t[0], x[0])

        pre_x = torch.zeros((len(time_scale) + (len(time_scale) - 1) * delta, x.size(1), x.size(2))).to(device)

        # configure training options
        options = {}
        options.update({'method': 'Dopri5'})
        options.update({'h': None})
        options.update({'rtol': 1e-3})
        options.update({'atol': 1e-5})
        options.update({'print_neval': False})
        options.update({'neval_max': 1000000})
        options.update({'safety': None})
        func_v = self.net_v

        pre_t = torch.linspace(t[0].detach(), t[-1].detach(), len(time_scale) + delta * (len(time_scale) - 1)).to(
            device)
        pre_x[0] = x[0]
        for i in range(pre_t.size(0) - 1):
            options.update({'t0': pre_t[i]})
            options.update({'t1': pre_t[i + 1]})
            pre_x[i + 1] = odesolve(func_v, y0=pre_x[i], options=options)
        pre_x[pre_x < 0] = 0

        pre_x_t = torch.zeros((len(time_scale), x.size(1), x.size(2))).float().to(device)
        pre_x_t_v = torch.zeros((len(time_scale), x.size(1), x.size(2))).float().to(device)
        for i in range(len(time_scale)):
            pre_x_t[i] = pre_x[i * (delta + 1)]
            pre_x_t_v[i] = self.net_v(t[i], pre_x_t[i])

        return pre_v, pre_x, pre_x_t, pre_t, pre_x_t_v


    def predict_t_point(self, ti, x_ti):
        ti = torch.tensor(ti).float().requires_grad_(True).to(device)
        xi = torch.tensor(x_ti).float().to(device)
        pre_v_t1 = self.net_v(ti, xi)
        return pre_v_t1

    def predict_g(self, ti, x_ti):
        ti = torch.tensor(ti).float().requires_grad_(True).to(device)
        xi = torch.tensor(x_ti).float().to(device)
        pre_g_t1 = self.net_g(ti, xi)
        return pre_g_t1


def plot_velo(data_umap, data, velocity, filname, args):
    data_plus_v = data + velocity
    umap_data_plus_v = x_to_umapx(data_plus_v, args)
    umap_v = umap_data_plus_v - data_umap

    data_umap = data_umap.cpu().detach().numpy()
    umap_v = umap_v.cpu().detach().numpy()
    time_points = len(data)
    fig, ax = plt.subplots()
    widths = 0.05  # arrow width
    plt.axis('on')
    v_scale = 1
    aa = 1
    color_wanted = [np.array([250, 187, 110]) / 255,
                    np.array([173, 219, 136]) / 255,
                    np.array([250, 199, 179]) / 255,
                    np.array([238, 68, 49]) / 255,
                    np.array([206, 223, 239]) / 255,
                    np.array([3, 149, 198]) / 255,
                    np.array([180, 180, 213]) / 255,
                    np.array([178, 143, 237]) / 255]
    labels = []
    labels_ins = []
    # add inferrred trajecotry
    for i in range(time_points):
        lab = ax.scatter(data_umap[i][:, 0], data_umap[i][:, 1], s=aa * 30, linewidth=0, color=color_wanted[i],
                    zorder=3)
        labels_ins.append(lab)
        labels.append("t%s" % (i))
        ax.quiver(data_umap[i][:, 0], data_umap[i][:, 1], umap_v[i][:, 0] / v_scale, umap_v[i][:, 1] / v_scale,
                   color='k', alpha=0.5, linewidths=widths, zorder=4, headwidth=2, headlength=3)
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0, handles=labels_ins, labels=labels)
    plt.tight_layout()
    path = os.path.join(filname, 'Cell_Sample_velocity.png')
    plt.savefig(path, bbox_inches='tight', dpi=600)
    plt.close()
    # plt.show()


def plot_velo_allcell(data, pre_v, data_umap, timepoints, time_allcell, filname, args):
    '''
    Plot the velocity of all the cells
    :param data_umap:
    :param data:
    :param velocity:
    :param genename:
    :param data_meta:
    :return:
    '''
    umap_v = []
    for i in range(timepoints):
        data_v_i = data[i].cpu() + pre_v[i]
        data_v_i = data_v_i.reshape(1, data_v_i.size(0), data_v_i.size(1))
        umap_data_plus_v_i = x_to_umapx(data_v_i, args)
        umap_data_plus_v_i = umap_data_plus_v_i.reshape(umap_data_plus_v_i.size(1), umap_data_plus_v_i.size(2))
        umap_v_i = umap_data_plus_v_i - data_umap[time_allcell==i]
        umap_v.append(umap_v_i)

    data_umap = data_umap.cpu().detach().numpy()
    time_points = len(data)
    fig, ax = plt.subplots()
    widths = 0.05  # arrow width
    plt.axis('on')
    v_scale = 1  # * max_vv  # 20
    aa = 1  # 3
    color_wanted = [np.array([250, 187, 110]) / 255,
                    np.array([173, 219, 136]) / 255,
                    np.array([250, 199, 179]) / 255,
                    np.array([238, 68, 49]) / 255,
                    np.array([206, 223, 239]) / 255,
                    np.array([3, 149, 198]) / 255,
                    np.array([180, 180, 213]) / 255,
                    np.array([178, 143, 237]) / 255]

    n_points = 3500
    n_arrows_to_show = 150
    if n_points <= n_arrows_to_show:
        selected_indices = np.arange(n_points)
    else:
        selected_indices = np.array([5, 6, 108, 119, 125, 175, 185, 225, 237, 257, 273, 286, 288, 321,
                                    380, 409, 413, 416, 543, 595, 663, 690, 705, 765, 803, 861, 867, 868,
                                    1148, 1186, 1198, 1246, 1255, 1320, 1339, 1352, 1373, 1396, 1425, 1431, 1437, 1450,
                                    1453, 1455, 1470, 1473, 1483, 1505, 1527, 1538, 1542, 1573, 1596, 1602, 1610, 1615,
                                    1642, 1651, 1675, 1690, 1710, 1737, 1747, 1810, 1837, 1857, 1859, 1888, 1893, 1911,
                                    1956, 1993, 2009, 2010, 2030, 2034, 2059, 2091, 2100, 2127, 2137, 2145, 2159, 2225,
                                    2685, 2705, 2707, 2724, 2731, 2733, 2796, 2823, 2824, 2835, 2847, 2854, 2882, 2889,
                                    2231, 2339, 2379, 2381, 2390, 2426, 2438, 2479, 2537, 2588, 2600, 2624, 2628, 2669,
                                    2898, 2902, 2904, 2942, 2969, 3019, 3033, 3054, 3087, 3117, 3133, 3171, 3222, 3233,
                                    3239, 3257, 3268, 3330, 3348, 3366, 3383, 3399, 3401, 3416,
                                    870, 879, 884, 909, 944, 969, 1001, 1027, 1083, 1086, 1101, 1107, 1115])  #3399

    labels = []
    labels_ins = []
    for i in range(time_points):
        lab = ax.scatter(data_umap[time_allcell==i][:, 0], data_umap[time_allcell==i][:, 1],
                         s=aa * 10, linewidth=0, color=color_wanted[i], alpha=0.7)  #
        labels_ins.append(lab)
        labels.append("t%s" % (i))
        ax.quiver(data_umap[time_allcell==i][selected_indices, 0], data_umap[time_allcell==i][selected_indices, 1],
                  umap_v[i].cpu().detach().numpy()[selected_indices, 0] / v_scale,
                  umap_v[i].cpu().detach().numpy()[selected_indices, 1] / v_scale,
                   color='k', alpha=0.8, linewidths=widths, zorder=4, scale=20, width=0.003)

        ax.set(xlabel=None, ylabel=None)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        arrow_length = 0.2
        ax.annotate('',
                    xy=(arrow_length, 0),
                    xytext=(-0.005, 0),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1),
                    xycoords='axes fraction')
        ax.text(arrow_length / 2, -0.03, 'UMAP 1',
                transform=ax.transAxes,
                ha='center',
                va='center')
        ax.annotate('',
                    xy=(0, arrow_length),
                    xytext=(0, -0.005),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1),
                    xycoords='axes fraction')
        ax.text(-0.03, arrow_length / 2, 'UMAP 2',  # 标签位置微调
                transform=ax.transAxes,
                ha='center',
                va='center',
                rotation=90)

    plt.tight_layout()
    path = os.path.join(filname, 'AllCellVelocity.png')
    plt.savefig(path, bbox_inches='tight', dpi=600)
    plt.close()
    # plt.show()


def plot_traj(samples_x, pre_x, samples_y, time, sample_plot, filname):
    samples_plot_ind = torch.randperm(samples_x.size(1))[:sample_plot]  # 选取的样本不重复
    real_data = samples_y
    data = samples_x.cpu().detach().numpy()
    pre_x = pre_x.detach().cpu().numpy()
    time_points = len(data)
    fig, ax = plt.subplots()
    plt.margins(0, 0)
    aa = 1  # 3

    plt.tight_layout()
    plt.margins(0, 0)
    color_wanted = [np.array([250, 187, 110]) / 255,
                    np.array([173, 219, 136]) / 255,
                    np.array([250, 199, 179]) / 255,
                    np.array([238, 68, 49]) / 255,
                    np.array([206, 223, 239]) / 255,
                    np.array([3, 149, 198]) / 255,
                    np.array([180, 180, 213]) / 255,
                    np.array([178, 143, 237]) / 255]

    labels = []
    labels_ins = []
    # 添加真实的点
    for i in range(time_points):
        ax.scatter(real_data[time==i].cpu()[:, 0], real_data[time==i].cpu()[:, 1],
                    c='none', marker='o', edgecolors=color_wanted[i], linewidth=0.5, alpha=0.2)

    for i in range(time_points):
        lab = ax.scatter(data[i][samples_plot_ind, 0], data[i][samples_plot_ind, 1],
                    s=aa * 10, linewidth=0, color=color_wanted[i], zorder=3)
        labels_ins.append(lab)
        labels.append("t%s" % (i))

    for j in range(sample_plot):
        ax.plot(pre_x[:, samples_plot_ind[j], 0], pre_x[:, samples_plot_ind[j], 1],
                 linewidth=0.5, color='black', zorder=2)
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0, handles=labels_ins, labels=labels)
    path = os.path.join(filname, 'Cell_Sample_trajectory.png')
    plt.savefig(path, bbox_inches='tight', dpi=600)
    # plt.show()
    plt.close()


def plot_LossAdam(loss_adam, timepoint):
    ## loss
    torch.save(loss_adam, "Output/DiscreteVelocity/Loss_adam.pt")
    plt.figure(figsize=(10, 4))
    plt.plot(loss_adam)  # linestyle为线条的风格（共五种）,color为线条颜色
    plt.title('Loss of Adam at time %s' % (timepoint + 1))
    plt.savefig("Output/ContinuousResults/Loss_adam_time%s.png" % (timepoint + 1), bbox_inches='tight', dpi=600)
    plt.close()

def Sampling(data, v_allcell, rho_allcell, cell_num, gene_num, time_points, samples_num, cluster_ids_allcell,
             cluster_sign_ids):
    cluster_sign_ids = cluster_sign_ids.long()
    samples_x = torch.zeros((time_points, samples_num, gene_num)).to(device)
    samples_v = torch.zeros((time_points, samples_num, gene_num)).to(device)
    samples_rho = torch.zeros((time_points, samples_num)).to(device)
    samples_index = torch.zeros((time_points, samples_num), dtype=torch.long).to(device)  # 每一行100个样本

    for i in range(time_points):
        if i == 0:
            if cell_num[i] < samples_num:
                samples_index[i, :] = torch.randint(0, cell_num[i], (samples_num,))  # k: 选取次数，选取的样本可重复
                sample_t0_index = samples_index[i, :]  # t0时刻的细胞位置
                sample_t0_ids = cluster_ids_allcell[i][sample_t0_index]  # t0时刻的细胞属于哪一类
                cluster_num = torch.unique(sample_t0_ids).size(0)  # 共有几类细胞 = 2
                cluster_cell_counts = torch.zeros((cluster_num), dtype=torch.long).to(device)  # 每一类有多少个细胞
                for j in range(cluster_num):
                    cluster_cell_counts[j] = torch.eq(sample_t0_ids, j).sum()
            else:
                samples_index[i, :] = torch.randperm(cell_num[i])[:samples_num]  # 选取的样本不重复
                sample_t0_index = samples_index[i, :]  # 元素的位置
                sample_t0_ids = cluster_ids_allcell[i][sample_t0_index]  # 0/1
                cluster_num = torch.unique(sample_t0_ids).size(0)  # 2
                cluster_cell_counts = torch.zeros((cluster_num), dtype=torch.long).to(device)  # [53, 47]
                for j in range(cluster_num):
                    cluster_cell_counts[j] = torch.eq(sample_t0_ids, j).sum()
        else:  # 其他时间的细胞与t0时刻的细胞为同一类型
            for j in range(cluster_num):
                index0_j = torch.nonzero(sample_t0_ids == j).squeeze()
                cell_ids_j = torch.nonzero(cluster_ids_allcell[i] == cluster_sign_ids[i, j]).squeeze()
                if cell_ids_j.size(0) < cluster_cell_counts[j]:
                    ids_ij_index = cell_ids_j[
                        torch.randint(0, cell_ids_j.size(0), (cluster_cell_counts[j],))]  # k: 选取次数，选取的样本可重复
                else:
                    ids_ij_index = cell_ids_j[torch.randperm(cell_ids_j.size(0))[:cluster_cell_counts[j]]]  # 选取的样本不重复
                samples_index[i][index0_j] = ids_ij_index

        samples_x[i] = data[i][samples_index[i, :], :]
        samples_v[i] = v_allcell[i].to(device)[samples_index[i, :], :]
        samples_rho[i] = rho_allcell[i].to(device)[samples_index[i, :]]

    return samples_x, samples_v, samples_rho, samples_index, sample_t0_ids


def plot_dis_density(data, dens_allcell, time_point, timepoints, filname):
    data = data.cpu().detach().numpy()
    dens_allcell = dens_allcell.cpu().detach().numpy()
    fig, ax = plt.subplots()
    plt.tight_layout()
    for i in range(timepoints):
        ax.scatter(data[i][:, 0], data[i][:, 1], color='gray', s=10, marker='o',
                   facecolors='none', edgecolors='gray', alpha=0.5)
    com = ax.scatter(data[time_point][:, 0], data[time_point][:, 1], c=dens_allcell[time_point],
                     s=10,
                     edgecolors='none', cmap='Blues', alpha=0.5)
    plt.colorbar(com)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.title('Discrete cell density at time %s' % (time_point + 1))
    ax.grid(False)
    path = os.path.join(filname, "Discrete cell density_time%s.png" % (time_point + 1))
    plt.savefig(path, bbox_inches='tight', dpi=600)
    # plt.show()
    plt.close()


def f(x, intercept, trend):  # 计算出拟合方程
    y = trend * x + intercept
    return y


def plot_dis_growth(data, growth_allcell, timepoints, filname):
    data = data.cpu().detach().numpy()
    growth_allcell = growth_allcell.cpu().detach().numpy()
    fig, ax = plt.subplots()

    g_max = np.max(growth_allcell)
    g_min = np.min(growth_allcell)

    for i in range(timepoints):
        ax.scatter(data[i][:, 0], data[i][:, 1], color='gray', s=10, marker='o',
                   facecolors='none', edgecolors='gray', alpha=0.5)
        com = ax.scatter(data[i][:, 0], data[i][:, 1], c=growth_allcell[i], s=10,
                         edgecolors='none', cmap='summer', alpha=0.7, vmin=g_min,
                         vmax=g_max)
        if i == 0:
            cbar = fig.colorbar(
                com,
                shrink=0.3,
                pad=0.02,
                location='right',
                anchor=(0.0, 0.2),
            )
            cbar.outline.set_edgecolor('none')
            cbar.locator = MaxNLocator(nbins=2)
            cbar.update_ticks()

    ax.set(xlabel=None, ylabel=None)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    arrow_length = 0.2
    ax.annotate('',
                xy=(arrow_length, 0),
                xytext=(-0.005, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1),
                xycoords='axes fraction')
    ax.text(arrow_length / 2, -0.03, 'UMAP 1',
            transform=ax.transAxes,
            ha='center',
            va='center')
    ax.annotate('',
                xy=(0, arrow_length),
                xytext=(0, -0.005),
                arrowprops=dict(arrowstyle='->', color='black', lw=1),
                xycoords='axes fraction')
    ax.text(-0.03, arrow_length / 2, 'UMAP 2',
            transform=ax.transAxes,
            ha='center',
            va='center',
            rotation=90)

    plt.title('Discrete cell growth')
    ax.grid(False)
    plt.tight_layout()
    path = os.path.join(filname, 'Discrete cell growth.png')
    plt.savefig(path, bbox_inches='tight', dpi=600)
    # plt.show()
    plt.close()


def gradG_ts(time_scale, sample_x, timepoints, model):
    dim = sample_x.size(2)
    grad = torch.zeros((sample_x.size(0), dim, 1)).to(device)
    for i in range(timepoints):
        t = time_scale[i]
        for j in range(sample_x.size(1)):
            x = sample_x[i][j].unsqueeze(0)
            pre_g = model.net_g(t, x)
            grad[i] = grad[i] + \
                      torch.autograd.grad(pre_g, x, torch.ones_like(pre_g), retain_graph=True, create_graph=True)[
                          0].reshape(dim, 1).detach()
        grad[i] = grad[i] / sample_x.size(1)
    return grad


def plot_gradg(grad_g, filname):
    grap_num = grad_g.size(0)
    grad_g = grad_g.cpu().numpy()
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(grap_num):
        if i < 4:
            index1 = 0
            index2 = i
        else:
            index1 = 1
            index2 = i - 4
        sns.heatmap(grad_g[i], ax=axs[index1, index2], cmap='coolwarm', square=True)
        axs[index1, index2].set_title('Grad g t%s' % i)
    plt.tight_layout()
    plt.axis('off')
    plt.margins(0, 0)
    path = os.path.join(filname, 'gradg_alltime.png')
    plt.savefig(path, bbox_inches='tight', dpi=600)
    # plt.show()
    plt.close()

def MultimodalGaussian_density(x, data_celltype, sigma):
    """density function for MultimodalGaussian
    """
    mu = data_celltype
    num_gaussian = mu.shape[0]
    dim = mu.shape[1]
    sigma_matrix = sigma * torch.eye(dim).type(torch.float32).to(device)
    p_unn = torch.zeros([x.shape[0]]).type(torch.float32).to(device)
    for i in range(num_gaussian):
        m = torch.distributions.multivariate_normal.MultivariateNormal(mu[i,:], sigma_matrix)
        p_unn = p_unn + torch.exp(m.log_prob(x)).type(torch.float32).to(device)
    p_n = p_unn/num_gaussian
    return p_n

def calculate_prob_neufate(x, data_neu, data_mo, sigma):
    den_neu = MultimodalGaussian_density(x, data_neu, sigma)
    den_mo = MultimodalGaussian_density(x, data_mo, sigma)
    prob = den_neu/(den_neu + den_mo)
    return prob


def x_to_umapx(x, args):
    x_reshape = x.reshape(-1, x.shape[-1])
    path_umap = os.path.join(args.data_dir, args.dataset_data, 'umap10to2_model.pkl')
    model_umap = joblib.load(path_umap)
    xu = model_umap.transform(x_reshape.cpu().detach().numpy())
    xu = torch.tensor(xu).to(device)
    xu_output = xu.reshape(x.size(0), x.size(1), 2)
    return xu_output

def main(args):
    seed = args.VelocityRefine_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    time_scale = ast.literal_eval(args.VelocityRefine_time)
    start = datetime.now()
    # load dataset
    adata, data = loaddata(args)
    time_points = len(data)
    time_alldata = adata.obs['time']
    umap_alldata = torch.tensor(adata.obsm['umap_10to2']).to(device)

    gene_num = data[0].size()[1]
    cell_num = [data[i].size()[0] for i in range(time_points)]
    output_num = gene_num  # \rho, g, v

    hidden_layers1 = ast.literal_eval(args.VelocityRefine_layers1)
    layers1 = [gene_num + 1] + hidden_layers1 + [output_num]  # v

    hidden_layers2 = ast.literal_eval(args.VelocityRefine_layers2)  # v
    layers2 = [gene_num + 1] + hidden_layers2 + [1]  # \rho g

    output_density_filname = os.path.join(args.out_dir, args.dataset_data, 'DiscreteDensity')
    output_velocity_filname = os.path.join(args.out_dir, args.dataset_data, 'DiscreteVelocity')
    output_trajectory_filname = os.path.join(args.out_dir, args.dataset_data, 'ContinuousResults')

    file_path_v = os.path.join(output_velocity_filname, 'velocity_allcell.pkl')
    with open(file_path_v, 'rb') as file:
        v_allcell = pickle.load(file)
    file_path_rho = os.path.join(output_density_filname, 'dens_allcell.pkl')
    with open(file_path_rho, 'rb') as file:
        rho_allcell = pickle.load(file)
    file_path_ids = os.path.join(output_velocity_filname, 'cluster_ids_allcell.pkl')
    with open(file_path_ids, 'rb') as file:
        cluster_ids_allcell = pickle.load(file)
    file_path_sign = os.path.join(output_velocity_filname, 'cluster_sign_ids.pkl')
    with open(file_path_sign, 'rb') as file:
        cluster_sign_ids = pickle.load(file)

    path = os.path.join(args.out_dir, args.dataset_data, 'ContinuousResults')
    if not os.path.exists(path):
        os.makedirs(path)

    samples_num = 300
    for ii in range(1):
        samples_x, samples_v, samples_rho, samples_index, sample_t0_ids = Sampling(data, v_allcell, rho_allcell,
                                                                                   cell_num, gene_num, time_points,
                                                                                   samples_num,
                                                                                   cluster_ids_allcell,
                                                                                   cluster_sign_ids)
        model = PhysicsInformedNN(time_scale, samples_x, samples_v, samples_rho, layers1, layers2)
        model.train(args.VelocityRefine_iters)
        torch.save(model, "model_PDE%d.pth"%(ii+1))
        # model = torch.load("model_PDE1.pth")  # 导入保存的模型和参数

        umap_samples_x = torch.zeros((samples_x.size(0), samples_x.size(1), 2)).to(device)
        for jj in range(time_points):
            umap_samples_x[jj] = umap_alldata[time_alldata == jj][samples_index[jj].cpu()]

        D, alpha, M = model.D.item(), model.alpha.item(), model.M.item()
        print("D=%.3f, alpha=%.3f, M=%.3f" % (D, alpha, M))

        delta = 9
        pre_rho, pre_g, pre_v, pre_x, pre_x_t, pre_t_details, pre_x_t_v = model.predict(time_scale, samples_x, delta)


        pre_v_alldata = []
        tt = torch.tensor(time_scale).to(device)
        for i in range(time_points):
            pre_v_ti = model.predict_t_point(time_scale[i], data[i])
            pre_v_alldata.append(pre_v_ti.cpu())

        file_path_v = os.path.join(output_trajectory_filname, 'pre_v_initialdata.pkl')
        with open(file_path_v, 'wb') as file:
            pickle.dump(pre_v_alldata, file)

        plot_velo_allcell(data, pre_v_alldata, umap_alldata, time_points, time_alldata, output_trajectory_filname, args)


        torch.save(samples_x, os.path.join(output_trajectory_filname, 'samples_x.pt'))
        torch.save(umap_samples_x, os.path.join(output_trajectory_filname, 'umap_samples_x.pt'))
        torch.save(pre_rho, os.path.join(output_trajectory_filname, "pre_rho.pt"))
        torch.save(pre_g, os.path.join(output_trajectory_filname, "pre_g.pt"))
        torch.save(pre_v, os.path.join(output_trajectory_filname, "pre_v.pt"))
        torch.save(pre_x, os.path.join(output_trajectory_filname, "pre_x.pt"))
        torch.save(pre_x_t, os.path.join(output_trajectory_filname, "pre_x_timepoints.pt"))
        torch.save(pre_x_t_v, os.path.join(output_trajectory_filname, "pre_x_v_timepoints.pt"))

        # plot velocity
        plot_velo(umap_samples_x, samples_x, pre_v, output_trajectory_filname, args)
        # plot rho
        for i in range(time_points):
            plot_dis_density(umap_samples_x, pre_rho, i, time_points, output_trajectory_filname)
        # plot g pre_v_initialdata
        plot_dis_growth(umap_samples_x, pre_g, time_points, output_trajectory_filname)
        samples_plot = samples_num

        # pre_x_t, pre_x ---- uamp
        umap_pre_x_t = x_to_umapx(pre_x_t, args)
        umap_pre_x = x_to_umapx(pre_x, args)

        for ll in range(time_points):
            umap_pre_x_t[ll] = umap_pre_x[ll * (delta + 1)]

        torch.save(umap_pre_x_t, os.path.join(output_trajectory_filname, "umap_pre_x_t.pt"))
        torch.save(umap_pre_x, os.path.join(output_trajectory_filname, "umap_pre_x.pt"))

        # plot trajectory
        plot_traj(umap_pre_x_t, umap_pre_x, umap_alldata, time_alldata, samples_plot, output_trajectory_filname)

        time_scale = torch.tensor(time_scale).float().requires_grad_(True).to(device)
        samples_x = samples_x.requires_grad_(True).to(device)

        grad_g = gradG_ts(time_scale, samples_x, time_points, model)

        pre_g_alldata = []
        tt = torch.tensor(time_scale).to(device)
        for i in range(time_points - 1):
            pre_g_ti = model.predict_g(time_scale[i], data[i])
            pre_g_alldata.append(pre_g_ti.cpu())

        file_path_g = os.path.join(output_trajectory_filname, 'pre_g_initialdata.pkl')
        with open(file_path_g, 'wb') as file:
            pickle.dump(pre_g_alldata, file)

        plot_gradg(grad_g, output_trajectory_filname)

        end = datetime.now()
        time_ = end - start
        print('')
        print('Run Time:', time_)

    path = os.path.join(output_trajectory_filname, 'seed.txt')
    with open(path, 'w') as file:
        file.write(str(seed))

if __name__ == "__main__":
    main(input())