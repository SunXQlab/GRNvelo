import os
import torch
import pickle
import random
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

device = torch.device('cpu')


def MultimodalGaussian_density(x,time_pt,data_train,sigma):
    """density function for MultimodalGaussian
    """
    mu = data_train[time_pt]
    num_gaussian = mu.shape[0]  # mu is number_sample * dimension
    dim = mu.shape[1]
    sigma_matrix = sigma * torch.eye(dim).type(torch.float32).to(device)
    p_unn = torch.zeros([x.shape[0]]).type(torch.float32).to(device)
    for i in range(num_gaussian):
        m = torch.distributions.multivariate_normal.MultivariateNormal(mu[i,:], sigma_matrix)
        p_unn = p_unn + torch.exp(m.log_prob(x)).type(torch.float32).to(device)
    p_n = p_unn/num_gaussian
    return p_n


def Sampling(num_samples, time_all, time_pt, data_train, sigma):
    # perturb the  coordinate x with Gaussian noise N (0, sigma*I )
    mu = data_train[time_all[time_pt]]
    num_gaussian = mu.shape[0]  # mu is number_sample * dimension
    dim = mu.shape[1]
    sigma_matrix = sigma * torch.eye(dim)
    m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(dim), sigma_matrix)
    noise_add = m.rsample(torch.Size([num_samples])).type(torch.float32).to(device)

    # check if number of points is <num_samples
    if num_gaussian < num_samples:
        samples = mu[random.choices(range(0, num_gaussian), k=num_samples)] + noise_add
    else:
        samples = mu[random.sample(range(0, num_gaussian), num_samples)] + noise_add
    return samples


def loaddata(args):
    adata_filname = os.path.join(args.data_dir, args.dataset_data, f"{args.dataset_data}.h5ad")
    adata = sc.read_h5ad(adata_filname)

    datafilname = os.path.join(args.data_dir, args.dataset_data, f"{args.dataset_data}.npy")
    data=np.load(datafilname, allow_pickle=True)
    data_train=[]
    for i in range(len(data)):
        data_train.append(torch.from_numpy(data[i]).type(torch.float32).to(device))
    return adata, data_train



def plot_dis_density(data, dens_allcell, time_point, timepoints, time_allcell, umap_, filname):
    fig, ax = plt.subplots()
    ax.scatter(umap_[:, 0], umap_[:, 1], color='gray', s=100, marker='o',
               facecolors='none', edgecolors='gray', alpha=0.5)
    com = ax.scatter(umap_[time_allcell==time_point, 0], umap_[time_allcell==time_point, 1], c=dens_allcell[time_point], s=100,
               edgecolors='none', cmap='Blues', alpha=0.5)
    plt.colorbar(com)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.title('Discrete cell density at time %s'%(time_point+1))
    ax.grid(False)
    plt.tight_layout()
    filname = os.path.join(filname, "Discrete cell density_time%s.png" % (time_point+1))
    plt.savefig(filname, bbox_inches='tight', dpi=600)
    plt.close()

def main(args):
    # Seed
    seed = args.DensityGeneration_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set the seed for Python's built-in random number generator
    random.seed(seed)
    # Set the Numpy random seed
    np.random.seed(seed)

    # load dataset
    adata, data = loaddata(args)
    time_points = len(data)
    sigma_value = args.DensityGeneration_sigma
    sigma_now = np.ones(time_points) * sigma_value  # 0.2

    # plot
    time_allcell = adata.obs['time']
    data_umap = adata.obsm['umap_10to2']

    output_density_filname = os.path.join(args.out_dir, args.dataset_data, 'DiscreteDensity')
    if not os.path.exists(output_density_filname):
        os.makedirs(output_density_filname)

    den_allcell_alltime = []
    for i in range(time_points):
        den_allcell_timei = []
        if i == 0:
            dens_0 = MultimodalGaussian_density(data[i], i, data, sigma_now[i])  # normalized density
            den_allcell_alltime.append(dens_0)
            den_allcell_timei.append(dens_0)
            for j in range(i + 1, time_points):
                dens_j = torch.zeros_like(data[j][:, 0])
                den_allcell_timei.append(dens_j)
        elif i == (time_points - 1):
            for j in range(i):
                dens_j = torch.zeros_like(data[j][:, 0])
                den_allcell_timei.append(dens_j)
            dens_i = MultimodalGaussian_density(data[i], i, data, sigma_now[i])  # normalized density
            den_allcell_alltime.append(dens_i)
            den_allcell_timei.append(dens_i)
        else:
            for j in range(i):
                dens_j = torch.zeros_like(data[j][:, 0])
                den_allcell_timei.append(dens_j)

            dens_i = MultimodalGaussian_density(data[i], i, data, sigma_now[i])  # normalized density
            den_allcell_timei.append(dens_i)
            den_allcell_alltime.append(dens_i)
            for j in range(i + 1, time_points):
                dens_j = torch.zeros_like(data[j][:, 0])
                den_allcell_timei.append(dens_j)

        plot_dis_density(data, den_allcell_timei, i, time_points, time_allcell, data_umap, output_density_filname)

    file_path = os.path.join(output_density_filname, 'dens_allcell.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(den_allcell_alltime, file)

if __name__=="__main__":
    main(input())
