import argparse
import os


def config():
    parser = argparse.ArgumentParser()
    # system config
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--device_num', default=1)
    parser.add_argument('--data_dir', default='../Data')
    parser.add_argument('--out_dir', default='../Output')

    # data config
    parser.add_argument('--dataset_data', type=str, default='Veres')

    # DensityGeneration config
    parser.add_argument('--DensityGeneration_seed', type=int, default=1)
    parser.add_argument('--DensityGeneration_sigma', type=int, default=1)


    # VelocityGeneration config
    parser.add_argument('--VelocityGeneration_seed', type=int, default=1)
    parser.add_argument('--VelocityGeneration_num_cluster', type=int, default=4)
    parser.add_argument('--VelocityGeneration_iters', type=int, default=1000)
    parser.add_argument('--VelocityGeneration_iters_thres', type=int, default=500)
    parser.add_argument('--VelocityGeneration_hidden_layers', type=str, default='[40, 40]')


    # VelocityRefine config
    parser.add_argument('--VelocityRefine_time', type=str, default='[0, 1, 2, 3, 4, 5, 6, 7]')
    parser.add_argument('--VelocityRefine_layers1', type=str, default='[30, 30]')
    parser.add_argument('--VelocityRefine_layers2', type=str, default='[20, 20]')
    parser.add_argument('--VelocityRefine_iters', type=int, default=1000)
    parser.add_argument('--VelocityRefine_seed', type=int, default=5537)



    parser.add_argument('--sigma_type', default='const')
    parser.add_argument('--sigma_const', type=float, default=0.1)
    parser.add_argument('--sde_adjoint', action='store_true', default=False)

    # training config
    parser.add_argument('--train_epochs', type=int, default=3000)
    parser.add_argument('--train_lr', type=float, default=1e-3)
    parser.add_argument('--train_batch', type=float, default=512)
    parser.add_argument('--train_dt', type=float, default=0.1)
    parser.add_argument('--train_clip', type=float, default=0.1)
    parser.add_argument('--save', type=int, default=500)
    parser.add_argument('--lambda_marginal', type=float, default=1)

    # loss config
    parser.add_argument('--sinkhorn_scaling', type=float, default=0.7)
    parser.add_argument('--sinkhorn_blur', type=float, default=0.1)
    parser.add_argument('--use_intLoss', default=True)

    # evaluation config
    parser.add_argument('--evaluate_n', type=int, default=2000)

    args = parser.parse_args()
    return args
