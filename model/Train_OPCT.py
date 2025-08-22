import config
import DensityGeneration
import VelocityGeneration
import VelocityRefine

if __name__ == "__main__":
    args = config.config()
    DensityGeneration.main(args)
    print('PINN-ODE...')
    VelocityGeneration.main(args)
    print('\nPINN-PDE...')
    VelocityRefine.main(args)
