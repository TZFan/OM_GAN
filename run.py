import argparse
from train import Experiment
import torch

parser = argparse.ArgumentParser(description="GAN setup with command line arguments.")
parser.add_argument("--dataset_root", type=str, default=r"D:\OM_TEST\data_ex\Synthetic-Data-Generation-for-Deep-Learning-Model-Enhancement-main\Synthetic-Data-Generation-for-Deep-Learning-Model-Enhancement-main", help="Root directory of the dataset.")
parser.add_argument("--num_epoch", type=int, default=1500, help="Number of epochs for training.")
parser.add_argument("--warmup_epoch", type=int, default=15, help="Number of warm-up epochs for the generator.")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training and validation.")
parser.add_argument("--G_path", type=str, default="./G_model.pth", help="Path to save the Generator model.")
parser.add_argument("--D_path", type=str, default="./D_model.pth", help="Path to save the Discriminator model.")
parser.add_argument("--D_lr", type=float, default=0.00004, help="Learning rate for the Discriminator.")
parser.add_argument("--G_lr", type=float, default=0.0002, help="Learning rate for the Generator.")
parser.add_argument("--D_step", type=int, nargs="+", default=[1000, 1000], help="Milestones for the Discriminator scheduler.")
parser.add_argument("--G_step", type=int, nargs="+", default=[1000, 1000], help="Milestones for the Generator scheduler.")
parser.add_argument("--D_gamma", type=float, default=0.5, help="Gamma value for the Discriminator scheduler.")
parser.add_argument("--G_gamma", type=float, default=0.5, help="Gamma value for the Generator scheduler.")
parser.add_argument("--content_loss_lambda", type=float, default=1.0, help="Lambda for content loss.")
args = parser.parse_args()

if __name__ == '__main__':
    experiment = Experiment(args)
    experiment.run()
    # experiment.display_real_images()
