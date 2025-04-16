import os
import torch
import argparse
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, ConcatDataset

import model
import utils
from dataset import MyDataset, Augment


class Experiment:

    def __init__(self, args):
        self.root = os.path.abspath(args.dataset_root)
        self.num_epoch = args.num_epoch
        self.warmup_epoch = args.warmup_epoch
        self.batch_size = args.batch_size
        self.G_path = args.G_path
        self.D_path = args.D_path
        self.content_loss_lambda = args.content_loss_lambda

        # utils.generate_edge_smoothed_dataset("data.csv", os.path.join(self.root, "edge_smoothed"))

        train_real_dataset = MyDataset(self.root, style="real", mode="train", augmentations=False)
        real_fake_dataset = MyDataset(self.root, style="real", mode="test", augmentations=False)

        combined_dataset = ConcatDataset([train_real_dataset, real_fake_dataset])
        print(f"Number of samples in combined_dataset: {len(combined_dataset)}")
        #self.combined_loader = DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=True, num_workers=28)
        self.combined_loader = DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=True, num_workers=3)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.D = model.discriminator().to(self.device)
        self.G = model.generator().to(self.device)

        #self.vgg19 = torchvision.models.vgg19(pretrained=True).features.to(self.device).eval()
        # 替换原来的VGG初始化代码,512*512
        self.vgg_extractor = model.VGGFeatureExtractor().to(self.device).eval()

        self.D_optimizer = optim.Adam(self.D.parameters(), args.D_lr, betas=(0.5, 0.99))
        self.G_optimizer = optim.Adam(self.G.parameters(), args.G_lr, betas=(0.5, 0.99))

        self.BCE_loss = nn.BCELoss().to(self.device)
        self.L1_loss = nn.L1Loss().to(self.device)

        self.D_scheduler = MultiStepLR(self.D_optimizer, milestones=args.D_step, gamma=args.D_gamma)
        self.G_scheduler = MultiStepLR(self.G_optimizer, milestones=args.G_step, gamma=args.G_gamma)


    def train_warming(self, e):
        self.G.train()

        Content_losses = []

        for i, real_imgs in enumerate(self.combined_loader):
            real_imgs = real_imgs.to(self.device)

            noise = torch.randn(real_imgs.size(0), 100).to(self.device)

            fake_imgs = self.G(noise)
            #灰度图拓展为3通道
            fake_imgs_rgb_norm = (fake_imgs.repeat(1, 3, 1, 1) + 1) / 2
            real_imgs_rgb_norm = (real_imgs.repeat(1, 3, 1, 1) + 1) / 2

            #fake_imgs_rgb_norm = (fake_imgs + 1) / 2  # 直接归一化
            #real_imgs_rgb_norm = (real_imgs + 1) / 2

            assert fake_imgs_rgb_norm.shape[1] == 3, f"期望3通道，实际得到{fake_imgs_rgb_norm.shape[1]}通道"

            #fake_imgs_features = self.vgg19(fake_imgs_rgb_norm)
            #real_imgs_features = self.vgg19(real_imgs_rgb_norm)

            #512*512
            fake_imgs_features = self.vgg_extractor(fake_imgs_rgb_norm)  # [B,256,128,128]
            real_imgs_features = self.vgg_extractor(real_imgs_rgb_norm) # 同尺寸
            #print(f"fake_imgs_features shape: {fake_imgs_features.shape}")
            #print(f"real_imgs_features shape: {real_imgs_features.shape}")

            content_loss = self.content_loss_lambda * self.L1_loss(fake_imgs_features, real_imgs_features.detach())

            self.G_optimizer.zero_grad()
            content_loss.backward()
            self.G_optimizer.step()

            Content_losses.append(content_loss.item())

        average_content_loss = np.mean(Content_losses)
        print(f"\nEpoch: {e}, Average content loss: {average_content_loss:.3f}\n")
        return average_content_loss
    

    def train(self, e, n_critic=5, lambda_gp=10):
        self.D.train()
        self.G.train()

        D_losses = []
        G_losses = []

        for i, real_imgs in enumerate(self.combined_loader):
            noise = torch.randn(real_imgs.size(0), 100).to(self.device)
            real_imgs = real_imgs.to(self.device)
            self.D_optimizer.zero_grad()
            # 真实图像判别器损失
            #print("D_fake",D_fake)
            D_real,_ = self.D(real_imgs)
            D_real_loss = -torch.mean(D_real)
            # 生成图像判别器损失
            fake_imgs = self.G(noise).detach()
            D_fake,_ = self.D(fake_imgs)
            D_fake_loss = torch.mean(D_fake)

            # 计算总的判别器损失，包括梯度惩罚
            D_loss = D_real_loss + D_fake_loss + lambda_gp * utils.compute_gradient_penalty(self.D, real_imgs.data, fake_imgs.data)

            D_loss.backward()
            self.D_optimizer.step()

            if i % n_critic == 0:
                self.G_optimizer.zero_grad()

                fake_imgs = self.G(noise)
                #G_loss = -torch.mean(self.D(fake_imgs))
                G_loss = -torch.mean(self.D(fake_imgs)[0])  # 只有对抗损失
                # 旋转分类损失
                _, rot_logits = self.D(fake_imgs)
                rot_labels = torch.randint(0, 4, (fake_imgs.size(0),)).to(self.device)  # 随机生成旋转标签
                G_rot_loss = torch.nn.functional.cross_entropy(rot_logits, rot_labels)

                # 总的生成器损失
                total_G_loss = G_loss + G_rot_loss
                total_G_loss.backward()
                self.G_optimizer.step()

                #G_loss.backward()
                #self.G_optimizer.step()
                G_losses.append(total_G_loss.item())

                G_losses.append(G_loss.item())

            D_losses.append(D_loss.item())

        average_D_loss = np.mean(D_losses)
        average_G_loss = np.mean(G_losses)
        print(f"Epoch: {e}, Average D loss: {average_D_loss:.3f}, G loss: {average_G_loss:.3f}")

        self.G_scheduler.step()
        self.D_scheduler.step()
        return average_D_loss, average_G_loss


    def valid(self, e):
        self.save_path = os.path.join(os.getcwd(), 'images')
        os.makedirs(self.save_path, exist_ok=True)
        grid_size = 3
        n_images = grid_size ** 2  

        with torch.no_grad():
            self.G.eval()  
            noise = torch.randn(n_images, 100).to(self.device)  
            gen_imgs = self.G(noise)

            # 获取旋转分类结果
            _, rot_logits = self.D(gen_imgs)
            rot_preds = torch.argmax(rot_logits, dim=1)  # 获取旋转分类结果

            grid_filename = f"epoch_{e}_generated_grid.png"
            grid_path = os.path.join(self.save_path, grid_filename)
            
            save_image(gen_imgs.data, grid_path, nrow=grid_size, normalize=True)

            # 打印旋转分类的预测
            print(f"Rotation predictions for generated images: {rot_preds}")


    def run(self):
        warm_up_content_losses = []     
        training_D_losses = []
        training_G_losses = []

        print("Warming Up")
        for e in range(self.warmup_epoch):
            curr_content_loss = self.train_warming(e)
            warm_up_content_losses.append(curr_content_loss)
            self.valid(e)

        print("Training and Validating")
        for e in range(self.num_epoch):
            curr_D_loss, curr_G_loss = self.train(e)

            training_D_losses.append(curr_D_loss)
            training_G_losses.append(curr_G_loss)
            self.valid(e)

        utils.save_model(self.num_epoch, self.D.state_dict(), self.G.state_dict(), self.D_optimizer.state_dict(), self.G_optimizer.state_dict(), self.G_path, self.D_path)
        return warm_up_content_losses, training_D_losses, training_G_losses
    

    def display_real_images(self):
        self.save_path = os.path.join(os.getcwd(), 'real_images')
        os.makedirs(self.save_path, exist_ok=True)
        grid_size = 3
        n_images = grid_size ** 2  

        real_imgs = next(iter(self.combined_real_loader))

        real_imgs = real_imgs[:n_images]
    
        real_imgs_normalized = (real_imgs + 1) / 2

        grid_filename = f"real_images_grid.png"
        grid_path = os.path.join(self.save_path, grid_filename)
        save_image(real_imgs_normalized.data, grid_path, nrow=grid_size, normalize=True)
        print(f"Saved real images grid to {grid_path}")
