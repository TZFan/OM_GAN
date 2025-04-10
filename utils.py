import os
import cv2
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.autograd as autograd

def rotate_batch(imgs):
    """对输入图像批量随机旋转 0°, 90°, 180°, 270°，并返回旋转标签"""
    angles = [0, 90, 180, 270]
    rotated_imgs = []
    labels = []
    for img in imgs:
        angle = np.random.choice(angles)
        labels.append(angles.index(angle))
        rotated_imgs.append(torch.rot90(img, k=angle // 90, dims=[1, 2]))
    return torch.stack(rotated_imgs), torch.tensor(labels)

def compute_gradient_penalty(D, real_samples, fake_samples):
    device = real_samples.device
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _ = D(interpolates)
    fake = torch.ones(d_interpolates.size(), device=device)

    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def save_model(epoch, D_state, G_state, D_optim_state, G_optim_state, D_path, G_path):
    torch.save({"epoch": epoch, "D_state": D_state, "D_optim_state": D_optim_state}, D_path)
    torch.save({"epoch": epoch, "G_state": G_state, "G_optim_state": G_optim_state}, G_path)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def edge_promoting(image_path, save_path, n):
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)

    rgb_img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    
    rgb_img = cv2.resize(rgb_img, (32, 32))
    pad_img = np.pad(rgb_img, ((2, 2), (2, 2), (0, 0)), mode='reflect')
    gray_img = cv2.resize(gray_img, (32, 32))
    
    edges = cv2.Canny(gray_img, 100, 200)
    dilation = cv2.dilate(edges, kernel)
    result = np.copy(rgb_img)
    #gauss_img = np.copy(rgb_img)
    idx = np.where(dilation != 0)
    """
    for i in range(np.sum(dilation != 0)):
        for j in range(3):  
            result[idx[0][i], idx[1][i], j] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, j], gauss))
    """
    for i in range(np.sum(dilation != 0)):
        for c in range(3):  # 对BGR三个通道分别处理
            result[idx[0][i], idx[1][i], c] = np.mean(
                rgb_img[idx[0][i]:idx[0][i] + 5, idx[1][i]:idx[1][i] + 5, c]
            )

    result = result
    cv2.imwrite(os.path.join(save_path, f'{n}.png'), result)  


def generate_edge_smoothed_dataset(csv_file, save_path):
    df = pd.read_csv(csv_file)

    if os.path.isdir(save_path):
        print(f"Edge-smoothed dataset already exists at {save_path}. Skipping generation.")
        return
    else:
        os.makedirs(save_path)
        print(f"Creating {save_path} and starting to generate edge-smoothed dataset...")
        
    existing_files = glob.glob(os.path.join(save_path, '*.png'))
    if existing_files:
        highest_number = max([int(os.path.splitext(os.path.basename(f))[0]) for f in existing_files])
    else:
        highest_number = 0  

    n = highest_number + 1  
    
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        edge_promoting(row['image'], save_path, n)
        n += 1