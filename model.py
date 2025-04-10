import torch.nn as nn
import torch.nn.functional as F
import utils
import torchvision

class residual_block(nn.Module):
    def __init__(self, channel):
        super(residual_block, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(channel)
        utils.initialize_weights(self)

    def forward(self, inputs):
        residual = F.relu(self.norm1(self.conv1(inputs)))
        residual = self.norm2(self.conv2(residual))
        return inputs + residual

#生成器类 generator
class generator(nn.Module):
    def __init__(self, noise_dim=100, out_channel=3, filters=64, res_num=4):
        super(generator, self).__init__()
        
        #self.fc = nn.Linear(noise_dim, filters * 8 * 8)
        self.fc = nn.Linear(noise_dim, filters * 64 * 64)
        
        self.initial_conv = nn.Sequential(
            nn.ConvTranspose2d(filters, filters, kernel_size=4, stride=2, padding=1), 
            nn.InstanceNorm2d(filters),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(filters, filters // 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(filters // 2),
            nn.ReLU(inplace=True),


            nn.ConvTranspose2d(filters // 2, filters // 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(filters // 4),
            nn.ReLU(inplace=True),

            # 新增 256x256 -> 512x512
            nn.ConvTranspose2d(filters // 4, filters // 8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(filters // 8),
            nn.ReLU(inplace=True),

            # 新增将 512x512 -> 1024x1024 的反卷积层
            nn.ConvTranspose2d(filters // 8, filters // 16, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(filters // 16),
            nn.ReLU(inplace=True),
        )
        
        self.res_blocks = nn.Sequential(*[residual_block(filters // 16) for _ in range(res_num)])

        self.final_conv = nn.Sequential(
            nn.Conv2d(filters // 16, out_channel, kernel_size=7, stride=1, padding=3),
            #nn.Conv2d(filters // 8, out_channel, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        utils.initialize_weights(self)

    def forward(self, x):
        x = self.fc(x)
        #x = x.view(-1, 64, 16, 16)
        x = x.view(-1, 64, 64, 64)
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        x = self.final_conv(x)
        return x

"""
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()

        self.model = nn.Sequential(
            #nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, 4, 2, 1),  # 32->16
            nn.InstanceNorm2d(1024),
            nn.LeakyReLU(0.2),

            nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )
        utils.initialize_weights(self)

    def forward(self, img):
        validity = self.model(img)
        validity = validity.squeeze(1).squeeze(2)
        return validity
"""
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.InstanceNorm2d(1024),
            nn.LeakyReLU(0.2),

            # 1024 -> 2048 转换至最终层
            nn.Conv2d(1024, 2048, 4, 2, 1),
            nn.InstanceNorm2d(2048),
            nn.LeakyReLU(0.2),
        )

        # 判别真实或伪造
        self.adv_head = nn.Conv2d(2048, 1, kernel_size=4, stride=1, padding=0)

        # 旋转分类（4类）
        self.rot_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 4)
        )

        utils.initialize_weights(self)

    def forward(self, img):
        feat = self.features(img)
        adv_out = self.adv_head(feat).squeeze(1).squeeze(1)
        rot_logits = self.rot_head(feat)
        return adv_out, rot_logits
"""
    def initialize_weights(self):
        # 自定义初始化函数，可以根据需求调整初始化方法
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.constant_(m.bias, 0)
"""



class VGGFeatureExtractor(nn.Module):

    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        vgg = torchvision.models.vgg19(pretrained=True).features

        # 精确截取到relu2_2层（输出128x128特征图）
        self.slice = nn.Sequential()
        for i in range(16):  # 前7层对应relu2_2
            self.slice.add_module(str(i), vgg[i])

        # 冻结参数（重要！）
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 输入: [B,3,512,512] → 输出: [B,256,128,128]
        return self.slice(x)
