import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from .blocks import DeconvBlock, ConvBlock


#####################################################################
# Generator (latent code c 추가)
#####################################################################

class InfoGenerator(nn.Module):
    def __init__(self, img_size=32, latent_dim=100, out_channels=3, base=64,
                 num_discrete=10, num_continuous=2):
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.num_discrete = num_discrete
        self.num_continuous = num_continuous

        num_blocks = {32: 2, 64: 3, 128: 4, 256: 5}
        if img_size not in num_blocks:
            raise ValueError(f"Unsupported img_size: {img_size}")

        self.num_blocks = num_blocks[img_size]
        in_channels = base * (2 ** self.num_blocks)
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + num_discrete + num_continuous, in_channels,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )

        blocks = []
        for _ in range(self.num_blocks):
            blocks.append(DeconvBlock(in_channels, in_channels // 2))
            in_channels //= 2
        self.blocks = nn.Sequential(*blocks)

        self.final = nn.ConvTranspose2d(in_channels, out_channels,
                                        kernel_size=4, stride=2, padding=1, bias=False)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, noises, c_discrete, c_continuous):
        noises = noises.view(-1, self.latent_dim, 1, 1)
        c_discrete = F.one_hot(c_discrete, num_classes=self.num_discrete).float()
        c_discrete = c_discrete.view(-1, self.num_discrete, 1, 1)
        c_continuous = c_continuous.view(-1, self.num_continuous, 1, 1)
        x = torch.cat([noises, c_discrete, c_continuous], dim=1)

        x = self.initial(x)
        x = self.blocks(x)
        x = self.final(x)
        return torch.tanh(x)


#####################################################################
# Discriminator + Q Network
#####################################################################

class InfoDiscriminator(nn.Module):
    def __init__(self, img_size=32, in_channels=3, base=64,
                 num_discrete=10, num_continuous=2):
        super().__init__()
        self.img_size = img_size
        self.num_discrete = num_discrete
        self.num_continuous = num_continuous

        num_blocks = {32: 2, 64: 3, 128: 4, 256: 5}
        if img_size not in num_blocks:
            raise ValueError(f"Unsupported img_size: {img_size}")

        self.num_blocks = num_blocks[img_size]
        out_channels = base
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        blocks = []
        for _ in range(self.num_blocks):
            blocks.append(ConvBlock(out_channels, out_channels * 2))
            out_channels *= 2
        self.blocks = nn.Sequential(*blocks)

        self.d_head = nn.Conv2d(out_channels, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.q_head = nn.Sequential(
            nn.Conv2d(out_channels, 128, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Flatten()
        )
        self.q_discrete = nn.Linear(128, num_discrete)
        self.q_continuous = nn.Linear(128, num_continuous)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.initial(x)
        x = self.blocks(x)

        adv_logits = self.d_head(x).view(-1, 1)     # (B, 1)
        q = self.q_head(x)                          # (B, 128)
        disc_logits = self.q_discrete(q)            # (B, num_discrete)
        cont_logits = self.q_continuous(q)          # (B, num_continuous)
        return adv_logits, disc_logits, cont_logits


#####################################################################
# InfoGAN Trainer
#####################################################################

class InfoGAN(nn.Module):
    def __init__(self, generator, discriminator, latent_dim=100,
                 num_discrete=10, num_continuous=2, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discriminator = discriminator.to(self.device)
        self.generator = generator.to(self.device)

        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))

        self.latent_dim = latent_dim
        self.num_discrete = num_discrete
        self.num_continuous = num_continuous
        self.global_epoch = 0
        self.q_lambda = 5.0

    def d_adv_loss_fn(self, real_logits, fake_logits):
        real_labels = torch.ones_like(real_logits)
        fake_labels = torch.zeros_like(fake_logits)
        d_real_loss = F.binary_cross_entropy_with_logits(real_logits, real_labels)
        d_fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_labels)
        return d_real_loss, d_fake_loss

    def g_adv_loss_fn(self, fake_logits):
        real_labels = torch.ones_like(fake_logits)
        return F.binary_cross_entropy_with_logits(fake_logits, real_labels)

    def q_discrete_loss_fn(self, q_disc_logits, codes_disc):
        return F.cross_entropy(q_disc_logits, codes_disc)

    def q_continuous_loss_fn(self, q_cont, codes_cont):
        return F.mse_loss(q_cont, codes_cont)

    def train_step(self, batch):
        images = batch["image"].to(self.device)
        batch_size = images.size(0)

        noises = torch.randn(batch_size, self.latent_dim).to(self.device)
        codes_disc = torch.randint(0, self.num_discrete, (batch_size,)).to(self.device)
        codes_cont = torch.randn(batch_size, self.num_continuous).to(self.device)

        # (1) Discriminator 업데이트
        real_logits, _, _ = self.discriminator(images)
        fake_images = self.generator(noises, codes_disc, codes_cont).detach()
        fake_logits, _, _ = self.discriminator(fake_images)

        d_real_adv, d_fake_adv = self.d_adv_loss_fn(real_logits, fake_logits)
        d_loss = d_real_adv + d_fake_adv

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # (2) Generator 업데이트
        fake_images = self.generator(noises, codes_disc, codes_cont)
        fake_logits, q_disc_logits, q_cont = self.discriminator(fake_images)
        g_adv_loss = self.g_adv_loss_fn(fake_logits)

        q_disc_loss = self.q_discrete_loss_fn(q_disc_logits, codes_disc)
        q_cont_loss = self.q_continuous_loss_fn(q_cont, codes_cont)
        q_loss = q_disc_loss + q_cont_loss

        g_loss = g_adv_loss + self.q_lambda * q_loss

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        return dict(
            d_loss=d_loss.detach().cpu().item(),
            g_loss=g_loss.detach().cpu().item(),
            q_loss=q_loss.detach().cpu().item(),
            # q_disc_loss=q_disc_loss.detach().cpu().item(),
            # q_cont_loss=q_cont_loss.detach().cpu().item(),
        )
