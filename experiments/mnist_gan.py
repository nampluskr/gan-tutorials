import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..")) 
SRC_DIR = os.path.abspath(os.path.join(ROOT_DIR, "src"))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import torch
import torchvision.transforms as T

from gan_tutorials.datasets import MNIST, get_train_loader
from gan_tutorials.utils import set_seed, create_images, sample_latent, update_history, plot_images
from gan_tutorials.gan import Generator, Discriminator, GAN
from gan_tutorials.trainer import fit


if __name__ == "__main__":

    SEED = 42
    DATA_DIR = "/mnt/d/deep_learning/datasets/mnist"
    BATCH_SIZE = 128

    set_seed(SEED)
    dataset = MNIST(root_dir=DATA_DIR, split="train")
    train_loader = get_train_loader(dataset, batch_size=BATCH_SIZE)

    IMG_SIZE = 32
    LATENT_DIM = 128
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    BASE = 64

    generator = Generator(
        img_size=IMG_SIZE,
        latent_dim=LATENT_DIM,
        out_channels=OUT_CHANNELS,
        base=BASE,
    )
    discriminator = Discriminator(
        img_size=IMG_SIZE,
        in_channels=IN_CHANNELS,
        base=BASE,
    )
    gan = GAN(discriminator, generator, loss_type="bce")

    NUM_EPOCHS = 2
    TOTAL_EPOCHS = 10
    NUM_SAMPLES = 100

    FILENAME = os.path.splitext(os.path.basename(__file__))[0]
    OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", FILENAME)
    IMAGE_NAME = FILENAME + ""

    noises = sample_latent(NUM_SAMPLES, LATENT_DIM)
    history = {}
    epoch = 0

    for _ in range(TOTAL_EPOCHS // NUM_EPOCHS):
        epoch_history = fit(gan, train_loader, num_epochs=NUM_EPOCHS, total_epochs=TOTAL_EPOCHS)
        update_history(history, epoch_history)

        images = create_images(gan.generator, noises)
        epoch = gan.global_epoch
        image_path = os.path.join(OUTPUT_DIR, f"{IMAGE_NAME}_epoch{epoch:03d}.png")
        plot_images(*images, ncols=10, save_path=image_path)
