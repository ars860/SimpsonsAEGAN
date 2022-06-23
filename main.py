import os
import json
import re
import sys
import time

import cv2
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import torchvision as tv
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from aegan import AEGAN

BATCH_SIZE = 16
LATENT_DIM = 32
EPOCHS = 20000


def save_images(GAN, vec, filename):
    images = GAN.generate_samples(vec)
    ims = tv.utils.make_grid(images[:36], normalize=True, nrow=6, )
    if GAN.channels_cnt == 5:
        ims = restore_simplified(ims)
    ims = ims.numpy().transpose((1, 2, 0))
    ims = np.array(ims * 255, dtype=np.uint8)
    image = Image.fromarray(ims)
    image.save(filename)


def gen_samples(dataset):
    dataloader = DataLoader(dataset,
                            batch_size=36,
                            shuffle=False,
                            num_workers=1,
                            drop_last=True
                            )
    X = iter(dataloader)
    test_ims, _ = next(X)
    # test_ims2, _ = next(X)
    # test_ims = torch.cat((test_ims1, test_ims2), 0)
    channels_cnt = test_ims.shape[1]
    test_ims_show = tv.utils.make_grid(test_ims[:36], normalize=True, nrow=6, )
    if channels_cnt == 5:
        test_ims_show = restore_simplified(test_ims_show)
    test_ims_show = test_ims_show.numpy().transpose((1, 2, 0)) * 255
    test_ims_show = test_ims_show.astype(np.uint8)
    #     .numpy().transpose((1, 2, 0))  # test_ims_show.numpy().transpose((1, 2, 0))
    # test_ims_show = np.array(test_ims_show * 255, dtype=np.uint8)
    image = Image.fromarray(test_ims_show)
    return test_ims, image


import wandb


class Simplify(torch.nn.Module):
    def __call__(self, img):
        img = np.array(img)
        # r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        # r, g, b = r[:, :, None], g[:, :, None], b[:, :, None]
        # img = np.concatenate([b, g, r], axis=2)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # img = np.moveaxis(img, 0, 2)

        black_pixels = np.where(
            (img[:, :, 2] < 160)
        )

        yellow_pixels = np.where(
            (img[:, :, 0] > 20) & (img[:, :, 0] < 30) &
            (img[:, :, 1] > 200) & (img[:, :, 1] < 300)
        )

        blue_pixels = np.where(
            (img[:, :, 0] > 110) & (img[:, :, 0] < 130) &
            (img[:, :, 1] > 100)  # & (img[:, :, 1] < 300)
        )

        red_pixels = np.where(
            (img[:, :, 0] > 0) & (img[:, :, 0] < 15) &
            (img[:, :, 1] > 180)  # & (img[:, :, 1] < 300)
            & (img[:, :, 2] > 100)
        )

        blank = np.ones_like(img) * 255
        blank[yellow_pixels] = [254, 215, 16]
        blank[blue_pixels] = [99, 111, 221]
        blank[red_pixels] = [255, 0, 0]
        blank[black_pixels] = 0

        # edges = find_edges(img)
        # blank[edges == 0] = 0

        # blank = cv2.fastNlMeansDenoisingColored(blank, None, 20)

        black_pixels = np.zeros_like(img)
        black_pixels[np.where((blank[:, :, 0] == 0) & (blank[:, :, 1] == 0) & (blank[:, :, 2] == 0))] = 1
        black_pixels = black_pixels[None, :, :, 0]

        yellow_pixels = np.zeros_like(img)
        yellow_pixels[np.where((blank[:, :, 0] == 254) & (blank[:, :, 1] == 215) & (blank[:, :, 2] == 16))] = 1
        yellow_pixels = yellow_pixels[None, :, :, 0]

        blue_pixels = np.zeros_like(img)
        blue_pixels[np.where((blank[:, :, 0] == 99) & (blank[:, :, 1] == 111) & (blank[:, :, 2] == 221))] = 1
        blue_pixels = blue_pixels[None, :, :, 0]

        red_pixels = np.zeros_like(img)
        red_pixels[np.where((blank[:, :, 0] == 255) & (blank[:, :, 1] == 0) & (blank[:, :, 2] == 0))] = 1
        red_pixels = red_pixels[None, :, :, 0]

        white_pixels = np.zeros_like(img)
        white_pixels[np.where((blank[:, :, 0] == 255) & (blank[:, :, 1] == 255) & (blank[:, :, 2] == 255))] = 1
        white_pixels = white_pixels[None, :, :, 0]

        pixels = np.concatenate([black_pixels, yellow_pixels, blue_pixels, red_pixels, white_pixels])

        return torch.Tensor(pixels)


def restore_simplified(x, threshold=0.5):
    black, yellow, blue, red, white = torch.split(x, [1, 1, 1, 1, 1], dim=0)

    def color(t, r, g, b):
        return torch.cat([t * r, t * g, t * b], dim=0)

    return color(white > threshold, 1, 1, 1) + \
           color(red > threshold, 1, 0, 0) + \
           color(black > threshold, 0, 0, 0) + \
           color(blue > threshold, 99 / 255, 111 / 255, 221 / 255) + \
           color(yellow > threshold, 254 / 255, 215 / 255, 16 / 255)


def main():
    results = "data_simplified_linear_upsampling_4colors_stolen_discriminator_results"  # "my_generator_upsampling_results"
    transfer_from, transfer_from_epoch = None, 0  # "data_linear_upsampling_colorspace_stolen_discriminator_results", 299
    root = "data_cleared"  # "data" # "small_data"  # os.path.join("data")
    channels_cnt = 5  # 3
    colors_cnt = 5  # 16

    os.makedirs(f"{results}/generated", exist_ok=True)
    os.makedirs(f"{results}/reconstructed", exist_ok=True)
    os.makedirs(f"{results}/checkpoints", exist_ok=True)

    use_wandb = True
    if use_wandb:
        wandb.init(project="gan_simpsons", entity="ars860",
                   config={"name": results, "transfer": str(transfer_from) + "_" + str(transfer_from_epoch),
                           "data": root})

    device = "cuda"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = tv.transforms.Compose([
        tv.transforms.RandomAffine(0, translate=(5 / 96, 5 / 96), fill=(255, 255, 255)),
        # tv.transforms.ColorJitter(hue=0.5),
        tv.transforms.RandomHorizontalFlip(p=0.5),
        # Simplify(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
    ])

    if channels_cnt == 5:
        transform = tv.transforms.Compose([
            tv.transforms.RandomAffine(0, translate=(5 / 96, 5 / 96), fill=(255, 255, 255)),
            # tv.transforms.ColorJitter(hue=0.5),
            tv.transforms.RandomHorizontalFlip(p=0.5),
            Simplify(),
            # tv.transforms.ToTensor(),
            # tv.transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
        ])

    dataset = ImageFolder(
        root=root,
        transform=transform
    )
    dataloader = DataLoader(dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=1,
                            drop_last=True
                            )
    # X = iter(dataloader)
    # test_ims1, _ = next(X)
    # test_ims2, _ = next(X)
    # test_ims = torch.cat((test_ims1, test_ims2), 0)
    # test_ims_show = tv.utils.make_grid(test_ims[:36], normalize=True, nrow=6, )
    # test_ims_show = test_ims_show.numpy().transpose((1, 2, 0))
    # test_ims_show = np.array(test_ims_show * 255, dtype=np.uint8)
    # image = Image.fromarray(test_ims_show)
    test_ims, image = gen_samples(dataset)
    image.save(f"{results}/reconstructed/test_images.png")

    torch.manual_seed(1)
    noise_fn = lambda x: torch.randn((x, LATENT_DIM), device=device)
    test_noise = noise_fn(36)
    gan = AEGAN(
        LATENT_DIM,
        noise_fn,
        dataloader,
        device=device,
        batch_size=BATCH_SIZE,
        channels_cnt=channels_cnt,
        colors_cnt=colors_cnt,
    )

    if use_wandb:
        wandb.watch(gan.generator, log="gradients", log_freq=1000)

    epoch_start = -1
    checkpoints = os.listdir(f"{results}/checkpoints")
    checkpoints = map(
        lambda checkpoint_name: re.match(r"(generator|encoder|discriminator_image|discriminator_latent)+_([^_]+)\.pt",
                                         checkpoint_name).group(2), checkpoints)
    checkpoints = list(map(int, checkpoints))
    if checkpoints:
        last_checkpoint = max(checkpoints)
        epoch_start = last_checkpoint
        print(f"Loaded from checkpoint: {last_checkpoint}")
        gan.load(os.path.join(results, "checkpoints"), last_checkpoint)
        # gan.generator.load_state_dict(torch.load("trained_generator_weights.pt", map_location=device))
    else:
        gan.init()
        if transfer_from is not None:
            print(f"No checkpoints found inializing from {transfer_from} on epoch {transfer_from_epoch}")
            gan.load(os.path.join(transfer_from, "checkpoints"), transfer_from_epoch)
    epoch_start += 1

    start = time.time()
    for i in range(epoch_start, EPOCHS):
        while True:
            try:
                with open("pause.json") as f:
                    pause = json.load(f)
                if pause['pause'] == 0:
                    break
                print(f"Pausing for {pause['pause']} seconds")
                time.sleep(pause["pause"])
            except (KeyError, json.decoder.JSONDecodeError, FileNotFoundError):
                break
        elapsed = int(time.time() - start)
        elapsed = f"{elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}"
        print(f"Epoch {i + 1}; Elapsed time = {elapsed}s")
        gan.train_epoch(max_steps=100, use_wandb=use_wandb)
        if (i + 1) % 50 == 0:
            # torch.save(
            #     gan.generator.state_dict(),
            #     os.path.join("results", "checkpoints", f"gen.{i:05d}.pt"))
            gan.save(os.path.join(results, "checkpoints"), i)
        save_images(gan, test_noise, os.path.join(results, "generated", f"gen.{i:04d}.png"))

        with torch.no_grad():
            reconstructed = gan.generator(gan.encoder(test_ims.to(device))).cpu()
        reconstructed = tv.utils.make_grid(reconstructed[:36], normalize=True, nrow=6, )
        if gan.channels_cnt == 5:
            reconstructed = restore_simplified(reconstructed)
        reconstructed = reconstructed.numpy().transpose((1, 2, 0))
        reconstructed = np.array(reconstructed * 255, dtype=np.uint8)
        reconstructed = Image.fromarray(reconstructed)
        reconstructed.save(os.path.join(results, "reconstructed", f"gen.{i:04d}.png"))

    images = gan.generate_samples()
    ims = tv.utils.make_grid(images, normalize=True)
    plt.imshow(ims.numpy().transpose((1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    main()
