from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch import optim
import torchvision as tv
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


class MyGenerator(nn.Module):
    # input is (batch, latent_dim, 1, 1)
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        channels = [latent_dim, 64 * 8, 64 * 8, 64 * 4, 64 * 2, 64, 64, 3]
        params = [(2, 1, 0), (2, 2, 0), (2, 2, 0), (2, 2, 0), (2, 2, 0), (3, 3, 0), (3, 1, 1)]
        self.layers = nn.Sequential()
        for cur_channels, next_channels, conv_params in zip(channels[:-1], channels[1:], params):
            kernel_size, stride, padding = conv_params
            self.layers.append(nn.ConvTranspose2d(cur_channels, next_channels, kernel_size=kernel_size, stride=stride,
                                                  padding=padding))
            self.layers.append(nn.BatchNorm2d(next_channels))
            self.layers.append(nn.ReLU(True))

    def forward(self, x):
        return self.layers(x)


class MyGeneratorUpsampling(nn.Module):
    # input is (batch, latent_dim, 1, 1)
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        channels = [latent_dim, 64 * 8, 64 * 8, 64 * 4, 64 * 4, 64 * 2, 64, 64, 3]
        params = [3, 2, 2, 2, 1, 2, 2, 1]
        self.layers = nn.Sequential()
        for i, (cur_channels, next_channels, scale) in enumerate(zip(channels[:-1], channels[1:], params)):
            if scale > 1:
                self.layers.append(nn.Upsample(scale_factor=scale))
            self.layers.append(nn.Conv2d(cur_channels, next_channels, kernel_size=5, stride=1, padding=2))

            if i == len(params) - 1:
                self.layers.append(nn.Tanh())
            else:
                self.layers.append(nn.BatchNorm2d(next_channels))
                self.layers.append(nn.LeakyReLU(0.2, True))

    def forward(self, x):
        return self.layers(x)


class MyDiscriminator(nn.Module):
    # input is (batch, 3, 96, 96)
    def __init__(self):
        super().__init__()

        channels = [3, 64, 64, 128, 256, 256, 512, 512, 1]
        params = [1, 2, 2, 1, 2, 2, 2, 3]
        self.layers = nn.Sequential()
        for i, (cur_channels, next_channels, scale) in enumerate(zip(channels[:-1], channels[1:], params)):
            self.layers.append(nn.Conv2d(cur_channels, next_channels, kernel_size=5, stride=scale, padding=2))

            if i < len(params) - 1:
                self.layers.append(nn.BatchNorm2d(next_channels))
                self.layers.append(nn.LeakyReLU(0.2, True))
            else:
                self.layers.append(nn.Sigmoid())

    def forward(self, x):
        return self.layers(x)


class DiscriminatorImage(nn.Module):
    """A discriminator for discerning real from generated images.

    Input shape: (?, 3, 96, 96)
    Output shape: (?, 1)
    """

    def __init__(self, device="cpu"):
        """Initialize the discriminator."""
        super().__init__()
        self.device = device
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        down_channels = [3, 64, 128, 256, 512]
        self.down = nn.ModuleList()
        leaky_relu = nn.LeakyReLU()
        for i in range(4):
            self.down.append(
                nn.Conv2d(
                    in_channels=down_channels[i],
                    out_channels=down_channels[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True,
                )
            )
            self.down.append(nn.BatchNorm2d(down_channels[i + 1]))
            self.down.append(leaky_relu)

        self.classifier = nn.ModuleList()
        self.width = down_channels[-1] * 6 ** 2
        self.classifier.append(nn.Linear(self.width, 1))
        self.classifier.append(nn.Sigmoid())

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
        rv = torch.randn(input_tensor.size(), device=self.device) * 0.02
        intermediate = input_tensor + rv
        for module in self.down:
            intermediate = module(intermediate)
            rv = torch.randn(intermediate.size(), device=self.device) * 0.02 + 1
            intermediate *= rv

        intermediate = intermediate.view(-1, self.width)

        for module in self.classifier:
            intermediate = module(intermediate)

        return intermediate


latent_vector = torch.rand([1, 32, 1, 1])


# with torch.no_grad():
#     img = generator.forward(latent_vector)
#     latent = discriminator.forward(img)
#     print(img.shape)
#     print(latent.shape)


def main():
    root = "small_data"
    BATCH_SIZE = 16

    transform = tv.transforms.Compose([
        # tv.transforms.RandomAffine(0, translate=(5/96, 5/96), fill=(255,255,255)),
        # tv.transforms.ColorJitter(hue=0.5),
        tv.transforms.RandomHorizontalFlip(p=0.5),
        tv.transforms.ToTensor(),
        # tv.transforms.Lambda(lambda img: img * 2 - 1),
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

    latent_dim = 100
    device = "cuda"
    lr = 0.0002

    generator = MyGeneratorUpsampling(latent_dim).to(device)
    # generator.load_state_dict(torch.load(Path() / "my_generator_upsampling_results" / "checkpoints" / "generator_199.pt"))
    discriminator = DiscriminatorImage(device=device).to(device)  # MyDiscriminator().to(device)

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    num_epochs = 10000

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            discriminator.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            real_cpu = real_cpu * 2 - 1
            b_size = real_cpu.size(0)
            label = torch.randn((b_size,), device=device) * 0.5 + 0.7
            # Forward pass real batch through D
            output = discriminator(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            # Generate fake image batch with G
            fake = generator(noise)
            label = torch.randn((b_size,)).to(device) * 0.3
            # Classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label = torch.randn((b_size,), device=device) * 0.5 + 0.7  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            iters += 1

        with torch.no_grad():
            fake = generator(fixed_noise).detach().cpu()
        ims = tv.utils.make_grid(fake[:36], nrow=6, )
        ims = ims.numpy().transpose((1, 2, 0))
        ims = np.array(((ims + 1) / 2) * 255, dtype=np.uint8)
        image = Image.fromarray(ims)
        image.save(f"dcgan/{epoch}.png")


if __name__ == '__main__':
    main()
