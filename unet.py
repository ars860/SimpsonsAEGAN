import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum

# N_CLASSES = 11


class SkipType(Enum):
    SKIP = 0
    NO_SKIP = 1
    ZERO_SKIP = 2


class UpBlock(nn.Module):
    def __init__(self, ch_out, skip=SkipType.SKIP, dropout=None):
        super(UpBlock, self).__init__()
        self.skip = skip
        self.upconv = nn.ConvTranspose2d(ch_out * 2, ch_out * 2, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv1 = nn.Conv2d(ch_out * 2 if skip == SkipType.NO_SKIP else ch_out * 3, ch_out, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        if dropout is not None and dropout != 0:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, u=None, padding=None):
        if u is None and self.skip:
            raise ValueError("Expected skip connection")

        if padding is None:
            padding = [x.shape[2] % 2, x.shape[3] % 2]

        u = self.upconv(u)
        u = F.pad(u, [padding[1], 0, padding[0], 0])

        if self.skip == SkipType.SKIP:
            x1 = torch.cat((x, u), dim=1)
        elif self.skip == SkipType.ZERO_SKIP:
            x1 = torch.cat((torch.zeros_like(x), u), dim=1)
        else:
            x1 = u

        x2 = self.relu1(self.conv1(x1))
        x3 = self.relu2(self.conv2(x2))
        if getattr(self, 'dropout', None) is not None:
            x3 = self.dropout(x3)
        return x3


class Unet(nn.Module):
    def __init__(self, layers, output_channels=3, skip=SkipType.SKIP, dropout=None):
        super(Unet, self).__init__()
        prev_layer = 3
        for i, layer in enumerate(layers, start=1):
            setattr(self, f'down{i}', self.enc_block(prev_layer, layer, max_pool=prev_layer != 1))
            prev_layer = layer
        for i, layer in reversed(list(enumerate(layers[:-1], start=1))):
            setattr(self, f'up{i}', UpBlock(layer, skip, dropout))
        self.final = nn.Conv2d(layers[0], output_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward_vars(self, x):
        children = list(self.named_children())
        vars = {}
        downs = list(map(lambda nm: nm[1], filter(lambda nm: 'down' in nm[0], children)))
        ups = list(map(lambda nm: nm[1], filter(lambda nm: 'up' in nm[0], children)))
        final = children[-2][1]
        sigmoid = children[-1][1]

        vars['x'] = x
        for i, down_i in enumerate(downs[:-1], start=1):
            vars[f'x{i}'] = down_i(vars[f'x{"" if i == 1 else i - 1}'])
            print("down", vars[f'x{i}'].shape)

        vars[f'u{len(downs)}'] = downs[-1](vars[f'x{len(downs) - 1}'])
        for i, up_i in enumerate(ups):
            vars[f'u{len(ups) - i}'] = up_i(vars[f'x{len(ups) - i}'], vars[f'u{len(ups) - i + 1}'])
            print("up", vars[f'u{len(ups) - i}'].shape)

        vars['res'] = final(vars['u1'])
        vars['res_sigmoid'] = sigmoid(vars['res'])
        return vars

    def forward(self, x):
        return self.forward_vars(x)['res_sigmoid']

    def enc_block(self, ch_in, ch_out, max_pool=True):
        layers = [nn.MaxPool2d(kernel_size=2)] if max_pool else []
        layers += [
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.ReLU(),
        ]
        return nn.Sequential(*layers)

    def get_last_block_inputs(self, x):
        with torch.no_grad():
            vars = self.forward_vars(x)

            return vars['x1'], vars['u2']
