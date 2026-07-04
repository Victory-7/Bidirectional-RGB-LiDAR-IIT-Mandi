import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


# ==========================================================
# POSITIONAL ENCODING
# ==========================================================

class PositionalEncoding2D(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.channels = channels

    def forward(self, x):

        B, C, H, W = x.shape

        device = x.device

        y_embed = torch.linspace(
            0,
            1,
            H,
            device=device
        ).view(1, 1, H, 1)

        x_embed = torch.linspace(
            0,
            1,
            W,
            device=device
        ).view(1, 1, 1, W)

        pos = torch.cat(
            [
                y_embed.repeat(B, 1, 1, W),
                x_embed.repeat(B, 1, H, 1)
            ],
            dim=1
        )

        return torch.cat(
            [x, pos],
            dim=1
        )


# ==========================================================
# TRANSFORMER BLOCK
# ==========================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        dim=512,
        heads=8,
        mlp_ratio=4
    ):

        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(

            nn.Linear(
                dim,
                dim * mlp_ratio
            ),

            nn.GELU(),

            nn.Linear(
                dim * mlp_ratio,
                dim
            )
        )

    def forward(self, x):

        attn_out, _ = self.attn(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x)
        )

        x = x + attn_out

        x = x + self.mlp(
            self.norm2(x)
        )

        return x


# ==========================================================
# TRANSFORMER BOTTLENECK
# ==========================================================

class TransformerBottleneck(nn.Module):

    def __init__(
        self,
        dim=512,
        heads=8,
        layers=6
    ):

        super().__init__()

        self.blocks = nn.ModuleList([

            TransformerBlock(
                dim=dim,
                heads=heads
            )

            for _ in range(layers)

        ])

    def forward(self, x):

        for block in self.blocks:

            x = block(x)

        return x


# ==========================================================
# DECODER BLOCK
# ==========================================================

class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)


# ==========================================================
# RANGE IMAGE GENERATOR
# ==========================================================

class LiDARRangeGenerator(nn.Module):

    def __init__(self, config):

        super().__init__()

        pretrained = config["model"][
            "pretrained"
        ]

        resnet = models.resnet50(
            weights=(
                models.ResNet50_Weights.DEFAULT
                if pretrained
                else None
            )
        )

        self.encoder0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )

        self.pool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.pos_encoding = (
            PositionalEncoding2D(
                2048
            )
        )

        self.feature_reduce = nn.Conv2d(
            2050,
            config["model"][
                "transformer_dim"
            ],
            kernel_size=1
        )

        self.transformer = (
            TransformerBottleneck(
                dim=config["model"][
                    "transformer_dim"
                ],
                heads=config["model"][
                    "transformer_heads"
                ],
                layers=config["model"][
                    "transformer_layers"
                ]
            )
        )

        decoder_channels = (
            config["model"][
                "decoder_channels"
            ]
        )

        self.decoder4 = DecoderBlock(
            config["model"][
                "transformer_dim"
            ],
            decoder_channels[0]
        )

        self.decoder3 = DecoderBlock(
            decoder_channels[0],
            decoder_channels[1]
        )

        self.decoder2 = DecoderBlock(
            decoder_channels[1],
            decoder_channels[2]
        )

        self.decoder1 = DecoderBlock(
            decoder_channels[2],
            decoder_channels[3]
        )

        self.range_head = nn.Sequential(

            nn.Conv2d(
                decoder_channels[3],
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                1,
                kernel_size=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        e0 = self.encoder0(x)

        e1 = self.encoder1(
            self.pool(e0)
        )

        e2 = self.encoder2(e1)

        e3 = self.encoder3(e2)

        e4 = self.encoder4(e3)

        e4 = self.pos_encoding(
            e4
        )

        e4 = self.feature_reduce(
            e4
        )

        B, C, H, W = e4.shape

        tokens = e4.flatten(
            2
        ).transpose(
            1,
            2
        )

        tokens = self.transformer(
            tokens
        )

        e4 = tokens.transpose(
            1,
            2
        ).reshape(
            B,
            C,
            H,
            W
        )

        d4 = self.decoder4(
            e4
        )

        d4 = F.interpolate(
            d4,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        d3 = self.decoder3(
            d4
        )

        d3 = F.interpolate(
            d3,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        d2 = self.decoder2(
            d3
        )

        d2 = F.interpolate(
            d2,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        d1 = self.decoder1(
            d2
        )

        range_map = self.range_head(
            d1
        )

        range_map = F.interpolate(
            range_map,
            size=(64, 2048),
            mode="bilinear",
            align_corners=False
        )

        return range_map


# ==========================================================
# BUILD MODEL
# ==========================================================

def build_model(config):

    model = LiDARRangeGenerator(
        config
    )

    return model