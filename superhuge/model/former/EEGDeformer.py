# This is the script of EEG-Deformer
# This is the network script
import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def cnn_block(self, in_chan, kernel_size, dp):
        return nn.Sequential(
            nn.Dropout(p=dp),
            nn.Conv1d(
                in_channels=in_chan,
                out_channels=in_chan,
                kernel_size=kernel_size,
                padding=self.get_padding_1D(kernel=kernel_size),
            ),
            nn.BatchNorm1d(in_chan),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        in_chan,
        fine_grained_kernel=11,
        dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            dim = int(dim * 0.5)
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                        self.cnn_block(
                            in_chan=in_chan, kernel_size=fine_grained_kernel, dp=dropout
                        ),
                    ]
                )
            )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        dense_feature = []
        for attn, ff, cnn in self.layers:
            x_cg = self.pool(x)
            x_cg = attn(x_cg) + x_cg
            x_fg = cnn(x)
            x_info = self.get_info(x_fg)  # (b, in_chan)
            dense_feature.append(x_info)
            x = ff(x_cg) + x_fg
        x_dense = torch.cat(dense_feature, dim=-1)  # b, in_chan*depth
        x = x.view(x.size(0), -1)  # b, in_chan*d_hidden_last_layer
        emd = torch.cat(
            (x, x_dense), dim=-1
        )  # b, in_chan*(depth + d_hidden_last_layer)
        return emd

    def get_info(self, x):
        # x: b, k, l
        x = torch.log(torch.mean(x.pow(2), dim=-1))
        return x

    def get_padding_1D(self, kernel):
        return int(0.5 * (kernel - 1))


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class Deformer(nn.Module):
    def cnn_block(self, out_chan, kernel_size, num_chan):
        return nn.Sequential(
            Conv2dWithConstraint(
                1,
                out_chan,
                kernel_size,
                padding=self.get_padding(kernel_size[-1]),
                max_norm=2,
            ),
            Conv2dWithConstraint(
                out_chan, out_chan, (num_chan, 1), padding=0, max_norm=2
            ),
            nn.BatchNorm2d(out_chan),
            nn.ELU(),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
        )

    def __init__(
        self,
        *,
        dropout=0.3,
        ff_hidden_dim=128,
        mha_depth=4,
        mha_embed_dim=256,
        mha_num_heads=8,
        num_kernel=32,
        temporal_kernel_size=13,
        **kwargs,
    ):
        super().__init__()

        num_chan = kwargs["num_channels"]
        num_time = kwargs["window_length"] * kwargs["fs"]

        self.cnn_encoder = self.cnn_block(
            out_chan=num_kernel,
            kernel_size=(1, temporal_kernel_size),
            num_chan=num_chan,
        )

        dim = int(0.5 * num_time)  # embedding size after the first cnn encoder

        self.to_patch_embedding = Rearrange("b k c f -> b k (c f)")

        self.pos_embedding = nn.Parameter(torch.randn(1, num_kernel, dim))

        self.transformer = Transformer(
            dim=dim,
            depth=mha_depth,
            heads=mha_num_heads,
            dim_head=mha_embed_dim,
            mlp_dim=ff_hidden_dim,
            dropout=dropout,
            in_chan=num_kernel,
            fine_grained_kernel=temporal_kernel_size,
        )

    def forward(self, eeg):
        # eeg: (b, time, chan)
        eeg = rearrange(eeg, "b t c -> b 1 c t")
        x = self.cnn_encoder(eeg)  # (b, num_kernel, 1, 0.5*num_time)

        x = self.to_patch_embedding(x)

        x += self.pos_embedding
        x = self.transformer(x)
        return x

    def get_padding(self, kernel):
        return (0, int(0.5 * (kernel - 1)))

    def get_hidden_size(self, input_size, num_layer):
        return [int(input_size * (0.5**i)) for i in range(num_layer + 1)]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    data = torch.ones((16, 32, 1000))
    emt = Deformer(
        num_chan=32,
        num_time=1000,
        temporal_kernel=11,
        num_kernel=64,
        num_classes=2,
        depth=4,
        heads=16,
        mlp_dim=16,
        dim_head=16,
        dropout=0.5,
    )
    print(emt)
    print(count_parameters(emt))

    out = emt(data)
