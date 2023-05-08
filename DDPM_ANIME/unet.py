"""
  * FileName: unet.py
  * Author:   Slatter
  * Date:     2023/5/6 00:03
  * Description:  
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import List


class TimeEmbedding(nn.Module):
    def __init__(self, num_steps: int, embed_dim: int, t_dim: int):
        super(TimeEmbedding, self).__init__()
        self.embed_dim = embed_dim
        half_dim = embed_dim // 2
        den = torch.exp(- torch.arange(half_dim) * math.log(10000) / (half_dim - 1))  # (half_dim)
        time = torch.arange(0, num_steps).view(num_steps, 1)  # (num_steps, 1)
        embedding = torch.cat([torch.sin(time * den), torch.cos(time * den)], dim=-1)  # (num_steps, embed_dim)

        self.time_embedding = nn.Sequential(
            nn.Embedding.from_pretrained(embedding),
            nn.Linear(embed_dim, t_dim),
            nn.Hardswish(),
            nn.Linear(t_dim, t_dim)
        )

    def forward(self, t: torch.Tensor):
        """
        :param t: t moment  (batch)
        :return: time embedding according to t moment   (batch, t_dim)
        """
        return self.time_embedding(t)


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super(DownSample, self).__init__()
        self.net = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x, temb):
        """
        :param x: (b, c, h, w)
        :param temb:
        :return: (b, c, h // 2, w // 2)
        """
        return self.net(x)


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super(UpSample, self).__init__()
        self.net = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, temb):
        """
        :param x: (b, c, h, w)
        :param temb:
        :return: (b, c, h * 2, w * 2)
        """
        _, _, H, W = x.shape
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.net(x)
        return out


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super(AttnBlock, self).__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: (b, c, h, w)
        :return: (b, c, h, w)
        """
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.proj_q(h)  # (b, c, h, w)
        k = self.proj_k(h)  # (b, c, h, w)
        v = self.proj_v(h)  # (b, c, h, w)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)  # (b, c, h, w) -> (b, h, w, c) -> (b, h * w, c)
        k = k.view(B, C, H * W)  # (b, c, h, w) -> (b, c, h * w)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))  # (b, h * w, h * w)
        assert list(w.shape) == [B, H * W, H * W]
        w = self.softmax(w)  # (b, h * w, h * w)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)  # (b, c, h, w) -> (b, h, w, c) -> (b, h * w, c)
        h = torch.bmm(w, v)  # (b, h * w, c)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)  # (b, h, w, c) -> (b, c, h, w)
        h = self.proj(h)

        return x + h


# original ddpm paper use swish as activation function, here we use hardswish to reduce flops
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim, dropout, attn=False):
        super(ResBlock, self).__init__()
        self.temb_proj = nn.Sequential(
            nn.Hardswish(),
            nn.Linear(time_embed_dim, out_ch)
        )

        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_ch),
            nn.Hardswish(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=out_ch),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        )

        if in_ch != out_ch:  # (batch, in_ch, h, w) -> (batch, out_ch, h, w)
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

        if attn:
            self.attn = AttnBlock(in_ch=out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x, t):
        """
        residual block process
        :param x: feature map (batch, c, h, w)
        :param t: time embedding  (batch, time_embed_dim)
        :return: h (batch, c, h, w)
        """
        h = self.block1(x)
        time_embed = self.temb_proj(t)[:, :, None, None]
        h += time_embed  # add time embeddding

        h = self.block2(h)
        h = self.shortcut(x) + h

        h = self.attn(h)
        return h


class UNet(nn.Module):
    def __init__(self, num_steps: int, ch: int, ch_mult: List[int], attn: List[int], num_res_blocks: int, dropout: float):
        """
        :param num_steps: num of time steps
        :param ch: initial num of channels  (b, 3, h, w) -> (b, ch, h, w)
        :param ch_mult: used to generate the corresponding number of channels (ch * ch_mult[0], ch * ch_mult[1], ...)
        :param attn: a list determine which feature map to use attention
        :param num_res_blocks: how many res blocks per downsample or upsample operation
        :param dropout: dropout rate
        """
        super(UNet, self).__init__()
        t_dim = ch * 4

        self.time_embedding = TimeEmbedding(num_steps, ch, t_dim)

        # head block
        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)

        # down sample blocks
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        cur_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(cur_ch, out_ch, t_dim, dropout, attn=(i in attn)))
                cur_ch = out_ch
                chs.append(cur_ch)

            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(cur_ch))
                chs.append(cur_ch)

        # bottom block
        self.middle_blocks = nn.ModuleList([
            ResBlock(cur_ch, cur_ch, t_dim, dropout, attn=True),
            ResBlock(cur_ch, cur_ch, t_dim, dropout, attn=False),
        ])

        # up sample blocks
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(chs.pop() + cur_ch, out_ch, t_dim, dropout, attn=(i in attn)))
                cur_ch = out_ch

            if i != 0:
                self.upblocks.append(UpSample(cur_ch))
        assert len(chs) == 0

        # end blocks
        self.tail = nn.Sequential(
            nn.GroupNorm(32, cur_ch),
            nn.Hardswish(),
            nn.Conv2d(cur_ch, 3, 3, stride=1, padding=1)
        )

    def forward(self, x, t):
        """
        :param x: (b, 3, h, w)
        :param t: t moment (b)
        :return: (b, 3, h, w)
        """
        t_embed = self.time_embedding(t)  # (b, t_dim)

        # down sample
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, t_embed)
            hs.append(h)

        # bottom layer
        for layer in self.middle_blocks:
            h = layer(h, t_embed)

        # up sample
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)    # wide residual connection
            h = layer(h, t_embed)

        # final layer
        out = self.tail(h)  # (b, 3, h, w)
        assert len(hs) == 0
        return out


if __name__ == '__main__':
    batch_size = 8
    model = UNet(num_steps=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1], num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size,))
    y = model(x, t)
