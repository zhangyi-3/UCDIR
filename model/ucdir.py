import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(nn.Linear(in_channels, out_channels * (1 + self.use_affine_level)))

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(nn.GroupNorm(groups, dim), Swish(),
                                   nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
                                   nn.Conv2d(dim, dim_out, 3, padding=1))  # print('**block')

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, nl_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(nl_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class ResnetBlockDY3h(nn.Module):
    def __init__(self, dim, dim_out, nl_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=1, nset=8):
        super().__init__()
        self.noise_func = nn.Sequential(nn.Linear(nl_emb_dim, nset), Swish(), nn.Linear(nset, nset), )
        self.nset = nset
        self.dim_out = dim_out
        self.norm1 = nn.GroupNorm(norm_groups, dim)
        self.conv1 = nn.Conv2d(dim, dim_out, 3, padding=1)

        self.norm2 = nn.GroupNorm(norm_groups, dim_out)
        self.conv2 = nn.Sequential(nn.Conv2d(3, self.nset * 2, 1), SimpleGate(),
                                   nn.Conv2d(self.nset, self.nset, kernel_size=3, padding=1), )

        self.spdyconv = nn.Conv2d(dim_out, dim_out * self.nset, kernel_size=3, padding=1, groups=self.nset)

        self.swish = Swish()

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb, guide):
        b, c, H, w = x.shape
        # print(time_emb.shape, self.noise_func)
        attw = self.noise_func(time_emb).view(b, -1)
        h = self.norm1(x)
        h = self.conv1(h)
        h = self.swish(h)
        h = self.norm2(h)
        # h = self.conv2(h, attw)

        # An equivalent implementation of AKGM by using group conv
        ratio = w / guide.shape[-1]
        guide = F.interpolate(guide, scale_factor=ratio, mode='bilinear', align_corners=False)
        att_sp = self.conv2(guide) * attw.view(b, self.nset, 1, 1)
        hset = self.spdyconv(h).view(b, self.dim_out, self.nset, H, w)
        h = torch.sum(hset * att_sp.unsqueeze(1), dim=2, keepdim=False)

        h = self.swish(h)
        return h + self.res_conv(x)


class SimpleGateFC(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * x2


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum("bnchw, bncyx -> bnhwyx", query, key).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, nl_emb_dim=None, norm_groups=1, dropout=0, with_attn=False,
                 resname='ResnetBlock'):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = eval(resname)(dim, dim_out, nl_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb, kwargs={}):
        x = self.res_block(x, time_emb, **kwargs)
        if (self.with_attn):
            x = self.attn(x)
        return x


from utils.util import patch_forward, patch_forward_guide


class DY3h(nn.Module):
    def __init__(self, in_channel=6, out_channel=3, inner_channel=32, norm_groups=1, channel_mults=[1, 2, 4, 8, 8],
                 attn_res=[8], res_blocks=3, dropout=0, with_noise_level_emb=True, image_size=128,
                 resname='ResnetBlockDY3h', ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(PositionalEncoding(inner_channel),
                                                 nn.Linear(inner_channel, inner_channel * 4), Swish(),
                                                 nn.Linear(inner_channel * 4, inner_channel))
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(pre_channel, channel_mult, nl_emb_dim=noise_level_channel,
                                                norm_groups=norm_groups, dropout=dropout, with_attn=use_attn,
                                                resname=resname))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, nl_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True, resname=resname),
            ResnetBlocWithAttn(pre_channel, pre_channel, nl_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False, resname=resname)])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(
                    ResnetBlocWithAttn(pre_channel + feat_channels.pop(), channel_mult, nl_emb_dim=noise_level_channel,
                                       norm_groups=norm_groups, dropout=dropout, with_attn=use_attn, resname=resname))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)
        self.prec = pre_channel
        # self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)
        # for dynamic networks
        dim = self.prec
        dim_out = default(out_channel, in_channel)
        self.final_conv = nn.Sequential(nn.GroupNorm(1, dim), Swish(),
                                        nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
                                        nn.Conv2d(dim, dim_out, 3, padding=1))

    def naiveforward(self, x, time, guide):
        t = self.noise_level_mlp(time) if exists(self.noise_level_mlp) else None

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t, {'guide': guide})
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t, {'guide': guide})
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t, {'guide': guide})
            else:
                x = layer(x)

        return self.final_conv(x)

    def forward(self, x, time, guide):
        _, _, h, w = x.shape

        if h * w > 1024 * 1024:  # guide image should be split to patches
            return patch_forward_guide(x, self.naiveforward, params={'time': time, 'guide': guide}, skip=1024,
                                       padding=64)
        else:
            # return self.naiveforward(x, time, guide)
            fac = 32
            padh, padw = (h // fac + 1) * fac - h, (w // fac + 1) * fac - w
            x = F.pad(x, (0, padw, 0, padh), mode='reflect')
            guide = F.pad(guide, (0, padw, 0, padh), mode='reflect')
            return self.naiveforward(x, time, guide)[..., :-padh, :-padw]


class UNetSeeInDark(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetSeeInDark, self).__init__()

        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.conv1_1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv10_1 = nn.Conv2d(32, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        _, _, h, w = x.shape
        fac = 32
        padh, padw = (h // fac + 1) * fac - h, (w // fac + 1) * fac - w
        x = F.pad(x, (0, padw, 0, padh), mode='reflect')
        # print('unet ', x.shape, padh, padw)
        return self.naive_forward(x)[..., :-padh, :-padw]

    def naive_forward(self, x):
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)

        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)

        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)

        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)

        conv5 = self.lrelu(self.conv5_1(pool4))
        conv5 = self.lrelu(self.conv5_2(conv5))

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(up8))
        conv8 = self.lrelu(self.conv8_2(conv8))

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))

        conv10 = self.conv10_1(conv9)
        # out = nn.functional.pixel_shuffle(conv10, 2)
        out = conv10
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    def lrelu(self, x):
        outt = torch.max(0.2 * x, x)
        return outt
