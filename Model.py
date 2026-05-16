import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TaskAdapter(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.dim = dim
        self.mid_channels = dim // reduction

        self.down_conv = nn.Conv2d(dim, self.mid_channels, 1, bias=False)
        self.conv_main = nn.Conv2d(self.mid_channels, self.mid_channels, 1, bias=False)
        self.conv_aux = nn.Conv2d(self.mid_channels, self.mid_channels, 3, padding=1, bias=False)

        self.gelu = nn.GELU()

        self.bn = nn.BatchNorm2d(self.mid_channels * 2)
        self.silu = nn.SiLU(inplace=True)
        self.up_conv = nn.Conv2d(self.mid_channels * 2, dim, 1, bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        x_down = self.down_conv(x)
        x_main = self.conv_main(x_down)
        x_aux = self.conv_aux(x_down)
        x_spatial = x_main + x_aux

        x_fft = torch.fft.fft2(x)
        x_mag = torch.abs(x_fft)
        x_freq = self.gelu(x_mag)
        x_freq = self.down_conv(x_freq)

        x_cat = torch.cat([x_spatial, x_freq], dim=1)
        x_cat = self.bn(x_cat)
        x_cat = self.silu(x_cat)
        x_fusion = self.up_conv(x_cat)

        x_att = self.avg_pool(x)
        x_att = self.silu(x_att)
        alpha = self.sigmoid(x_att)
        x_out = x_fusion * alpha

        return x_out

class MSTAF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.scale = (dim // 8) ** -0.5

        self.q1 = nn.Linear(dim, dim)
        self.kv1 = nn.Linear(dim, dim*2)
        self.q2 = nn.Linear(dim, dim)
        self.kv2 = nn.Linear(dim, dim*2)
        self.attn_proj = nn.Conv2d(dim*2, dim, 1)

        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv3 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv5 = nn.Conv2d(dim, dim, 5, padding=2)
        self.multi_proj = nn.Conv2d(dim*3, dim, 1)

        self.freq_proj = nn.Conv2d(dim, dim, 1)

        self.gate_conv = nn.Conv2d(dim, dim, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.silu = nn.SiLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        self.weight_net = nn.Linear(dim, 2)

        self.res_conv_inter = nn.Conv2d(dim, dim, 1)
        self.res_conv_seg = nn.Conv2d(dim, dim, 1)

    def forward(self, F_inter, F_seg):
        B, C, H, W = F_inter.shape

        f_inter = F_inter.flatten(2).transpose(-2, -1)
        f_seg = F_seg.flatten(2).transpose(-2, -1)

        Q1 = self.q1(f_inter)
        K1, V1 = self.kv1(f_seg).chunk(2, dim=-1)
        A1 = (Q1 @ K1.transpose(-2,-1) * self.scale).softmax(dim=-1) @ V1

        Q2 = self.q2(f_seg)
        K2, V2 = self.kv2(f_inter).chunk(2, dim=-1)
        A2 = (Q2 @ K2.transpose(-2,-1) * self.scale).softmax(dim=-1) @ V2

        A = torch.cat([A1, A2], dim=-1)
        A = A.transpose(-2,-1).reshape(B, 2*C, H, W)
        F_att = self.attn_proj(A)

        f1 = self.conv1(F_att)
        f3 = self.conv3(F_att)
        f5 = self.conv5(F_att)
        F_multi = self.multi_proj(torch.cat([f1,f3,f5], dim=1))

        fft = torch.fft.fft2(F_multi)
        mag, phase = torch.abs(fft), torch.angle(fft)
        mag = self.freq_proj(mag)
        F_freq = torch.fft.ifft2(mag * torch.exp(1j * phase)).real

        gate_feat = self.gate_conv(F_seg)
        g = self.global_pool(gate_feat).flatten(1)
        g = self.silu(g)
        w1, w2 = self.softmax(self.weight_net(g)).chunk(2, dim=1)
        w1 = w1.view(B,1,1,1)
        w2 = w2.view(B,1,1,1)

        F_fusion = w1 * F_freq + w2 * F_freq
        Out_int = F_fusion + self.res_conv_inter(F_inter)
        Out_seg = F_fusion + self.res_conv_seg(F_seg)

        return Out_int, Out_seg

class CrossTaskInteraction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.dilated_conv1 = nn.Conv2d(dim, dim, 3, padding=1, dilation=1)
        self.dilated_conv2 = nn.Conv2d(dim, dim, 3, padding=2, dilation=2)
        self.dilated_conv3 = nn.Conv2d(dim, dim, 3, padding=3, dilation=3)
        self.silu = nn.SiLU(inplace=True)

        self.temp_proj = nn.Conv2d(dim, dim, 1)

        self.bn = nn.BatchNorm2d(dim)

        self.ca_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_conv = nn.Conv2d(dim, dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.gate_conv = nn.Conv2d(dim*2, dim, 1)
        self.gelu = nn.GELU()

        self.out_conv_inter = nn.Conv2d(dim, dim, 1)
        self.out_conv_seg = nn.Conv2d(dim, dim, 1)

    def forward(self, F_interp, F_seg):

        x_s = self.silu(self.dilated_conv1(F_interp))
        x_s = self.silu(self.dilated_conv2(x_s))
        x_s = self.silu(self.dilated_conv3(x_s))

        x_t = self.temp_proj(F_seg)

        fft_s = torch.fft.fft2(x_s)
        fft_t = torch.fft.fft2(x_t)
        f_align = torch.abs(fft_s * torch.conj(fft_t))

        f_bn = self.bn(f_align)
        w_ca = self.sigmoid(self.ca_conv(self.ca_pool(f_bn)))
        f_ca = f_bn * w_ca

        f_cat = torch.cat([f_ca, f_align], dim=1)
        g = self.sigmoid(self.gelu(self.gate_conv(f_cat)))
        f_gate = g * f_align

        out_int = self.out_conv_inter(f_gate)
        out_seg = self.out_conv_seg(f_gate)

        return out_int, out_seg

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        coords = torch.meshgrid([torch.arange(window_size), torch.arange(window_size)], indexing='ij')
        coords = torch.stack(coords)
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim)
        )

    def window_partition(self, x):
        B, C, H, W = x.shape
        x = x.permute(0,2,3,1)
        x = x.view(B, H//self.window_size, self.window_size, W//self.window_size, self.window_size, C)
        windows = x.permute(0,1,3,2,4,5).contiguous().view(-1, self.window_size**2, C)
        return windows

    def window_reverse(self, windows, B, H, W):
        C = windows.shape[-1]
        x = windows.view(B, H//self.window_size, W//self.window_size, self.window_size, self.window_size, C)
        x = x.permute(0,1,3,2,4,5).contiguous().view(B, H, W, C)
        return x.permute(0,3,1,2)

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x

        x = x.permute(0,2,3,1).reshape(B, H*W, C)
        x = self.norm1(x).reshape(B, H, W, C).permute(0,3,1,2)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2,3))

        x_windows = self.window_partition(x)
        attn_windows = self.attn(x_windows)
        x = self.window_reverse(attn_windows, B, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2,3))

        x = shortcut + x

        shortcut = x
        x = x.permute(0,2,3,1).reshape(B, H*W, C)
        x = self.norm2(x)
        x = self.mlp(x).reshape(B, H, W, C).permute(0,3,1,2)
        x = shortcut + x

        return x

class PatchEmbed(nn.Module):
    def __init__(self, in_chans=2, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, 4, stride=4)
    def forward(self, x):
        return self.proj(x)

class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4*dim)
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0,2,3,1)
        x = torch.cat([x[:,0::2,0::2], x[:,1::2,0::2], x[:,0::2,1::2], x[:,1::2,1::2]], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x.permute(0,3,1,2)

class PatchExpand(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.expand = nn.Linear(dim, 2*dim, bias=False)
        self.norm = nn.LayerNorm(dim//2)
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0,2,3,1)
        x = self.expand(x).view(B, H, W, 2, 2, C//2).permute(0,1,3,2,4,5).reshape(B, H*2, W*2, C//2)
        x = self.norm(x)
        return x.permute(0,3,1,2)

class MultiTaskSwinUNet(nn.Module):
    def __init__(self, img_size=512, in_chans=2, out_chans=1):
        super().__init__()
        self.embed_dim = 96
        self.depths = [2,4,8,16]
        self.num_heads = [2,4,8,16]
        self.dims = [96,192,384,768]
        self.window_size = 8

        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=self.embed_dim)

        self.encoder_layers = nn.ModuleList()
        self.talas = nn.ModuleList()
        dim = self.embed_dim
        for i in range(4):
            self.encoder_layers.append(SwinTransformerBlock(dim, self.num_heads[i], self.window_size, shift_size=0 if i%2==0 else 4))
            self.talas.append(TaskAdapter(dim))
            if i < 3:
                self.encoder_layers.append(PatchMerging(dim))
                dim *= 2

        self.mstaf_layers = nn.ModuleList([MSTAF(d) for d in [384,192,96]])
        self.ctci_layers = nn.ModuleList([CrossTaskInteraction(d) for d in [384,192,96]])
        self.decoder_expands = nn.ModuleList([PatchExpand(768), PatchExpand(384), PatchExpand(192)])
        self.decoder_blocks = nn.ModuleList([
            SwinTransformerBlock(384, 8, 8),
            SwinTransformerBlock(192, 4, 8),
            SwinTransformerBlock(96, 2, 8)
        ])

        self.interp_head = nn.Sequential(
            nn.Conv2d(96, 384, 3, 1, 1), nn.PixelShuffle(2), nn.GELU(),
            nn.Conv2d(96, 384, 3, 1, 1), nn.PixelShuffle(2), nn.GELU(),
            nn.Conv2d(96, out_chans, 3, 1, 1), nn.Sigmoid()
        )
        self.seg_head = nn.Sequential(
            nn.Conv2d(96, 384, 3, 1, 1), nn.PixelShuffle(2), nn.GELU(),
            nn.Conv2d(96, 384, 3, 1, 1), nn.PixelShuffle(2), nn.GELU(),
            nn.Conv2d(96, 3*out_chans, 3, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.patch_embed(x)
        skips = []
        idx = 0

        for i in range(4):
            x = self.encoder_layers[idx](x)
            idx +=1
            x = self.talas[i](x)
            skips.append(x)
            if i <3:
                x = self.encoder_layers[idx](x)
                idx +=1

        f_inter, f_seg = x, x

        for i in range(3):
            f_inter = self.decoder_expands[i](f_inter)
            f_seg = self.decoder_expands[i](f_seg)

            skip = skips[2-i]
            f_inter, f_seg = self.mstaf_layers[i](f_inter, skip)

            f_inter = self.decoder_blocks[i](f_inter)
            f_seg = self.decoder_blocks[i](f_seg)

            f_inter, f_seg = self.ctci_layers[i](f_inter, f_seg)

        interp_pred = self.interp_head(f_inter)
        seg_pred = self.seg_head(f_seg)

        return interp_pred, seg_pred

def test_model_pipeline():
    print("="*60)
    print("TASC-SwinMT Paper-Aligned Version - Test Pipeline")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MultiTaskSwinUNet().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params/1e6:.2f}M")
    
    model.eval()
    with torch.no_grad():
        x = torch.randn(2,2,512,512).to(device)
        interp, seg = model(x)
        
        print(f"\nInput shape: {x.shape}")
        print(f"Interpolation output shape: {interp.shape}")
        print(f"Segmentation output shape: {seg.shape}")
    

if __name__ == "__main__":
    test_model_pipeline()