import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange
import math
from torchstat import stat

from basicsr.archs.arch_util import flow_warp 

def to(x):
    return {'device': x.device, 'dtype': x.dtype}

def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim = 2)
    flat_x = rearrange(x, 'b l c -> b (l c)')
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x

def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim = 2, k = r)
    return logits

class RelPosEmb(nn.Module):
    def __init__(
        self,
        block_size,
        rel_size,
        dim_head
    ):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, 'b (x y) c -> b x y c', x = block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')

        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        return rel_logits_w + rel_logits_h

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class RestormerLayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(RestormerLayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    
    def compute_flops(self, x):
        N,C,H,W = x.shape
        hidden_features = int(C*2.66)
        # project_in
        project_in_flops = C * 2*hidden_features * H * W
        # dwconv
        dwconv_flops = hidden_features * 2 * H * W * 9
        # project_out
        project_out_flops = hidden_features * C * H * W
        flops = project_in_flops + dwconv_flops + project_out_flops

        return flops


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(ChannelAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
    def compute_flops(self, x):
        N,C,H,W = x.shape
        proj_qkv_flops = C * 3 * C * H * W
        dwconv_flops = C * 3 * H * W * 9
        project_out_flops = C * C * H * W

        attn_matmul_flops = 2 * C * C * H * W
        attn_out_flops = C * C * H * W

        flops = proj_qkv_flops + dwconv_flops + project_out_flops + attn_matmul_flops + attn_out_flops
        return flops
        
##########################################################################
## Overlapping Cross-Attention (OCA)
class OCAB(nn.Module):
    def __init__(self, dim, window_size, overlap_ratio, num_heads, dim_head, bias):
        super(OCAB, self).__init__()
        self.num_spatial_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.dim_head = dim_head
        self.inner_dim = self.dim_head * self.num_spatial_heads
        self.scale = self.dim_head**-0.5

        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size, padding=(self.overlap_win_size-window_size)//2)
        self.qkv = nn.Conv2d(self.dim, self.inner_dim*3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=bias)
        self.rel_pos_emb = RelPosEmb(
            block_size = window_size,
            rel_size = window_size + (self.overlap_win_size - window_size),
            dim_head = self.dim_head
        )
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        qs, ks, vs = qkv.chunk(3, dim=1)

        # spatial attention
        qs = rearrange(qs, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1 = self.window_size, p2 = self.window_size)
        ks, vs = map(lambda t: self.unfold(t), (ks, vs))
        ks, vs = map(lambda t: rearrange(t, 'b (c j) i -> (b i) j c', c = self.inner_dim), (ks, vs))

        # print(f'qs.shape:{qs.shape}, ks.shape:{ks.shape}, vs.shape:{vs.shape}')
        #split heads
        qs, ks, vs = map(lambda t: rearrange(t, 'b n (head c) -> (b head) n c', head = self.num_spatial_heads), (qs, ks, vs))

        # attention
        qs = qs * self.scale
        spatial_attn = (qs @ ks.transpose(-2, -1))
        spatial_attn += self.rel_pos_emb(qs)
        spatial_attn = spatial_attn.softmax(dim=-1)

        out = (spatial_attn @ vs)

        out = rearrange(out, '(b h w head) (p1 p2) c -> b (head c) (h p1) (w p2)', head = self.num_spatial_heads, h = h // self.window_size, w = w // self.window_size, p1 = self.window_size, p2 = self.window_size)

        # merge spatial and channel
        out = self.project_out(out)

        return out
def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError

def batch_index_fill(x, x1, x2, idx1, idx2):
    B, N, C = x.size()
    B, N1, C = x1.size()
    B, N2, C = x2.size()

    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1)
    idx1 = idx1 + offset * N
    idx2 = idx2 + offset * N

    x = x.reshape(B*N, C)

    x[idx1.reshape(-1)] = x1.reshape(B*N1, C)
    x[idx2.reshape(-1)] = x2.reshape(B*N2, C)

    x = x.reshape(B, N, C)
    return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
class PredictorLG(nn.Module):
    """ Importance Score Predictor
    """
    def __init__(self, dim, window_size=8, k=4,ratio=0.5):
        super().__init__()

        self.ratio = ratio
        self.window_size = window_size
        cdim = dim + k
        embed_dim = window_size**2
        
        self.in_conv = nn.Sequential(
            nn.Conv2d(cdim, cdim//4, 1),
            LayerNorm(cdim//4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.out_mask = nn.Sequential(
            nn.Linear(embed_dim, window_size),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(window_size, 2),
            nn.Softmax(dim=-1)
        )

        self.out_SA = nn.Sequential(
            nn.Conv2d(cdim//4, 1, 3, 1, 1),
            nn.Sigmoid(),
        )        


    def forward(self, input_x, mask=None, ratio=0.5, training = False):

        x = self.in_conv(input_x)

        sa = self.out_SA(x)
        
        x = torch.mean(x, keepdim=True, dim=1) 

        x = rearrange(x,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        B, N, C = x.size()

        pred_score = self.out_mask(x)
        mask = F.gumbel_softmax(pred_score, hard=True, dim=2)[:, :, 0:1]

        if training:
            return mask, sa
        else:
            score = pred_score[:, : , 0]
            B, N = score.shape
            r = torch.mean(mask,dim=(0,1))*1.0
            if self.ratio == 1:
                num_keep_node = N #int(N * r) #int(N * r)
            else:
                num_keep_node = min(int(N * r * 2 * self.ratio), N)
            idx = torch.argsort(score, dim=1, descending=True)
            idx1 = idx[:, :num_keep_node]
            idx2 = idx[:, num_keep_node:]
            return [idx1, idx2], sa

    def compute_flops(self, x):
        N,C,H,W = x.shape
        C = C + 4
        p = self.window_size
        h, w = H//p, W//p
        # in_conv
        in_conv_flops = C * C//4 * H * W

        # out_mask
        linear1_flops = h * w * p**3
        linear2_flops = 2* h * w * p

        # out_SA
        out_SA_flops = C//4 * H * W * 9

        flops = in_conv_flops + linear1_flops + linear2_flops + out_SA_flops

        return flops

class CAMixer(nn.Module):
    def __init__(self, dim, window_size=8, bias=True, is_deformable=True, num_heads = 4, dim_head = 16,overlap_ratio = 0.5,  ratio=0.5):
        super().__init__()    

        self.dim = dim
        self.window_size = window_size
        self.is_deformable = is_deformable
        self.ratio = ratio

        self.num_heads = num_heads
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.dim_head = dim_head
        self.inner_dim = self.dim_head * self.num_heads
        self.scale = self.dim_head**-0.5


        self.inner_dim = self.dim_head * self.num_heads

        k = 3
        d = 2

        #self.proj_qkv = nn.Conv2d(self.dim, self.inner_dim*3, kernel_size=1, bias=bias)
        self.proj_v = nn.Conv2d(self.dim, self.inner_dim, kernel_size=1, bias=bias)
        self.proj_q = nn.Conv2d(self.dim, self.inner_dim, kernel_size=1, bias=bias)
        self.proj_k = nn.Conv2d(self.dim, self.inner_dim, kernel_size=1, bias=bias)

        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size, padding=(self.overlap_win_size-window_size)//2)
        self.project_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=bias)
        self.rel_pos_emb = RelPosEmb(
            block_size = window_size,
            rel_size = window_size + (self.overlap_win_size - window_size),
            dim_head = self.dim_head
        )

        # Predictor
        self.route = PredictorLG(dim = self.inner_dim,window_size = window_size,ratio=ratio)

    def forward(self,x,condition_global=None, mask=None, training = False):
        N,C,H,W = x.shape

        qs = self.proj_q(x)
        ks = self.proj_k(x)
        vs = self.proj_v(x)

        if self.is_deformable:
            condition_wind = torch.stack(torch.meshgrid(torch.linspace(-1,1,self.window_size),torch.linspace(-1,1,self.window_size)))\
                    .type_as(x).unsqueeze(0).repeat(N, 1, H//self.window_size, W//self.window_size)
            if condition_global is None:
                _condition = torch.cat([vs, condition_wind], dim=1)
            else:
                _condition = torch.cat([vs, condition_global, condition_wind], dim=1)

        mask, sa = self.route(_condition,ratio=self.ratio, training=training)
        # easy attn
        v_out_easy = vs*sa
        # #print("mask", mask)
        # if training:
        #     # spatial attention
        #     qs = rearrange(qs, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1 = self.window_size, p2 = self.window_size)
        #     ks, vs = map(lambda t: self.unfold(t), (ks, vs))
        #     ks, vs = map(lambda t: rearrange(t, 'b (c j) i -> (b i) j c', c = self.inner_dim), (ks, vs))

        #     # print(f'qs.shape:{qs.shape}, ks.shape:{ks.shape}, vs.shape:{vs.shape}')
        #     #split heads
        #     qs, ks, vs = map(lambda t: rearrange(t, 'b n (head c) -> (b head) n c', head = self.num_heads), (qs, ks, vs))

        #     # attention
        #     qs = qs * self.scale
        #     spatial_attn = (qs @ ks.transpose(-2, -1))
        #     spatial_attn += self.rel_pos_emb(qs)
        #     spatial_attn = spatial_attn.softmax(dim=-1)

        #     v_out_hard = (spatial_attn @ vs)
        #     v_out_hard = rearrange(v_out_hard, '(b h w head) (p1 p2) c -> b (head c) (h p1) (w p2)', head = self.num_heads, h = H // self.window_size, w = W // self.window_size, p1 = self.window_size, p2 = self.window_size)

        #     v_out_easy = rearrange(v_out_easy,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        #     v_out_hard = rearrange(v_out_hard,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        
            
        #     out = v_out_hard*mask + v_out_easy*(1-mask) 
                
        #     out = rearrange(out, 'b (h w) (dh dw c) -> b c (h dh) (w dw)', dh=self.window_size, dw=self.window_size,h = H // self.window_size, w = W // self.window_size)
        #     out = self.project_out(out)
            
        #     return out, torch.mean(mask,dim=1)
            
        # else:
        if True:
            qs = rearrange(qs, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.window_size, p2 = self.window_size)
            ks, vs = map(lambda t: self.unfold(t), (ks, vs))
            ks, vs = map(lambda t: rearrange(t, 'b (c j) i -> b i (j c)', c = self.inner_dim), (ks, vs))

            idx1, idx2 = mask
            qs = batch_index_select(qs, idx1)
            ks = batch_index_select(ks, idx1)
            vs = batch_index_select(vs, idx1)

            qs, ks, vs = map(lambda t: rearrange(t, 'b n (j c) -> (b n) j c', c = self.inner_dim), (qs, ks, vs))

            qs, ks, vs = map(lambda t: rearrange(t, 'b n (head c) -> (b head) n c', head = self.num_heads), (qs, ks, vs))

            # attention
            print(f'qs.shape:{qs.shape}, ks.shape:{ks.shape}, vs.shape:{vs.shape}')
            qs = qs * self.scale
            spatial_attn = (qs @ ks.transpose(-2, -1))
            spatial_attn += self.rel_pos_emb(qs)
            spatial_attn = spatial_attn.softmax(dim=-1)

            v_out_hard = (spatial_attn @ vs)
            v1 = rearrange(v_out_hard, '(b j head) (p1 p2) c -> b j (p1 p2 head c)',b = N, head = self.num_heads, p1 = self.window_size, p2 = self.window_size)
            v2 = rearrange(v_out_easy,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
            v2 = batch_index_select(v2, idx2)
            v_out = torch.cat([v1, v2], dim=1)
            out = batch_index_fill(v_out.clone(), v1.clone(), v2.clone(), idx1, idx2)

            out = rearrange(out, 'b (h w) (dh dw c) -> b c (h dh) (w dw)', dh=self.window_size, dw=self.window_size,h = H // self.window_size, w = W // self.window_size)
            out = self.project_out(out)


        return out

    def compute_flops(self, x):
        N,C,H,W = x.shape
        # proj_qkv
        proj_qkv_flops = 2 * C * 3* self.inner_dim * H * W

        # predictor
        predictor_flops = self.route.compute_flops(x)

        # easy attn
        easy_attn_flops = C * H * W

        # hard attn
        hard_attn_flops = 0
        h, w = H//self.window_size, W//self.window_size
        p = self.window_size
        p2 = self.overlap_win_size
        # qs = qs * self.scale
        # spatial_attn = (qs @ ks.transpose(-2, -1))
        hard_attn_flops += self.ratio*(H * W * C + h * w * p**2 * self.inner_dim * p2**2)
        # spatial_attn += self.rel_pos_emb(qs)
        hard_attn_flops += self.ratio* h * w * p**2 * p2**2  
        # v_out_hard = (spatial_attn @ vs)
        hard_attn_flops += self.ratio* h * w * p**2 * self.inner_dim * p2**2    
        # proj_out
        proj_out_flops = self.inner_dim * C * H * W
        flops = proj_qkv_flops + predictor_flops + easy_attn_flops + hard_attn_flops + proj_out_flops
        return flops
        
##########################################################################
class CATransformerBlock(nn.Module):
    def __init__(self, dim, window_size, ratio, num_channel_heads,  ffn_expansion_factor, bias, LayerNorm_type, num_heads = 4, dim_head = 16, overlap_ratio = 0.5):
        super(CATransformerBlock, self).__init__()


        #self.spatial_attn = OCAB(dim, window_size, overlap_ratio, num_spatial_heads, spatial_dim_head, bias)
        self.spatial_attn = CAMixer(dim,window_size=window_size,ratio=ratio,num_heads = num_heads, dim_head = dim_head,overlap_ratio = overlap_ratio)
        self.channel_attn = ChannelAttention(dim, num_channel_heads, bias)

        self.norm1 = RestormerLayerNorm(dim, LayerNorm_type)
        self.norm2 = RestormerLayerNorm(dim, LayerNorm_type)
        self.norm3 = RestormerLayerNorm(dim, LayerNorm_type)
        self.norm4 = RestormerLayerNorm(dim, LayerNorm_type)

        self.channel_ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.spatial_ffn = FeedForward(dim, ffn_expansion_factor, bias)



    def forward(self, x, global_condition=None, training = False):
        if training:
            x = x + self.channel_attn(self.norm1(x))
            x = x + self.channel_ffn(self.norm2(x))
            y, decision = self.spatial_attn(self.norm3(x), global_condition, training=training)
            x  = x + y
            x = x + self.spatial_ffn(self.norm4(x))
            return x, decision
        else:
            x = x + self.channel_attn(self.norm1(x))
            x = x + self.channel_ffn(self.norm2(x))
            x = x + self.spatial_attn(self.norm3(x), global_condition, training=training)
            x = x + self.spatial_ffn(self.norm4(x))
            return x
    
    def compute_flops(self, x):
        N,C,H,W = x.shape
        # channel_attn
        channel_attn_flops = self.channel_attn.compute_flops(x)
        # channel_ffn
        channel_ffn_flops = self.channel_ffn.compute_flops(x)
        # spatial_attn
        spatial_attn_flops = self.spatial_attn.compute_flops(x)
        # spatial_ffn
        spatial_ffn_flops = self.spatial_ffn.compute_flops(x)
        flops = channel_attn_flops + channel_ffn_flops + spatial_attn_flops + spatial_ffn_flops
        print("channel_attn_flops",channel_attn_flops)  
        print("channel_ffn_flops",channel_ffn_flops)
        print("spatial_attn_flops",spatial_attn_flops)
        print("spatial_ffn_flops",spatial_ffn_flops)
        print("\n")
        return flops

    

##########################################################################
class ChannelTransformerBlock(nn.Module):
    def __init__(self, dim, num_channel_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(ChannelTransformerBlock, self).__init__()

        self.channel_attn = ChannelAttention(dim, num_channel_heads, bias)
        self.norm1 = RestormerLayerNorm(dim, LayerNorm_type)
        self.norm2 = RestormerLayerNorm(dim, LayerNorm_type)

        self.channel_ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.channel_attn(self.norm1(x))
        x = x + self.channel_ffn(self.norm2(x))
        return x
    


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class SR_Upsample(nn.Sequential):
    """SR_Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of features.
    """

    def __init__(self, scale, num_feat):
        m = []

        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, kernel_size = 3, stride = 1, padding = 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(SR_Upsample, self).__init__(*m)


##---------- Prompt Module -----------------------
class PromptBlock(nn.Module):
    def __init__(self,  window_size, overlap_ratio, num_channel_heads, num_spatial_heads, 
                 spatial_dim_head, ffn_expansion_factor, bias, LayerNorm_type,
                 prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192, 
                ):
        super(PromptBlock,self).__init__()

        # prompt generation
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)

        # prompt interaction
        self.attn = ChannelTransformerBlock(dim=lin_dim + prompt_dim, window_size = window_size, 
                                     overlap_ratio=overlap_ratio,  num_channel_heads=num_channel_heads, 
                                     num_spatial_heads=num_spatial_heads, spatial_dim_head = spatial_dim_head, 
                                     ffn_expansion_factor=ffn_expansion_factor, bias=bias, 
                                     LayerNorm_type=LayerNorm_type) 
        self.conv = nn.Conv2d(prompt_dim+lin_dim,lin_dim,kernel_size=3,stride=1,padding=1,bias=False)
        

    def forward(self,x):
        # input x shape is [B, HW, C]
        B, C, H, W = x.shape
        # prompt generation
        emb = x.mean(dim=(-2,-1))
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear", align_corners=True)
        prompt = self.conv3x3(prompt)

        # x shape [B, C + C_p, H, W]
        x = torch.cat([x, prompt], 1)
        x = self.attn(x)
        x = self.conv(x)

        return x
    
##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(PromptGenBlock,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        

    def forward(self,x):
        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1))
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt



class XRestormerLayer(nn.Module):
    def __init__(self, dim, depth, window_size, ratio,  num_channel_heads, ffn_expansion_factor, bias, LayerNorm_type, num_heads, dim_head, overlap_ratio):
        super(XRestormerLayer, self).__init__()
        self.layer = nn.Sequential(*[CATransformerBlock(dim=dim, window_size = window_size, ratio = ratio, num_channel_heads=num_channel_heads,  ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, num_heads = num_heads, dim_head = dim_head, overlap_ratio = overlap_ratio) for i in range(depth)])

    def forward(self, x, global_condition=None, training = False):
        if training:
            decision_avg = 0
            for layer in self.layer:
                x, decision = layer(x, global_condition, training = training)
                decision_avg += decision
            decision_avg /= len(self.layer)
            return x, decision_avg
        else:
            for layer in self.layer:
                x = layer(x, global_condition, training = training)
            return x

    def compute_flops(self, x):
        flops = 0
        for layer in self.layer:
            flops += layer.compute_flops(x)
        return flops
                



##########################################################################


class CAPromptXRestormerEffv2(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        dim = 48,
        num_blocks = [4,6,6,8],
        num_refinement_blocks = 4,
        channel_heads = [1,2,4,8],
        spatial_heads = [1,2,4,8],
        overlap_ratio = 0.5,
        dim_head = 16,
        ratio = 0.5,
        window_size = 8,
        bias = False,
        ffn_expansion_factor = 2.66,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        scale = 1,
        prompt = True
    ):

        super(CAPromptXRestormerEffv2, self).__init__()
        print("Initializing XRestormer")
        self.scale = scale
        self.ratio = ratio

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = XRestormerLayer(dim=dim, window_size = window_size, ratio = ratio, num_channel_heads=channel_heads[0],  ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, depth=num_blocks[0], num_heads = spatial_heads[0], dim_head = dim_head, overlap_ratio = overlap_ratio)

        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = XRestormerLayer(dim=int(dim*2**1), window_size = window_size, ratio = ratio,   num_channel_heads=channel_heads[1],  ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, depth=num_blocks[1], num_heads = spatial_heads[1], dim_head = dim_head, overlap_ratio = overlap_ratio)

        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = XRestormerLayer(dim=int(dim*2**2), window_size = window_size, ratio = ratio,   num_channel_heads=channel_heads[2],  ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, depth=num_blocks[2], num_heads = spatial_heads[2], dim_head = dim_head, overlap_ratio = overlap_ratio)

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = XRestormerLayer(dim=int(dim*2**3), window_size = window_size, ratio = ratio,   num_channel_heads=channel_heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, depth=num_blocks[3], num_heads = spatial_heads[3], dim_head = dim_head, overlap_ratio = overlap_ratio)

        #self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), window_size = window_size, overlap_ratio=0.5,  num_channel_heads=channel_heads[3], num_spatial_heads=8, spatial_dim_head = 16, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])


        self.up4_3 = Upsample(int(dim*2**2)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**1) + 192, int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = XRestormerLayer(dim=int(dim*2**2), window_size = window_size, ratio = ratio,  num_channel_heads=channel_heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, depth = num_blocks[2], num_heads = spatial_heads[2], dim_head = dim_head, overlap_ratio = overlap_ratio)


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = XRestormerLayer(dim=int(dim*2**1), window_size = window_size, ratio = ratio, num_channel_heads=channel_heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, depth = num_blocks[1], num_heads = spatial_heads[1], dim_head = dim_head, overlap_ratio = overlap_ratio)

        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = XRestormerLayer(dim=int(dim*2**1), window_size = window_size, ratio = ratio,  num_channel_heads=channel_heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, depth = num_blocks[0], num_heads = spatial_heads[0], dim_head = dim_head, overlap_ratio = overlap_ratio)

        self.refinement = XRestormerLayer(dim=int(dim*2**1), window_size = window_size, ratio = ratio,  num_channel_heads=channel_heads[0],  ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, depth=num_refinement_blocks, num_heads = spatial_heads[0], dim_head = dim_head, overlap_ratio = overlap_ratio)

        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.prompt = prompt
        if prompt:
            self.prompt1 = PromptGenBlock(prompt_dim=64,prompt_len=5,prompt_size = 64,lin_dim = 96)
            self.prompt2 = PromptGenBlock(prompt_dim=128,prompt_len=5,prompt_size = 32,lin_dim = 192)
            self.prompt3 = PromptGenBlock(prompt_dim=320,prompt_len=5,prompt_size = 16,lin_dim = 384)

            self.noise_level1 = ChannelTransformerBlock(dim=int(dim*2**1)+64, num_channel_heads = 1, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            self.reduce_noise_level1 = nn.Conv2d(int(dim*2**1)+64,int(dim*2**1),kernel_size=1,bias=bias)

            self.noise_level2 = ChannelTransformerBlock(dim=int(dim*2**1) + 224,  num_channel_heads = 1, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            self.reduce_noise_level2 = nn.Conv2d(int(dim*2**1)+224,int(dim*2**2),kernel_size=1,bias=bias)

            self.noise_level3 = ChannelTransformerBlock(dim=int(dim*2**2) + 512,  num_channel_heads = 1, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            self.reduce_noise_level3 = nn.Conv2d(int(dim*2**2)+512,int(dim*2**2),kernel_size=1,bias=bias)
        
        self.global_predictor = nn.Sequential(nn.Conv2d(dim, 8, 1, 1, 0, bias=True),
                                        nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                        nn.Conv2d(8, 2, 3, 1, 1, bias=True),
                                        nn.LeakyReLU(negative_slope=0.1, inplace=True))

                

    def forward(self, inp_img, training = False):
        if self.scale > 1:
            inp_img = F.interpolate(inp_img, scale_factor=self.scale, mode='bilinear', align_corners=False)
        B, C, H, W = inp_img.shape
        inp_enc_level1 = self.patch_embed(inp_img)
        condition_global = self.global_predictor(inp_enc_level1)
        condition_global_level2  = F.interpolate(condition_global, size=(H //2, W //2), mode='bilinear', align_corners=False)
        condition_global_level3  = F.interpolate(condition_global, size=(H //4, W //4), mode='bilinear', align_corners=False)
        condition_global_level4  = F.interpolate(condition_global, size=(H //8, W //8), mode='bilinear', align_corners=False)
        if training:
            # Encoder1
            
            decision_avg = 0
            out_enc_level1, decision = self.encoder_level1(inp_enc_level1, condition_global, training = training)
            inp_enc_level2 = self.down1_2(out_enc_level1)
            decision_avg += decision
            

            # Encoder2
            out_enc_level2, decision = self.encoder_level2(inp_enc_level2, condition_global_level2, training = training)
            inp_enc_level3 = self.down2_3(out_enc_level2)
            decision_avg += decision
            
            # Encoder3
            out_enc_level3, decision = self.encoder_level3(inp_enc_level3, condition_global_level3, training = training)
            inp_enc_level4 = self.down3_4(out_enc_level3)
            decision_avg += decision
            
            # Bottleneck
            latent, decision = self.latent(inp_enc_level4, condition_global_level4, training = training)
            decision_avg += decision

            if self.prompt:
                dec3_param = self.prompt3(latent)
                latent = torch.cat([latent, dec3_param], 1)
                latent = self.noise_level3(latent)
                latent = self.reduce_noise_level3(latent)


            inp_dec_level3 = self.up4_3(latent)
            inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
            inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
            out_dec_level3, decision = self.decoder_level3(inp_dec_level3, condition_global_level3, training = training)
            decision_avg += decision

            if self.prompt:
                dec2_param = self.prompt2(out_dec_level3)
                out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
                out_dec_level3 = self.noise_level2(out_dec_level3)
                out_dec_level3 = self.reduce_noise_level2(out_dec_level3)
            

            inp_dec_level2 = self.up3_2(out_dec_level3)
            inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
            inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
            out_dec_level2, decision = self.decoder_level2(inp_dec_level2, condition_global_level2, training = training)
            decision_avg += decision

            if self.prompt:
                dec1_param = self.prompt1(out_dec_level2)
                out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
                out_dec_level2 = self.noise_level1(out_dec_level2)
                out_dec_level2 = self.reduce_noise_level1(out_dec_level2)


            inp_dec_level1 = self.up2_1(out_dec_level2)
            inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
            out_dec_level1, decision = self.decoder_level1(inp_dec_level1, condition_global, training = training)
            decision_avg += decision

            out_dec_level1, decision = self.refinement(out_dec_level1, condition_global, training = training)
            decision_avg += decision
            out_dec_level1 = self.output(out_dec_level1) + inp_img

            decision_avg /= 8

            ratio_loss =  2*self.ratio*(torch.mean(decision_avg)-0.5)**2 

            return out_dec_level1, ratio_loss

        else:
            out_enc_level1 = self.encoder_level1(inp_enc_level1, condition_global, training = training)
            inp_enc_level2 = self.down1_2(out_enc_level1)
            out_enc_level2 = self.encoder_level2(inp_enc_level2, condition_global_level2, training = training)
            inp_enc_level3 = self.down2_3(out_enc_level2)
            out_enc_level3 = self.encoder_level3(inp_enc_level3, condition_global_level3, training = training)
            inp_enc_level4 = self.down3_4(out_enc_level3)
            latent = self.latent(inp_enc_level4, condition_global_level4, training = training)

            if self.prompt:
                dec3_param = self.prompt3(latent)
                latent = torch.cat([latent, dec3_param], 1)
                latent = self.noise_level3(latent)
                latent = self.reduce_noise_level3(latent)

            inp_dec_level3 = self.up4_3(latent)
            inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
            inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
            out_dec_level3 = self.decoder_level3(inp_dec_level3, condition_global_level3, training = training)

            if self.prompt:
                dec2_param = self.prompt2(out_dec_level3)
                out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
                out_dec_level3 = self.noise_level2(out_dec_level3)
                out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

            inp_dec_level2 = self.up3_2(out_dec_level3)
            inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
            inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
            out_dec_level2 = self.decoder_level2(inp_dec_level2, condition_global_level2, training = training)

            if self.prompt:
                dec1_param = self.prompt1(out_dec_level2)
                out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
                out_dec_level2 = self.noise_level1(out_dec_level2)
                out_dec_level2 = self.reduce_noise_level1(out_dec_level2)

            inp_dec_level1 = self.up2_1(out_dec_level2)
            inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
            out_dec_level1 = self.decoder_level1(inp_dec_level1, condition_global, training = training)

            out_dec_level1 = self.refinement(out_dec_level1, condition_global, training = training)
            out_dec_level1 = self.output(out_dec_level1) + inp_img

            return out_dec_level1

    def compute_flops(self, x):
        N,C,H,W = x.shape
        x_level1 = self.patch_embed(x)
        x_level2 = self.down1_2(x_level1)
        x_level3 = self.down2_3(x_level2)
        x_level4 = self.down3_4(x_level3)
        # patch_embed
        patch_embed_flops = 3 * H * W * C * 3 * 3 * 48
        # encoder_level1
        encoder_level1_flops = self.encoder_level1.compute_flops(x_level1)
        # encoder_level2
        encoder_level2_flops = self.encoder_level2.compute_flops(x_level2)
        # encoder_level3
        encoder_level3_flops = self.encoder_level3.compute_flops(x_level3)
        # latent
        latent_flops = self.latent.compute_flops(x_level4)
        # decoder_level3
        decoder_level3_flops = self.decoder_level3.compute_flops(x_level3)
        # decoder_level2
        decoder_level2_flops = self.decoder_level2.compute_flops(x_level2)
        # decoder_level1
        decoder_level1_flops = self.decoder_level1.compute_flops(x_level1)
        # refinement
        refinement_flops = self.refinement.compute_flops(x_level1)
        flops = patch_embed_flops + encoder_level1_flops + encoder_level2_flops + encoder_level3_flops + latent_flops + decoder_level3_flops + decoder_level2_flops + decoder_level1_flops + refinement_flops
        return flops

if __name__ == "__main__":
    training = False
    model = CAPromptXRestormerEffv2(
        inp_channels=3,
        out_channels=3,
        dim = 48,
        num_blocks = [2,4,4,4],
        num_refinement_blocks = 4,
        channel_heads = [1,1,1,1],
        spatial_heads = [1,2,4,8],
        overlap_ratio = 0.5,
        dim_head = 16,
        ratio = 0.5,
        window_size = 8,
        bias = False,
        ffn_expansion_factor = 2.66,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        scale = 1,
        prompt = True
        )
    
    # torchstat
    # x = torch.randn(1, 3, 64, 64)
    # if training:
    #     y, ratio_loss = model(x, training=True)
    
    #     print("output shape", y.shape)
    #     print("Complex ratio: %.2f"%ratio_loss.item())
    # else:
    #     y = model(x, training=False)
    #     print("output shape", y.shape)
    # print('# model_restoration parameters: %.2f M'%(sum(param.numel() for param in model.parameters())/ 1e6))
    # stat(model, (3, 512, 512))

    # COMPUTE FLOPS 
    from utils_modelsummary import get_model_activation, get_model_flops
    with torch.no_grad():
        input_dim = (3, 64, 64)  # set the input dimension
        activations, num_conv2d = get_model_activation(model, input_dim)
        print('{:>16s} : {:<.4f} [M]'.format('#Activations', activations/10**6))
        print('{:>16s} : {:<d}'.format('#Conv2d', num_conv2d))
        flops = get_model_flops(model, input_dim, False)
        print('{:>16s} : {:<.4f} [G]'.format('FLOPs', flops/10**9))
        num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
        print('{:>16s} : {:<.4f} [M]'.format('#Params', num_parameters/10**6))