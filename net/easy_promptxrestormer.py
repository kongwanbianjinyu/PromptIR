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

def round_to_nearest_power_of_2(x):
    if x & (x - 1) == 0:  # Step 1: Check if x is already a power of 2
        return x
    msb_pos = x.bit_length() - 1  # Step 2: Find MSB position
    lower_bound = 1 << msb_pos  # Step 3: Calculate lower bound
    upper_bound = 1 << (msb_pos + 1)  # Step 4: Calculate upper bound
    midpoint = (upper_bound + lower_bound) // 2  # Calculate midpoint
    if x < midpoint:  # Step 5 & 6: Compare and decide to round down or up
        return lower_bound
    else:
        return upper_bound
##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class EasyFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(EasyFeedForward, self).__init__()

        ffn_channel = int(ffn_expansion_factor * dim)
        ffn_channel = round_to_nearest_power_of_2(ffn_channel)
        #print("FFN Channel: ", ffn_channel)
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.sg = SimpleGate()
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):

        x = self.conv1(x)
        x = self.sg(x)
        x = self.conv2(x)
        x = self.project_out(x)
        return x
    

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



class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class EasyChannelAttention(nn.Module):
    def __init__(self, dim,num_channel_heads, bias):
        super(EasyChannelAttention, self).__init__()
        dw_channel = dim
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        out = self.project_out(x)
        return out



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
        
class EasySpatialAttention(nn.Module):
    def __init__(self, dim, inner_dim = 64, bias = True):
        super().__init__()    

        self.dim = dim
        self.inner_dim = inner_dim
        self.proj_v = nn.Conv2d(self.dim, self.inner_dim, kernel_size=1, bias=bias)
        

        self.in_conv = nn.Sequential(
            nn.Conv2d(inner_dim, inner_dim//4, 1),
            LayerNorm(inner_dim//4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.out_SA = nn.Sequential(
            nn.Conv2d(inner_dim//4, 1, 3, 1, 1),
            nn.Sigmoid(),
        )        
        self.project_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=bias)


    def forward(self,x):
        N,C,H,W = x.shape
        vs = self.proj_v(x)
        sa = self.in_conv(vs)
        sa = self.out_SA(sa)

        v_out_easy = vs*sa

        out = self.project_out(v_out_easy)

        return out
##########################################################################
class EasyTransformerBlock(nn.Module):
    def __init__(self, dim, num_channel_heads,  ffn_expansion_factor, bias, LayerNorm_type, inner_dim):
        super(EasyTransformerBlock, self).__init__()

        self.spatial_attn = EasySpatialAttention(dim,inner_dim = inner_dim, bias = bias)
        self.channel_attn = EasyChannelAttention(dim, num_channel_heads, bias)

        self.norm1 = RestormerLayerNorm(dim, LayerNorm_type)
        self.norm2 = RestormerLayerNorm(dim, LayerNorm_type)
        self.norm3 = RestormerLayerNorm(dim, LayerNorm_type)
        self.norm4 = RestormerLayerNorm(dim, LayerNorm_type)

        self.channel_ffn = EasyFeedForward(dim, ffn_expansion_factor, bias)
        self.spatial_ffn = EasyFeedForward(dim, ffn_expansion_factor, bias)



    def forward(self, x):
        x = x + self.channel_attn(self.norm1(x))
        x = x + self.channel_ffn(self.norm2(x))
        x = x + self.spatial_attn(self.norm3(x))
        x = x + self.spatial_ffn(self.norm4(x))
        return x

    

##########################################################################
class ChannelTransformerBlock(nn.Module):
    def __init__(self, dim, num_channel_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(ChannelTransformerBlock, self).__init__()

        self.channel_attn = EasyChannelAttention(dim, num_channel_heads, bias)
        self.norm1 = RestormerLayerNorm(dim, LayerNorm_type)
        self.norm2 = RestormerLayerNorm(dim, LayerNorm_type)

        self.channel_ffn = EasyFeedForward(dim, ffn_expansion_factor, bias)

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
    def __init__(self, dim, depth, num_channel_heads, ffn_expansion_factor, bias, LayerNorm_type, inner_dim):
        super(XRestormerLayer, self).__init__()
        self.layer = nn.Sequential(*[EasyTransformerBlock(dim=dim, num_channel_heads = num_channel_heads,  ffn_expansion_factor= ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, inner_dim=inner_dim) for i in range(depth)])

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x
                



##########################################################################


class EasyPromptXRestormer(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        dim = 48,
        num_blocks = [2,4,4,4],
        num_refinement_blocks = 4,
        channel_heads = [1,1,1,1],
        inner_dim = [16,32,64,128],
        bias = False,
        ffn_expansion_factor = 2.66,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        scale = 1,
        prompt = True
    ):

        super(EasyPromptXRestormer, self).__init__()
        print("Initializing XRestormer")
        self.scale = scale

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = XRestormerLayer(dim=dim,  num_channel_heads=channel_heads[0],  ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, depth=num_blocks[0], inner_dim = inner_dim[0])

        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = XRestormerLayer(dim=int(dim*2**1), num_channel_heads=channel_heads[1],  ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, depth=num_blocks[1], inner_dim = inner_dim[1])

        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = XRestormerLayer(dim=int(dim*2**2), num_channel_heads=channel_heads[2],  ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, depth=num_blocks[2], inner_dim = inner_dim[2])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = XRestormerLayer(dim=int(dim*2**3),  num_channel_heads=channel_heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, depth=num_blocks[3], inner_dim = inner_dim[3])

        #self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), window_size = window_size, overlap_ratio=0.5,  num_channel_heads=channel_heads[3], num_spatial_heads=8, spatial_dim_head = 16, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])


        self.up4_3 = Upsample(int(dim*2**2)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**1) + 192, int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = XRestormerLayer(dim=int(dim*2**2),  num_channel_heads=channel_heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, depth = num_blocks[2], inner_dim = inner_dim[2])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = XRestormerLayer(dim=int(dim*2**1), num_channel_heads=channel_heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, depth = num_blocks[1], inner_dim = inner_dim[1])

        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = XRestormerLayer(dim=int(dim*2**1),  num_channel_heads=channel_heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, depth = num_blocks[0], inner_dim = inner_dim[0])

        self.refinement = XRestormerLayer(dim=int(dim*2**1), num_channel_heads=channel_heads[0],  ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, depth=num_refinement_blocks, inner_dim = inner_dim[0])

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

                

    def forward(self, inp_img):
        if self.scale > 1:
            inp_img = F.interpolate(inp_img, scale_factor=self.scale, mode='bilinear', align_corners=False)
        B, C, H, W = inp_img.shape
        inp_enc_level1 = self.patch_embed(inp_img)
        
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        if self.prompt:
            dec3_param = self.prompt3(latent)
            latent = torch.cat([latent, dec3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        if self.prompt:
            dec2_param = self.prompt2(out_dec_level3)
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        if self.prompt:
            dec1_param = self.prompt1(out_dec_level2)
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1

if __name__ == "__main__":
    device = "cuda:0"
    model = EasyPromptXRestormer(
        inp_channels=3,
        out_channels=3,
        dim = 48,
        num_blocks = [2,4,4,4],
        num_refinement_blocks = 4,
        channel_heads= [1,1,1,1],
        inner_dim = [16, 32, 64, 128],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        scale = 1,prompt = True
        ).to(device)
    
    # torchstat
    x = torch.randn(1, 3, 64, 64).to(device)
    y = model(x)
    # if training:
    #     y, ratio_loss = model(x, training=True)
    
    #     print("output shape", y.shape)
    #     print("Complex ratio: %.2f"%ratio_loss.item())
    # else:
    #     y = model(x, training=False)
    #     print("output shape", y.shape)
    # print('# model_restoration parameters: %.2f M'%(sum(param.numel() for param in model.parameters())/ 1e6))
    # # stat(model, (3, 512, 512))

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

    # Result
    # Spatial Attention: 0.2237 [G]
    # Spatial FFN: 2.1706 [G]
    # Channel Attention: 1.1219 [G]
    # Channel FFN: 2.1706 [G]
    # Others: 1.4205 [G]
    # Total: 7.1071 [G]
