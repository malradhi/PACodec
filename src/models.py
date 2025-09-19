import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d, BatchNorm2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from utils import init_weights, get_padding
from stft import TorchSTFT

LRELU_SLOPE = 0.1




def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))

        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.in_ch=in_ch
        self.out_ch=out_ch
        self.conv1 = conv3x3(in_ch, out_ch)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.ca = ChannelAttention(out_ch)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride
        self.down_conv=conv3x3(in_ch,out_ch)
    def forward(self, x):
        residual = x


        out = self.conv1(x)

        out = self.bn1(out)
        
        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        out = self.ca(out) * out
        
        out = self.sa(out) * out

        if self.in_ch == self.out_ch:
            out = residual + out
            out = self.relu(out)
        else:
            out = self.relu(out + self.down_conv(residual))

        return out
    

    
class Encoder(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        
        self.encs = nn.ModuleList()
        middle = h.n_blocks//2 + 1
        for i in range(1, h.n_blocks+1):
            if i < middle:
                self.encs.append(ResBlock(3,3))
            elif i == middle:
                self.encs.append(ResBlock(3,1))


        self.encs_masked = nn.ModuleList()
        for i in range(2):
            self.encs_masked.append(ResBlock(3,3))
        self.encs_masked.append(ResBlock(3,1))
            
        self.final_encs = nn.ModuleList()

        self.final_encs.append(ResBlock(2,1))
        for i in range(5):
            self.final_encs.append(ResBlock(1,1))


        self.linear = nn.Linear(h.win_size//2+1,h.latent_dim)
        self.dropout = nn.Dropout(h.latent_dropout)

    def apply_f0_mask(self, x, f0, sample_rate=22050, n_fft=1024, sigma=2.0):
        f0=f0.squeeze(1)
        # Extract tensor dimensions and device
        batch_size, channels, n_freq_bins, time_steps = x.shape
        device = x.device
        
        # Convert f0 from Hz to frequency bin indices
        hz_per_bin = sample_rate / n_fft
        f0_bins = f0 / hz_per_bin  # Shape: (B, T)  
        # Create frequency bin indices tensor
        freq_indices = torch.arange(n_freq_bins, device=device).view(1, 1, n_freq_bins)  # Shape: (1, 1, N)
              
        # Reshape f0_bins for broadcasting
        f0_bins = f0_bins.unsqueeze(-1)  # Shape: (B, T, 1)
        
        # Create Gaussian mask centered at f0_bins
        # exp(-((x - μ)² / (2σ²)))
        gaussian_mask = torch.exp(-((freq_indices - f0_bins) ** 2) / (2 * sigma ** 2))  # Shape: (B, T, N)
        
        # Reshape mask to match the input tensor dimensions
        # From (B, T, N) to (B, 1, N, T)

        mask = gaussian_mask.permute(0, 2, 1).unsqueeze(1)  # Shape: (B, 1, N, T)
        
        
        # Broadcasting: The mask has shape (B, 1, N, T) and will be broadcast to (B, C, N, T)
        # This applies the same mask to all channels
        masked_output= x * mask




        return masked_output

    def forward(self, x,f0):
        # x: (B, 4, N, T)
        masked_x=self.apply_f0_mask(x,f0)
        for enc_block in self.encs:
          x = enc_block(x)

        for block in self.encs_masked:
            masked_x=block(masked_x)
        
        z=torch.cat([x,masked_x],dim=1)
        for block in self.final_encs:
            z=block(z)
        
        z = z.squeeze(1).transpose(1,2) # (B, 1, N, T) -> (B, T, N)
        z = self.linear(z)
        #! Apply dropout (according to DAE) to increase decoder robustness,
        #! because representation predicted from AM is used in TTS application.
        z = self.dropout(z)
        
        return z
    
class Generator(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        
        self.linear = nn.Linear(h.latent_dim, h.win_size//2+1)
        self.decs = nn.ModuleList()
        middle = h.n_blocks//2 + 1
        for i in range(1, h.n_blocks+1):
            if i < middle:
                self.decs.append(ResBlock(1,1))
            elif i == middle:
                self.decs.append(ResBlock(1,4))
            else:
                self.decs.append(ResBlock(4,4))
                
        self.dec_istft_input = h.dec_istft_input

        self.conv_post = Conv2d(4,2,3,1,padding=1) # Predict Real/Img (default) or Magitude/Phase

        
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft = TorchSTFT(filter_length=h.n_fft, hop_length=h.hop_size, win_length=h.win_size)
        
    
    def forward(self, x):
        # x: (B, T, D)
        x = self.linear(x)
        x = x.transpose(1,2).unsqueeze(1)
        for dec_block in self.decs:
            x = dec_block(x)
        
        # (B, 4, N, T)
        x = F.leaky_relu(x)
        x = x.contiguous().view(x.size(0),-1,x.size(-1)) # (B, 4N, T)
        x = self.reflection_pad(x)
        x = x.contiguous().view(x.size(0),4,-1,x.size(-1)) # (B, 4N, T') -> (B, 4, N, T')
        # (B, 4, N, T') -> (B, 2, N, T') (default) or (B, 4, N, T')
        x = self.conv_post(x)
        


        magnitude = x[:,0,:,:]
        phase = x[:,1,:,:]
        wav = self.stft.polar_inverse(magnitude, phase)

            
        return wav

class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses