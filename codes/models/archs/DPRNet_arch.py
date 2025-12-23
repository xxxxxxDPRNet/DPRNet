import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
import cv2
from models.archs.arch_util import do_interpolation_lut
import pdb
from models.archs.CSRNet_arch import CSRNet


class DPRNet(nn.Module):

    def __init__(self, lay_nums=4, gridnum=8):
        super(DPRNet, self).__init__() 

        self.HF_Net = HF_Net()
        self.LF_Net = LF_Net(num_high=3)

        self.LF_mapping_Net = LF_mapping_v2()

        self.Recon_Net = Recon_Net_v2()

    def forward(self, x, phase):
        HF = self.HF_Net(x)  # HF-List:[0]-[B, 3, 480, 640], [1]-[B, 3, 240, 320],[2]-[B, 3, 120, 160]
        LF = self.LF_Net(x)  # [1, 3, 60, 80]
        sr_mapping = self.LF_mapping_Net(LF)  # [1, 3, 60, 80]

        # sr-List:[3]-[B, 3, 480, 640], [2]-[B, 3, 240, 320],[1]-[B, 3, 120, 160], [0]-[1, 3, 60, 80]
        sr_imgs = self.Recon_Net(sr_mapping, HF)  
        if phase == 'train':
            return sr_imgs, HF
        else:
            return sr_imgs[-1]


# get input HF
class HF_Net(nn.Module):
    def __init__(self):
        super(HF_Net, self).__init__()    
        self.init_conv = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.doglayers = nn.ModuleList([DoGlayer(in_channels=12, out_channels=12) for _ in range(3)])
        
        self.hfmaplayers = nn.ModuleList([nn.Sequential(
                                        nn.Conv2d(12 * 3, 12, 3, padding=1),
                                        nn.LeakyReLU(),
                                        ResidualBlock(12),
                                        nn.Conv2d(12, 3, 3, padding=1)) 
                                        for _ in range(3)])
    def forward(self, x):
        x = self.init_conv(x)
        outputs = []
        for doglayer, hfmaplayer in zip(self.doglayers, self.hfmaplayers):
            conv_feat, hf_feat = doglayer(x)
            hf = hfmaplayer(hf_feat)
            outputs.append(hf)
            x = conv_feat

        return outputs
    
# get input LF
class LF_Net(nn.Module):
    def __init__(self, num_high=3):
        super(LF_Net, self).__init__()    
        self.num_high = num_high
        self.kernel = self.gauss_kernel()
    
    def gauss_kernel(self, device=torch.device('cuda'), channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.cuda()
        return kernel
    
    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out
    
    def downsample(self, x):
        return x[:, :, ::2, ::2]
    
    def forward(self, x):
        down = x
        for _ in range(self.num_high):
            filtered = self.conv_gauss(down, self.kernel)
            down = self.downsample(filtered)      
        return down
    
class LF_mapping_v2(nn.Module):
    def __init__(self):
        super(LF_mapping_v2, self).__init__()    

        self.globaltmo = CSRNet()
        self.grid = [4,4]

        self.lightcontext = nn.Sequential(
                                nn.Conv2d(3, 8, 3, stride=2, padding=1),
                                nn.LeakyReLU(0.2),
                                nn.InstanceNorm2d(8, affine=True),

                                nn.Conv2d(8, 16, 3, stride=2, padding=1),
                                nn.LeakyReLU(0.2),
                                nn.InstanceNorm2d(16, affine=True), 

                                nn.Conv2d(16, 24, 3, stride=2, padding=1),
                                nn.LeakyReLU(0.2),
                                nn.InstanceNorm2d(24, affine=True),              
                                nn.Dropout(p=0.5),
                                nn.AdaptiveAvgPool2d(2)
                                )        
        # From SepLUT
        self.lut3d_generator = nn.ModuleList([
            LUT3DGenerator(n_colors=3, n_vertices=9, n_feats=6, n_ranks=3) for _ in range(16)
        ])
           
        self.init_weights()


    def forward(self, input):
        globaltmo = self.globaltmo(input)

        input_lut = F.interpolate(globaltmo, size=(60,80), mode='bilinear', align_corners=False)        
        context = self.lightcontext(input_lut).view(input.shape[0], 16, -1)

        output = torch.zeros_like(input)
        for i in range(input.shape[0]):
            lut3ds = []
            for j in range(len(self.lut3d_generator)):
                _, lut3d = self.lut3d_generator[j](context[i,j].unsqueeze(0))
                lut3ds.append(lut3d.squeeze())
            lut3ds = torch.stack(lut3ds, dim=0)
            outs = do_interpolation_lut(globaltmo[i].permute(1,2,0), lut3ds, [4, 4])
            output[i] = outs.permute(2,0,1)
             
        return output 
    def init_weights(self):

        def special_initilization(m):
            classname = m.__class__.__name__
            if 'Conv' in classname:
                if hasattr(m, 'weight'):
                    nn.init.xavier_normal_(m.weight.data)
            elif 'InstanceNorm' in classname:
                if m.weight is not None:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
        self.apply(special_initilization)

class Recon_Net_v2(nn.Module):
    def __init__(self):
        super(Recon_Net_v2, self).__init__() 
        in_channels = 6
        f_number = 16
        self.padding_mode = 'reflect'
        self.act = nn.GELU()

        
        self.reconmodel = nn.Sequential(nn.Conv2d(in_channels, f_number, 3, padding=1),
                          nn.LeakyReLU(),
                          ResidualBlock(f_number),
                          ResidualBlock(f_number),
                          ResidualBlock(f_number),
                          nn.Conv2d(f_number, 3, 3, padding=1),
                          nn.LeakyReLU())
        
        self.trans_mask_blocks = nn.Sequential(
                nn.Conv2d(in_channels, 16, 1),
                ResidualBlock(16),
                ResidualBlock(16),
                ResidualBlock(16),
                nn.LeakyReLU(),
                nn.Conv2d(16, 3, 1))
        
        self.trans_mask_blocks_v1 = nn.Sequential(
                nn.Conv2d(in_channels, 8, 1),
                ResidualBlock(8),
                ResidualBlock(8),
                ResidualBlock(8),
                nn.LeakyReLU(),
                nn.Conv2d(8, 3, 1))

    def forward(self, translf, input_hfs):
        outputlayers = [translf]
        for i in range(len(input_hfs)):
            if i == 0:
                upsampled_lf = F.interpolate(translf, scale_factor=2)
                hf_concat = torch.cat([upsampled_lf, input_hfs[-(i+1)]], dim=1)
                current_output = self.reconmodel(hf_concat) + upsampled_lf
                outputlayers.append(current_output)
            elif i ==1:
                upsampled_lf = F.interpolate(current_output, scale_factor=2)
                hf_concat = torch.cat([upsampled_lf, input_hfs[-(i+1)]], dim=1)
                current_output = self.trans_mask_blocks(hf_concat) + upsampled_lf
                outputlayers.append(current_output)
            else:
                upsampled_lf = F.interpolate(current_output, scale_factor=2)
                hf_concat = torch.cat([upsampled_lf, input_hfs[-(i+1)]], dim=1)
                current_output = self.trans_mask_blocks_v1(hf_concat) + upsampled_lf
                outputlayers.append(current_output)
        return outputlayers

class DoGlayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoGlayer, self).__init__()

        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels ,kernel_size=3, stride=1, padding=1),
                                    nn.LeakyReLU(inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=out_channels, out_channels=out_channels ,kernel_size=3, stride=1, padding=1),
                                    nn.LeakyReLU(inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=out_channels, out_channels=out_channels ,kernel_size=3, stride=1, padding=1),
                                    nn.LeakyReLU(inplace=True))
        self.pool = nn.MaxPool2d(2)


    def forward(self, x):
        
        g1 = self.conv_1(x)
        g2 = self.conv_2(g1)
        g3 = self.conv_3(g2)
        
        l1 = x - g1
        l2 = g1 - g2
        l3 = g2 - g3

        feat_cat = torch.cat([l1, l2, l3], dim=1)

        g3 = self.pool(g3)
        return g3, feat_cat
    
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_features, in_features, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)
    
class LUT3DGenerator(nn.Module):
    r"""The 3DLUT generator module.

    Args:
        n_colors (int): Number of input color channels.
        n_vertices (int): Number of sampling points along each lattice dimension.
        n_feats (int): Dimension of the input image representation vector.
        n_ranks (int): Number of ranks (or the number of basis LUTs).
    """

    def __init__(self, n_colors, n_vertices, n_feats, n_ranks) -> None:
        super().__init__()

        # h0
        self.weights_generator = nn.Linear(n_feats, n_ranks)
        # h1
        self.basis_luts_bank = nn.Linear(
            n_ranks, n_colors * (n_vertices ** n_colors), bias=False)
        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.n_ranks = n_ranks
        #Ȩ�س�ʼ��
        self.init_weights()


    def init_weights(self):
        r"""Init weights for models.

        For the mapping f (`backbone`) and h (`lut_generator`), we follow the initialization in
            [TPAMI 3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).

        """
        nn.init.ones_(self.weights_generator.bias)
        identity_lut = torch.stack([
            torch.stack(
                torch.meshgrid(*[torch.arange(self.n_vertices) for _ in range(self.n_colors)]),
                dim=0).true_divide(self.n_vertices - 1).flip(0),
            *[torch.zeros(
                self.n_colors, *((self.n_vertices,) * self.n_colors)) for _ in range(self.n_ranks - 1)]
            ], dim=0).view(self.n_ranks, -1)
        self.basis_luts_bank.weight.data.copy_(identity_lut.t())

    def forward(self, x):
        
        weights = self.weights_generator(x)
        luts = self.basis_luts_bank(weights)
        luts = luts.view(x.shape[0], -1, *((self.n_vertices,) * self.n_colors))
        return weights, luts

class Trans_low_net(nn.Module):
    def __init__(self, num_residual_blocks):
        super(Trans_low_net, self).__init__()

        model = [nn.Conv2d(3, 16, 3, padding=1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 64, 3, padding=1),
            nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]

        model += [nn.Conv2d(64, 16, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 3, 3, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = x + self.model(x)
        out = torch.tanh(out)
        return out   

if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis
    model = DPRNet()
    inputs = torch.randn(1, 3, 480, 640)  
    for name, module in model.named_children():
        flops = FlopCountAnalysis(module, inputs)
        print(f"{name}: {flops.total() / 1e9:.2f} GFLOPs")