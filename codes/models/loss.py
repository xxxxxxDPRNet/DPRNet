import torch
import torch.nn as nn
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from kornia.color import rgb_to_lab

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss
class SSIM_Loss(nn.Module):
    def __init__(self): 
        super(SSIM_Loss, self).__init__()
        self.ssim = SSIM(data_range=1.0, size_average=True)
    def forward(self, x, y):

        return 1 - self.ssim(x,y)
class MS_SSIM_Loss(nn.Module):
    def __init__(self, channel):
        super(MS_SSIM_Loss, self).__init__()
        self.mssim = MS_SSIM(data_range=1.0, size_average=True, channel=channel)
    def forward(self, x, y):

        return 1 - self.mssim(x, y)

# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss
    
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.vgg.eval()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4]    

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg) - 1):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss
    
# class VGGLoss(nn.Module):
#     def __init__(self):
#         super(VGGLoss, self).__init__()        
#         self.vgg = Vgg11().cuda()
#         self.vgg.eval()

#         self.criterion = nn.L1Loss()
#         self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4]        

#     def forward(self, x, y):
#         # print(y.shape)
#         y_vgg = self.vgg(y)

#         for i in range(len(y_vgg)):
#             y_vgg[i] = y_vgg[i].detach().cpu()
#         torch.cuda.empty_cache()

#         for i in range(len(y_vgg)):
#             y_vgg[i] = y_vgg[i].to(x)

#         x_vgg = self.vgg(x)
#         loss = 0
#         for i in range(len(x_vgg) - 1):
#             loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i])      
  
#         return loss
    
class ColorLossDetaE(nn.Module):
    def __init__(self):
        super(ColorLossDetaE, self).__init__()

    def forward(self, inp, tar):
        lab_inp, lab_tar = rgb_to_lab(inp), rgb_to_lab(tar)
        l1, a1, b1 = torch.moveaxis(lab_inp, 1, 0)[:3]
        l2, a2, b2 = torch.moveaxis(lab_tar, 1, 0)[:3]
        return torch.mean(torch.sqrt((l2 - l1) ** 2 + (a2 - a1) ** 2 + (b2 - b1) ** 2))

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
    
class Vgg11(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg11, self).__init__()
        vgg_pretrained_features = models.vgg11(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        # self.slice3 = torch.nn.Sequential()
        # self.slice4 = torch.nn.Sequential()
        # self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(1, 2):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(7, 12):
        #     self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(12, 21):
        #     self.slice4.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(21, 30):
        #     self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        # h_relu3 = self.slice3(h_relu2)        
        # h_relu4 = self.slice4(h_relu3)        
        # h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2]
        return out