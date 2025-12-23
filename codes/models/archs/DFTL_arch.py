import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import cv2 as cv

class CNN3(nn.Module):
	def __init__(self, channels=[3*42*42, 3*42*7, 3*7*7, 3*7, 3]):
		super(CNN3, self).__init__()
		self.relu = nn.ReLU(inplace=True)
		self.e_conv1 = nn.Conv2d(channels[0],channels[1],1,1,0,bias=True)
		self.e_conv2 = nn.Conv2d(channels[1],channels[2],1,1,0,bias=True) 
		self.e_conv3 = nn.Conv2d(channels[2],channels[3],1,1,0,bias=True)
		self.e_conv4 = nn.Conv2d(channels[3],channels[4],1,1,0,bias=True)
	
	def forward(self, x):
		x1 = self.relu((self.e_conv1(x))) #8
		x2 = self.relu((self.e_conv2(x1))) #4
		x3 = self.relu((self.e_conv3(x2))) #2
		y = self.e_conv4(x3) #1
		return y



class denoiseNet_adaptive_thre6(nn.Module):

	def __init__(self, tsize=16, stride=4, in_size=224):
		super(denoiseNet_adaptive_thre6, self).__init__()

		self.L = ((in_size-tsize)//stride+1)**2
		channels = [3*self.L, 3*self.L//4, 3*self.L//16, 3*self.L//4, 3*self.L]
		self.cnn = CNN3(channels)
		self.sigmoid = nn.Sigmoid()
		self.tsize = tsize
		self.stride = stride
		self.unfold = nn.Unfold(tsize, padding=0, stride=stride)
		self.fold = nn.Fold(in_size, tsize, padding=0, stride=stride)
		self.eps = 1e-20
	
	def forward(self, x):
		b,ch,h,w = x.shape
		scale = self.fold(self.unfold(torch.ones(x.shape, requires_grad=False)))
		if torch.cuda.is_available():
			scale = scale.cuda()

		tiles = self.unfold(x).permute(0,2,1).reshape(b,-1,self.tsize,self.tsize)
		dct_tiles = dct.dct_2d(tiles)
		masks = self.sigmoid(self.cnn(dct_tiles/(self.tsize*self.tsize)))
		clean_dct_tiles = masks * dct_tiles
		clean_tiles = dct.idct_2d(clean_dct_tiles)
		clean_imgs = self.fold(clean_tiles.reshape(b,-1,ch*self.tsize*self.tsize).permute(0,2,1))
		return clean_imgs / (scale)


class TFDL(nn.Module):
	def __init__(self):
		super(TFDL, self).__init__()
		self.max_lv = 3
		tsize = 16
		stride = 4
		self.scale1 = denoiseNet_adaptive_thre6(tsize, stride, 224)
		self.scale2 = denoiseNet_adaptive_thre6(tsize, stride, 112)
		self.scale3 = denoiseNet_adaptive_thre6(tsize, stride, 56)
		self.enhance = CSRNet()
		self.enhance1 = CSRNet()
		self.enhance2 = CSRNet()
		self.enhance3 = CSRNet()

	def forward(self, x):
		base_in, x3_in, x2_in, x1_in = build_laplacian_pyramid(x, max_level=self.max_lv)
		base = self.enhance(base_in)
		x3 = self.enhance1(x3_in)
		x3 = self.scale3(x3)
		x2 = self.enhance2(x2_in)
		x2 = self.scale2(x2)
		x1 = self.enhance3(x1_in)
		x1 = self.scale1(x1)
		enhanced = reconstruct([base, x3, x2, x1])
		return enhanced


class DFTL(nn.Module):
	def __init__(self):
		super(DFTL, self).__init__()
		self.max_lv = 3
		tsize = 16
		stride = 4
		self.scale1 = denoiseNet_adaptive_thre6(tsize, stride, 224)
		self.scale2 = denoiseNet_adaptive_thre6(tsize, stride, 112)
		self.scale3 = denoiseNet_adaptive_thre6(tsize, stride, 56)
		self.enhance = CSRNet()
		self.enhance1 = CSRNet()
		self.enhance2 = CSRNet()
		self.enhance3 = CSRNet()

	def forward(self, x):
		base_in, x3_in, x2_in, x1_in = build_laplacian_pyramid(x, max_level=self.max_lv)
		base = self.enhance(base_in)
		x3_d = self.scale3(x3_in)
		x3 = self.enhance3(x3_d)
		x2_d = self.scale2(x2_in)
		x2 = self.enhance2(x2_d)
		x1_d = self.scale1(x1_in)
		x1 = self.enhance1(x1_d)
		enhanced = reconstruct([base, x3, x2, x1])
		return enhanced
	
import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Condition(nn.Module):
    def __init__(self, in_nc=3, nf=32):
        super(Condition, self).__init__()
        stride = 2
        pad = 0
        self.pad = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(in_nc, nf, 7, stride, pad, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.conv3 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1_out = self.act(self.conv1(self.pad(x)))
        conv2_out = self.act(self.conv2(self.pad(conv1_out)))
        conv3_out = self.act(self.conv3(self.pad(conv2_out)))
        out = torch.mean(conv3_out, dim=[2, 3], keepdim=False)

        return out


# 3layers with control
class CSRNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, base_nf=64, cond_nf=32):
        super(CSRNet, self).__init__()

        self.base_nf = base_nf
        self.out_nc = out_nc

        self.cond_net = Condition(in_nc=in_nc, nf=cond_nf)
      
        self.cond_scale1 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_scale2 = nn.Linear(cond_nf, base_nf,  bias=True)
        self.cond_scale3 = nn.Linear(cond_nf, 3, bias=True)

        self.cond_shift1 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_shift2 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_shift3 = nn.Linear(cond_nf, 3, bias=True)

        self.conv1 = nn.Conv2d(in_nc, base_nf, 1, 1, bias=True) 
        self.conv2 = nn.Conv2d(base_nf, base_nf, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(base_nf, out_nc, 1, 1, bias=True)

        self.act = nn.ReLU(inplace=True)


    def forward(self, x):
        cond = self.cond_net(x)

        scale1 = self.cond_scale1(cond)
        shift1 = self.cond_shift1(cond)

        scale2 = self.cond_scale2(cond)
        shift2 = self.cond_shift2(cond)

        scale3 = self.cond_scale3(cond)
        shift3 = self.cond_shift3(cond)

        out = self.conv1(x)
        out = out * scale1.view(-1, self.base_nf, 1, 1) + shift1.view(-1, self.base_nf, 1, 1) + out
        out = self.act(out)
        

        out = self.conv2(out)
        out = out * scale2.view(-1, self.base_nf, 1, 1) + shift2.view(-1, self.base_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv3(out)
        out = out * scale3.view(-1, self.out_nc, 1, 1) + shift3.view(-1, self.out_nc, 1, 1) + out
        return out
	


from typing import Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d, filter2D

__all__ = [
    "PyrDown",
    "PyrUp",
    "ScalePyramid",
    "pyrdown",
    "pyrup",
    "build_pyramid",
    "build_laplacian_pyramid",
    "reconstruct",
    "reconstruct_laplacian",
]


def _get_pyramid_gaussian_kernel() -> torch.Tensor:
    """Utility function that return a pre-computed gaussian kernel."""
    return torch.tensor([[
        [1., 4., 6., 4., 1.],
        [4., 16., 24., 16., 4.],
        [6., 24., 36., 24., 6.],
        [4., 16., 24., 16., 4.],
        [1., 4., 6., 4., 1.]
    ]]) / 256.


class PyrDown(nn.Module):
    r"""Blurs a tensor and downsamples it.

    Args:
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail

    Return:
        torch.Tensor: the downsampled tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H / 2, W / 2)`

    Examples:
        >>> input = torch.rand(1, 2, 4, 4)
        >>> output = PyrDown()(input)  # 1x2x2x2
    """

    def __init__(self, border_type: str = 'reflect', align_corners: bool = False) -> None:
        super(PyrDown, self).__init__()
        self.border_type: str = border_type
        self.align_corners: bool = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return pyrdown(input, self.border_type, self.align_corners)



class PyrUp(nn.Module):
    r"""Upsamples a tensor and then blurs it.

    Args:
        borde_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail

    Return:
        torch.Tensor: the upsampled tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H * 2, W * 2)`

    Examples:
        >>> input = torch.rand(1, 2, 4, 4)
        >>> output = PyrUp()(input)  # 1x2x8x8
    """

    def __init__(self, border_type: str = 'reflect', align_corners: bool = False):
        super(PyrUp, self).__init__()
        self.border_type: str = border_type
        self.align_corners: bool = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return pyrup(input, self.border_type, self.align_corners)



class ScalePyramid(nn.Module):
    r"""Creates an scale pyramid of image, usually used for local feature
    detection. Images are consequently smoothed with Gaussian blur and
    downscaled.
    Arguments:
        n_levels (int): number of the levels in octave.
        init_sigma (float): initial blur level.
        min_size (int): the minimum size of the octave in pixels. Default is 5
        double_image (bool): add 2x upscaled image as 1st level of pyramid. OpenCV SIFT does this. Default is False
    Returns:
        Tuple(List(Tensors), List(Tensors), List(Tensors)):
        1st output: images
        2nd output: sigmas (coefficients for scale conversion)
        3rd output: pixelDists (coefficients for coordinate conversion)

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output 1st: :math:`[(B, C, NL, H, W), (B, C, NL, H/2, W/2), ...]`
        - Output 2nd: :math:`[(B, NL), (B, NL), (B, NL), ...]`
        - Output 3rd: :math:`[(B, NL), (B, NL), (B, NL), ...]`

    Examples::
        >>> input = torch.rand(2, 4, 100, 100)
        >>> sp, sigmas, pds = ScalePyramid(3, 15)(input)
    """

    def __init__(self,
                 n_levels: int = 3,
                 init_sigma: float = 1.6,
                 min_size: int = 15,
                 double_image: bool = False):
        super(ScalePyramid, self).__init__()
        # 3 extra levels are needed for DoG nms.
        self.n_levels = n_levels
        self.extra_levels: int = 3
        self.init_sigma = init_sigma
        self.min_size = min_size
        self.border = min_size // 2 - 1
        self.sigma_step = 2 ** (1. / float(self.n_levels))
        self.double_image = double_image
        return

    def __repr__(self) -> str:
        return self.__class__.__name__ +\
            '(n_levels=' + str(self.n_levels) + ', ' +\
            'init_sigma=' + str(self.init_sigma) + ', ' + \
            'min_size=' + str(self.min_size) + ', ' + \
            'extra_levels=' + str(self.extra_levels) + ', ' + \
            'border=' + str(self.border) + ', ' + \
            'sigma_step=' + str(self.sigma_step) + ', ' + \
            'double_image=' + str(self.double_image) + ')'

    def get_kernel_size(self, sigma: float):
        ksize = int(2.0 * 4.0 * sigma + 1.0)

        #  matches OpenCV, but may cause padding problem for small images
        #  PyTorch does not allow to pad more than original size.
        #  Therefore there is a hack in forward function

        if ksize % 2 == 0:
            ksize += 1
        return ksize

    def get_first_level(self, input):
        pixel_distance = 1.0
        cur_sigma = 0.5
        # Same as in OpenCV up to interpolation difference
        if self.double_image:
            x = F.interpolate(input, scale_factor=2.0, mode='bilinear', align_corners=False)
            pixel_distance = 0.5
            cur_sigma *= 2.0
        else:
            x = input
        if self.init_sigma > cur_sigma:
            sigma = max(math.sqrt(self.init_sigma**2 - cur_sigma**2), 0.01)
            ksize = self.get_kernel_size(sigma)
            cur_level = gaussian_blur2d(x, (ksize, ksize), (sigma, sigma))
            cur_sigma = self.init_sigma
        else:
            cur_level = x
        return cur_level, cur_sigma, pixel_distance

    def forward(self, x: torch.Tensor) -> Tuple[  # type: ignore
            List, List, List]:
        bs, ch, h, w = x.size()
        cur_level, cur_sigma, pixel_distance = self.get_first_level(x)

        sigmas = [cur_sigma * torch.ones(bs, self.n_levels + self.extra_levels).to(x.device).to(x.dtype)]
        pixel_dists = [pixel_distance * torch.ones(
                       bs,
                       self.n_levels + self.extra_levels).to(
                       x.device).to(x.dtype)]
        pyr = [[cur_level]]
        oct_idx = 0
        while True:
            cur_level = pyr[-1][0]
            for level_idx in range(1, self.n_levels + self.extra_levels):
                sigma = cur_sigma * math.sqrt(self.sigma_step**2 - 1.0)
                ksize = self.get_kernel_size(sigma)

                # Hack, because PyTorch does not allow to pad more than original size.
                # But for the huge sigmas, one needs huge kernel and padding...

                ksize = min(ksize, min(cur_level.size(2), cur_level.size(3)))
                if ksize % 2 == 0:
                    ksize += 1

                cur_level = gaussian_blur2d(
                    cur_level, (ksize, ksize), (sigma, sigma))
                cur_sigma *= self.sigma_step
                pyr[-1].append(cur_level)
                sigmas[-1][:, level_idx] = cur_sigma
                pixel_dists[-1][:, level_idx] = pixel_distance
            _pyr = pyr[-1][-self.extra_levels]
            nextOctaveFirstLevel = F.interpolate(_pyr, size=(_pyr.size(-2) // 2, _pyr.size(-1) // 2),
                                                 mode='nearest')  # Nearest matches OpenCV SIFT
            pixel_distance *= 2.0
            cur_sigma = self.init_sigma
            if (min(nextOctaveFirstLevel.size(2),
                    nextOctaveFirstLevel.size(3)) <= self.min_size):
                break
            pyr.append([nextOctaveFirstLevel])
            sigmas.append(cur_sigma * torch.ones(
                          bs,
                          self.n_levels + self.extra_levels).to(
                          x.device))
            pixel_dists.append(
                pixel_distance * torch.ones(
                    bs,
                    self.n_levels + self.extra_levels).to(
                    x.device))
            oct_idx += 1
        for i in range(len(pyr)):
            pyr[i] = torch.stack(pyr[i], dim=2)  # type: ignore
        return pyr, sigmas, pixel_dists



def pyrdown(
        input: torch.Tensor,
        border_type: str = 'reflect', align_corners: bool = False) -> torch.Tensor:
    r"""Blurs a tensor and downsamples it.

    Args:
        input (tensor): the tensor to be downsampled.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail.

    Return:
        torch.Tensor: the downsampled tensor.

    Examples:
        >>> input = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        >>> pyrdown(input, align_corners=True)
        tensor([[[[ 3.7500,  5.2500],
                  [ 9.7500, 11.2500]]]])
    """
    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")
    kernel: torch.Tensor = _get_pyramid_gaussian_kernel()
    b, c, height, width = input.shape
    # blur image
    x_blur: torch.Tensor = filter2D(input, kernel, border_type)

    # downsample.
    out: torch.Tensor = F.interpolate(x_blur, size=(height // 2, width // 2), mode='bilinear',
                                      align_corners=align_corners)
    return out



def pyrup(input: torch.Tensor, border_type: str = 'reflect', align_corners: bool = False) -> torch.Tensor:
    r"""Upsamples a tensor and then blurs it.

    Args:
        input (tensor): the tensor to be downsampled.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail.

    Return:
        torch.Tensor: the downsampled tensor.

    Examples:
        >>> input = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
        >>> pyrup(input, align_corners=True)
        tensor([[[[0.7500, 0.8750, 1.1250, 1.2500],
                  [1.0000, 1.1250, 1.3750, 1.5000],
                  [1.5000, 1.6250, 1.8750, 2.0000],
                  [1.7500, 1.8750, 2.1250, 2.2500]]]])
    """
    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")
    kernel: torch.Tensor = _get_pyramid_gaussian_kernel()
    # upsample tensor
    b, c, height, width = input.shape
    x_up: torch.Tensor = F.interpolate(input, size=(height * 2, width * 2),
                                       mode='bilinear', align_corners=align_corners)

    # blurs upsampled tensor
    x_blur: torch.Tensor = filter2D(x_up, kernel, border_type)
    return x_blur



def build_pyramid(
        input: torch.Tensor,
        max_level: int,
        border_type: str = 'reflect', align_corners: bool = False) -> List[torch.Tensor]:
    r"""Constructs the Gaussian pyramid for an image.

    The function constructs a vector of images and builds the Gaussian pyramid
    by recursively applying pyrDown to the previously built pyramid layers.

    Args:
        input (torch.Tensor): the tensor to be used to construct the pyramid.
        max_level (int): 0-based index of the last (the smallest) pyramid layer.
          It must be non-negative.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        align_corners(bool): interpolation flag. Default: False. See

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output :math:`[(B, C, H, W), (B, C, H/2, W/2), ...]`
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    if not isinstance(max_level, int) or max_level < 0:
        raise ValueError(f"Invalid max_level, it must be a positive integer. Got: {max_level}")

    # create empty list and append the original image
    pyramid: List[torch.Tensor] = []
    pyramid.append(input)

    # iterate and downsample

    for _ in range(max_level - 1):
        img_curr: torch.Tensor = pyramid[-1]
        img_down: torch.Tensor = pyrdown(img_curr, border_type, align_corners)
        pyramid.append(img_down)

    return pyramid


def reconstruct(
        l_pyramid: List[torch.Tensor]) -> torch.Tensor:

    out = l_pyramid[0]
    for i in range(1, len(l_pyramid)):
        out = pyrup(out) + l_pyramid[i]

    return out


def build_laplacian_pyramid(
        input: torch.Tensor,
        max_level: int,
        border_type: str = 'reflect', align_corners: bool = False) -> List[torch.Tensor]:

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    if not isinstance(max_level, int) or max_level < 0:
        raise ValueError(f"Invalid max_level, it must be a positive integer. Got: {max_level}")

    g_pyramid = build_pyramid(input, max_level+1, border_type)

    # create empty list and append the original image
    l_pyramid: List[torch.Tensor] = []
    l_pyramid.append(g_pyramid[-1])

    # iterate and downsample

    for i in range(max_level, 0, -1):
        GE: torch.Tensor = pyrup(g_pyramid[i])
        L: torch.Tensor = g_pyramid[i-1] - GE
        l_pyramid.append(L)

    return l_pyramid


def reconstruct_laplacian(
        l_pyramid: List[torch.Tensor]) -> torch.Tensor:

    out = l_pyramid[0]
    for i in range(1, len(l_pyramid)):
        out = pyrup(out) + l_pyramid[i]

    return out

def dct1(x):
    """
    Discrete Cosine Transform, Type I
    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    y = torch.view_as_real(torch.fft.fft([x, x.flip([1])[:, 1:-1]], dim=1))[:, :, 0].view(*x_shape)
    return y


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I
    Our definition if idct1 is such that idct1(dct1(x)) == x
    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_3d(dct_3d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)


class LinearDCT(nn.Linear):
    """Implement any DCT as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DCT matrix is stored, which will 
    increase memory usage.
    :param in_features: size of expected input
    :param type: which dct function in this file to use"""
    def __init__(self, in_features, type, norm=None, bias=False):
        self.type = type
        self.N = in_features
        self.norm = norm
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        # initialise using dct function
        I = torch.eye(self.N)
        if self.type == 'dct1':
            self.weight.data = dct1(I).data.t()
        elif self.type == 'idct1':
            self.weight.data = idct1(I).data.t()
        elif self.type == 'dct':
            self.weight.data = dct(I, norm=self.norm).data.t()
        elif self.type == 'idct':
            self.weight.data = idct(I, norm=self.norm).data.t()
        self.weight.requires_grad = False # don't learn this!


def apply_linear_2d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 2D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 2 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    return X2.transpose(-1, -2)

def apply_linear_3d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 3D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 3 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    X3 = linear_layer(X2.transpose(-1, -3))
    return X3.transpose(-1, -3).transpose(-1, -2)

if __name__ == '__main__':
    x = torch.Tensor(1000,4096)
    x.normal_(0,1)
    linear_dct = LinearDCT(4096, 'dct')
    error = torch.abs(dct(x) - linear_dct(x))
    assert error.max() < 1e-3, (error, error.max())
    linear_idct = LinearDCT(4096, 'idct')
    error = torch.abs(idct(x) - linear_idct(x))
    assert error.max() < 1e-3, (error, error.max())