import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output

"""
Copyright (c) 2022 Samsung Electronics Co., Ltd.

Author(s):
Luxi Zhao (lucy.zhao@samsung.com; lucyzhao.zlx@gmail.com)
Abdelrahman Abdelhamed (abdoukamel@gmail.com)

Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F









class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.process = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        y = self.process(x)
        y = self.avg_pool(y)
        z = self.conv_du(y)
        return z * y + x


class Refine(nn.Module):

    def __init__(self, n_feat, out_channel):
        super(Refine, self).__init__()

        self.conv_in = nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1)
        self.process = nn.Sequential(
            CALayer(n_feat, 4))
        self.conv_last = nn.Conv2d(in_channels=n_feat, out_channels=out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.process(out)
        out = self.conv_last(out)

        return out
    
def utils_get_image_stats(image_shape, grid_size):
    """
    Information about the cropped image.
    :return: grid size, tile size, sizes of the 4 margins, meshgrids.
    """

    grid_rows = grid_size[0]
    grid_cols = grid_size[1]

    residual_height = image_shape[0] % grid_rows
    residual_width = image_shape[1] % grid_cols

    tile_height = image_shape[0] // grid_rows
    tile_width = image_shape[1] // grid_cols

    margin_top = tile_height // 2
    margin_left = tile_width // 2

    margin_bot = tile_height + residual_height - margin_top
    margin_right = tile_width + residual_width - margin_left

    return tile_height, tile_width, margin_top, margin_left, margin_bot, margin_right

def apply_ltm_lut(imgs, luts):

        imgs = (imgs - .5) * 2.

        grids = imgs.unsqueeze(0).unsqueeze(0)
        luts = luts.unsqueeze(0)

        outs = F.grid_sample(luts, grids,
        mode='bilinear', padding_mode='border', align_corners=True)

        return outs.squeeze(0).squeeze(1).permute(1,2,0)


def apply_ltm(image, tone_curve, num_curves):
    """
    Apply tone curve to an image (patch).
    :param image: (h, w, 3) if num_curves == 3, else (h, w)
    :param tone_curve: (num_curves, control_points)
    :param num_curves: 3 for 1 curve per channel, 1 for 1 curve for all channels.
    :return: tone-mapped image.
    """
    
    if image.shape[-1] == 3:
        if type(image) == np.ndarray:
            r = tone_curve[0][image[..., 0]]
            g = tone_curve[1][image[..., 1]]
            b = tone_curve[2][image[..., 2]]
            new_image = np.stack((r, g, b), axis=-1)
        else:
            r = tone_curve[0][image[..., 0].reshape(-1).long()].reshape(image[..., 0].shape)
            g = tone_curve[1][image[..., 1].reshape(-1).long()].reshape(image[..., 1].shape)
            b = tone_curve[2][image[..., 2].reshape(-1).long()].reshape(image[..., 2].shape)
            new_image = torch.stack((r, g, b), axis=-1)
            # new_image = np.stack((r, g, b), axis=-1)
    else:
        new_image = tone_curve[0][image[..., 0].reshape(-1).long()].reshape(image[..., 0].shape).unsqueeze(dim=2)
        #tone_curve[0][image[..., 0].reshape(-1).long()].reshape(image[..., 0].shape)
    
    return new_image


def apply_gtm(image, tone_curve, num_curves):
    """
    Apply a single tone curve to an image.
    :param image: (h, w, 3) if num_curves == 3, else (h, w)
    :param tone_curve: (1, num_curves, control_points)
    :param num_curves: 3 for 1 curve per channel, 1 for 1 curve for all channels.
    :return: tone-mapped image.
    """
    tone_curve = tone_curve[0]
    out = apply_ltm(image, tone_curve, num_curves)
    return out


def apply_ltm_center(image, tone_curves, stats, num_curves):
    """
    Apply tone curves to the center region of an image.
    :param image: the original image.
    :param tone_curves: a list of all tone curves in row scan order.
    :return: interpolated center region of an image.
    """
    grid_rows, grid_cols, tile_height, tile_width, margin_top, margin_left, margin_bot, margin_right, meshgrids = stats
    xs_tl, ys_tl, xs_br, ys_br = meshgrids['center']

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    xs_tl = xs_tl.to(device)
    ys_tl = ys_tl.to(device)
    xs_br = xs_br.to(device)
    ys_br = ys_br.to(device)


    # Get neighbourhoods
    neighbourhoods = []
    for y in range(margin_top, image.shape[0]-margin_bot, tile_height):
        for x in range(margin_left, image.shape[1]-margin_right, tile_width):
            neighbourhoods.append(image[y:y + tile_height, x:x + tile_width, :])

    assert len(neighbourhoods) == (grid_rows-1) * (grid_cols-1)

    # Get indices for all 4-tile neighbourhoods
    tile_ids = []
    for i in range(grid_rows - 1):
        for j in range(grid_cols - 1):
            start = i * grid_cols + j
            tile_ids.append([start, start + 1, start + grid_cols, start + grid_cols + 1])

    # Apply LTM and interpolate
    new_ns = []
    for i, n in enumerate(neighbourhoods):
        n_tile_ids = tile_ids[i]  # ids of the 4 tone curves (tiles) of the neighbourhood
        # n_4versions = [apply_ltm(n, tone_curves[j], num_curves) for j in n_tile_ids]  # tl, tr, bl, br
        n_4versions = [apply_ltm_lut(n, tone_curves[j])for j in n_tile_ids]
        out = ys_br * xs_br * n_4versions[0] + ys_br * xs_tl * n_4versions[1] + ys_tl * xs_br * n_4versions[2] + ys_tl * xs_tl * n_4versions[3]
        out /= (tile_height-1) * (tile_width-1)

        new_ns.append(out)

    # Stack the interpolated neighbourhoods together
    rows = []
    for i in range(grid_rows - 1):
        cols = [new_ns[i * (grid_cols - 1) + j] for j in range(grid_cols - 1)]
        row = torch.cat(cols, dim=1)
        rows.append(row)
    out = torch.cat(rows, dim=0)
    return out


def apply_ltm_border(image, tone_curves, stats, num_curves=3):
    """
    Apply tone curves to the border, not including corner areas.
    :param image: the original image.
    :param tone_curves: a list of all tone curves in row scan order.
    :return: interpolated border regions of the image. In order of top, bottom, left, right.
    """
    grid_rows, grid_cols, tile_height, tile_width, margin_top, margin_left, margin_bot, margin_right, meshgrids = stats
    (top_xs_l, top_xs_r), (bot_xs_l, bot_xs_r), (left_ys_t, left_ys_b), (right_ys_t, right_ys_b) = meshgrids['border']

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    top_xs_l = top_xs_l.to(device) 
    top_xs_r = top_xs_r.to(device)
    bot_xs_l = bot_xs_l.to(device)
    bot_xs_r = bot_xs_r.to(device) 

    left_ys_t = left_ys_t.to(device)
    left_ys_b = left_ys_b.to(device)  
    right_ys_t = right_ys_t.to(device)
    right_ys_b = right_ys_b.to(device)
    # top, bottom, left, right neighbourhoods to be interpolated
    ntop = []
    nbot = []
    nleft = []
    nright = []

    for x in range(margin_left, image.shape[1] - margin_right, tile_width):
        ntop.append(image[:margin_top, x:x + tile_width, :])
        nbot.append(image[-margin_bot:, x:x + tile_width, :])

    for y in range(margin_top, image.shape[0] - margin_bot, tile_height):
        nleft.append(image[y:y + tile_height, :margin_left, :])
        nright.append(image[y:y + tile_height, -margin_right:, :])

    def apply_ltm_two_tiles(tc1, tc2, meshgrid1, meshgrid2, nbhd, interp_length, num_curves):
        """
        Apply tone curve to, and interpolate a two-tile neighbourhood, either horizontal or vertical
        :param tc1: left / top tone curves
        :param tc2: right / bottom tone curves
        :param meshgrid1: left / top meshgrids (leftmost / topmost positions are 0)
        :param meshgrid2: right / bottom meshgrids (rightmost / bottommost positions are 0)
        :param nbhd: neighbourhood to interpolate
        :param interp_length: normalizing factor of the meshgrid.
               Example: if xs = np.meshgrid(np.arange(10)), then interp_length = 9
        :return: interpolated neighbourhood
        """

        # new_nbhd1 = apply_ltm(nbhd, tc1, num_curves)
        # new_nbhd2 = apply_ltm(nbhd, tc2, num_curves)

        new_nbhd1 = apply_ltm_lut(nbhd, tc1)
        new_nbhd2 = apply_ltm_lut(nbhd, tc2)

        out = meshgrid1 * new_nbhd2 + meshgrid2 * new_nbhd1
        out /= interp_length
        return out

    new_ntop = [apply_ltm_two_tiles(tone_curves[i],  # left tone curve
                                    tone_curves[i + 1],  # right tone curve
                                    top_xs_l, top_xs_r,
                                    n, tile_width - 1, num_curves) for i, n in enumerate(ntop)]

    new_nbot = [apply_ltm_two_tiles(tone_curves[(grid_rows - 1) * grid_cols + i],  # left tone curve
                                    tone_curves[(grid_rows - 1) * grid_cols + i + 1],  # right tone curve
                                    bot_xs_l, bot_xs_r,
                                    n, tile_width - 1, num_curves) for i, n in enumerate(nbot)]

    new_nleft = [apply_ltm_two_tiles(tone_curves[i * grid_cols],  # top tone curve
                                     tone_curves[(i + 1) * grid_cols],  # bottom tone curve
                                     left_ys_t, left_ys_b,
                                     n, tile_height - 1, num_curves) for i, n in enumerate(nleft)]

    new_nright = [apply_ltm_two_tiles(tone_curves[(i + 1) * grid_cols - 1],  # top tone curve
                                      tone_curves[(i + 2) * grid_cols - 1],  # bottom tone curve
                                      right_ys_t, right_ys_b,
                                      n, tile_height - 1, num_curves) for i, n in enumerate(nright)]

    new_ntop = torch.cat(new_ntop, dim=1)
    new_nbot = torch.cat(new_nbot, dim=1)
    new_nleft = torch.cat(new_nleft, dim=0)
    new_nright = torch.cat(new_nright, dim=0)
    return new_ntop, new_nbot, new_nleft, new_nright


def apply_ltm_corner(image, tone_curves, stats, num_curves=3):
    """
    tone_curves: a list of all tone curves in row scan order.
    return: interpolated corner tiles in the order of top left, top right, bot left, bot right
    """
    grid_rows, grid_cols, tile_height, tile_width, margin_top, margin_left, margin_bot, margin_right, _ = stats

    corner_ids = [0, grid_cols - 1, -grid_rows, -1]
    tl_tile = image[:margin_top, :margin_left]
    tr_tile = image[:margin_top, -margin_right:]
    bl_tile = image[-margin_bot:, :margin_left]
    br_tile = image[-margin_bot:, -margin_right:]

    corner_tiles = [tl_tile, tr_tile, bl_tile, br_tile]
    corner_tcs = [tone_curves[i] for i in corner_ids]  # tcs: (grid_size, num_curves, control_points)
    #new_tiles = [apply_ltm(corner_tiles[i], corner_tcs[i], num_curves) for i in range(len(corner_tcs))]
    new_tiles = [apply_ltm_lut(corner_tiles[i],corner_tcs[i]) for i in range(len(corner_tcs))]

    return new_tiles[0], new_tiles[1], new_tiles[2], new_tiles[3]


# def get_meshgrids(height, width):
#     """
#     Get two meshgrids of size (height, width). One with top left corner being (0, 0),
#     the other with bottom right corner being (0, 0).
#     :return: top left xs, ys, bottom right xs, ys
#     """
#     xs, ys = np.meshgrid(np.arange(width), np.arange(height))
#     newys, newxs = torch.meshgrid(torch.arange(height, dtype=torch.int32), torch.arange(width, dtype=torch.int32))
#     # mesh grid for top left corner
#     xs_tl = np.tile(np.abs(xs)[..., np.newaxis], 3)  # [0, 1, 2, ..., tile_width-1]
#     ys_tl = np.tile(np.abs(ys)[..., np.newaxis], 3)
#     new_xs_tl = newxs[..., None].abs().repeat(1, 1, 3)
#     new_ys_tl = newys[..., None].abs().repeat(1, 1, 3)
#     # mesh grid for bottom right corner
#     xs_br = np.tile(np.abs(xs - width + 1)[..., np.newaxis], 3)  # [-(tile_width-1), ..., -2, -1, 0]
#     ys_br = np.tile(np.abs(ys - height + 1)[..., np.newaxis], 3)

#     new_xs_br = (newxs - width + 1).abs()[..., None].repeat(1, 1, 3)
#     new_ys_br = (newys - width + 1).abs()[..., None].repeat(1, 1, 3)
#     # return xs_tl, ys_tl, xs_br, ys_br
#     return new_xs_tl, new_ys_tl, new_xs_br, new_ys_br
def get_meshgrids(height, width):
    """
    Get two meshgrids of size (height, width). One with top left corner being (0, 0),
    the other with bottom right corner being (0, 0).
    :return: top left xs, ys, bottom right xs, ys
    """
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    # mesh grid for top left corner
    xs_tl = np.tile(np.abs(xs)[..., np.newaxis], 3)  # [0, 1, 2, ..., tile_width-1]
    ys_tl = np.tile(np.abs(ys)[..., np.newaxis], 3)
    # mesh grid for bottom right corner
    xs_br = np.tile(np.abs(xs - width + 1)[..., np.newaxis], 3)  # [-(tile_width-1), ..., -2, -1, 0]
    ys_br = np.tile(np.abs(ys - height + 1)[..., np.newaxis], 3)
    
    return torch.tensor(xs_tl), torch.tensor(ys_tl), torch.tensor(xs_br), torch.tensor(ys_br)



def get_meshgrid_center(tile_height, tile_width):
    return get_meshgrids(tile_height, tile_width)


def get_meshgrid_border(tile_height, tile_width, margin_top, margin_left, margin_bot, margin_right):
    """
    :return: meshgrids for the 4 border regions, in the order of top, bottom, left, right
    """
    # top
    top_xs_l, _, top_xs_r, _ = get_meshgrids(margin_top, tile_width)

    # bottom
    bot_xs_l, _, bot_xs_r, _ = get_meshgrids(margin_bot, tile_width)

    # left
    _, left_ys_t, _, left_ys_b = get_meshgrids(tile_height, margin_left)

    # right
    _, right_ys_t, _, right_ys_b = get_meshgrids(tile_height, margin_right)

    return (top_xs_l, top_xs_r), (bot_xs_l, bot_xs_r), (left_ys_t, left_ys_b), (right_ys_t, right_ys_b)


def get_image_stats(image, grid_size):
    """
    Information about the cropped image.
    :param image: the original image
    :return: grid size, tile size, sizes of the 4 margins, meshgrids.
    """

    grid_rows = grid_size[0]
    grid_cols = grid_size[1]

    tile_height, tile_width, margin_top, margin_left, margin_bot, margin_right = utils_get_image_stats(image.shape,
                                                                                                       grid_size)

    meshgrid_center = get_meshgrid_center(tile_height, tile_width)
    meshgrid_border = get_meshgrid_border(tile_height, tile_width, margin_top, margin_left, margin_bot, margin_right)

    meshgrids = {
        'center': meshgrid_center,
        'border': meshgrid_border
    }

    return grid_rows, grid_cols, tile_height, tile_width, margin_top, margin_left, margin_bot, margin_right, meshgrids


#自己构造image 1 * 512 * 512 *3,tone_curve 1 * 64 * 3 * 256 维度得tf tensor 然后只在这里debug  
def do_interpolation_lut(image, tone_curves, grid_size, num_curves=3):
    """
    Perform tone mapping and interpolation on an image.
    Center region: bilinear interpolation.
    Border region: linear interpolation.
    Corner region: no interpolation.
    :param num_curves: 3 -> 1 curve for each R,G,B channel, 1 -> 1 curve for all channels
    :param image: input int8
    :param tone_curves: (grid_size, num_curves, control_points)
    :param grid_size: (ncols, nrows)
    :return: image: float32, between [0-1]
    """
    if grid_size[0] == 1 and grid_size[1] == 1:
        return apply_gtm(image, tone_curves, num_curves).astype(np.float64)

    # get image statistics
    stats = get_image_stats(image, grid_size)



    # Center area:
    center = apply_ltm_center(image, tone_curves, stats, num_curves)

    # Border area:
    b_top, b_bot, b_left, b_right = apply_ltm_border(image, tone_curves, stats, num_curves)

    # Corner area:
    tlc, trc, blc, brc = apply_ltm_corner(image, tone_curves, stats, num_curves)

    # stack the corners, borders, and center together
    row_t = torch.cat([tlc, b_top, trc], dim=1)
    row_c = torch.cat([b_left, center, b_right], dim=1)
    row_b = torch.cat([blc, b_bot, brc], dim=1)
    out = torch.cat([row_t, row_c, row_b], dim=0)

    assert out.shape == image.shape

    return out



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)  
        # self.pool1 = nn.MaxPool2d(2)
        
        # self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        # self.pool2 = nn.MaxPool2d(2) 
        
        # self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        # self.pool3 = nn.MaxPool2d(2)

        # self.conv4 = nn.Conv2d(16, 32, kernel_size=3, padding=1) 
        # self.pool4 = nn.MaxPool2d(2)

        # self.conv5 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.pool5 = nn.MaxPool2d(2)
        
        # self.conv6 = nn.Conv2d(64, 768, kernel_size=3, padding=1)
        # self.pool6 = nn.MaxPool2d(2)

        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),  
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),  
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),  
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer_4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer_5 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer_6 = nn.Sequential(
            nn.Conv2d(64, 768, kernel_size=3, padding=1),
            nn.BatchNorm2d(768),  
            nn.Sigmoid(),
            nn.MaxPool2d(2)
        )

        
    def forward(self, x):
        
        '''

        original = x
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2(x) 
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.pool4(x)

        x = self.conv5(x) 
        x = self.pool5(x)

        x = self.conv6(x)
        x = self.pool6(x)
        oldres = x
        x = original
        '''

        x = self.layer_1(x)

        x = self.layer_2(x)

        x = self.layer_3(x)

        x = self.layer_4(x)

        x = self.layer_5(x)

        x = self.layer_6(x)

        x = x.reshape(x.shape[0], x.shape[2] * x.shape[3], 3, int(x.shape[1] / 3))
        return x


def _lut_transform(imgs, luts):
    # img (b, 3, h, w), lut (b, c, m, m, m)
    if imgs.shape[1]==1:

        #for gray image pro-processs
        luts = luts.expand(1,1,64,64,64)
        # normalize pixel values
        imgs = (imgs - .5) * 2.
        grids = (imgs.unsqueeze(4)).repeat(1,1,1,1,3)
    else:
        # normalize pixel values
        imgs = (imgs - .5) * 2.
        # reshape img to grid of shape (b, 1, h, w, 3)
        # imgs = imgs.permute(2,0,1).unsqueeze(dim=0)
        # grids = imgs.permute(0, 2, 3, 1).unsqueeze(1)
        grids = imgs.unsqueeze(0).unsqueeze(0)
        luts = luts.unsqueeze(0)
        # after gridsampling, output is of shape (b, c, 1, h, w)
    outs = F.grid_sample(luts, grids,
        mode='bilinear', padding_mode='border', align_corners=True)
    return outs.squeeze(2)


if __name__ == '__main__':

    import torch
    import cv2

    grid_size = [8,8]
    
    np.random.seed(42)
    rand_img  = np.random.random((512, 512, 3))
    luts_np = np.random.random((64, 3, 9))

    img_torch = torch.tensor(rand_img, dtype=torch.float32).cuda()
    luts_torch = torch.tensor(luts_np, dtype=torch.float32).cuda()


    iluts = []
    for i in range(luts_torch.shape[0]):
        iluts.append(torch.stack(
            torch.meshgrid(*(luts_torch[i].unbind(0)[::-1])),
            dim=0).flip(0))
    iluts = torch.stack(iluts, dim=0)
 

    result = do_interpolation_lut(img_torch, iluts, grid_size)
    print(result)






