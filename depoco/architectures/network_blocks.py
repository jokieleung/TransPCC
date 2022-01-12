

import torch.nn as nn
import torch
import numpy as np
import octree_handler
import depoco.architectures.original_kp_blocks as o_kp_conv
##################################################
############ NETWORK BLOCKS Dictionary############
##################################################

from .pointnet_util import index_points, square_distance, PointNetFeaturePropagation, PointNetSetAbstraction
#from pointnet_util import PointNetFeaturePropagation, PointNetSetAbstraction
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        
    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)
        
        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f
        
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res, attn


def printNan(bla: torch.tensor, pre=''):
    if (bla != bla).any():
        print(pre, 'NAN')


class Network(nn.Module):
    def __init__(self, config_list: list):
        super().__init__()
        blocks = []
        for config in config_list:
            blocks.append(getBlocks(config))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input_dict: dict):
        return self.blocks(input_dict)


def getBlocks(config: dict):
    config_it, block_type = blockConfig2Params(config)
    blocks = []
    for c in config_it:
        blocks.append(eval(block_type)(c))
    if(len(config_it) == 1):
        return blocks[0]
    return nn.Sequential(*blocks)


def blockConfig2Params(config: dict):
    """converts the config to a list of dicts
        it needs at least the keys 'type', 'parameters' and 'number_blocks'

    Arguments:
        config {dict} -- dictionary with the parameters for the block specified in 'type'

    Returns:
        {list} -- parameters [param_set1, param_set2, ...]
        {string} -- block_type
    """
    nr_blocks = config['number_blocks']
    if nr_blocks == 1:
        return [config['parameters']], config['type']

    config_list = []
    for i in range(nr_blocks):
        new_config = config["parameters"].copy()
        for k, v in zip(config["parameters"].keys(), config["parameters"].values()):
            if (type(v) is list):
                if(len(v) == nr_blocks):
                    new_config[k] = v[i]
                if k == 'subsampling_ratio':
                    new_config['cum_subsampling_ratio'] = np.cumprod([1.0]+v)[i]  # HACK: to specivic
        config_list.append(new_config)
    return config_list, config['type']


def dict2initParams(dict_, class_):
    init_params = class_.__init__.__code__.co_varnames
    print(f'init vars: \n {init_params}')
    params = {k: dict_[k] for k in dict_ if k in init_params}
    print(params)
    return params

def gridSampling(pcd: torch.tensor, resolution_meter=1.0, map_size=40):
    resolution = resolution_meter/map_size

    # v_size = torch.full(size=[3], fill_value=1/resolution,
    #                     dtype=pcd.dtype, device=pcd.device)

    grid = torch.floor(pcd/resolution)
    center = (grid+0.5)*resolution
    dist = ((pcd-center)**2).sum(dim=1)
    dist = dist/dist.max()*0.7

    # grid_idx = grid[:, 0] + grid[:, 1] * \
    #     v_size[0] + grid[:, 2] * v_size[0] * v_size[1]
    v_size = np.ceil(1/resolution)
    grid_idx = grid[:, 0] + grid[:, 1] * \
        v_size + grid[:, 2] * v_size * v_size
    grid_d = grid_idx+dist
    idx_orig = torch.argsort(grid_d)

    # trick from https://github.com/rusty1s/pytorch_unique
    unique, inverse = torch.unique_consecutive(
        grid_idx[idx_orig], return_inverse=True)
    perm = torch.arange(inverse.size(
        0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])

    # idx = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    """
    HACK: workaround to get the first item. scatter overwrites indices on gpu not sequentially
           -> you get random points in the voxel not the first one
    """ 
    p= perm.cpu()
    i=inverse.cpu()
    idx = torch.empty(unique.shape,dtype=p.dtype).scatter_(0, i, p)
    # idx= torch.empty(unique.shape,dtype=long)
    return idx_orig[idx].tolist()


class GridSampleConv(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.relu = nn.LeakyReLU()
        ### Preactivation ####
        in_fdim = config['in_fdim']
        out_fdim = config['out_fdim']
        self.preactivation = nn.Identity()
        if in_fdim > 1:
            pre_blocks = [
                nn.Linear(in_features=in_fdim, out_features=out_fdim)]
            if config['batchnorm']:
                pre_blocks.append(nn.BatchNorm1d(out_fdim))
            if config['relu']:
                pre_blocks.append(self.relu)
            self.preactivation = nn.Sequential(*pre_blocks)
        # KP Conv
        conf_in_fdim = out_fdim if in_fdim > 1 else in_fdim
        self.subsampling_dist = config['subsampling_dist'] * config['subsampling_factor']
        # self.kernel_radius = max(config['kernel_radius'], self.subsampling_dist/40 )
        self.kernel_radius = max(config['min_kernel_radius'],config['kernel_radius']*self.subsampling_dist)/40
        KP_extent = self.kernel_radius / \
            (config['num_kernel_points']**(1/3)-1)*1.5
        self.kp_conv = o_kp_conv.KPConv(kernel_size=config['num_kernel_points'],
                                        p_dim=3, in_channels=conf_in_fdim,
                                        out_channels=out_fdim,
                                        KP_extent=KP_extent, radius=self.kernel_radius,
                                        deformable=config['deformable'])
        self.max_nr_neighbors = config['max_nr_neighbors']
        self.map_size = config['map_size']
        self.octree = octree_handler.Octree()

        print('kernel radius',self.kernel_radius)
        # Post linear
        post_layer = []
        if config['batchnorm']:
            post_layer.append(nn.BatchNorm1d(out_fdim))
        if config['relu']:
            post_layer.append(self.relu)
        post_layer.append(
            nn.Linear(in_features=out_fdim, out_features=out_fdim))
        if config['batchnorm']:
            post_layer.append(nn.BatchNorm1d(out_fdim))
        self.post_layer = nn.Sequential(*post_layer)

        # Shortcut
        self.shortcut = nn.Identity()
        if in_fdim != out_fdim:
            sc_blocks = [nn.Linear(in_features=in_fdim,
                                   out_features=out_fdim)]
            if config['batchnorm']:
                sc_blocks.append(nn.BatchNorm1d(out_fdim))
            self.shortcut = nn.Sequential(*sc_blocks)

    def forward(self, input_dict: dict) -> dict:
        source = input_dict['points']
        source_np = source.detach().cpu().numpy()
        sample_idx = gridSampling(source,resolution_meter=self.subsampling_dist,map_size=self.map_size)
        # get neighbors
        self.octree.setInput(source_np)
        neighbors_index = self.octree.radiusSearchIndices(
            sample_idx, self.max_nr_neighbors, self.kernel_radius)
        neighbors_index = torch.from_numpy(
            neighbors_index).long().to(source.device)

        features = self.preactivation(input_dict['features'])
        features = self.kp_conv.forward(q_pts=input_dict['points'][sample_idx, :],
                                        s_pts=input_dict['points'],
                                        neighb_inds=neighbors_index,
                                        x=features)
        
        features = self.post_layer(features)
        input_dict['features'] = self.relu(
            self.shortcut(input_dict['features'][sample_idx, :]) + features)
        input_dict['points'] = input_dict['points'][sample_idx, :]
        return input_dict

# https://discuss.pytorch.org/t/apply-mask-softmax/14212/14



class LinearLayer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        blocks = [nn.Linear(in_features=config['in_fdim'],
                            out_features=config['out_fdim'])]
        if 'relu' in config:
            if config['relu']:
                blocks.append(nn.LeakyReLU())
        if 'batchnorm' in config:
            if config['batchnorm']:
                blocks.append(nn.BatchNorm1d(num_features=config['out_fdim']))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input_dict: dict):
        input_dict['features'] = self.blocks(input_dict['features'])
        return input_dict


class LinearDeconv(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        if config['estimate_radius']:
            self.kernel_radius = nn.Parameter(torch.tensor(
                [config['kernel_radius']]), requires_grad=True).float()
        else:
            self.kernel_radius = config['kernel_radius']
        # self.kernel_radius = torch.tensor()

        feature_space = config['inter_fdim'] if 'inter_fdim' in config.keys(
        ) else 128

        self.upsampling_rate = config['upsampling_rate']
        trans_blocks = [nn.Linear(config['in_fdim'], out_features=feature_space),
                        nn.LeakyReLU(),
                        nn.Linear(in_features=feature_space,
                                  out_features=3*self.upsampling_rate),
                        nn.Tanh()]
        self.transl_nn = nn.Sequential(*trans_blocks)

        feature_blocks = [nn.Linear(config['in_fdim'], out_features=feature_space),
                          nn.LeakyReLU(),
                          nn.Linear(
            in_features=feature_space, out_features=config['out_fdim']*self.upsampling_rate),
            nn.LeakyReLU()]
        if config['use_batch_norm']:
            feature_blocks.append(nn.BatchNorm1d(
                config['out_fdim']*self.upsampling_rate))
        self.feature_nn = nn.Sequential(*feature_blocks)
        self.tmp_i = 0

        self.points = None

    def forward(self, input_dict: dict):
        p = input_dict['points']
        f = input_dict['features']
        # print('p', p, 'f', f)
        delta = self.transl_nn(f)
        delta = delta.reshape(
            (delta.shape[0], self.upsampling_rate, 3))*self.kernel_radius
        p_new = (p.unsqueeze(1) +
                 delta).reshape((delta.shape[0]*self.upsampling_rate, 3))
        f_new = self.feature_nn(f).reshape(
            (delta.shape[0]*self.upsampling_rate, self.config['out_fdim']))

        self.tmp_i += 1
        # if(self.tmp_i % 100 == 0) and self.config['estimate_radius']:
        #     print('learned kernel radius', self.kernel_radius)
        self.points = p_new
        input_dict['points'] = p_new
        input_dict['features'] = f_new
        return input_dict

def getScalingFactor(upsampling_rate, nr_layer,layer=0):
    sf = upsampling_rate**(1/nr_layer)
    factors = nr_layer*[round(sf)]
    # factors[-1]=round(upsampling_rate/np.prod(factors[:-1]))

    sampling_factor = np.prod(factors)
    print(f'factors {factors}, upsampling rate {sampling_factor}, should: {upsampling_rate}')
    return factors[layer]

class AdaptiveDeconv(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        if config['estimate_radius']:
            self.kernel_radius = nn.Parameter(torch.tensor(
                [config['kernel_radius']]), requires_grad=True).float()
        else:
            self.kernel_radius = config['kernel_radius']

        feature_space = config['inter_fdim'] if 'inter_fdim' in config.keys(
        ) else 128


        sub_rate = config['subsampling_fct_p1']*config['subsampling_dist']**(-config['subsampling_fct_p2'])
        print('sub rate',sub_rate)

        self.upsampling_rate = getScalingFactor(upsampling_rate=1/sub_rate,nr_layer=config['number_blocks'],layer=config['block_id'])
        trans_blocks = [nn.Linear(config['in_fdim'], out_features=feature_space),
                        nn.LeakyReLU(),
                        nn.Linear(in_features=feature_space,
                                  out_features=3*self.upsampling_rate),
                        nn.Tanh()]
        self.transl_nn = nn.Sequential(*trans_blocks)

        feature_blocks = [nn.Linear(config['in_fdim'], out_features=feature_space),
                          nn.LeakyReLU(),
                          nn.Linear(
            in_features=feature_space, out_features=config['out_fdim']*self.upsampling_rate),
            nn.LeakyReLU()]
        if config['use_batch_norm']:
            feature_blocks.append(nn.BatchNorm1d(
                config['out_fdim']*self.upsampling_rate))
        self.feature_nn = nn.Sequential(*feature_blocks)
        self.tmp_i = 0

        self.points = None

    def forward(self, input_dict: dict):
        p = input_dict['points']
        f = input_dict['features']
        delta = self.transl_nn(f)
        delta = delta.reshape(
            (delta.shape[0], self.upsampling_rate, 3))*self.kernel_radius
        p_new = (p.unsqueeze(1) +
                 delta).reshape((delta.shape[0]*self.upsampling_rate, 3))
        f_new = self.feature_nn(f).reshape(
            (delta.shape[0]*self.upsampling_rate, self.config['out_fdim']))

        self.tmp_i += 1
        # if(self.tmp_i % 100 == 0) and self.config['estimate_radius']:
        #     print('learned kernel radius', self.kernel_radius)
        self.points = p_new
        input_dict['points'] = p_new
        input_dict['features'] = f_new
        return input_dict

#------------------------------- point transformer part by Jokie 220111
class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)

    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])

    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2


class Encoder_PointTrans(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        npoints, nblocks, nneighbor, d_points, transformer_dim = cfg['num_point'], cfg['nblocks'], cfg['nneighbor'], cfg['input_dim'], cfg['transformer_dim']
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 2 ** nblocks)
        )
        self.transformer2 = TransformerBlock(32 * 2 ** nblocks, transformer_dim, nneighbor)
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor))
        self.nblocks = nblocks

    def forward(self, x):
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        xyz = xyz_and_feats[-1][0]
        points = self.transformer2(xyz, self.fc2(points))[0]
        return xyz, points, xyz_and_feats

class Decoder_PointTrans(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        npoints, nblocks, nneighbor, transformer_dim = cfg['num_point'], cfg['nblocks'], cfg['nneighbor'], cfg['transformer_dim']
        #npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.nblocks = nblocks
        self.transition_ups = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in reversed(range(nblocks)):
            channel = 32 * 2 ** i
            self.transition_ups.append(TransitionUp(channel * 2, channel, channel))
            self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor))

    def forward(self, xyz, points, xyz_and_feats):

        for i in range(self.nblocks):
            points = self.transition_ups[i](xyz, points, xyz_and_feats[- i - 2][0], xyz_and_feats[- i - 2][1])
            xyz = xyz_and_feats[- i - 2][0]
            points = self.transformers[i](xyz, points)[0]

        return xyz, points



