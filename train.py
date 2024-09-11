import numpy as np
import glob
import os

import sys

from models.FourierEncoding import PositionalEncoding, GaussianEncoding, BasicEncoding
from models.LatentEncoder import LatentEncoder
from models.CascadedEDSRNet import CascadedEDSR
from models.INRNet import INR
from models.Global_Encoder import Global_Encoder

from datamodules.data_set import load_data, get_coords, LIIFDataset, LIIFVizDataset

import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
import torch
import torch.nn as nn
import math
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.transforms.functional import InterpolationMode
from typing import Optional

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import json

import argparse

parser = argparse.ArgumentParser('Train Multi-Modality Super Resolution')

parser.add_argument('--data-dir', help='path to training data')

parser.add_argument('--use-gpu', action='store_true', help='Whether to use GPU')
parser.add_argument('--use-global-encoder', action='store_true', help='Whether to use Pre-trained Global Encoder')
parser.add_argument('--use-pos-encoder', action='store_true', help='Whether to use Positional Encoding')

# parser.add_argument('--train-size', type=int, default=80, help='Size of Training Data Size')
# parser.add_argument('--val-size', type=int, default=20, help='Size of Training Data Size')
parser.add_argument('--batch-size', type=int, default=8, help='Batch size of Training Data Loader')

parser.add_argument('--train-sr', type=float, default=5.0)

parser.add_argument('--file-prefix', type=str, choices=['ua', 'va'])

parser.add_argument('--height-0', type=int, choices=[10, 60, 160, 200])
parser.add_argument('--height-1', type=int, choices=[10, 60, 160, 200])

parser.add_argument('--train-frac', type=float, default=0.8)

parser.add_argument('--h-LR', type=int, default=120)
parser.add_argument('--w-LR', type=int, default=160)

parser.add_argument('--latent-dim', type=int, default=1)

parser.add_argument('--sigma', type=float, default=30.0)
parser.add_argument('--m', type=int, default=50)

# parser.add_argument('--encoded-h-LR', type=int, default=30)
# parser.add_argument('--encoded-w-LR', type=int, default=40)

parser.add_argument('--n-feature-encoder-blocks', type=int, default=3)

parser.add_argument('--encoder-scale', type=int, default=2)
parser.add_argument('--residual-scale', type=float, default=1.0)

parser.add_argument('--input-dim', type=int, default=1, help='Number of Input Data Channels')
parser.add_argument('--n-encoder-features', type=int, default=64, help='Number of Encoder Features')
parser.add_argument('--n-global-encoder-features', type=int, default=64)
parser.add_argument('--encoder-kernel-size', type=int, default=3)
parser.add_argument('--n-encoder-resblocks', type=int, default=16)
parser.add_argument('--encoder-non-lin', type=str, default='tanh')

# parser.add_argument('--max-grad-norm', type=float, default=2.0)

parser.add_argument('--decoder-non-lin', type=str, default='relu')

parser.add_argument('--num-epochs', type=int, default=200)

parser.add_argument('--encoder-model', type=str, choices=['fadn', 'edsr'])

parser.add_argument('--encoder-lr', type=float, default=1e-3)
parser.add_argument('--encoder-wd', type=float, default=1e-5)

parser.add_argument('--encoder-bias', action='store_true')
parser.add_argument('--encoder-batch-norm', action='store_true')

parser.add_argument('--encoder-upsampling', action='store_true')

parser.add_argument('--decoder-lr', type=float, default=1e-3)
parser.add_argument('--decoder-wd', type=float, default=1e-5)

parser.add_argument('--learning-rate-decay', type=float, default=0.9999)
parser.add_argument('--start-lr-decay', type=int, default=0)
parser.add_argument('--lr-decay-interval', type=int, default=1)

parser.add_argument('--lambda-s', type=float, default=1.0)
parser.add_argument('--lambda-c', type=float, default=1.0)
parser.add_argument('--lambda-l', type=float, default=0.1)
parser.add_argument('--lambda-kl', type=float, default=1.0)

parser.add_argument('--model-save-dir', type=str, help='Directory to Save Models')
parser.add_argument('--load-pretrained-model', action='store_true', help='Whether to use a pretrained model')
parser.add_argument('--pretrained-epoch', type=int, default=500, help='Number of epochs the pretrained model was trained')

parser.add_argument('--save-interval', type=int, help='Intervals at which to save the models')

parser.add_argument('--query-points', type=int, default=512)

args = parser.parse_args()

non_act = {'relu': partial(nn.ReLU),
       'sigmoid': partial(nn.Sigmoid),
       'tanh': partial(nn.Tanh),
       'selu': partial(nn.SELU),
       'softplus': partial(nn.Softplus),
       'gelu': partial(nn.GELU),
       'elu': partial(nn.ELU),
       'silu': partial(nn.SiLU)}

def look_up_feature(
    coordinate: torch.Tensor, 
    feature: torch.Tensor, 
    feat_coord: torch.Tensor
):
    '''
    Args:
        coordinate (torch.tensor) - (b, n_query_pts, 2)
        feature    (torch.tensor) - (b, n_feature, H, W)
    Returns:
        feature      (torch.tensor) - (b, n_query_pts, n_feature)
        f_coordinate (torch.tensor) - (b, n_query_pts, 2)
    '''
    feature = F.grid_sample(
                feature, 
                coordinate.flip(-1).unsqueeze(1), # (b, 1, n_feature, 2)
                mode='nearest', 
                align_corners=False)[:, :, 0, :].permute(0, 2, 1) # (b, n_feature, n_feature)

    f_coordinate = F.grid_sample(
                    feat_coord, 
                    coordinate.flip(-1).unsqueeze(1),
                    mode='nearest', 
                    align_corners=False)[:, :, 0, :].permute(0, 2, 1) # (b, n_feature, 2)

    return feature, f_coordinate

def pred_HR(batch,
            coord_encoder,
            m0_latent_encoder,
            m1_latent_encoder,
            m021_latent_encoder,
            m120_latent_encoder,
            m0_global_encoder,
            m1_global_encoder,
            m0_feature_encoder,
            m1_feature_encoder,
            m0_decoder,
            m1_decoder,
            device,
            encoder_model: str = 'fadn',
            use_global_encoder: bool = True,
            use_pos_encoder: bool = True):
    
    m0_data_LR, m1_data_LR = batch['m0_img_LR'].to(device), batch['m1_img_LR'].to(device)
    m0_data_HR, m1_data_HR = batch['m0_img_HR'].to(device), batch['m1_img_HR'].to(device)
    
    coord = batch['grid_HR'].to(device)
    cell = batch['cell'].to(device)

    b, q = coord.shape[:2]

    if use_pos_encoder:
        encoded_coord = coord_encoder(coord)

    m0_encoded_LR = m0_latent_encoder(m0_data_LR)
    m1_encoded_LR = m1_latent_encoder(m1_data_LR)

    ### 
    # Cross Encoder
    m0_cross_encoded_LR = m120_latent_encoder(m1_data_LR)
    m1_cross_encoded_LR = m021_latent_encoder(m0_data_LR)
    ###

    if use_global_encoder:
        m0_global_encoded_LR = m0_global_encoder(m0_encoded_LR).unsqueeze(1)#.repeat(1, coord.shape[1], 1)
        m1_global_encoded_LR = m1_global_encoder(m1_encoded_LR).unsqueeze(1)#.repeat(1, coord.shape[1], 1)
        
        m0_global_encoded_LR = m0_global_encoded_LR.expand(b, q, m0_global_encoded_LR.size(2)).contiguous()
        m1_global_encoded_LR = m1_global_encoded_LR.expand(b, q, m1_global_encoded_LR.size(2)).contiguous()

    # Cross Encoder
    if use_global_encoder:
        m0_cross_global_encoded_LR = m0_global_encoder(m0_cross_encoded_LR).unsqueeze(1)#.repeat(1, coord.shape[1], 1)
        m1_cross_global_encoded_LR = m1_global_encoder(m1_cross_encoded_LR).unsqueeze(1)#.repeat(1, coord.shape[1], 1)

        m0_cross_global_encoded_LR = m0_cross_global_encoded_LR.expand(b, q, m0_cross_global_encoded_LR.size(2)).contiguous()
        m1_cross_global_encoded_LR = m1_cross_global_encoded_LR.expand(b, q, m1_cross_global_encoded_LR.size(2)).contiguous()

    if encoder_model == 'fadn':
        m0_feature, m0_sparsity_avg = m0_feature_encoder(m0_encoded_LR)
        m1_feature, m1_sparsity_avg = m1_feature_encoder(m1_encoded_LR)

        ### 
        # Cross Encoder
        m0_cross_feature, m0_cross_sparsity_avg = m0_feature_encoder(m0_cross_encoded_LR)
        m1_cross_feature, m1_cross_sparsity_avg = m1_feature_encoder(m1_cross_encoded_LR)
        ###
    else:
        m0_feature = m0_feature_encoder(m0_encoded_LR)
        m1_feature = m1_feature_encoder(m1_encoded_LR)

        ###
        # Cross Encoder
        m0_cross_feature = m0_feature_encoder(m0_cross_encoded_LR)
        m1_cross_feature = m1_feature_encoder(m1_cross_encoded_LR)
        ###
    
    m0_feature_data = F.unfold(m0_feature, 3, padding=1).view(
        m0_feature.shape[0], 
        m0_feature.shape[1] * 9, 
        m0_feature.shape[2], 
        m0_feature.shape[3]
    )
    
    m1_feature_data = F.unfold(m1_feature, 3, padding=1).view(
        m1_feature.shape[0], 
        m1_feature.shape[1] * 9, 
        m1_feature.shape[2], 
        m1_feature.shape[3]
    )

    ###
    # Cross Encoder
    m0_cross_feature_data = F.unfold(m0_cross_feature, 3, padding=1).view(
        m0_cross_feature.shape[0], 
        m0_cross_feature.shape[1] * 9, 
        m0_cross_feature.shape[2], 
        m0_cross_feature.shape[3]
    )
    
    m1_cross_feature_data = F.unfold(m1_cross_feature, 3, padding=1).view(
        m1_cross_feature.shape[0], 
        m1_cross_feature.shape[1] * 9, 
        m1_cross_feature.shape[2], 
        m1_cross_feature.shape[3]
    )
    ###
    
    # field radius (global: [-1, 1])
    rx = 2 / m0_feature_data.shape[-2] / 2
    ry = 2 / m0_feature_data.shape[-1] / 2

    feat_coord = get_coords(m0_feature_data.shape[-2:], flatten=False).type_as(m0_feature_data) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(m0_feature_data.shape[0], 2, *m0_feature_data.shape[-2:]) # (b, 2, H, W)
    
    vx_lst = [-1, 1]
    vy_lst = [-1, 1]
    eps_shift = 1e-6
    
    self_m0_preds, self_m1_preds = [], []
    m0_areas, m1_areas = [], []

    ###
    # Cross Encoder
    self_m0_cross_preds, self_m1_cross_preds = [], []
    m0_cross_areas, m1_cross_areas = [], []
    ###

    for vx in vx_lst:
        for vy in vy_lst:
            #########
            coord_ = coord.clone() # (b, n_query_pts, 2)
            # left-top, left-down, right-top, right-down move one radius.
            coord_[:, :, 0] += vx * rx + eps_shift
            coord_[:, :, 1] += vy * ry + eps_shift
            coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

            m0_q_feat, q_coord = look_up_feature(coord_, m0_feature_data, feat_coord)
            m1_q_feat, _ = look_up_feature(coord_, m1_feature_data, feat_coord)

            ### 
            # Cross Encoder
            m0_cross_q_feat, _ = look_up_feature(coord_, m0_cross_feature_data, feat_coord)
            m1_cross_q_feat, _ = look_up_feature(coord_, m1_cross_feature_data, feat_coord)
            ###

            m0_relative_offset = coord - q_coord
            m0_relative_offset[:, :, 0] *= m0_feature_data.shape[-2]
            m0_relative_offset[:, :, 1] *= m0_feature_data.shape[-1]

            m1_relative_offset = coord - q_coord
            m1_relative_offset[:, :, 0] *= m1_feature_data.shape[-2]
            m1_relative_offset[:, :, 1] *= m1_feature_data.shape[-1]

            ###
            # Cross Encoder
            m0_cross_relative_offset = coord - q_coord
            m0_cross_relative_offset[:, :, 0] *= m0_cross_feature_data.shape[-2]
            m0_cross_relative_offset[:, :, 1] *= m0_cross_feature_data.shape[-1]

            m1_cross_relative_offset = coord - q_coord
            m1_cross_relative_offset[:, :, 0] *= m1_cross_feature_data.shape[-2]
            m1_cross_relative_offset[:, :, 1] *= m1_cross_feature_data.shape[-1]
            ###

            m0_area = torch.abs(m0_relative_offset[:, :, 0] * m0_relative_offset[:, :, 1])
            m1_area = torch.abs(m1_relative_offset[:, :, 0] * m1_relative_offset[:, :, 1])

            ###
            # Cross Encoder
            m0_cross_area = torch.abs(m0_cross_relative_offset[:, :, 0] * m0_cross_relative_offset[:, :, 1])
            m1_cross_area = torch.abs(m1_cross_relative_offset[:, :, 0] * m1_cross_relative_offset[:, :, 1])
            ###

            m0_inp = torch.cat([m0_q_feat, m0_relative_offset], dim=-1)
            m1_inp = torch.cat([m1_q_feat, m1_relative_offset], dim=-1)

            ###
            # Cross Encoder
            m0_cross_inp = torch.cat([m0_cross_q_feat, m0_cross_relative_offset], dim=-1)
            m1_cross_inp = torch.cat([m1_cross_q_feat, m1_cross_relative_offset], dim=-1)
            ###

            decoded_cell = 2 / (cell.unsqueeze(1).repeat(1, coord.shape[1], 1))

            if use_global_encoder and use_pos_encoder:
                m0_inp = torch.cat([m0_inp, m0_global_encoded_LR, encoded_coord, decoded_cell], dim=-1)
                m1_inp = torch.cat([m1_inp, m1_global_encoded_LR, encoded_coord, decoded_cell], dim=-1)
            elif not use_global_encoder and use_pos_encoder:
                m0_inp = torch.cat([m0_inp, encoded_coord, decoded_cell], dim=-1)
                m1_inp = torch.cat([m1_inp, encoded_coord, decoded_cell], dim=-1)
            elif use_global_encoder and not use_pos_encoder:
                m0_inp = torch.cat([m0_inp, m0_global_encoded_LR, decoded_cell], dim=-1)
                m1_inp = torch.cat([m1_inp, m1_global_encoded_LR, decoded_cell], dim=-1)
            else:
                m0_inp = torch.cat([m0_inp, decoded_cell], dim=-1)
                m1_inp = torch.cat([m1_inp, decoded_cell], dim=-1)

            ###
            # Cross Encoder
            if use_global_encoder and use_pos_encoder:
                m0_cross_inp = torch.cat([m0_cross_inp, m0_cross_global_encoded_LR, encoded_coord, decoded_cell], dim=-1)
                m1_cross_inp = torch.cat([m1_cross_inp, m1_cross_global_encoded_LR, encoded_coord, decoded_cell], dim=-1)
            elif not use_global_encoder and use_pos_encoder:
                m0_cross_inp = torch.cat([m0_cross_inp, encoded_coord, decoded_cell], dim=-1)
                m1_cross_inp = torch.cat([m1_cross_inp, encoded_coord, decoded_cell], dim=-1)
            elif use_global_encoder and not use_pos_encoder:
                m0_cross_inp = torch.cat([m0_cross_inp, m0_cross_global_encoded_LR, decoded_cell], dim=-1)
                m1_cross_inp = torch.cat([m1_cross_inp, m1_cross_global_encoded_LR, decoded_cell], dim=-1)
            else:
                m0_cross_inp = torch.cat([m0_cross_inp, decoded_cell], dim=-1)
                m1_cross_inp = torch.cat([m1_cross_inp, decoded_cell], dim=-1)
            ###

            self_m0_pred = m0_decoder(m0_inp.view(b * q, -1)).view(b, q, -1)
            self_m1_pred = m1_decoder(m1_inp.view(b * q, -1)).view(b, q, -1)

            ###
            # Cross Encoder
            self_m0_cross_pred = m0_decoder(m0_cross_inp.view(b * q, -1)).view(b, q, -1)
            self_m1_cross_pred = m1_decoder(m1_cross_inp.view(b * q, -1)).view(b, q, -1)
            ###

            self_m0_preds.append(self_m0_pred)
            self_m1_preds.append(self_m1_pred)

            ###
            # Cross Encoder
            self_m0_cross_preds.append(self_m0_cross_pred)
            self_m1_cross_preds.append(self_m1_cross_pred)
            ###

            m0_areas.append(m0_area + 1e-9)
            m1_areas.append(m1_area + 1e-9)

            ###
            # Cross Encoder
            m0_cross_areas.append(m0_cross_area + 1e-9)
            m1_cross_areas.append(m1_cross_area + 1e-9)
            ###
            
            #########
            
    m0_tot_area = torch.stack(m0_areas).sum(dim=0)
    m1_tot_area = torch.stack(m1_areas).sum(dim=0)

    ###
    # Cross Encoder
    m0_cross_tot_area = torch.stack(m0_cross_areas).sum(dim=0)
    m1_cross_tot_area = torch.stack(m1_cross_areas).sum(dim=0)
    ###
    
    t = m0_areas[0]; m0_areas[0] = m0_areas[3]; m0_areas[3] = t
    t = m0_areas[1]; m0_areas[1] = m0_areas[2]; m0_areas[2] = t
    
    t = m1_areas[0]; m1_areas[0] = m1_areas[3]; m1_areas[3] = t
    t = m1_areas[1]; m1_areas[1] = m1_areas[2]; m1_areas[2] = t

    ###
    # Cross Encoder
    t = m0_cross_areas[0]; m0_cross_areas[0] = m0_cross_areas[3]; m0_cross_areas[3] = t
    t = m0_cross_areas[1]; m0_cross_areas[1] = m0_cross_areas[2]; m0_cross_areas[2] = t
    
    t = m1_cross_areas[0]; m1_cross_areas[0] = m1_cross_areas[3]; m1_cross_areas[3] = t
    t = m1_cross_areas[1]; m1_cross_areas[1] = m1_cross_areas[2]; m1_cross_areas[2] = t
    ###
    
    self_m0_ret = 0
    for pred, area in zip(self_m0_preds, m0_areas):
        self_m0_ret = self_m0_ret + pred * (area / m0_tot_area).unsqueeze(-1)
        
    self_m1_ret = 0
    for pred, area in zip(self_m1_preds, m1_areas):
        self_m1_ret = self_m1_ret + pred * (area / m1_tot_area).unsqueeze(-1)

    ###
    # Cross Encoder
    self_m0_cross_ret = 0
    for pred, area in zip(self_m0_cross_preds, m0_cross_areas):
        self_m0_cross_ret = self_m0_cross_ret + pred * (area / m0_cross_tot_area).unsqueeze(-1)
        
    self_m1_cross_ret = 0
    for pred, area in zip(self_m1_cross_preds, m1_cross_areas):
        self_m1_cross_ret = self_m1_cross_ret + pred * (area / m1_cross_tot_area).unsqueeze(-1)
    ###

    if encoder_model == 'fadn':
        return m0_data_HR, m0_encoded_LR, m0_cross_encoded_LR, self_m0_ret, m0_sparsity_avg, self_m0_cross_ret, m0_cross_sparsity_avg, m1_data_HR, m1_encoded_LR, m1_cross_encoded_LR, self_m1_ret, m1_sparsity_avg, self_m1_cross_ret, m1_cross_sparsity_avg, b
    else:
        return m0_data_HR, m0_encoded_LR, m0_cross_encoded_LR, self_m0_ret, self_m0_cross_ret, m1_data_HR, m1_encoded_LR, m1_cross_encoded_LR, self_m1_ret, self_m1_cross_ret, b


data_dir = args.data_dir #'./data'

print('Start Loading Data')

train_dataset = LIIFDataset(
    m0_data_dir=os.path.join(data_dir, "wind-{}m/{}".format(args.height_0, args.file_prefix)),
    m1_data_dir=os.path.join(data_dir, "wind-{}m/{}".format(args.height_1, args.file_prefix)),
    file_prefix=args.file_prefix,
    liif_scales=[1, args.train_sr],
    low_resol=[args.h_LR, args.w_LR],
    query_points=args.query_points,
    max_val=50,
    train_frac=args.train_frac,
    train=True
)

val_dataset_eval = LIIFDataset(
    m0_data_dir=os.path.join(data_dir, "wind-{}m/{}".format(args.height_0, args.file_prefix)),
    m1_data_dir=os.path.join(data_dir, "wind-{}m/{}".format(args.height_1, args.file_prefix)),
    file_prefix=args.file_prefix,
    liif_scales=[1, args.train_sr],
    low_resol=[args.h_LR, args.w_LR],
    query_points=args.query_points,
    max_val=50,
    train_frac=args.train_frac,
    train=False
)

batch_size = args.batch_size

train_dataloader = DataLoader(dataset=train_dataset, 
                              batch_size=batch_size, 
                              num_workers=4, 
                              pin_memory=True, 
                              shuffle=True)

val_dataloader_eval = DataLoader(dataset=val_dataset_eval, 
                                 batch_size=1, 
                                 num_workers=4, 
                                 pin_memory=True, 
                                 shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

if torch.cuda.is_available():
    torch.cuda.empty_cache()

if args.use_pos_encoder:
    coord_encoder = PositionalEncoding(sigma=args.sigma, m=args.m)
else:
    coord_encoder = None

m0_latent_encoder = LatentEncoder(
    in_dim=args.input_dim,
    latent_dim=args.latent_dim).to(device)
if args.load_pretrained_model:
    m0_latent_encoder = torch.load(
        os.path.join(args.model_save_dir, 'm0_latent_encoder_epoch_{}.sav'.format(args.pretrained_epoch)),
        map_location=device
    )

m1_latent_encoder = LatentEncoder(
    in_dim=args.input_dim,
    latent_dim=args.latent_dim).to(device)
if args.load_pretrained_model:
    m1_latent_encoder = torch.load(
        os.path.join(args.model_save_dir, 'm1_latent_encoder_epoch_{}.sav'.format(args.pretrained_epoch)),
        map_location=device
    )

m021_latent_encoder = LatentEncoder(
    in_dim=args.input_dim,
    latent_dim=args.latent_dim).to(device)
if args.load_pretrained_model:
    m021_latent_encoder = torch.load(
        os.path.join(args.model_save_dir, 'm021_latent_encoder_epoch_{}.sav'.format(args.pretrained_epoch)),
        map_location=device
    )

m120_latent_encoder = LatentEncoder(
    in_dim=args.input_dim,
    latent_dim=args.latent_dim).to(device)
if args.load_pretrained_model:
    m120_latent_encoder = torch.load(
        os.path.join(args.model_save_dir, 'm120_latent_encoder_epoch_{}.sav'.format(args.pretrained_epoch)),
        map_location=device
    )

if args.use_global_encoder:
    m0_global_encoder = Global_Encoder(
        n_inps = args.latent_dim,
        n_feats= args.n_encoder_features,
        out_feats= args.n_global_encoder_features
    ).to(device)
    if args.load_pretrained_model:
        m0_global_encoder = torch.load(
            os.path.join(args.model_save_dir, 'm0_global_encoder_epoch_{}.sav'.format(args.pretrained_epoch)),
            map_location=device
        )

    m1_global_encoder = Global_Encoder(
        n_inps = args.latent_dim,
        n_feats= args.n_encoder_features,
        out_feats= args.n_global_encoder_features
    ).to(device)
    if args.load_pretrained_model:
        m1_global_encoder = torch.load(
            os.path.join(args.model_save_dir, 'm1_global_encoder_epoch_{}.sav'.format(args.pretrained_epoch)),
            map_location=device
        )
else:
    m0_global_encoder = None
    m1_global_encoder = None

m0_feature_encoder = CascadedEDSR(
    n_blocks=args.n_feature_encoder_blocks,
    n_inputs=args.latent_dim,
    n_feats=args.n_encoder_features,
    kernel_size=args.encoder_kernel_size, 
    n_resblocks=args.n_encoder_resblocks,
    bias=args.encoder_bias,
    bn=args.encoder_batch_norm,
    act=args.encoder_non_lin,
    res_scale=args.residual_scale,
    scale=args.encoder_scale,
    upsampling=args.encoder_upsampling,
    weighted_upsampling=False
).to(device)
if args.load_pretrained_model:
    m0_feature_encoder = torch.load(
        os.path.join(args.model_save_dir, 'm0_feature_encoder_epoch_{}.sav'.format(args.pretrained_epoch)),
        map_location=device
    )

m1_feature_encoder = CascadedEDSR(
    n_blocks=args.n_feature_encoder_blocks,
    n_inputs=args.latent_dim,
    n_feats=args.n_encoder_features,
    kernel_size=args.encoder_kernel_size, 
    n_resblocks=args.n_encoder_resblocks,
    bias=args.encoder_bias,
    bn=args.encoder_batch_norm,
    act=args.encoder_non_lin,
    res_scale=args.residual_scale,
    scale=args.encoder_scale,
    upsampling=args.encoder_upsampling,
    weighted_upsampling=False
).to(device)
if args.load_pretrained_model:
    m1_feature_encoder = torch.load(
        os.path.join(args.model_save_dir, 'm1_feature_encoder_epoch_{}.sav'.format(args.pretrained_epoch)),
        map_location=device
    )

if args.use_global_encoder and args.use_pos_encoder:
    in_dim = args.n_encoder_features*9 + 4 + 4*args.m + args.n_global_encoder_features #1000 is for resnet50
elif args.use_global_encoder and not args.use_pos_encoder:
    in_dim = args.n_encoder_features*9 + 4 + args.n_global_encoder_features #1000 is for resnet50
elif not args.use_global_encoder and args.use_pos_encoder:
    in_dim = args.n_encoder_features*9 + 4 + 4*args.m
else:
    in_dim = args.n_encoder_features*9 + 4

m0_decoder = INR(
    in_dim=in_dim, #args.n_encoder_features*9 + 4 + 4*args.m + 1000, #1000 is for resnet50
    act=args.decoder_non_lin,
).to(device)
if args.load_pretrained_model:
    m0_decoder = torch.load(
        os.path.join(args.model_save_dir, 'm0_decoder_epoch_{}.sav'.format(args.pretrained_epoch)),
        map_location=device
    )

m1_decoder = INR(
    in_dim=in_dim, #args.n_encoder_features*9 + 4 + 4*args.m + 1000, # #1000 is for resnet50
    act=args.decoder_non_lin,
).to(device)
if args.load_pretrained_model:
    m1_decoder = torch.load(
        os.path.join(args.model_save_dir, 'm1_decoder_epoch_{}.sav'.format(args.pretrained_epoch)),
        map_location=device
    )

param_list = [
    {'params': m0_latent_encoder.parameters(), 'lr': args.encoder_lr, 'weight_decay': args.encoder_wd}, 
    {'params': m1_latent_encoder.parameters(), 'lr': args.encoder_lr, 'weight_decay': args.encoder_wd}, 
    {'params': m021_latent_encoder.parameters(), 'lr': args.encoder_lr, 'weight_decay': args.encoder_wd},
    {'params': m120_latent_encoder.parameters(), 'lr': args.encoder_lr, 'weight_decay': args.encoder_wd},
    {'params': m0_feature_encoder.parameters(), 'lr': args.encoder_lr, 'weight_decay': args.encoder_wd}, 
    {'params': m1_feature_encoder.parameters(), 'lr': args.encoder_lr, 'weight_decay': args.encoder_wd},
    {'params': m0_decoder.parameters(), 'lr': args.decoder_lr, 'weight_decay': args.decoder_wd}, 
    {'params': m1_decoder.parameters(), 'lr': args.decoder_lr, 'weight_decay': args.decoder_wd}
]

if args.use_global_encoder:
    param_list.append(
        {'params': m0_global_encoder.parameters(), 'lr': args.encoder_lr, 'weight_decay': args.encoder_wd}, 
    )
    param_list.append(
        {'params': m1_global_encoder.parameters(), 'lr': args.encoder_lr, 'weight_decay': args.encoder_wd},
    )

optimizer = torch.optim.Adam(param_list)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.learning_rate_decay, verbose=True)
num_epochs = args.num_epochs

output = sys.stdout
print('\t'.join(['Epoch', 'Mode', 'Reconstruction Loss', 'Cross Reconstruction Loss', 'Latent Loss']), file=output)

# recon_loss = nn.MSELoss(reduction='mean')
recon_loss = nn.L1Loss(reduction='mean')
LR_loss = nn.MSELoss(reduction='mean')

if args.model_save_dir is not None:
    os.makedirs(args.model_save_dir, exist_ok=True)

for epoch in np.arange(num_epochs):
    m0_latent_encoder.train() 
    m1_latent_encoder.train()
    m021_latent_encoder.train()
    m120_latent_encoder.train()
    if args.use_global_encoder:
        m0_global_encoder.train()
        m1_global_encoder.train()
    m0_feature_encoder.train()
    m1_feature_encoder.train()
    m0_decoder.train()
    m1_decoder.train()
    
    n_train_points = 0

    total_self_loss = 0
    total_cross_loss = 0
    total_latent_loss = 0

    for batch in train_dataloader:
        optimizer.zero_grad()
        if args.encoder_model == 'fadn':
            m0_data_HR, m0_encoded_LR, m0_cross_encoded_LR, self_m0_ret, m0_sparsity_avg, self_m0_cross_ret, m0_cross_sparsity_avg, m1_data_HR, m1_encoded_LR, m1_cross_encoded_LR, self_m1_ret, m1_sparsity_avg, self_m1_cross_ret, m1_cross_sparsity_avg, b = pred_HR(batch, 
                                                                                                    coord_encoder,
                                                                                                    m0_latent_encoder,
                                                                                                    m1_latent_encoder,
                                                                                                    m021_latent_encoder,
                                                                                                    m120_latent_encoder,
                                                                                                    m0_global_encoder,
                                                                                                    m1_global_encoder,
                                                                                                    m0_feature_encoder,
                                                                                                    m1_feature_encoder,
                                                                                                    m0_decoder,
                                                                                                    m1_decoder,
                                                                                                    device,
                                                                                                    args.encoder_model,
                                                                                                    use_global_encoder=args.use_global_encoder,
                                                                                                    use_pos_encoder=args.use_pos_encoder)
        else:
            m0_data_HR, m0_encoded_LR, m0_cross_encoded_LR, self_m0_ret, self_m0_cross_ret, m1_data_HR, m1_encoded_LR, m1_cross_encoded_LR, self_m1_ret, self_m1_cross_ret, b = pred_HR(batch, 
                                                                    coord_encoder,
                                                                    m0_latent_encoder,
                                                                    m1_latent_encoder,
                                                                    m021_latent_encoder,
                                                                    m120_latent_encoder,
                                                                    m0_global_encoder,
                                                                    m1_global_encoder,
                                                                    m0_feature_encoder,
                                                                    m1_feature_encoder,
                                                                    m0_decoder,
                                                                    m1_decoder,
                                                                    device,
                                                                    args.encoder_model,
                                                                    use_global_encoder=args.use_global_encoder,
                                                                    use_pos_encoder=args.use_pos_encoder)

        self_loss = recon_loss(
            input=torch.cat([self_m0_ret, self_m1_ret], dim=0),
            target=torch.cat([m0_data_HR, m1_data_HR], dim=0)
            )
        
        cross_loss = recon_loss(
            input=torch.cat([self_m0_cross_ret, self_m1_cross_ret], dim=0),
            target=torch.cat([m0_data_HR, m1_data_HR], dim=0)
            )
            
        mean_m0_encoded_LR = torch.cat([m0_encoded_LR.unsqueeze(dim=0), m0_cross_encoded_LR.unsqueeze(dim=0)], dim=0).mean(dim=0).detach()
        
        mean_m1_encoded_LR = torch.cat([m1_encoded_LR.unsqueeze(dim=0), m1_cross_encoded_LR.unsqueeze(dim=0)], dim=0).mean(dim=0).detach()
        
        latent_loss = LR_loss(
        	input=torch.cat([m0_encoded_LR, m0_cross_encoded_LR, m1_encoded_LR, m1_cross_encoded_LR], dim=0),
        	target=torch.cat([mean_m0_encoded_LR, mean_m0_encoded_LR, mean_m1_encoded_LR, mean_m1_encoded_LR], dim=0)
        	)
        
        # self_loss = self_loss/b

        # cross_loss = cross_loss/b

        spar_loss = m0_sparsity_avg**2 + m1_sparsity_avg**2 + m0_cross_sparsity_avg**2 + m1_cross_sparsity_avg**2 if args.encoder_model=='fadn' else None

        loss = args.lambda_s*self_loss + args.lambda_c*cross_loss + args.lambda_l*latent_loss + args.lambda_kl*spar_loss.mean() if args.encoder_model=='fadn' else args.lambda_s*self_loss + args.lambda_c*cross_loss + args.lambda_l*latent_loss

        loss.backward()

        optimizer.step()

        template = '# [{}/{}]: Self-Loss={:.5f}, Cross-Loss={:.5f}, Latent-Loss={:.5f}'
        line = template.format(epoch+1, num_epochs, self_loss, cross_loss, latent_loss)
        print(line, end='\r', file=sys.stderr)

        total_self_loss += self_loss.item() * b
        total_cross_loss += cross_loss.item() * b
        total_latent_loss += latent_loss.item() *b

        n_train_points += b

    print(' '*90, end='\r', file=sys.stderr)
    
    line = '\t'.join(
        [str(epoch+1), 
         'train', 
         str(round(total_self_loss/n_train_points, 4)), 
         str(round(total_cross_loss/n_train_points, 4)), 
         str(round(total_latent_loss/n_train_points, 4))
        ]
        )
    print(line, file=output)
    output.flush()
    if epoch > args.start_lr_decay and (epoch+1+args.start_lr_decay)%args.lr_decay_interval==0:
        scheduler.step()

    m0_latent_encoder.eval() 
    m1_latent_encoder.eval()
    m021_latent_encoder.eval()
    m120_latent_encoder.eval()
    if args.use_global_encoder:
        m0_global_encoder.eval()
        m1_global_encoder.eval()
        # global_encoder.eval()
    m0_feature_encoder.eval()
    m1_feature_encoder.eval()
    m0_decoder.eval()
    m1_decoder.eval()

    n_val_points = 0

    total_self_loss = 0
    total_cross_loss = 0
    total_latent_loss = 0

    for batch in val_dataloader_eval:
        with torch.no_grad():
            if args.encoder_model == 'fadn':
                m0_data_HR, m0_encoded_LR, m0_cross_encoded_LR, self_m0_ret, m0_sparsity_avg, self_m0_cross_ret, m0_cross_sparsity_avg, m1_data_HR, m1_encoded_LR, m1_cross_encoded_LR, self_m1_ret, m1_sparsity_avg, self_m1_cross_ret, m1_cross_sparsity_avg, b = pred_HR(batch, 
                                                                                                        coord_encoder,
                                                                                                        m0_latent_encoder,
                                                                                                        m1_latent_encoder,
                                                                                                        m021_latent_encoder,
                                                                                                        m120_latent_encoder,
                                                                                                        m0_global_encoder,
                                                                                                        m1_global_encoder,
                                                                                                        # global_encoder,
                                                                                                        m0_feature_encoder,
                                                                                                        m1_feature_encoder,
                                                                                                        m0_decoder,
                                                                                                        m1_decoder,
                                                                                                        device,
                                                                                                        args.encoder_model,
                                                                                                        use_global_encoder=args.use_global_encoder,
                                                                                                        use_pos_encoder=args.use_pos_encoder)
            else:
                m0_data_HR, m0_encoded_LR, m0_cross_encoded_LR, self_m0_ret, self_m0_cross_ret, m1_data_HR, m1_encoded_LR, m1_cross_encoded_LR, self_m1_ret, self_m1_cross_ret, b = pred_HR(batch, 
                                                                        coord_encoder,
                                                                        m0_latent_encoder,
                                                                        m1_latent_encoder,
                                                                        m021_latent_encoder,
                                                                        m120_latent_encoder,
                                                                        m0_global_encoder,
                                                                        m1_global_encoder,
                                                                        # global_encoder,
                                                                        m0_feature_encoder,
                                                                        m1_feature_encoder,
                                                                        m0_decoder,
                                                                        m1_decoder,
                                                                        device,
                                                                        args.encoder_model,
                                                                        use_global_encoder=args.use_global_encoder,
                                                                        use_pos_encoder=args.use_pos_encoder)

            self_loss = recon_loss(
                input=torch.cat([self_m0_ret, self_m1_ret], dim=0),
                target=torch.cat([m0_data_HR, m1_data_HR], dim=0)
                )
            
            cross_loss = recon_loss(
                input=torch.cat([self_m0_cross_ret, self_m1_cross_ret], dim=0),
                target=torch.cat([m0_data_HR, m1_data_HR], dim=0)
                )
                
            mean_m0_encoded_LR = torch.cat([m0_encoded_LR.unsqueeze(dim=0), m0_cross_encoded_LR.unsqueeze(dim=0)], dim=0).mean(dim=0)
        
            mean_m1_encoded_LR = torch.cat([m1_encoded_LR.unsqueeze(dim=0), m1_cross_encoded_LR.unsqueeze(dim=0)], dim=0).mean(dim=0)
        
            latent_loss = LR_loss(
                input=torch.cat([m0_encoded_LR, m0_cross_encoded_LR, m1_encoded_LR, m1_cross_encoded_LR], dim=0),
                target=torch.cat([mean_m0_encoded_LR, mean_m0_encoded_LR, mean_m1_encoded_LR, mean_m1_encoded_LR], dim=0)
                )
            
            # self_loss = self_loss/b

            # cross_loss = cross_loss/b

            total_self_loss += self_loss.item() * b
            total_cross_loss += cross_loss.item() * b
            total_latent_loss += latent_loss.item() * b

            n_val_points += b
    line = '\t'.join(
        [str(epoch+1), 
         'val', 
         str(round(total_self_loss/n_val_points, 4)),
         str(round(total_cross_loss/n_val_points, 4)),
         str(round(total_latent_loss/n_val_points, 4))
        ]
        )
    print(line, file=output)
    output.flush()

    ## save the models
    if args.model_save_dir is not None and (epoch+1)%args.save_interval == 0:

        path = os.path.join(args.model_save_dir, 'm0_latent_encoder_epoch_{}.sav'.format(epoch+1))
        torch.save(m0_latent_encoder, path)

        path = os.path.join(args.model_save_dir, 'm1_latent_encoder_epoch_{}.sav'.format(epoch+1))
        torch.save(m1_latent_encoder, path)

        path = os.path.join(args.model_save_dir, 'm021_latent_encoder_epoch_{}.sav'.format(epoch+1))
        torch.save(m021_latent_encoder, path)

        path = os.path.join(args.model_save_dir, 'm120_latent_encoder_epoch_{}.sav'.format(epoch+1))
        torch.save(m120_latent_encoder, path)

        if args.use_global_encoder:
            path = os.path.join(args.model_save_dir, 'm0_global_encoder_epoch_{}.sav'.format(epoch+1))
            torch.save(m0_global_encoder, path)

            path = os.path.join(args.model_save_dir, 'm1_global_encoder_epoch_{}.sav'.format(epoch+1))
            torch.save(m1_global_encoder, path)

            # path = os.path.join(args.model_save_dir, 'global_encoder_epoch_{}.sav'.format(epoch+1))
            # torch.save(global_encoder, path)

        path = os.path.join(args.model_save_dir, 'm0_feature_encoder_epoch_{}.sav'.format(epoch+1))
        torch.save(m0_feature_encoder, path)

        path = os.path.join(args.model_save_dir, 'm1_feature_encoder_epoch_{}.sav'.format(epoch+1))
        torch.save(m1_feature_encoder, path)

        path = os.path.join(args.model_save_dir, 'm0_decoder_epoch_{}.sav'.format(epoch+1))
        torch.save(m0_decoder, path)

        path = os.path.join(args.model_save_dir, 'm1_decoder_epoch_{}.sav'.format(epoch+1))
        torch.save(m1_decoder, path)
