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
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.transforms.functional import InterpolationMode
from typing import Optional

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import json

import argparse

from skimage.metrics import structural_similarity as ssim

parser = argparse.ArgumentParser('Train Multi-Modality Super Resolution')

parser.add_argument('--data-dir', help='path to training data')

parser.add_argument('--use-gpu', action='store_true', help='Whether to use GPU')
parser.add_argument('--use-global-encoder', action='store_true', help='Whether to use Pre-trained Global Encoder')
parser.add_argument('--use-pos-encoder', action='store_true', help='Whether to use Positional Encoding')
parser.add_argument('--saved-model-epoch', type=int, default=500)

parser.add_argument('--file-prefix', type=str, choices=['ua', 'va'])

parser.add_argument('--height-0', type=int, choices=[10, 60, 160, 200])
parser.add_argument('--height-1', type=int, choices=[10, 60, 160, 200])

parser.add_argument('--train-frac', type=float, default=0.8)

parser.add_argument('--h-LR', type=int, default=120)
parser.add_argument('--w-LR', type=int, default=160)

parser.add_argument('--sr-scale', type=float, default=1.0)

parser.add_argument('--sigma', type=float, default=30.0)
parser.add_argument('--m', type=int, default=50)

parser.add_argument('--model-save-dir', type=str, help='Directory to Save Models')

# parser.add_argument('--save-interval', type=int, help='Intervals at which to save the models')

parser.add_argument('--result-save-dir', help='path to save results')

args = parser.parse_args()

non_act = {'relu': partial(nn.ReLU),
       'sigmoid': partial(nn.Sigmoid),
       'tanh': partial(nn.Tanh),
       'selu': partial(nn.SELU),
       'softplus': partial(nn.Softplus),
       'gelu': partial(nn.GELU),
       'elu': partial(nn.ELU)}

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

def pointwise_decode(decoder,
                     device,
                     feature,
                     batch_size:int,
                     query_points:int,
                     max_points_to_decode:int=32):
    # pass
    feature = feature.view(batch_size * query_points, -1)
    feature_splits = torch.split(feature, split_size_or_sections=max_points_to_decode, dim=0)
    pred = []
    for feature_split in feature_splits:
        pred.append(decoder(feature_split.to(device)).cpu())
    pred = torch.cat(pred, dim=0)
    return pred.view(batch_size, query_points, -1)

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
            use_global_encoder: bool = True,
            use_pos_encoder: bool = True):
    
    m0_data_LR, m1_data_LR = batch['m0_img_LR'].to(device), batch['m1_img_LR'].to(device)
    m0_data_HR, m1_data_HR = batch['m0_img_HR'].to(device), batch['m1_img_HR'].to(device)
    
    coord = batch['grid_HR'].to(device)
    cell = batch['cell'].to(device)

    b, q = coord.shape[:2]

    if use_pos_encoder:
        encoded_coord = coord_encoder(coord).to(device)

    m0_encoded_LR = m0_latent_encoder(m0_data_LR)
    m1_encoded_LR = m1_latent_encoder(m1_data_LR)

    ### 
    # Cross Encoder
    m0_cross_encoded_LR = m120_latent_encoder(m1_data_LR)
    m1_cross_encoded_LR = m021_latent_encoder(m0_data_LR)
    ###

    if use_global_encoder:
        m0_global_encoded_LR = m0_global_encoder(m0_encoded_LR).unsqueeze(1)#.cpu()#.repeat(1, coord.shape[1], 1)
        m1_global_encoded_LR = m1_global_encoder(m1_encoded_LR).unsqueeze(1)#.cpu()#.repeat(1, coord.shape[1], 1)
        
        m0_global_encoded_LR = m0_global_encoded_LR.expand(b, q, m0_global_encoded_LR.size(2)).contiguous()
        m1_global_encoded_LR = m1_global_encoded_LR.expand(b, q, m1_global_encoded_LR.size(2)).contiguous()

    ###
    # Cross Encoder
    if use_global_encoder:
        m0_cross_global_encoded_LR = m0_global_encoder(m0_cross_encoded_LR).unsqueeze(1)#.cpu()#.repeat(1, coord.shape[1], 1)
        m1_cross_global_encoded_LR = m1_global_encoder(m1_cross_encoded_LR).unsqueeze(1)#.cpu()#.repeat(1, coord.shape[1], 1)

        m0_cross_global_encoded_LR = m0_cross_global_encoded_LR.expand(b, q, m0_cross_global_encoded_LR.size(2)).contiguous()
        m1_cross_global_encoded_LR = m1_cross_global_encoded_LR.expand(b, q, m1_cross_global_encoded_LR.size(2)).contiguous()
    ###

    m0_feature = m0_feature_encoder(m0_encoded_LR)#.cpu()
    m1_feature = m1_feature_encoder(m1_encoded_LR)#.cpu()

    ###
    # Cross Encoder
    m0_cross_feature = m0_feature_encoder(m0_cross_encoded_LR)#.cpu()
    m1_cross_feature = m1_feature_encoder(m1_cross_encoded_LR)#.cpu()
    ###
    
    m0_feature_data = F.unfold(m0_feature, 3, padding=1).view(
        m0_feature.shape[0], 
        m0_feature.shape[1] * 9, 
        m0_feature.shape[2], 
        m0_feature.shape[3]
    )#.cpu()
    
    m1_feature_data = F.unfold(m1_feature, 3, padding=1).view(
        m1_feature.shape[0], 
        m1_feature.shape[1] * 9, 
        m1_feature.shape[2], 
        m1_feature.shape[3]
    )#.cpu()

    ###
    # Cross Encoder
    m0_cross_feature_data = F.unfold(m0_cross_feature, 3, padding=1).view(
        m0_cross_feature.shape[0], 
        m0_cross_feature.shape[1] * 9, 
        m0_cross_feature.shape[2], 
        m0_cross_feature.shape[3]
    )#.cpu()
    
    m1_cross_feature_data = F.unfold(m1_cross_feature, 3, padding=1).view(
        m1_cross_feature.shape[0], 
        m1_cross_feature.shape[1] * 9, 
        m1_cross_feature.shape[2], 
        m1_cross_feature.shape[3]
    )#.cpu()
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
            # self_m0_pred = pointwise_decode(decoder=m0_decoder, device=device, feature=m0_inp, batch_size=b, query_points=q, max_points_to_decode=480*640)
            self_m1_pred = m1_decoder(m1_inp.view(b * q, -1)).view(b, q, -1)
            # self_m1_pred = pointwise_decode(decoder=m1_decoder, device=device, feature=m1_inp, batch_size=b, query_points=q, max_points_to_decode=480*640)

            ###
            # Cross Encoder
            self_m0_cross_pred = m0_decoder(m0_cross_inp.view(b * q, -1)).view(b, q, -1)
            # self_m0_cross_pred = pointwise_decode(decoder=m0_decoder, device=device, feature=m0_cross_inp, batch_size=b, query_points=q, max_points_to_decode=480*640)
            self_m1_cross_pred = m1_decoder(m1_cross_inp.view(b * q, -1)).view(b, q, -1)
            # self_m1_cross_pred = pointwise_decode(decoder=m1_decoder, device=device, feature=m1_cross_inp, batch_size=b, query_points=q, max_points_to_decode=480*640)
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
        
    b, _, h_HR, w_HR = m0_data_HR.shape 

    self_m0_ret = self_m0_ret.view(b, 1, h_HR, w_HR)
    self_m1_ret = self_m0_ret.view(b, 1, h_HR, w_HR)

    self_m0_cross_ret = self_m0_cross_ret.view(b, 1, h_HR, w_HR)
    self_m1_cross_ret = self_m1_cross_ret.view(b, 1, h_HR, w_HR)

    return m0_data_HR, \
            self_m0_ret, \
            self_m0_cross_ret, \
            m1_data_HR, \
            self_m1_ret, \
            self_m1_cross_ret, \
            b, \
            m0_encoded_LR, \
            m1_encoded_LR, \
            m0_cross_encoded_LR, \
            m1_cross_encoded_LR

@torch.no_grad()
def evaluate_model(
        val_dataloader_viz,
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
        use_global_encoder:bool,
        use_pos_encoder:bool,
        sample_size:int,
        ch_HR:int,
        cw_HR:int,
        plot_image:bool=False,
    ):

    m0_latent_encoder.eval() 
    m1_latent_encoder.eval()
    m021_latent_encoder.eval()
    m120_latent_encoder.eval()
    if args.use_global_encoder:
        m0_global_encoder.eval()
        m1_global_encoder.eval()
    m0_feature_encoder.eval()
    m1_feature_encoder.eval()
    m0_decoder.eval()
    m1_decoder.eval()

    data = {}

    if plot_image:
        data["target_0"] = np.empty(shape=(min(sample_size, val_dataloader_viz.__len__()),ch_HR,cw_HR))
        data["self_prediction_0"] = np.empty(shape=(min(sample_size, val_dataloader_viz.__len__()),ch_HR,cw_HR))
        data["target_1"] = np.empty(shape=(min(sample_size, val_dataloader_viz.__len__()),ch_HR,cw_HR))
        data["self_prediction_1"] = np.empty(shape=(min(sample_size, val_dataloader_viz.__len__()),ch_HR,cw_HR))

        # data["cross_target_0"] = np.empty(shape=(min(sample_size, val_dataloader_viz.__len__()),ch_HR,cw_HR))
        data["cross_prediction_0"] = np.empty(shape=(min(sample_size, val_dataloader_viz.__len__()),ch_HR,cw_HR))
        # data["cross_target_1"] = np.empty(shape=(min(sample_size, val_dataloader_viz.__len__()),ch_HR,cw_HR))
        data["cross_prediction_1"] = np.empty(shape=(min(sample_size, val_dataloader_viz.__len__()),ch_HR,cw_HR))

        data["self_encoded_LR_0"] = np.empty(shape=(min(sample_size, val_dataloader_viz.__len__()),15,20))
        data["self_encoded_LR_1"] = np.empty(shape=(min(sample_size, val_dataloader_viz.__len__()),15,20))

        data["cross_encoded_LR_0"] = np.empty(shape=(min(sample_size, val_dataloader_viz.__len__()),15,20))
        data["cross_encoded_LR_1"] = np.empty(shape=(min(sample_size, val_dataloader_viz.__len__()),15,20))
    
    data["self_PSNR_0"] = np.empty(shape=(min(sample_size, val_dataloader_viz.__len__())))
    data["self_PSNR_1"] = np.empty(shape=(min(sample_size, val_dataloader_viz.__len__())))
    data["cross_PSNR_0"] = np.empty(shape=(min(sample_size, val_dataloader_viz.__len__())))
    data["cross_PSNR_1"] = np.empty(shape=(min(sample_size, val_dataloader_viz.__len__())))

    data["self_SSIM_0"] = np.empty(shape=(min(sample_size, val_dataloader_viz.__len__())))
    data["self_SSIM_1"] = np.empty(shape=(min(sample_size, val_dataloader_viz.__len__())))
    data["cross_SSIM_0"] = np.empty(shape=(min(sample_size, val_dataloader_viz.__len__())))
    data["cross_SSIM_1"] = np.empty(shape=(min(sample_size, val_dataloader_viz.__len__())))

    # n_val_points = 0

    with tqdm(total=min(sample_size, val_dataloader_viz.__len__())) as pbar:
        for sample_idx, batch in enumerate(val_dataloader_viz):
            m0_data_HR, self_m0_ret, self_m0_cross_ret, m1_data_HR, self_m1_ret, self_m1_cross_ret, _, m0_encoded_LR, m1_encoded_LR, m0_cross_encoded_LR, m1_cross_encoded_LR = pred_HR(batch=batch,
                coord_encoder=coord_encoder,
                m0_latent_encoder=m0_latent_encoder,
                m1_latent_encoder=m1_latent_encoder,
                m021_latent_encoder=m021_latent_encoder,
                m120_latent_encoder=m120_latent_encoder,
                m0_global_encoder=m0_global_encoder,
                m1_global_encoder=m1_global_encoder,
                m0_feature_encoder=m0_feature_encoder,
                m1_feature_encoder=m1_feature_encoder,
                m0_decoder=m0_decoder,
                m1_decoder=m1_decoder,
                device=device,
                use_global_encoder=use_global_encoder,
                use_pos_encoder=use_pos_encoder)
            
            if plot_image:
                data["target_0"][sample_idx] = m0_data_HR.cpu().numpy().squeeze()
                data["target_1"][sample_idx] = m1_data_HR.cpu().numpy().squeeze()

                data["self_prediction_0"][sample_idx] = self_m0_ret.cpu().numpy().squeeze()
                data["self_prediction_1"][sample_idx] = self_m1_ret.cpu().numpy().squeeze()

                data["cross_prediction_0"][sample_idx] = self_m0_cross_ret.cpu().numpy().squeeze()
                data["cross_prediction_1"][sample_idx] = self_m1_cross_ret.cpu().numpy().squeeze()

                data["self_encoded_LR_0"][sample_idx] = m0_encoded_LR.cpu().numpy().squeeze()
                data["self_encoded_LR_1"][sample_idx] = m1_encoded_LR.cpu().numpy().squeeze()

                data["cross_encoded_LR_0"][sample_idx] = m0_cross_encoded_LR.cpu().numpy().squeeze()
                data["cross_encoded_LR_1"][sample_idx] = m1_cross_encoded_LR.cpu().numpy().squeeze()
            
            val_range = 2.0
            data_range = 2.0

            noise = m0_data_HR.cpu().numpy().squeeze() - self_m0_ret.cpu().numpy().squeeze()
            data["self_PSNR_0"][sample_idx] = 10*np.log10((val_range**2)/(noise ** 2).mean())
            data["self_SSIM_0"][sample_idx] = ssim(m0_data_HR.cpu().numpy().squeeze(), self_m0_ret.cpu().numpy().squeeze(), data_range=data_range)

            noise = m1_data_HR.cpu().numpy().squeeze() - self_m1_ret.cpu().numpy().squeeze()
            data["self_PSNR_1"][sample_idx] = 10*np.log10((val_range**2)/(noise ** 2).mean())
            data["self_SSIM_1"][sample_idx] = ssim(m1_data_HR.cpu().numpy().squeeze(), self_m1_ret.cpu().numpy().squeeze(), data_range=data_range)

            noise = m0_data_HR.cpu().numpy().squeeze() - self_m0_cross_ret.cpu().numpy().squeeze()
            data["cross_PSNR_0"][sample_idx] = 10*np.log10((val_range**2)/(noise ** 2).mean())
            data["cross_SSIM_0"][sample_idx] = ssim(m0_data_HR.cpu().numpy().squeeze(), self_m0_cross_ret.cpu().numpy().squeeze(), data_range=data_range)

            noise = m1_data_HR.cpu().numpy().squeeze() - self_m1_cross_ret.cpu().numpy().squeeze()
            data["cross_PSNR_1"][sample_idx] = 10*np.log10((val_range**2)/(noise ** 2).mean())
            data["cross_SSIM_1"][sample_idx] = ssim(m1_data_HR.cpu().numpy().squeeze(), self_m1_cross_ret.cpu().numpy().squeeze(), data_range=data_range)

            pbar.update(1)
            
            if (sample_idx+1)%sample_size==0:
                break

    return data

data_dir = args.data_dir #'./data'

print('Start Loading Data')

device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

if torch.cuda.is_available():
    torch.cuda.empty_cache()

epoch=args.saved_model_epoch 

if args.use_pos_encoder:
    coord_encoder = PositionalEncoding(sigma=args.sigma, m=args.m)
else:
    coord_encoder = None

m0_latent_encoder = torch.load(
    os.path.join(args.model_save_dir, 'm0_latent_encoder_epoch_{}.sav'.format(epoch)),
    map_location=device
)

m1_latent_encoder = torch.load(
    os.path.join(args.model_save_dir, 'm1_latent_encoder_epoch_{}.sav'.format(epoch)),
    map_location=device
)

m021_latent_encoder = torch.load(
    os.path.join(args.model_save_dir, 'm021_latent_encoder_epoch_{}.sav'.format(epoch)),
    map_location=device
)

m120_latent_encoder = torch.load(
    os.path.join(args.model_save_dir, 'm120_latent_encoder_epoch_{}.sav'.format(epoch)),
    map_location=device
)

if args.use_global_encoder:
    m0_global_encoder = torch.load(
        os.path.join(args.model_save_dir, 'm0_global_encoder_epoch_{}.sav'.format(epoch)),
        map_location=device
    )

    m1_global_encoder = torch.load(
        os.path.join(args.model_save_dir, 'm1_global_encoder_epoch_{}.sav'.format(epoch)),
        map_location=device
    )
else:
    m0_global_encoder = None
    m1_global_encoder = None

m0_feature_encoder = torch.load(
    os.path.join(args.model_save_dir, 'm0_feature_encoder_epoch_{}.sav'.format(epoch)),
    map_location=device
)

m1_feature_encoder = torch.load(
    os.path.join(args.model_save_dir, 'm1_feature_encoder_epoch_{}.sav'.format(epoch)),
    map_location=device
)

m0_decoder = torch.load(
    os.path.join(args.model_save_dir, 'm0_decoder_epoch_{}.sav'.format(epoch)),
    map_location=device
)

m1_decoder = torch.load(
    os.path.join(args.model_save_dir, 'm1_decoder_epoch_{}.sav'.format(epoch)),
    map_location=device
)

scale = args.sr_scale

print("Super Resolution Scale = {}".format(scale))

val_dataset_viz = LIIFVizDataset(
    m0_data_dir=os.path.join(data_dir, "wind-{}m/{}".format(args.height_0, args.file_prefix)),
    m1_data_dir=os.path.join(data_dir, "wind-{}m/{}".format(args.height_1, args.file_prefix)),
    file_prefix=args.file_prefix,
    up_scale=scale,
    low_resol=[args.h_LR, args.w_LR],
    max_val=50,
    train_frac=args.train_frac,
    train=False
)

val_dataloader_viz = DataLoader(dataset=val_dataset_viz, 
                                batch_size=1, 
                                num_workers=4, 
                                pin_memory=True, 
                                shuffle=True) #False) 

if args.use_global_encoder and args.use_pos_encoder:
    result_file_name = "result-GPEI-LIIF-scale-{}-height0-{}-height1-{}.npz".format(scale, args.height_0, args.height_1)
elif args.use_global_encoder and not args.use_pos_encoder:
    result_file_name = "result-GEI-LIIF-scale-{}-height0-{}-height1-{}.npz".format(scale, args.height_0, args.height_1)
elif not args.use_global_encoder and args.use_pos_encoder:
    result_file_name = "result-PEI-LIIF-wKAN-scale-{}-height0-{}-height1-{}.npz".format(scale, args.height_0, args.height_1)
else:
    result_file_name = "result-LIIF-scale-{}-height0-{}-height1-{}.npz".format(scale, args.height_0, args.height_1)

os.makedirs(args.result_save_dir, exist_ok=True)

# if not os.path.exists(os.path.join(args.result_save_dir, result_file_name)):
data = evaluate_model(
    val_dataloader_viz=val_dataloader_viz,
    coord_encoder=coord_encoder,
    m0_latent_encoder=m0_latent_encoder,
    m1_latent_encoder=m1_latent_encoder,
    m021_latent_encoder=m021_latent_encoder,
    m120_latent_encoder=m120_latent_encoder,
    m0_global_encoder=m0_global_encoder,
    m1_global_encoder=m1_global_encoder,
    m0_feature_encoder=m0_feature_encoder,
    m1_feature_encoder=m1_feature_encoder,
    m0_decoder=m0_decoder,
    m1_decoder=m1_decoder,
    device=device,
    use_global_encoder=args.use_global_encoder,
    use_pos_encoder=args.use_pos_encoder,
    sample_size=300,
    ch_HR=int(scale*args.h_LR),
    cw_HR=int(scale*args.w_LR),
    plot_image=False,
)    

np.savez(os.path.join(args.result_save_dir, result_file_name), **data)