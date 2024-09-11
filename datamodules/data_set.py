import torch
import math
import random
import numpy as np
# import pytorch_lightning as pl
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.transforms.functional import InterpolationMode
# from pytorch_lightning.trainer.supporters import CombinedLoader
from typing import Optional
# from hydra.utils import log
import glob
import os
import pywt
from tqdm import tqdm

def load_data(
    m0_data_dir:str, 
    m1_data_dir:str,
    m0_file_prefix:str,
    m1_file_prefix:str,
    total_size:int=None,
    visualization:bool=False,
    val_size:int=250
):
    """
    Load images and return the image stack.
    Args:
        m0_data_dir (string): first modality dataset file path
        m1_data_dir (string): second modality dataset file path
        m0_file_prefix (string): first modality dataset filename prefix
        m1_file_prefix (string): second modality dataset filename prefix
    Returns:
        (mm_m0_dat, mm_m1_dat) (tensor,tensor): (num * H * W, num * H * W)
        m0_only_dat (tensor): num * H * W
        m1_only_dat (tensor): num * H * W
    """  

    m0_idx_list = [
        idx.replace(m0_data_dir+ '/'+m0_file_prefix+'_', "").replace('.npy', "") 
        for idx in glob.glob(m0_data_dir+'/*.npy')
    ]
    m0_idx_list = list(map(int, m0_idx_list))
    m0_idx_list.sort()

    m1_idx_list = [
        idx.replace(m1_data_dir+ '/'+m1_file_prefix+'_', "").replace('.npy', "") 
        for idx in glob.glob(m1_data_dir+'/*.npy')
    ]
    m1_idx_list = list(map(int, m1_idx_list))
    m1_idx_list.sort()

    if m0_idx_list == m1_idx_list:
        if total_size is not None:
            m0_idx_list = m0_idx_list[:total_size]
            m1_idx_list = m1_idx_list[:total_size]
        
        # To reduce the time to load the data at visualization
        if visualization:
            total_datapoints = len(m0_idx_list)
            m0_idx_list = m0_idx_list[total_datapoints-val_size:]
            m1_idx_list = m1_idx_list[total_datapoints-val_size:]
        # =================== #
        m0_dat, m1_dat = [], []
        with tqdm(total=len(m0_idx_list)) as pbar:
            for idx, timestep in enumerate(m0_idx_list):
                m0_tmp = np.load(os.path.join(m0_data_dir, m0_file_prefix+'_{}.npy'.format(timestep)))
                m1_tmp = np.load(os.path.join(m1_data_dir, m1_file_prefix+'_{}.npy'.format(timestep)))

                # m0_tmp = np.expand_dims(m0_tmp, axis=0)
                # m1_tmp = np.expand_dims(m1_tmp, axis=0)

                # if idx == 0:
                #     m0_dat = m0_tmp
                #     m1_dat = m1_tmp
                # else:
                #     m0_dat = np.vstack((m0_dat, m0_tmp))
                #     m1_dat = np.vstack((m1_dat, m1_tmp))

                m0_dat.append(m0_tmp)
                m1_dat.append(m1_tmp)
                pbar.update(1)
        m0_dat = np.stack(m0_dat, axis=0)
        m1_dat = np.stack(m1_dat, axis=0)
    else:
        m0_dat, m1_dat = None, None

    return ((torch.from_numpy(m0_dat), torch.from_numpy(m1_dat)), (m0_idx_list, m1_idx_list))

def get_coords(shape, ranges=None, flatten=True):
    """ 
    Make coordinates at grid centers.
    Args:
        shape   (list): image size [H, W]
        ranges  (list): grid boundaries [[left, right], [down, up]] 
        flatten (bool): True
    Returns:
        coords  (torch.tensor): H * W, 2
    """
    # determine the center of each grid
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -0.99995, 1 #-1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    
    # make mesh
    coords = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        coords = coords.view(-1, coords.shape[-1])
    return coords

class LIIFDataset(Dataset):
    def __init__(self, 
            m0_data_dir: str,
            m1_data_dir: str,
            file_prefix: str,
            liif_scales: list = [1, 5],
            low_resol: list = [75, 100],
            sampling_rate: float = 1.0,
            query_points: int = 256,
            max_val: float = 50,
            train_frac: float = 0.8,
            total_size: int = None,
            train: bool = True
    ):
        self.liif_scales = liif_scales
        self.low_resol = low_resol
        self.sampling_rate = sampling_rate
        self.query_points = query_points
        self.max_val = max_val

        self.m0_data_dir = m0_data_dir
        self.m1_data_dir = m1_data_dir

        self.m0_file_prefix, self.m1_file_prefix = file_prefix, file_prefix

        m0_idx_list = [
            idx.replace(self.m0_data_dir+ '/'+self.m0_file_prefix+'_', "").replace('.npy', "") 
            for idx in glob.glob(self.m0_data_dir+'/*.npy')
        ]
        m0_idx_list = list(map(int, m0_idx_list))
        # self.m0_idx_list.sort()

        m1_idx_list = [
            idx.replace(self.m1_data_dir+ '/'+self.m1_file_prefix+'_', "").replace('.npy', "") 
            for idx in glob.glob(self.m1_data_dir+'/*.npy')
        ]
        m1_idx_list = list(map(int, m1_idx_list))
        # self.m1_idx_list.sort()

        self.idx_list = list(set(m0_idx_list) & set(m1_idx_list))
        self.idx_list.sort()

        if total_size is not None:
            total_size = min(len(self.idx_list, total_size))
        else:
            total_size = len(self.idx_list)

        train_size = int(total_size * train_frac)
        val_size = total_size - train_size

        if train:
            self.idx_list = self.idx_list[:train_size]
        else:
            self.idx_list = self.idx_list[train_size:train_size+val_size]

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        # high resolution sample: H, W -> 1, H, W

        timestep = self.idx_list[idx]

        m0_tmp = np.load(os.path.join(self.m0_data_dir, self.m0_file_prefix+'_{}.npy'.format(timestep)))
        m1_tmp = np.load(os.path.join(self.m1_data_dir, self.m1_file_prefix+'_{}.npy'.format(timestep)))

        # high resolution sample: H, W -> 1, H, W
        m0_img_HR = torch.unsqueeze(torch.FloatTensor(m0_tmp), 0) / self.max_val
        
        # high resolution sample: H, W -> 1, H, W
        m1_img_HR = torch.unsqueeze(torch.FloatTensor(m1_tmp), 0) / self.max_val

        # paired sample 
        up_scale = random.uniform(self.liif_scales[0], self.liif_scales[1])
        h_LR, w_LR = self.low_resol
        h_HR, w_HR = round(h_LR * up_scale), round(w_LR * up_scale)
        
        # random crop
        # img_HR: 1, h_HR, w_HR
        # img_LR: 1, h_LR, w_LR
        m0_img_HR, m1_img_HR = transforms.RandomCrop((h_HR, w_HR))(torch.vstack((m0_img_HR,m1_img_HR)))
        
        m0_img_HR = torch.unsqueeze(m0_img_HR, 0)
        m1_img_HR = torch.unsqueeze(m1_img_HR, 0)
        
        m0_img_LR = transforms.Resize((h_LR, w_LR), interpolation=InterpolationMode.BICUBIC, antialias=None)(m0_img_HR)
        m1_img_LR = transforms.Resize((h_LR, w_LR), interpolation=InterpolationMode.BICUBIC, antialias=None)(m1_img_HR)
        
        # get HR coordinates: h_HR * w_HR, 2
        grid_HR = get_coords([h_HR, w_HR])
        
        # subset of grid_HR and img_HR 
        # grid_HR: n_query_pts, 2 
        # img_HR:  n_query_pts, 1
        n_query_pts = self.query_points
        query_pts = np.random.choice(len(grid_HR), n_query_pts, replace=False)
        grid_HR = grid_HR[query_pts]

        if not m0_img_HR.is_contiguous():
            m0_img_HR = m0_img_HR.contiguous()
        m0_img_HR = m0_img_HR.view(1, -1).permute(1, 0)
        m0_img_HR = m0_img_HR[query_pts]
        
        if not m1_img_HR.is_contiguous():
            m1_img_HR = m1_img_HR.contiguous()
        m1_img_HR = m1_img_HR.view(1, -1).permute(1, 0)
        m1_img_HR = m1_img_HR[query_pts]

        # get cell
        cell = torch.ones(2).int()
        cell[0] = h_HR
        cell[1] = w_HR

        return {
                "m0_img_LR": m0_img_LR,
                "m1_img_LR": m1_img_LR,
                "grid_HR": grid_HR,
                "m0_img_HR": m0_img_HR,
                "m1_img_HR": m1_img_HR,
                "cell": cell
                }


class LIIFVizDataset(Dataset):
    def __init__(self, 
        m0_data_dir: str, 
        m1_data_dir: str,
        file_prefix: str,
        up_scale: float = 1,
        low_resol: list = [75, 100],
        max_val: float = 50.0,
        total_size: int = None,
        train_frac: float = 0.8,
        train: bool = False,
    ):
      
        self.m0_data_dir = m0_data_dir
        self.m1_data_dir = m1_data_dir

        self.m0_file_prefix, self.m1_file_prefix = file_prefix, file_prefix

        m0_idx_list = [
            idx.replace(self.m0_data_dir+ '/'+self.m0_file_prefix+'_', "").replace('.npy', "") 
            for idx in glob.glob(self.m0_data_dir+'/*.npy')
        ]
        m0_idx_list = list(map(int, m0_idx_list))
        # self.m0_idx_list.sort()

        m1_idx_list = [
            idx.replace(self.m1_data_dir+ '/'+self.m1_file_prefix+'_', "").replace('.npy', "") 
            for idx in glob.glob(self.m1_data_dir+'/*.npy')
        ]
        m1_idx_list = list(map(int, m1_idx_list))
        # self.m1_idx_list.sort()

        self.idx_list = list(set(m0_idx_list) & set(m1_idx_list))
        self.idx_list.sort()

        if total_size is not None:
            total_size = min(len(self.idx_list, total_size))
        else:
            total_size = len(self.idx_list)

        train_size = int(total_size * train_frac)
        val_size = total_size - train_size

        if train:
            self.idx_list = self.idx_list[:train_size]
        else:
            self.idx_list = self.idx_list[train_size:train_size+val_size]
        self.up_scale = up_scale
        self.low_resol = low_resol
        self.max_val = max_val

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        # # high resolution sample: H, W -> 1, H, W
        # m0_img = self.m0_dataset[idx] / self.max_val
        # m0_img = torch.unsqueeze(m0_img, 0)
        
        # # high resolution sample: H, W -> 1, H, W
        # m1_img = self.m1_dataset[idx] / self.max_val
        # m1_img = torch.unsqueeze(m1_img, 0)

        # paired sample
        h_LR, w_LR = self.low_resol
        h_HR, w_HR = round(h_LR * self.up_scale), round(w_LR * self.up_scale)

        # paired sample
        # img_HR: 1, h_HR, w_HR
        # img_LR: 1, h_LR, w_LR
        # m0_img_HR, m1_img_HR = transforms.RandomCrop((h_HR, w_HR))(torch.vstack((m0_img_HR,m1_img_HR)))
        
        # m0_img_HR = torch.unsqueeze(m0_img_HR, 0)
        # m1_img_HR = torch.unsqueeze(m1_img_HR, 0)

        timestep = self.idx_list[idx]

        m0_tmp = np.load(os.path.join(self.m0_data_dir, self.m0_file_prefix+'_{}.npy'.format(timestep)))
        m1_tmp = np.load(os.path.join(self.m1_data_dir, self.m1_file_prefix+'_{}.npy'.format(timestep)))

        # high resolution sample: H, W -> 1, H, W
        m0_img = torch.unsqueeze(torch.FloatTensor(m0_tmp), 0) / self.max_val
        
        # high resolution sample: H, W -> 1, H, W
        m1_img = torch.unsqueeze(torch.FloatTensor(m1_tmp), 0) / self.max_val
        
        m0_img_HR = transforms.Resize((h_HR, w_HR), interpolation=InterpolationMode.BICUBIC, antialias=None)(m0_img)
        m1_img_HR = transforms.Resize((h_HR, w_HR), interpolation=InterpolationMode.BICUBIC, antialias=None)(m1_img)
        
        m0_img_LR = transforms.Resize((h_LR, w_LR), interpolation=InterpolationMode.BICUBIC, antialias=None)(m0_img)
        m1_img_LR = transforms.Resize((h_LR, w_LR), interpolation=InterpolationMode.BICUBIC, antialias=None)(m1_img)
        
        # grid_HR: h_HR * w_HR, 2 
        # img_HR:  h_HR * w_HR, 1
        grid_HR = get_coords([h_HR, w_HR])

        # if not m0_img_HR.is_contiguous():
        #     m0_img_HR = m0_img_HR.contiguous()
        # m0_img_HR = m0_img_HR.view(1, -1).permute(1, 0)
        
        # if not m1_img_HR.is_contiguous():
        #     m1_img_HR = m1_img_HR.contiguous()
        # m1_img_HR = m1_img_HR.view(1, -1).permute(1, 0)

        # get cell
        cell = torch.ones(2).int()
        cell[0] = h_HR
        cell[1] = w_HR

        return {
                "m0_img_LR": m0_img_LR,
                "m1_img_LR": m1_img_LR,
                "grid_HR": grid_HR,
                "m0_img_HR": m0_img_HR,
                "m1_img_HR": m1_img_HR,
                "cell": cell
                }

