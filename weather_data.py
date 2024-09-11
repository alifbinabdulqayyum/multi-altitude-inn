import h5pyd
import numpy as np
import dateutil

import timeit
from time import time

from multiprocessing import Pool

import os

import argparse

parser = argparse.ArgumentParser('Creating Weather Data')

parser.add_argument('--start-idx', type=int, default=0)
parser.add_argument('--skip-idx', type=int, default=40)
parser.add_argument('--height', type=int, choices=[10,60,160,200])

args = parser.parse_args()

# directory to save the data
def mkdir(path):
    os.makedirs(path, exist_ok=True)

data_dir = "data/wind-{}m".format(args.height)
ua_data_dir = os.path.join(data_dir, "ua")
va_data_dir = os.path.join(data_dir, "va")

temp_data_dir = "data/temperature-{}m".format(args.height)

pressure_data_dir = "data/pressure-{}m".format(args.height)

mkdir(ua_data_dir)
mkdir(va_data_dir)

# mkdir(temp_data_dir)

# mkdir(pressure_data_dir)

def process(idx_arr):
    for idx in idx_arr:
        # wind_speed_data_check(idx)
        # temperature_data_check(idx)
        # pressure_data_check(idx)
        wind_speed_data(idx)

def wind_speed_data_check(timestep):
    ua_filename = "ua_{}.npy".format(timestep)
    va_filename = "va_{}.npy".format(timestep)
    if os.path.exists(os.path.join(ua_data_dir, ua_filename)) and os.path.exists(os.path.join(va_data_dir, va_filename)):
        pass
    else:
        print("Timestep {} does not exist".format(timestep))

def temperature_data_check(timestep):
    temp_filename = "temp_{}.npy".format(timestep)
    if os.path.exists(os.path.join(temp_data_dir, temp_filename)):
        pass
    else:
        print("Timestep {} does not exist".format(timestep))
        
def pressure_data_check(timestep):
    pressure_filename = "pressure_{}.npy".format(timestep)
    if os.path.exists(os.path.join(pressure_data_dir, pressure_filename)):
        pass
    else:
        print("Timestep {} does not exist".format(timestep))

def wind_speed_data(timestep):
    start = time()
    
    ua_filename = "ua_{}.npy".format(timestep)
    va_filename = "va_{}.npy".format(timestep)

    if os.path.exists(os.path.join(ua_data_dir, ua_filename)) and os.path.exists(os.path.join(va_data_dir, va_filename)):
        pass
    else:
        # time instance data
        speed_HR = dset_speed[timestep,::,::]
        direction_HR = dset_dir[timestep,::,::]
        
        # crop region
        speed_HR = speed_HR[-1500:,-2500:-500]
        direction_HR = direction_HR[-1500:,-2500:-500]
        
        # direction
        ua_HR = np.multiply(speed_HR, np.cos(np.radians(direction_HR+np.pi/2)))
        va_HR = np.multiply(speed_HR, np.sin(np.radians(direction_HR+np.pi/2)))
        
        np.save(os.path.join(ua_data_dir, ua_filename), ua_HR)
        np.save(os.path.join(va_data_dir, va_filename), va_HR)

    stop = time()
    print("Timestep {} time {}".format(timestep, stop-start))

# def temperature_data(timestep):
#     start = time()

#     temp_filename = "temp_{}.npy".format(timestep)

#     if os.path.exists(os.path.join(temp_data_dir, temp_filename)):
#         pass
#     else:
#         # time instance data
#         temp_HR = dset_temp[timestep,::,::]
        
#         # crop region
#         temp_HR = temp_HR[-1500:,-2500:-500]

#         np.save(os.path.join(temp_data_dir, temp_filename), temp_HR)

#     stop = time()
#     print("Timestep {} time {}".format(timestep, stop-start))
    
# def pressure_data(timestep):
#     start = time()

#     pressure_filename = "pressure_{}.npy".format(timestep)

#     if os.path.exists(os.path.join(pressure_data_dir, pressure_filename)):
#         pass
#     else:
#         # time instance data
#         pressure_HR = dset_pressure[timestep,::,::]
        
#         # crop region
#         pressure_HR = pressure_HR[-1500:,-2500:-500]

#         np.save(os.path.join(pressure_data_dir, pressure_filename), pressure_HR)

#     stop = time()
#     print("Timestep {} time {}".format(timestep, stop-start))

ncpu = 1

# Open the wind data "file"
# server endpoint, username, password is found via a config file
f = h5pyd.File("/nrel/wtk-us.h5", 'r', bucket="nrel-pds-hsds")  

dset_speed = f["windspeed_{}m".format(args.height)]
dset_dir = f["winddirection_{}m".format(args.height)]
# dset_temp = f["temperature_{}m".format(args.height)]
# dset_pressure = f["pressure_{}m".format(args.height)]

start_idx = args.start_idx
skip_idx = args.skip_idx

idx_arr = np.arange(start_idx, start_idx+1000, 40)

batch_size = int(np.ceil(len(idx_arr) / ncpu))

batches = [idx_arr[i : i + batch_size] for i in range(0, len(idx_arr), batch_size)]

pool = Pool(ncpu)
pool.map(process, batches)
