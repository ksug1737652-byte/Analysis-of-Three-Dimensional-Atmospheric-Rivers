import warnings
warnings.filterwarnings('ignore')
# Import modules and packages
import cartopy.crs as ccrs
import cartopy.util as util
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import csv
from datetime import date
import datetime
from dateutil.relativedelta import relativedelta
import geopy.distance
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import matplotlib.path as mpath
import numpy as np
import operator
import os
import os.path
from scipy import ndimage
from shapely.geometry import Point
import skimage
from skimage.segmentation import find_boundaries
import xarray as xr
import pandas as pd

seasons = ['ALL']#,'DJF','MAM','JJA','SON']#,'01','02','03','04','05','06','07','08','09','10','11','12']


for period in [24,0,54][:]: # hours
    g = 9.80665
    #data_dir = '/media/takahashi/HDPH-UT/Kyoto-U/AR_jra55/AR_detection/AR_detection_Kennet/data/'
    #data_dir = '/Volumes/PromisePegasus/takahashi/JRA55/AR-2022-2023/'#,min_length,min_aspect,eqw_lat,plw_lat,dist_lat,distance_between)
    inputdata_dir = '/Volumes/Pegasus32 R6/takahashi/era5/'
    input_data_dir = '/Volumes/Pegasus32 R6/takahashi/era5/input_data/'
    mask_dir = '{}ar_out_era5_SH_20S/ar_ivt_data/'.format(inputdata_dir)
    data_dir = '{}ar_out_era5_SH_20S/'.format(inputdata_dir)

    data_dir_inv = '/Volumes/Pegasus32 R8/takahashi/era5/ar_out_3d_monthly_threshold/'#.format(inputdata_dir)
    mask_inv_dir = '/Volumes/Pegasus32 R8/takahashi/era5/ar_out_3d_monthly_threshold/ar_ivt_data/'#.format(inputdata_dir)

    if not os.path.exists('{}ar_frequency'.format(data_dir_inv)):#,Date.year,Date.month)):
        os.makedirs('{}ar_frequency'.format(data_dir_inv))#,Date.year,Date.month))
    if not os.path.exists('{}ar_frequency'.format(data_dir)):#,Date.year,Date.month)):
        os.makedirs('{}ar_frequency'.format(data_dir))#,Date.year,Date.month))
        
    dataout_dir = '{}ar_frequency/'.format(data_dir)#,Date.year,Date.month)
    dataout_dir_inv = '{}ar_frequency/'.format(data_dir_inv)#,Date.year,Date.month)

    z_file = '{}data/z_202301.nc'.format(inputdata_dir)
    z_data = xr.open_dataset(z_file)['z'].sel(latitude=slice(-20,-90)).values[0]/g

    data_dir = '{}data/'.format(inputdata_dir)
    sp_file = '{}z_202301.nc'.format(data_dir)
    dataset = xr.open_dataset(sp_file)
    sp = dataset['z'].sel(latitude=slice(-20,-90)).values[0]/9.8
    print(sp.shape)

    # years = [2021,1985,1986]#,1981,1982,1982,1983,1995,1996,1997,2005,2006,2007,2008,2009,2016,2017,2018,2019,2020][:1]       #np.arange(2023,2024,1)
    # years += [1979,1980,1988,1989,1990,1991,1992,1993,1994,1998,1999,2000,2001,2002,2003,2004,2010,2011,2012,2013,2014,2015,2022,2023]#np.arange(2023,2024,1)
    years = np.arange(1979,2024,1)#[1998,1999,2000,2001,2002,2019,2020,2018,2021]
    #[1998,1999,2015,2016,2017,2000]
    #[1990,1991,1992,1993,1994,1995,1996,1997,2007,2008,2009,2010,2011,2012,2013,2014]
    #1989,2022,2021]#1986,1987,2023,2022,1988]#1983,1984,1985]#np.arange(1979,2024,1)[::1]
    #1979,1980,1981,1982,1983,1984,1985,2003,2004,2005,2006]#np.arange(1979,2024,1)
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    season = 'Annual'
    timestep = 24
    timeweight = 1/timestep

    for season in seasons:
        print(season)
        if season == '01':
            months = np.array([1])
        elif season == '02':
            months = np.array([2])
        elif season == '03':
            months = np.array([3])
        elif season == '04':
            months = np.array([4])
        elif season == '05':
            months = np.array([5])
        elif season == '06':
            months = np.array([6])
        elif season == '07':
            months = np.array([7])
        elif season == '08':
            months = np.array([8])
        elif season == '09':
            months = np.array([9])
        elif season == '10':
            months = np.array([10])
        elif season == '11':
            months = np.array([11])
        elif season == '12':
            months = np.array([12])
        elif season == 'DJF':
            months = np.array([1,2,12])
        elif season == 'MAM':
            months = np.array([3,4,5])
        elif season == 'JJA':
            months = np.array([6,7,8])
        elif season == 'SON':
            months = np.array([9,10,11])
        if season == 'ALL':
            months = np.arange(1,13,1)

        time_length = 0
        ar_ivt_dataset = 0
        ar_dataset = 0
        total_dataset = 0
        sp_dataset = 0
        for year in years:
            print(year)
            ar_pr_dataset = []
            ar_pr_days_dataset = []
            m_list = []
            
            for month in months:

                if os.path.exists('{}ar_characteristics/pr{}-{}.nc'.format(data_dir_inv,period,year)):
                    continue

                if (year == 1979)& (month == 1):
                    sD = datetime.datetime(year,month,1,0)
                else: 
                    sD = datetime.datetime(year,month,1,0)-datetime.timedelta(hours=period)
                D_for = datetime.datetime(year,month,1,0)+datetime.timedelta(days=31)
                eD = datetime.datetime(D_for.year,D_for.month,1,0)-datetime.timedelta(hours=1)
                print(sD,eD)
                
                pr_file = '{}data/surface/pr/pr_{}{:0=2}.nc'.format(inputdata_dir,year,month)
                dataset = xr.open_dataset(pr_file)['tp'].sel(latitude=slice(-20,-90))
                pr = dataset.values*1000
                
                
                if not period == 0:
                    file = ['{}select_AR_mask-{}{:0=2}.nc'.format(mask_dir,sD.year,sD.month),
                            '{}select_AR_mask-{}{:0=2}.nc'.format(mask_dir,eD.year,eD.month)]
                    if (( year == 1979)& ( month == 1)):            
                        file = ['{}select_AR_mask-{}{:0=2}.nc'.format(mask_dir,sD.year,sD.month)]
                        print('True')
                    dataset = xr.open_mfdataset(file)['ar_mask'].sel(latitude=slice(-20,-90)).sel(time=slice('{:0=4}-{:0=2}-{:0=2}-{:0=2}'.format(sD.year,sD.month,sD.day,sD.hour),'{:0=4}-{:0=2}-{:0=2}-{:0=2}'.format(eD.year,eD.month,eD.day,eD.hour)))
                    ar_mask_dataset = dataset.values
                    time = dataset.shape[0]
                    lat = dataset.latitude
                    lon = dataset.longitude
                    
                    # 出力用コピー
                    ar2d_out = ar_mask_dataset.copy()*0

                    # 24時間フラグを付与
                    for t in range(time-1):
                        # tでTrueがあれば、t+1~t+24をTrueにする
                        if ar_mask_dataset[t].any():
                            if period + t >= time-1:
                                ar2d_out[t+1:] |= ar_mask_dataset[t]
                            else:                            
                                ar2d_out[t+1:t+(period+1)] |= ar_mask_dataset[t]
                            
                    

                    file = ['{}select_groundAR_mask-{}-{}.nc'.format(mask_inv_dir,sD.year,sD.month),
                            '{}select_groundAR_mask-{}-{}.nc'.format(mask_inv_dir,eD.year,eD.month)]
                    if (( year == 1979)& ( month == 1)):            
                        file = ['{}select_groundAR_mask-{}-{}.nc'.format(mask_inv_dir,sD.year,sD.month)]
                    dataset = xr.open_mfdataset(file)['ar_mask'].sel(latitude=slice(-20,-90)).sel(time=slice('{:0=4}-{:0=2}-{:0=2}-{:0=2}'.format(sD.year,sD.month,sD.day,sD.hour),'{:0=4}-{:0=2}-{:0=2}-{:0=2}'.format(eD.year,eD.month,eD.day,eD.hour)))
                    ar_mask_dataset = dataset.values
                    ar_mask_dataset[ar_mask_dataset==np.nan] = 0
                    time = dataset.shape[0]
                    lat = dataset.latitude
                    lon = dataset.longitude
                    ar_mask = ar_mask_dataset.max(axis=1)
                    
                    ar3d_out = ar_mask.copy()*0
                    time_length += time

                    # 24時間フラグを付与
                    for t in range(time-(period+1)):
                        # tでTrueがあれば、t+1~t+24をTrueにする
                        if ar_mask[t].any():
                            if period + t >= time-1:
                                ar3d_out[t+1:] |= ar_mask[t]
                            else:                            
                                ar3d_out[t+1:t+(period+1)] |= ar_mask[t]

                    if (not year == 1979)| (not month == 1):
                        ar3d_out  = ar3d_out[period:]     
                        ar2d_out  = ar2d_out[period:]     
                    else:
                        ar3d_out  = ar3d_out[:]
                        ar2d_out  = ar2d_out[:]
                        
                elif period == 54:
                    file = ['{}select_AR_mask-{}{:0=2}.nc'.format(mask_dir,sD.year,sD.month),
                            '{}select_AR_mask-{}{:0=2}.nc'.format(mask_dir,year,month),
                            '{}select_AR_mask-{}{:0=2}.nc'.format(mask_dir,eD.year,eD.month)]
                    if (( year == 1979)& ( month == 1)):            
                        file = ['{}select_AR_mask-{}{:0=2}.nc'.format(mask_dir,year,month),
                                '{}select_AR_mask-{}{:0=2}.nc'.format(mask_dir,sD.year,sD.month)]
                    dataset = xr.open_mfdataset(file)['ar_mask'].sel(latitude=slice(-20,-90)).sel(time=slice('{:0=4}-{:0=2}-{:0=2}-{:0=2}'.format(sD.year,sD.month,sD.day,sD.hour),'{:0=4}-{:0=2}-01-06'.format(eD.year,eD.month)))
                    ar_mask_dataset = dataset.values
                    time = dataset.shape[0]
                    lat = dataset.latitude
                    lon = dataset.longitude
                    
                    # 出力用コピー
                    ar2d_out = ar_mask_dataset.copy()*0

                    # 24時間フラグを付与
                    for t in range(time-1):
                        # tでTrueがあれば、t+1~t+24をTrueにする
                        if ar_mask_dataset[t].any():
                            if period + t >= time-1:
                                ar2d_out[t-6:] |= ar_mask_dataset[t]
                            else:                            
                                ar2d_out[t-6:t+(period+1)] |= ar_mask_dataset[t]
                            
                    

                    file = ['{}select_groundAR_mask-{}-{}.nc'.format(mask_inv_dir,sD.year,sD.month),
                            '{}select_groundAR_mask-{}-{}.nc'.format(mask_inv_dir,year,month),
                            '{}select_groundAR_mask-{}-{}.nc'.format(mask_inv_dir,eD.year,eD.month)]
                    if (( year == 1979)& ( month == 1)):            
                        file = ['{}select_groundAR_mask-{}-{}.nc'.format(mask_inv_dir,year,month),
                                '{}select_groundAR_mask-{}-{}.nc'.format(mask_inv_dir,sD.year,sD.month)]
                    dataset = xr.open_mfdataset(file)['ar_mask'].sel(latitude=slice(-20,-90)).sel(time=slice('{:0=4}-{:0=2}-{:0=2}-{:0=2}'.format(sD.year,sD.month,sD.day,sD.hour),'{:0=4}-{:0=2}-01-06'.format(eD.year,eD.month)))
                    ar_mask_dataset = dataset.values
                    ar_mask_dataset[ar_mask_dataset==np.nan] = 0
                    time = dataset.shape[0]
                    lat = dataset.latitude
                    lon = dataset.longitude
                    ar_mask = ar_mask_dataset.max(axis=1)
                    
                    ar3d_out = ar_mask.copy()*0
                    time_length += time

                    # 24時間フラグを付与
                    for t in range(time-(period+1)):
                        # tでTrueがあれば、t+1~t+24をTrueにする
                        if ar_mask[t].any():
                            if period + t >= time-1:
                                ar3d_out[t-6:] |= ar_mask[t]
                            else:                            
                                ar3d_out[t-6:t+(period+1)] |= ar_mask[t]

                    if (not year == 1979)| (not month == 1):
                        ar3d_out  = ar3d_out[period:-6]     
                        ar2d_out  = ar2d_out[period:-6]     
                    else:
                        ar3d_out  = ar3d_out[:-6]
                        ar2d_out  = ar2d_out[:-6]    


                else:
                    
                    file = '{}select_AR_mask-{}{:0=2}.nc'.format(mask_dir,sD.year,sD.month)
                    dataset = xr.open_dataset(file)['ar_mask'].sel(latitude=slice(-20,-90))
                    ar_mask_dataset = dataset.values
                    time = dataset.shape[0]
                    lat = dataset.latitude
                    lon = dataset.longitude
                    ar2d_out = ar_mask_dataset
                    

                    file = '{}select_groundAR_mask-{}-{}.nc'.format(mask_inv_dir,sD.year,sD.month)
                    dataset = xr.open_dataset(file)['ar_mask'].sel(latitude=slice(-20,-90))
                    ar_mask_dataset = dataset.values
                    ar_mask_dataset[ar_mask_dataset==np.nan] = 0
                    time = dataset.shape[0]
                    lat = dataset.latitude
                    lon = dataset.longitude
                    ar_mask = ar_mask_dataset.max(axis=1)
                    
                    ar3d_out = ar_mask


                ar_ivt_year = (ar3d_out*pr).sum(axis=0)
                ar_year = (ar2d_out*pr).sum(axis=0)
                # AR precipitatioin days
                ar3d_days = ar3d_out.sum(axis=0)
                ar2d_days = ar2d_out.sum(axis=0)
                total_out = ar2d_out
                total_out[:,:,:] = 1
                total_days = total_out.sum(axis=0)
                
                total_year = pr.sum(axis=0)
                
                Dataset = np.stack([total_year,ar_year,ar_ivt_year],axis=0)
                Dataset2 = np.stack([total_days,ar2d_days,ar3d_days],axis=0)

                Time = dataset.time
                m_list += [Time.values[0]]

                ar_pr_dataset += [ar_ivt_year]
                ar_pr_days_dataset += [ar3d_days]

                # ar_dataset += ar2d_days
                # ar_ivt_dataset += ar3d_days
                ar_dataset += ar_year
                ar_ivt_dataset += ar_ivt_year
                total_dataset += total_year

            print(ar_pr_dataset)
            m_axis_pr = np.stack(ar_pr_dataset,axis=0)
            m_axis_days = np.stack(ar_pr_days_dataset,axis=0)
            m_axis = np.stack(m_list,axis=0)

            out_AR_mask = xr.DataArray(
                m_axis_pr,
                name = 'tp',
                dims = ['time','latitude','longitude'],
                coords = (
                    m_axis,lat,lon,
                ),
                attrs = {
                    'long_name':'Precipitations 3D AR',
                    'units':'mm'
                },
            )
            outfile_ar_mask = '{}ar_characteristics/pr{}-{}.nc'.format(data_dir_inv,period,year)
            out_AR_mask.to_netcdf(outfile_ar_mask)
            
            out_AR_mask = xr.DataArray(
                m_axis_days,
                name = 'tp_days',
                dims = ['time','latitude','longitude'],
                coords = (
                    m_axis,lat,lon,
                ),
                attrs = {
                    'long_name':'Precipitation days 3D AR',
                    'units':'h'
                },
            )
            outfile_ar_mask = '{}ar_characteristics/pr_days{}-{}.nc'.format(data_dir_inv,period,year)
            out_AR_mask.to_netcdf(outfile_ar_mask)
            