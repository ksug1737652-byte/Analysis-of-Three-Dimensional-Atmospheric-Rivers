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

seasons = ['ALL']#'01','02','03','04','05','06','07','08','09','10','11','12']#'ALL','DJF','MAM','JJA','SON']#,

se = [0,3,4,2][1:2]
g = 9.80665
#data_dir = '/media/takahashi/HDPH-UT/Kyoto-U/AR_jra55/AR_detection/AR_detection_Kennet/data/'
#data_dir = '/Volumes/PromisePegasus/takahashi/JRA55/AR-2022-2023/'#,min_length,min_aspect,eqw_lat,plw_lat,dist_lat,distance_between)

for s in se:
    inputdata_dir = '/Volumes/Pegasus32 R6/takahashi/era5/'
    input_data_dir = '/Volumes/Pegasus32 R6/takahashi/era5/input_data/'
    mask_dir = '{}ar_out_era5_SH_20S/ar_ivt_data/'.format(inputdata_dir)
    mask_inv_dir = '{}ar_out_3d_{:0=2}/mask_data/'.format(inputdata_dir,s)

    data_dir = '{}ar_out_era5_SH_20S/'.format(inputdata_dir)
    data_dir_inv = '{}ar_out_3d_{:0=2}/'.format(inputdata_dir,s)
    if s == 3:
        data_dir_inv = '/Volumes/Pegasus32 R8/takahashi/era5/ar_out_3d_monthly_threshold/'#.format(inputdata_dir)
        mask_inv_dir = '/Volumes/Pegasus32 R8/takahashi/era5/ar_out_3d_monthly_threshold/mask_data/'#.format(inputdata_dir)

    elif s == 0:
        data_dir_inv = '{}ar_out_era5_SH_20S/'.format(inputdata_dir)
        mask_inv_dir = '{}ar_out_era5_SH_20S/ar_ivt_data/'.format(inputdata_dir)

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
    years = np.arange(1979,2024,1)[::1]
    #[2000,2001,2002,2019,2020,2018]
    #[1998,1999,2015,2016,2017,2000]
    #[1990,1991,1992,1993,1994,1995,1996,1997,2007,2008,2009,2010,2011,2012,2013,2014]
    #1989,2022,2021]#1986,1987,2023,2022,1988]#1983,1984,1985]#
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    season = 'Annual'
    timestep = 24
    timeweight = 1/timestep

    # n_ivt_mean = np.load('calc_files/climatology_n_ivt.npy')
    # e_ivt_mean = np.load('calc_files/climatology_e_ivt.npy')

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


        ar_eq_dataset = 0
        ar_eq_neg_dataset = 0
        ar_eq_pos_dataset = 0
        ar_ivt_dataset = 0
        ar_dataset = 0
        sp_dataset = 0
        for year in years:
            print(year)
            ivt_m = 0
            ar_ivt_year = []
            m_list = []
            ar_year = []
            for month in months:
                print(year, month)
                
                # ivt_m += np.power((np.power(n_ivt_mean[month-1],2)+np.power(e_ivt_mean[month-1],2)),0.5)/len(months)
                # file = '{}select_AR_mask-{}-{}.nc'.format(mask_eq_dir,year,month)
                # dataset = xr.open_dataset(file)
                # ar_mask_dataset = dataset['ar_mask'].sel(latitude=slice(-20,-90))
                # ar_mask = ar_mask_dataset.sum(axis=0).values
                # ar_eq_dataset += ar_mask/24
                
                # file = '{}select_AR_mask-{}-{}.nc'.format(mask_eq_neg_dir,year,month)
                # dataset = xr.open_dataset(file)
                # ar_mask_dataset = dataset['ar_mask'].sel(latitude=slice(-20,-90))
                # ar_mask = ar_mask_dataset.sum(axis=0).values
                # ar_eq_neg_dataset += ar_mask/24
                
                # file = '{}select_AR_mask-{}-{}.nc'.format(mask_eq_pos_dir,year,month)
                # dataset = xr.open_dataset(file)
                # ar_mask_dataset = dataset['ar_mask'].sel(latitude=slice(-20,-90))
                # ar_mask = ar_mask_dataset.sum(axis=0).values
                # ar_eq_pos_dataset += ar_mask/24
                
                # file = '{}select_AR_mask-{}{:0=2}.nc'.format(mask_dir,year,month)
                # dataset = xr.open_dataset(file)
                # ar_mask_dataset = dataset['ar_mask'].sel(latitude=slice(-20,-90))
                # time = ar_mask_dataset.shape[0]
                # lat = ar_mask_dataset.latitude
                # lon = ar_mask_dataset.longitude
                # ar_mask = ar_mask_dataset.sum(axis=0).values
                # # print(ar_mask.shape)
                
                if (s == 2)|(s == 4):
                    file = '{}select_AR_mask_num-{}-{}.nc'.format(mask_inv_dir,year,month)
                    dataset = xr.open_dataset(file)
                    ar_mask_dataset = dataset['ar_mask'].sel(latitude=slice(-20,-90))
                    ar_mask_dataset.values[ar_mask_dataset.values==np.nan] = 0
                    # ar_mask = np.where(ar_mask_dataset.values>0,1,0).max(axis=1).sum(axis=0)
                    # ar_year += ar_mask/24
                    time = ar_mask_dataset.time
                    lat = ar_mask_dataset.latitude
                    lon = ar_mask_dataset.longitude
                    pres = ar_mask_dataset.level
                    pres_data = (ar_mask_dataset.values*0+1)*pres.values[None,:,None,None]
                    
                    sp_file = '{}data/surface/sp/sp_{}{:0=2}.nc'.format(inputdata_dir,year,month)
                    dataset = xr.open_dataset(sp_file)
                    sp = dataset['sp'].sel(latitude=slice(-20,-90)).values/100
                    sp_data = (ar_mask_dataset.values*0+1)*sp[:,None,:,:]-100
                    ar_test = ar_mask_dataset.values.copy()
                    ar_test[pres_data<sp_data] = 0
                    ob_list = [np.unique(ar_test[i])[~(np.unique(ar_test[i])==0)] for i in range(ar_test.shape[0])]
                    # print(ob_list)

                    masked = np.empty_like(ar_test)

                    for t in range(ar_test.shape[0]):  # time方向にループ
                        # print(t)
                        valid_values = ob_list[t]
                        masked[t] = np.where(np.isin(ar_mask_dataset.values[t], valid_values), ar_mask_dataset.values[t], 0)
                    ar_mask_data = np.where(masked>0,1,0)
                    
                    # if s == 3:
                    #     out_AR_mask = xr.DataArray(
                    #         ar_mask_data,
                    #         name = 'ar_mask',
                    #         dims = ['time','level','latitude','longitude'],
                    #         coords = (
                    #             time,pres,lat,lon,
                    #         ),
                    #         attrs = {
                    #             'long_name':'ARs Mask Data',
                    #             'units':'#'
                    #         },
                    #     )
                    #     outfile_ar_mask = '{}ar_ivt_data/select_groundAR_mask-{}-{}.nc'.format(data_dir_inv,year,month)
                    #     out_AR_mask.to_netcdf(outfile_ar_mask)


                    ar_mask_dataset.values = ar_mask_data
                    ar_year += [ar_mask_data.max(axis=1).sum(axis=0)/24]
                elif s == 3:
                    sp_file = '{}data/surface/sp/sp_{}{:0=2}.nc'.format(inputdata_dir,year,month)
                    dataset = xr.open_dataset(sp_file)
                    sp = dataset['sp'].sel(latitude=slice(-20,-90)).values/100

                    file = '{}ar_ivt_data/select_groundAR_mask-{}-{}.nc'.format(data_dir_inv,year,month)
                    dataset = xr.open_dataset(file)
                    ar_mask_dataset = dataset['ar_mask'].sel(latitude=slice(-20,-90))
                    ar_year += [ar_mask_dataset.max(axis=1).sum(axis=0).values/24]
                    
                    dir = '{}pressure/q/q_{}{:0=2}.nc'.format(data_dir,year,month)
                    q = xr.open_dataset(dir)['q'].sel(level=slice(300,1000)).sel(latitude=slice(-20,-90)).values#70-120, 20-50

                    iwv_dataset = '{}IWV-{}-{}.nc'.format(input_data_dir,year,month)
                    IWV = xr.open_dataset(iwv_dataset)['IWV'].sel(latitude=slice(-20,-90)).values
    

                elif s == 0:
                    sp_file = '{}data/surface/sp/sp_{}{:0=2}.nc'.format(inputdata_dir,year,month)
                    dataset = xr.open_dataset(sp_file)
                    sp = dataset['sp'].sel(latitude=slice(-20,-90)).values/100

                    file = '{}select_AR_mask-{}{:0=2}.nc'.format(mask_inv_dir,year,month)
                    dataset = xr.open_dataset(file)
                    ar_mask_dataset = dataset['ar_mask'].sel(latitude=slice(-20,-90))
                    ar_year += [ar_mask_dataset.sum(axis=0).values/24]
                    Time = ar_mask_dataset.time
                    m_list += [Time.values[0]]
                    lat = ar_mask_dataset.latitude
                    lon = ar_mask_dataset.longitude
                    continue
                ar_mask_dataset.values[ar_mask_dataset.values==np.nan] = 0
                time = ar_mask_dataset.shape[0]
                Time = ar_mask_dataset.time
                lat = ar_mask_dataset.latitude
                lon = ar_mask_dataset.longitude
                prec_v = ar_mask_dataset.level.values
                sp_data = (ar_mask_dataset.values*0+1)*sp[:,None,:,:]
                pres_data = ((ar_mask_dataset.values*0+1)*prec_v[None,:,None,None]).astype(float)
                pres_data[pres_data>sp_data] = np.nan

                prec = np.insert(prec_v,0,prec_v[0])
                prec = np.insert(prec,-1,prec[-1])
                delta_p = (np.roll(prec,-1,axis=0)- np.roll(prec,1,axis=0))/2
                delta_p = delta_p[1:-1]
                # print('from here')
                ar_mask = (ar_mask_dataset.values*delta_p[None,:,None,None]*q*100/g).sum(axis=1)/IWV
                print(ar_mask.dtype)
                print(ar_mask_dataset.sum(axis=0).values.max())
                # print('to here')
                # ar_mask_mean_p = ar_mask_dataset.values*prec_v[None,:,None,None]
                # ar_mask_mean_p[ar_mask_mean_p==0] = np.nan
                # ar_mask_mean_p = np.nanmean(ar_mask_mean_p,axis=1)
                # ar_mask_mean_p = np.nanmean(ar_mask_mean_p,axis=0)
                # ar_mask_mean_p[ar_mask_mean_p==np.nan] = 0
                
                # print(ar_mask.shape)
                ar_ivt_year += [(ar_mask).sum(axis=0)]
                m_list += [Time.values[0]]
                # ar_year += ar_mask_data.max(axis=1).sum(axis=0)/24/years.shape[0]
                
                # n_ivt_dataset = '{}n-ivt-{}-{}.nc'.format(input_data_dir,year,month)
                # e_ivt_dataset = '{}e-ivt-{}-{}.nc'.format(input_data_dir,year,month)

                # n_ivt_data = xr.open_dataset(n_ivt_dataset)['VIVT'].sel(latitude=slice(-20,-90))
                # e_ivt_data = xr.open_dataset(e_ivt_dataset)['UIVT'].sel(latitude=slice(-20,-90))

                # ivt = np.power(np.power(n_ivt_data, 2) + np.power(e_ivt_data, 2), 1/2)
                # ar_ivt_dataset += (ar_mask*ivt.values).mean(axis=0)
            if not s == 0:
                ar_ivt_year = np.stack(ar_ivt_year,axis=0)
                ar_ivt_dataset += ar_ivt_year.sum(axis=0)
            m_axis = np.stack(m_list,axis=0)
            ar_year = np.stack(ar_year,axis=0)
            ar_dataset += ar_year.sum(axis=0)
            # out = xr.DataArray(
            #     ar_year,
            #     name = 'ar',
            #     dims = ['latitude','longitude'],
            #     coords = (
            #         lat,lon,
            #     ),
            #     attrs = {
            #         'long_name':'AR_percent_for_{}_{}'.format(season,year),
            #         'units':'days year**-1'
            #     },
            # )
            # outfile = '{}AR_frequency_weighted-thickness_{}_{}.nc'.format(dataout_dir,season,year)
            # out.to_netcdf(outfile)
            if not s == 0:
                out = xr.DataArray(
                    ar_ivt_year,
                    name = 'ar',
                    dims = ['time','latitude','longitude'],
                    coords = (
                        m_axis,lat,lon,
                    ),
                    attrs = {
                        'long_name':'AR_percent_for_water_vapor_weighted_{}_{}'.format(season,year),
                        'units':'#'
                    },
                )
                outfile = '{}AR_ground_frequency_iwv_weighted_{}_{}.nc'.format(dataout_dir_inv,season,year)
                out.to_netcdf(outfile)

            out = xr.DataArray(
                ar_year,
                name = 'ar',
                dims = ['time','latitude','longitude'],
                coords = (
                    m_axis,lat,lon,
                ),
                attrs = {
                    'long_name':'AR_percent_for_{}_{}'.format(season,year),
                    'units':'#'
                },
            )
            outfile = '{}AR_ground_frequency_{}_{}.nc'.format(dataout_dir_inv,season,year)
            out.to_netcdf(outfile)

            # out = xr.DataArray(
            #     ar_year,
            #     name = 'ar',
            #     dims = ['latitude','longitude'],
            #     coords = (
            #         lat,lon,
            #     ),
            #     attrs = {
            #         'long_name':'AR_percent_for_{}_{}'.format(season,year),
            #         'units':'days year**-1'
            #     },
            # )
            # outfile = '{}AR_ground_frequency_{}_{}.nc'.format(dataout_dir_inv,season,year)
            # out.to_netcdf(outfile)
