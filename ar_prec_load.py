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
period = 0 # hours
g = 9.80665
#data_dir = '/media/takahashi/HDPH-UT/Kyoto-U/AR_jra55/AR_detection/AR_detection_Kennet/data/'
#data_dir = '/Volumes/PromisePegasus/takahashi/JRA55/AR-2022-2023/'#,min_length,min_aspect,eqw_lat,plw_lat,dist_lat,distance_between)
inputdata_dir = '/Volumes/Pegasus32 R6/takahashi/era5/'
input_data_dir = '/Volumes/Pegasus32 R6/takahashi/era5/input_data/'
mask_dir = '{}ar_out_era5_SH_20S/ar_ivt_data/'.format(inputdata_dir)
mask_inv_dir = '{}ar_out_3d/ar_ivt_data/'.format(inputdata_dir)

data_dir = '{}ar_out_era5_SH_20S/'.format(inputdata_dir)
data_dir_inv = '/Volumes/Pegasus32 R8/takahashi/era5/ar_out_3d_monthly_threshold/'.format(inputdata_dir)

if not os.path.exists('{}ar_frequency'.format(data_dir_inv)):#,Date.year,Date.month)):
    os.makedirs('{}ar_frequency'.format(data_dir_inv))#,Date.year,Date.month))
if not os.path.exists('{}ar_frequency'.format(data_dir)):#,Date.year,Date.month)):
    os.makedirs('{}ar_frequency'.format(data_dir))#,Date.year,Date.month))
    
dataout_dir = '{}ar_frequency/'.format(data_dir)#,Date.year,Date.month)
dataout_dir_inv = '{}ar_frequency/'.format(data_dir_inv)#,Date.year,Date.month)

z_file = '{}data/z_202301.nc'.format(inputdata_dir)
z_data = xr.open_dataset(z_file)['z'].sel(latitude=slice(-20,-89)).values[0]/g

data_dir = '{}data/'.format(inputdata_dir)
sp_file = '{}z_202301.nc'.format(data_dir)
dataset = xr.open_dataset(sp_file)
sp = dataset['z'].sel(latitude=slice(-20,-90)).values[0]/9.8
print(sp.shape)

# years = [2021,1985,1986]#,1981,1982,1982,1983,1995,1996,1997,2005,2006,2007,2008,2009,2016,2017,2018,2019,2020][:1]      
# years += [1979,1980,1988,1989,1990,1991,1992,1993,1994,1998,1999,2000,2001,2002,2003,2004,2010,2011,2012,2013,2014,2015,2022,2023]#np.arange(2023,2024,1)
years =  np.arange(1979,2024,1)
#[1979,1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023]#np.arange(1979,2024,1)[::1]
months = [1,2,3,4,5,6,7,8,9,10,11,12]
season = 'Annual'
timestep = 24
timeweight = 1/timestep

n_ivt_mean = np.load('calc_files/climatology_n_ivt.npy')
e_ivt_mean = np.load('calc_files/climatology_e_ivt.npy')

for t in ['24','0','54','48'][:-1]:
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


        ar_ivt_dataset = 0
        ar_dataset = 0
        total_dataset = 0
        sp_dataset = 0
        for year in years:
            print(year)
            for month in months:
    
                #    'long_name':'Precipitations [0]Total, [1]2D AR, [2]3D AR',
                pr_file = '{}ar_characteristics/pr{}-{}-{}.nc'.format(data_dir_inv,t,year,month)
                dataset = xr.open_dataset(pr_file)['tp'].sel(latitude=slice(-20,-89))
                lat = dataset.latitude
                lon = dataset.longitude
                total_pr = dataset.values[0]
                ar2d_pr = dataset.values[1]
                ar3d_pr = dataset.values[2]
                
                ar_dataset += ar2d_pr
                ar_ivt_dataset += ar3d_pr
                total_dataset += total_pr

        ar_ivt_dataset[ar_ivt_dataset==0] = np.nan
        ar_dataset[ar_dataset==0] = np.nan
        # plot = 100*ar_ivt_dataset/total_dataset
        # plot = 100*ar_dataset/total_dataset
        
        # plot = ar_ivt_dataset/years.shape[0]
        # plot = ar_dataset/years.shape[0]
        plot = total_dataset/years.shape[0]
        
        dataarray = xr.DataArray(
            plot,
            name = 'ar_p',
            dims = ['latitude','longitude'],
            coords = (
                lat,lon,
            ),
        )

        print(dataarray.interp(latitude=-77.3,longitude=39.4))
        ar_inv_cyc, lon_cyc_v = util.add_cyclic_point(plot,coord=lon.values)
        
        center, radius = [0.5,0.5],0.5
        theta = np.linspace(0,2*np.pi,100)
        verts = np.vstack([np.sin(theta),np.cos(theta)]).T
        circle = mpath.Path(verts*radius+center)

        fig = plt.figure(figsize=[5,5])

        ax1 = fig.add_subplot(111,projection=ccrs.SouthPolarStereo())
        ax1.set_extent([-180,180,-90,-60],ccrs.PlateCarree())
        # levels = np.linspace(0.00001,20000,21)
        # levels = np.linspace(-1,1,21)
        # levels = np.linspace(0,1000,21)*len(months)/12
        # levels = list(np.linspace(0.01,0.1,10)*100)+list(np.linspace(0.02,0.1,9)*1000)+list(np.linspace(0.02,0.1,9)*10000)#+list(np.linspace(0.02,0.1,9)*100000)
        levels = list(np.linspace(0.01,0.1,20)*1000)+list(np.linspace(0.02,0.1,18)*10000)#+list(np.linspace(0.02,0.1,9)*100000)
        # levels = np.linspace(0,100,21)
        ct_levels = np.arange(500,4000.1,500)
        area_levels = np.linspace(0,1,3)
        cf1 = ax1.contourf(lon_cyc_v,lat.values,ar_inv_cyc,transform=ccrs.PlateCarree(),cmap='jet',levels=levels,extend='max',norm=LogNorm())
        # cf1 = ax1.contourf(lon_cyc_v,lat.values,ar_inv_cyc,transform=ccrs.PlateCarree(),cmap='turbo',levels=levels)#,extend='max')#,norm=LogNorm())
        # ct1_2 = ax1.contour(lon_cyc_v,lat.values,z,transform=ccrs.PlateCarree(),levels=ct_levels,linewidths=0.2,colors='k')
        # ct1_1 = ax1.contour(lon_cyc_v,lat_v,sp_cyc_v,transform=ccrs.PlateCarree(),levels=ct2_levels,linewidths=0.1,colors='k')
        cbar = plt.colorbar(cf1,ax=ax1,shrink=0.7,orientation='vertical',pad=0.15)
        cbar.set_label(r'Annual precipitation [$mm\,year^{-1}$]',fontsize=10)
        # cbar.set_label('Contribution [%]',fontsize=10)
        ticks = [10,100,1000]
        # labels = [r'$10^{1}$',r'$10^{2}$',r'$10^{3}$',r'$10^{4}$']
        labels = [r'$10^{1}$',r'$10^{2}$',r'$10^{3}$']
        cbar.set_ticks(ticks,labels=labels)
        # ax1.clabel(ct1_1,fontsize=5)
        # ax1.clabel(ct1_2,fontsize=5)
        # pl = ax1.scatter(70,-70,transform=ccrs.PlateCarree())
        # pl = ax1.scatter(71,-70,transform=ccrs.PlateCarree())
        # pl = ax1.scatter(72,-70,transform=ccrs.PlateCarree())
        # pl = ax1.scatter(103,-65,transform=ccrs.PlateCarree())
        ax1.coastlines()
        gl1 = ax1.gridlines(linestyle=':',crs=ccrs.PlateCarree(), draw_labels=True)

        xticks = np.arange(-180,180,90)
        yticks = np.arange(-80,-9,10)

        gl1.xlocator = ticker.FixedLocator(xticks)
        gl1.ylocator = ticker.FixedLocator(yticks)

        # ax1.set_xticks(xticks,crs=ccrs.PlateCarree())
        # ax1.set_yticks(yticks,crs=ccrs.PlateCarree())
        #ax2.set_xticks(xticks,crs=ccrs.PlateCarree())
        #ax2.set_yticks(yticks,crs=ccrs.PlateCarree())

        ax1.axes.tick_params(labelsize=5)
        #ax2.axes.tick_params(labelsize=5)
        #cax1.axes.tick_params(labelsize=8)
        ax1.set_boundary(circle,transform=ax1.transAxes)
        # ax1.set_title('AR precipitation')

        out = '/Users/takahashikazu/Desktop/NIPR/fig/'
        if not os.path.exists('{}AR_Detection_3d_monthly/3d_frequency/AR_precipitation'.format(out)):#,Date.year,Date.month)):
            os.makedirs('{}AR_Detection_3d_monthly/3d_frequency/AR_precipitation'.format(out))#,Date.year,Date.month))
        out_dir = '{}AR_Detection_3d_monthly/3d_frequency/AR_precipitation/'.format(out)#,Date.year,Date.month)

        plt.savefig('{}Total3d_precipitation{}h_cont_{}.png'.format(out_dir,t,season),dpi=700)
        # plt.show()
        plt.close('all')
        
        

        # plot = 100*ar_ivt_dataset/total_dataset
        plot = 100*ar_dataset/total_dataset
        
        # plot = ar_ivt_dataset/years.shape[0]
        # plot = ar_dataset/years.shape[0]
        # plot = total_dataset/years.shape[0]
        
        dataarray = xr.DataArray(
            plot,
            name = 'ar_p',
            dims = ['latitude','longitude'],
            coords = (
                lat,lon,
            ),
        )

        print(dataarray.interp(latitude=-77.3,longitude=39.4))
        ar_inv_cyc, lon_cyc_v = util.add_cyclic_point(plot,coord=lon.values)
        
        center, radius = [0.5,0.5],0.5
        theta = np.linspace(0,2*np.pi,100)
        verts = np.vstack([np.sin(theta),np.cos(theta)]).T
        circle = mpath.Path(verts*radius+center)

        fig = plt.figure(figsize=[5,5])

        ax1 = fig.add_subplot(111,projection=ccrs.SouthPolarStereo())
        ax1.set_extent([-180,180,-90,-60],ccrs.PlateCarree())
        # levels = np.linspace(0.00001,20000,21)
        # levels = np.linspace(-1,1,21)
        # levels = np.linspace(0,1000,21)*len(months)/12
        # levels = list(np.linspace(0.01,0.1,9)*100)+list(np.linspace(0.02,0.1,9)*1000)+list(np.linspace(0.02,0.1,9)*10000)#+list(np.linspace(0.02,0.1,9)*100000)
        levels = np.linspace(0,100,21)
        ct_levels = np.arange(500,4000.1,500)
        area_levels = np.linspace(0,1,3)
        # cf1 = ax1.contourf(lon_cyc_v,lat.values,ar_inv_cyc,transform=ccrs.PlateCarree(),cmap='jet',levels=levels,extend='max',norm=LogNorm())
        cf1 = ax1.contourf(lon_cyc_v,lat.values,ar_inv_cyc,transform=ccrs.PlateCarree(),cmap='turbo',levels=levels)#,extend='max')#,norm=LogNorm())
        # ct1_2 = ax1.contour(lon_cyc_v,lat.values,z,transform=ccrs.PlateCarree(),levels=ct_levels,linewidths=0.2,colors='k')
        # ct1_1 = ax1.contour(lon_cyc_v,lat_v,sp_cyc_v,transform=ccrs.PlateCarree(),levels=ct2_levels,linewidths=0.1,colors='k')
        cbar = plt.colorbar(cf1,ax=ax1,shrink=0.7,orientation='vertical',pad=0.15)
        # cbar.set_label('Annual precipitation [mm]',fontsize=10)
        cbar.set_label('Contribution [%]',fontsize=10)
        # ticks = [1,10,100,1000]
        # labels = [r'$10^{0}$',r'$10^{1}$',r'$10^{2}$',r'$10^{3}$']
        # cbar.set_ticks(ticks,labels=labels)
        # ax1.clabel(ct1_1,fontsize=5)
        # ax1.clabel(ct1_2,fontsize=5)
        # pl = ax1.scatter(70,-70,transform=ccrs.PlateCarree())
        # pl = ax1.scatter(71,-70,transform=ccrs.PlateCarree())
        # pl = ax1.scatter(72,-70,transform=ccrs.PlateCarree())
        # pl = ax1.scatter(103,-65,transform=ccrs.PlateCarree())
        ax1.coastlines()
        gl1 = ax1.gridlines(linestyle=':',crs=ccrs.PlateCarree(), draw_labels=True)

        xticks = np.arange(-180,180,90)
        yticks = np.arange(-80,-9,10)

        gl1.xlocator = ticker.FixedLocator(xticks)
        gl1.ylocator = ticker.FixedLocator(yticks)

        # ax1.set_xticks(xticks,crs=ccrs.PlateCarree())
        # ax1.set_yticks(yticks,crs=ccrs.PlateCarree())
        #ax2.set_xticks(xticks,crs=ccrs.PlateCarree())
        #ax2.set_yticks(yticks,crs=ccrs.PlateCarree())

        ax1.axes.tick_params(labelsize=5)
        #ax2.axes.tick_params(labelsize=5)
        #cax1.axes.tick_params(labelsize=8)
        ax1.set_boundary(circle,transform=ax1.transAxes)
        # ax1.set_title('AR precipitation')

        out = '/Users/takahashikazu/Desktop/NIPR/fig/'
        if not os.path.exists('{}AR_Detection_3d_monthly/3d_frequency/AR_precipitation'.format(out)):#,Date.year,Date.month)):
            os.makedirs('{}AR_Detection_3d_monthly/3d_frequency/AR_precipitation'.format(out))#,Date.year,Date.month))
        out_dir = '{}AR_Detection_3d_monthly/3d_frequency/AR_precipitation/'.format(out)#,Date.year,Date.month)

        plt.savefig('{}AR2d_precipitation{}h_cont_{}.png'.format(out_dir,t,season),dpi=700)
        # plt.show()
        plt.close('all')
