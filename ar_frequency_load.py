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

seasons = ['ALL']#,'DJF','JJA']#'01','02','03','04','05','06','07','08','09','10','11','12']#'ALL','DJF','MAM','JJA','SON']#,

se = [3,4,2][:1]
g = 9.80665
#data_dir = '/media/takahashi/HDPH-UT/Kyoto-U/AR_jra55/AR_detection/AR_detection_Kennet/data/'
#data_dir = '/Volumes/PromisePegasus/takahashi/JRA55/AR-2022-2023/'#,min_length,min_aspect,eqw_lat,plw_lat,dist_lat,distance_between)

for s in se:
    inputdata_dir = '/Volumes/Pegasus32 R6/takahashi/era5/'
    input_data_dir = '/Volumes/Pegasus32 R6/takahashi/era5/input_data/'
    mask_dir = '{}ar_out_era5_SH_20S/ar_ivt_data/'.format(inputdata_dir)
    mask_inv_dir = '{}ar_out_3d_{:0=2}/ar_ivt_data/'.format(inputdata_dir,s)

    data_dir = '{}ar_out_era5_SH_20S/'.format(inputdata_dir)
    data_dir_inv = '{}ar_out_3d_{:0=2}/'.format(inputdata_dir,s)
    if s == 3:
        data_dir_inv = '/Volumes/Pegasus32 R8/takahashi/era5/ar_out_3d_monthly_threshold/'#.format(inputdata_dir)
        mask_inv_dir = '/Volumes/Pegasus32 R8/takahashi/era5/ar_out_3d_monthly_threshold/ar_ivt_data/'#.format(inputdata_dir)

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
    sp = dataset['z'].sel(latitude=slice(-20,-89)).values[0]/9.8
    print(sp.shape)

    # years = [2021,1985,1986]#,1981,1982,1982,1983,1995,1996,1997,2005,2006,2007,2008,2009,2016,2017,2018,2019,2020][:1]       #np.arange(2023,2024,1)
    # years += [1979,1980,1988,1989,1990,1991,1992,1993,1994,1998,1999,2000,2001,2002,2003,2004,2010,2011,2012,2013,2014,2015,2022,2023]#np.arange(2023,2024,1)
    years = np.arange(1979,2024,1)[::1]
    #[1990,1991,1992,1993,1994,1995,1996,1997,2007,2008,2009,2010,2011,2012,2013,2014]
    #1989,2022,2021]#1986,1987,2023,2022,1988]#1983,1984,1985]#np.arange(1979,2024,1)[::1]
    #[1979,1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,2003,2004,2005,2006,2021,2022,2023]#
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

        ar_mask = 0
        ar_mask2 = 0
        ar_mask4 = 0
        for year in years:
            # for month in months:
            file1 = '{}AR_ground_frequency_ALL_{}.nc'.format(dataout_dir,year)
            file_ivt = '{}AR_ground_frequency_ALL_{}.nc'.format(dataout_dir_inv,year)
            file_ivt2 = '{}AR_ground_frequency_iwv_weighted_ALL_{}.nc'.format(dataout_dir_inv,year)
            # file3 = '{}AR_frequency_{}_1979_2023.nc'.format(dataout_dir_eq_pos,season)
            # file4 = '{}AR_frequency_{}_1979_2023.nc'.format(dataout_dir_eq_neg,season)
            # file2 = '{}AR_frequency_{}_1979_2023.nc'.format(dataout_dir_eq,season)
            dataset = xr.open_dataset(file1)
            dataset_ivt = xr.open_dataset(file_ivt)
            dataset_ivt2 = xr.open_dataset(file_ivt2)
            ar_mask_dataset = dataset['ar'].sel(latitude=slice(-20,-89))
            ar_mask_dataset_ivt = dataset_ivt['ar'].sel(latitude=slice(-20,-89))
            ar_mask_dataset_ivt2 = dataset_ivt2['ar'].sel(latitude=slice(-20,-89))
            # print(ar_mask_dataset_ivt.values.dtype)
            # ar_mask_dataset_ivt2 = dataset_ivt2['ar'].sel(latitude=slice(-20,-90))
            # dataset2 = xr.open_dataset(file2)
            # ar_mask_dataset2 = dataset2['ar'].sel(latitude=slice(-20,-90))
            # dataset3 = xr.open_dataset(file3)
            # ar_mask_dataset3 = dataset3['ar'].sel(latitude=slice(-20,-90))
            # dataset4 = xr.open_dataset(file4)
            # ar_mask_dataset4 = dataset4['ar'].sel(latitude=slice(-20,-90))
            # time = ar_mask_dataset.shape[0]
            lat = ar_mask_dataset.latitude
            lon = ar_mask_dataset.longitude
            
            for month in months:
                ar_mask += ar_mask_dataset.values[month-1]/len(years)
                ar_mask2 += (ar_mask_dataset_ivt.sel(time='{}-{:0=2}-01-00'.format(year,month)).values/len(years))#/ np.timedelta64(1, "D")).astype(float)
                ar_mask4 += (ar_mask_dataset_ivt2.sel(time='{}-{:0=2}-01-00'.format(year,month)).values/len(years))#/ np.timedelta64(1, "D")).astype(float)
            # ar_mask4 += ar_mask_dataset_ivt2.values/len(years)
            # ar_mask3 = ar_mask_dataset3.values
            # ar_mask4 = ar_mask_dataset4.values

        # print(ar_mask.min())
        # print(ar_mask.dtype)
        # print(ar_mask2.max())
        # print((ar_mask2/ np.timedelta64(1, "D")).astype(float).max())
        lat_v = lat.values
        lon_v = lon.values
        ar_mask3 = ar_mask4/3.65#(ar_mask4-ar_mask)/np.where(ar_mask==0,np.nan,ar_mask)#np.abs(ar_mask4-ar_mask)/ar_mask
        # ar_mask4= (ar_mask4-ar_mask2)/ar_mask2
        # ar_mask2= (ar_mask2-ar_mask2)/ar_mask2
        ar_dir = '/Volumes/Pegasus32 R6/takahashi/era5/' 
        input_data_dir = '/Volumes/Pegasus32 R6/takahashi/era5/input_data/' 
        grid_areas_file = '{}input_data/grid_areas.nc'.format(ar_dir)
        grid_areas = xr.open_dataset(grid_areas_file)['unknown'].sel(latitude=slice(-20,-89)).values
        land_mask_file = '{}input_data/lsm_2023-1.nc'.format(ar_dir)
        land_mask_data = xr.open_dataset(land_mask_file)['lsm'].sel(latitude=slice(-20,-89))
        ocean = 1-land_mask_data.values

        dataarray = xr.DataArray(
            ar_mask4,
            name = 'ar_p',
            dims = ['latitude','longitude'],
            coords = (
                lat,lon,
            ),
        )

        print(dataarray.interp(latitude=-77.3,longitude=39.4))

        # print(np.nansum(ar_mask3*grid_areas*ocean)/np.nansum(np.where(ar_mask==0,np.nan,grid_areas*ocean)))

        ar_inv_cyc, lon_cyc_v = util.add_cyclic_point(ar_mask3,coord=lon_v)
        # ar_cyc_v2, lon_cyc_v = util.add_cyclic_point(ar_mask2,coord=lon_v)
        # ar_cyc_v3, lon_cyc_v = util.add_cyclic_point(ar_mask3,coord=lon_v)
        # ar_cyc_v4, lon_cyc_v = util.add_cyclic_point(ar_mask4,coord=lon_v)
        # sp_cyc_v, lon_cyc_v = util.add_cyclic_point(sp,coord=lon_v)
        z, lon_cyc_v = util.add_cyclic_point(z_data,coord=lon_v)

        #c_time = mask_dev.variables['time'].values
        #print(mask_dev1.shape)
        #computing longitutinal mask ratio of ar
        #data_file = '/media/takahashi/HDPH-UT/Kyoto-U/AR_jra55/AR_detection/AR_detection_Kennet/data/GP.nc'
        #gp_dataset = xr.open_dataset(data_file)['z'][0].values
        #lat = xr.open_dataset(data_file)['latitude'].values
        #lon = xr.open_dataset(data_file)['longitude'].values
        g = 9.8

        #gp = gp_dataset/g

        #area_map = 0*gp
        #area_map[128:144,144:240] = 1

        center, radius = [0.5,0.5],0.5
        theta = np.linspace(0,2*np.pi,100)
        verts = np.vstack([np.sin(theta),np.cos(theta)]).T
        circle = mpath.Path(verts*radius+center)


        # fig = plt.figure(figsize=(4,3))
        #ax1 = fig.add_subplot(311,projection=ccrs.PlateCarree(central_longitude=0))
        #ax1 = fig.add_subplot(111,projection=ccrs.NearsidePerspective(central_latitude=-90,central_longitude=0))
        fig = plt.figure(figsize=[5,5])

        ax1 = fig.add_subplot(111,projection=ccrs.SouthPolarStereo())
        ax1.set_extent([-180,180,-90,-20],ccrs.PlateCarree())
        # ax1 = fig.add_subplot(111,projection=ccrs.LambertConformal(central_longitude=75,central_latitude=-65,
        #                                                             standard_parallels=(-65,-30),cutoff=-10))
        # ax1.set_extent([30,120,-85,-40],ccrs.PlateCarree())
        # fig.subplots_adjust(bottom=0.3)
        # cax1 = fig.add_axes((0.1,0.12,0.37,0.02))
        # cax2 = fig.add_axes((0.55,0.12,0.37,0.02))

        #ax2 = fig.add_subplot(312)
        #ax3 = fig.add_subplot(212)

        # levels = np.linspace(0.00001,20000,21)
        # levels = np.linspace(-1,1,21)
        # levels = list(np.linspace(0.01,0.1,10)*10)+list(np.linspace(0.02,0.1,9)*100)+list(np.linspace(0.02,0.1,9)*1000)+list(np.linspace(0.02,0.1,9)*10000)#+list(np.linspace(0.02,0.1,9)*100000)
        # levels = list(np.linspace(0.01,0.1,10))+list(np.linspace(0.02,0.1,9)*9)+list(np.linspace(0.02,0.1,9)*100)+list(np.linspace(0.02,0.1,9)*1000)
        # print(levels)
        # levels = np.linspace(0,200,21)*len(months)/12
        levels = np.linspace(0,30,21)*len(months)/12
        # levels = np.linspace(0,0.0001,21)*len(months)/12
        # levels = np.linspace(0,20,21)*len(months)/12
        ct_levels = np.arange(500,4000.1,500)
        area_levels = np.linspace(0,1,3)
        # cf1 = ax1.contourf(lon_cyc_v,lat_v,ar_inv_cyc,transform=ccrs.PlateCarree(),levels=levels,cmap='seismic',extend='both')
        cf1 = ax1.contourf(lon_cyc_v,lat_v,ar_inv_cyc,transform=ccrs.PlateCarree(),cmap='turbo',levels=levels,extend='max')#,norm=LogNorm())
        # cf1 = ax1.contourf(lon_cyc_v,lat_v,ar_inv_cyc,transform=ccrs.PlateCarree(),cmap='seismic',levels=levels,extend='both')#,norm=LogNorm())
        # cf1 = ax1.contourf(lon_cyc_v,lat_v,ar_inv_cyc,transform=ccrs.PlateCarree(),cmap='hot_r',levels=levels,extend='max',norm=LogNorm())
        # cf2 = ax1.contourf(lon_cyc_v,lat_v,-ar_inv_cyc,transform=ccrs.PlateCarree(),cmap='ocean_r',levels=levels,extend='max',norm=LogNorm())
        ct1_2 = ax1.contour(lon_cyc_v,lat_v,z,transform=ccrs.PlateCarree(),levels=ct_levels,linewidths=0.2,colors='k')
        # ct1_1 = ax1.contour(lon_cyc_v,lat_v,sp_cyc_v,transform=ccrs.PlateCarree(),levels=ct2_levels,linewidths=0.1,colors='k')
        cbar1 = plt.colorbar(cf1,ax=ax1,shrink=0.7,orientation='horizontal',pad=0.15)
        # cbar2 = plt.colorbar(cf2,cax=cax1,shrink=0.7,orientation='horizontal',pad=0.15)
        # ticks = [0.01,0.1,1,10,100]#,1000]
        # labels1 = [r'$10^{-2}$',r'$10^{-1}$',r'$10^{0}$',r'$10^{1}$',r'$10^{2}$']#,r'$10^{3}$']
        # labels2 = [r'$-10^{-1}$',r'$-10^{0}$',r'$-10^{1}$',r'$-10^{2}$',r'$-10^{3}$']
        # ticks = [1,10,100,1000]
        # labels = [r'$10^{0}$',r'$10^{1}$',r'$10^{2}$',r'$10^{3}$']
        ticks = np.linspace(0,30,6)
        labels1 = ticks
        cbar1.set_ticks(ticks,labels=labels1,fontsize=9)
        # cbar2.set_ticks(ticks,labels=labels2,fontsize=9)
        # cbar2.ax.invert_xaxis()
        # cbar.set_label(r'Frequency [$days\,year^{-1}$]',fontsize=10)
        cbar1.set_label('Frequency [%]',fontsize=10)
        # cbar2.set_label(r'Frequency difference [dimensionless]',fontsize=10)
        # ax1.clabel(ct1_1,fontsize=5)
        ax1.clabel(ct1_2,fontsize=5)
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
        # ax1.set_title('AR Frequency')


        # ax1 = fig.add_subplot(132,projection=ccrs.SouthPolarStereo())
        # ax1.set_extent([-180,180,-90,-40],ccrs.PlateCarree())

        # cf1 = ax1.contourf(lon_cyc_v,lat_v,ar_cyc_v3,transform=ccrs.PlateCarree(),levels=levels,cmap='seismic',extend='both')
        # ct1_2 = ax1.contour(lon_cyc_v,lat_v,z,transform=ccrs.PlateCarree(),levels=ct_levels,linewidths=0.2,colors='k')
        # # ct1_1 = ax1.contour(lon_cyc_v,lat_v,sp_cyc_v,transform=ccrs.PlateCarree(),levels=ct2_levels,linewidths=0.1,colors='k')
        # cbar = plt.colorbar(cf1,cax=cax1,shrink=0.7,orientation='horizontal',pad=0.1)
        # cbar.set_label('Days / year',fontsize=10)
        # # ax1.clabel(ct1_1,fontsize=5)
        # ax1.clabel(ct1_2,fontsize=5)
        # # pl = ax1.scatter(70,-70,transform=ccrs.PlateCarree())
        # # pl = ax1.scatter(71,-70,transform=ccrs.PlateCarree())
        # # pl = ax1.scatter(72,-70,transform=ccrs.PlateCarree())
        # # pl = ax1.scatter(103,-65,transform=ccrs.PlateCarree())
        # ax1.coastlines()
        # gl1 = ax1.gridlines(linestyle='--',crs=ccrs.PlateCarree(), draw_labels=True)

        # xticks = np.arange(-180,180,90)
        # yticks = np.arange(-80,-9,20)

        # gl1.xlocator = ticker.FixedLocator(xticks)
        # gl1.ylocator = ticker.FixedLocator(yticks)

        # # ax1.set_xticks(xticks,crs=ccrs.PlateCarree())
        # # ax1.set_yticks(yticks,crs=ccrs.PlateCarree())
        # #ax2.set_xticks(xticks,crs=ccrs.PlateCarree())
        # #ax2.set_yticks(yticks,crs=ccrs.PlateCarree())

        # ax1.axes.tick_params(labelsize=5)
        # #ax2.axes.tick_params(labelsize=5)
        # #cax1.axes.tick_params(labelsize=8)
        # ax1.set_boundary(circle,transform=ax1.transAxes)
        # ax1.set_title('295K')

        # ax1 = fig.add_subplot(133,projection=ccrs.SouthPolarStereo())
        # ax1.set_extent([-180,180,-90,-40],ccrs.PlateCarree())

        # cf1 = ax1.contourf(lon_cyc_v,lat_v,ar_cyc_v4,transform=ccrs.PlateCarree(),levels=levels,cmap='seismic',extend='both')
        # ct1_2 = ax1.contour(lon_cyc_v,lat_v,z,transform=ccrs.PlateCarree(),levels=ct_levels,linewidths=0.2,colors='k')
        # # ct1_1 = ax1.contour(lon_cyc_v,lat_v,sp_cyc_v,transform=ccrs.PlateCarree(),levels=ct2_levels,linewidths=0.1,colors='k')
        # # cbar = plt.colorbar(cf1,ax=ax1,shrink=0.7,orientation='vertical',pad=0.1)
        # # cbar.set_label('Difference rate',fontsize=10)
        # # ax1.clabel(ct1_1,fontsize=5)
        # ax1.clabel(ct1_2,fontsize=5)
        # # pl = ax1.scatter(70,-70,transform=ccrs.PlateCarree())
        # # pl = ax1.scatter(71,-70,transform=ccrs.PlateCarree())
        # # pl = ax1.scatter(72,-70,transform=ccrs.PlateCarree())
        # # pl = ax1.scatter(103,-65,transform=ccrs.PlateCarree())
        # ax1.coastlines()
        # gl1 = ax1.gridlines(linestyle='--',crs=ccrs.PlateCarree(), draw_labels=True)

        # xticks = np.arange(-180,180,90)
        # yticks = np.arange(-80,-9,20)

        # gl1.xlocator = ticker.FixedLocator(xticks)
        # gl1.ylocator = ticker.FixedLocator(yticks)

        # ax1.set_xticks(xticks,crs=ccrs.PlateCarree())
        # ax1.set_yticks(yticks,crs=ccrs.PlateCarree())
        #ax2.set_xticks(xticks,crs=ccrs.PlateCarree())
        #ax2.set_yticks(yticks,crs=ccrs.PlateCarree())

        # ax1.axes.tick_params(labelsize=5)
        # #ax2.axes.tick_params(labelsize=5)
        # #cax1.axes.tick_params(labelsize=8)
        # ax1.set_boundary(circle,transform=ax1.transAxes)
        # ax1.set_title('285K')


        #ax1.set_title('(a)')
        #ax2.set_title('(b)')
        #ax1.set_yticklabels(['0,01','100','200'])
        #ax2.legend()
        #ax2.set_title('ar mask ratio along latitude')
        #ax2.grid()
        #ax3.legend()
        #ax3.set_title('ar frequency changing with time lat=-65')
        #ax3.grid()
        #ax1.set_title('(a)',x=-0.1,y=0.9)
        # fig.tight_layout()
        out = '/Users/takahashikazu/Desktop/NIPR/fig/'
        if not os.path.exists('{}AR_Detection_3d_monthly/3d_frequency/Frequency'.format(out)):#,Date.year,Date.month)):
            os.makedirs('{}AR_Detection_3d_monthly/3d_frequency/Frequency'.format(out))#,Date.year,Date.month))
        out_dir = '{}AR_Detection_3d_monthly/3d_frequency/Frequency/'.format(out)#,Date.year,Date.month)

        plt.savefig('{}AR3d_percent_{:0=2}_Q2Dfrequency_{}.png'.format(out_dir,s,season),dpi=700)
        # plt.close('all')
        plt.show()