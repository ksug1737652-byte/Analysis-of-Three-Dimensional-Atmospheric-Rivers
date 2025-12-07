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
from scipy import stats
from scipy.stats import pearsonr
from scipy.signal import detrend
from matplotlib.colors import BoundaryNorm

seasons = ['ALL','DJF','MAM','JJA','SON']#'01','02','03','04','05','06','07','08','09','10','11','12']#'ALL','DJF','MAM','JJA','SON']#,
t = 24 # period of precipitation after AR condition 0,24,48
s = 3
g = 9.80665
#data_dir = '/media/takahashi/HDPH-UT/Kyoto-U/AR_jra55/AR_detection/AR_detection_Kennet/data/'
#data_dir = '/Volumes/PromisePegasus/takahashi/JRA55/AR-2022-2023/'#,min_length,min_aspect,eqw_lat,plw_lat,dist_lat,distance_between)

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

    ar2d_tp = []
    ar3d_tp = []
    total_tp = []
    ar2d_mmt = []
    ar3d_mmt = []
    total_mmt = []
    ar_mask = []
    ar_mask2 = []
    ar_mask4 = []

    ar2d_tp_monthly = []
    ar3d_tp_monthly = []
    total_tp_monthly = []
    ar2d_mmt_monthly = []
    ar3d_mmt_monthly = []
    total_mmt_monthly = []
    ar_mask_monthly = []
    ar_mask2_monthly = []
    ar_mask4_monthly = []
    T_month = []

    for year in years:
        ar_ivt_dataset = 0
        ar_dataset = 0
        total_dataset = 0
        sp_dataset = 0
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
        ar2d_tp += [ar_dataset]
        ar3d_tp += [ar_ivt_dataset]
        total_tp += [total_dataset]



    a1 = tp2d_time_series = np.stack(ar2d_tp,axis=0)
    a2 = tp3d_time_series = np.stack(ar3d_tp,axis=0)
    a3 = totaltp_time_series = np.stack(total_tp,axis=0)
    
    lat_v = lat.values
    lon_v = lon.values
    
    arrays = [a1, a2, a3]

    # 変数とタイプの対応表
    labels = {
        0: ('AR2d', 'PREC', r'$mm\,year^{-1}\,decade^{-1}$'),
        1: ('AR3d', 'PREC', r'$mm\,year^{-1}\,decade^{-1}$'),
        2: ('Total', 'PREC', r'$mm\,year^{-1}\,decade^{-1}$'),
    }

    # ループで処理
    for i, data in enumerate(arrays, start=1):
        variable, Type, Unit = labels[i-1]   # i=1 のとき labels[0]
        print(f"a{i}: variable={variable}, Type={Type}")
            
        ntime,nlat,nlon = data.shape
        time = years
        # x = (time - time.mean())  # 中心化
        # n = len(time)

        # # ==== 2次元化 ====
        # Y = data.reshape(n, -1)   # shape = (ntime, npoints)

        # # 欠損値処理（例: NaNを無視できるようマスク）
        # mask = np.isfinite(Y)
        # Y = np.where(mask, Y, np.nan)

        # # ==== 傾き (slope) の計算 ====
        # # slope = Σ(x*y) / Σ(x^2)
        # denom = np.nansum(x[:, None]**2 * mask, axis=0)
        # num = np.nansum(x[:, None] * Y, axis=0)
        # slope = num / denom

        # # ==== 残差と標準誤差 ====
        # yhat = slope * x[:, None]
        # resid = Y - yhat
        # rss = np.nansum(resid**2, axis=0)
        # dof = np.sum(mask, axis=0) - 2
        # stderr = np.sqrt(rss / dof / denom)

        # # ==== t値・p値 ====
        # tval = slope / stderr
        # pval = 2 * stats.sf(np.abs(tval), dof)

        # # ==== reshape back ====
        # slope_map = slope.reshape(nlat, nlon)  # 年あたり変化量（timeが月なら×12）
        # pval_map  = pval.reshape(nlat, nlon)

        slope_map = np.full((nlat, nlon), np.nan)
        pval_map = np.full((nlat, nlon), np.nan)

        for j in range(nlat):
            for i in range(nlon):
                y = data[:, j, i]
                if np.all(np.isnan(y)):
                    continue

                mask = ~np.isnan(y)
                if np.sum(mask) < 3:  # サンプルが少なすぎる場合はスキップ
                    continue

                slope, intercept, r_value, p_value, std_err = stats.linregress(time[mask], y[mask])
                slope_map[j, i] = slope*10
                pval_map[j, i] = p_value   
        
        # pval_map = (pval_map<=0.05).astype(int)

        np.save('{}{}_{}_{}_slope.npy'.format(dataout_dir,Type,variable,season),slope_map)
        np.save('{}{}_{}_{}_pvalue.npy'.format(dataout_dir,Type,variable,season),pval_map)

        # ar_inv_cyc, lon_cyc_v = util.add_cyclic_point(ar_mask3,coord=lon_v)
        # ar_cyc_v2, lon_cyc_v = util.add_cyclic_point(ar_mask2,coord=lon_v)
        # ar_cyc_v3, lon_cyc_v = util.add_cyclic_point(ar_mask3,coord=lon_v)
        # ar_cyc_v4, lon_cyc_v = util.add_cyclic_point(ar_mask4,coord=lon_v)
        # sp_cyc_v, lon_cyc_v = util.add_cyclic_point(sp,coord=lon_v)
        # z, lon_cyc_v = util.add_cyclic_point(z_data,coord=lon_v)

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
        # data1 = detrend(totaltp_time_series)
        # data2 = detrend(ar3d_time_series)
        # corr_map = np.full((nlat, nlon), np.nan)
        # p_map = np.full((nlat, nlon), np.nan)
        # for j in range(nlat):
        #     for i in range(nlon):
        #         ts1 = data1[:, j, i]
        #         ts2 = data2[:, j, i]

        #         # NaN を無視
        #         mask = ~np.isnan(ts1) & ~np.isnan(ts2)
        #         if np.sum(mask) < 3:  # サンプル不足はスキップ
        #             continue

        #         r, p = pearsonr(ts1[mask], ts2[mask])
        #         corr_map[j, i] = r
        #         p_map[j, i] = p






        lat = -77
        lon = 39
        
        i = -lat-20
        j = lon


        center, radius = [0.5,0.5],0.5
        theta = np.linspace(0,2*np.pi,100)
        verts = np.vstack([np.sin(theta),np.cos(theta)]).T
        circle = mpath.Path(verts*radius+center)


        # fig = plt.figure(figsize=(4,3))
        #ax1 = fig.add_subplot(311,projection=ccrs.PlateCarree(central_longitude=0))
        #ax1 = fig.add_subplot(111,projection=ccrs.NearsidePerspective(central_latitude=-90,central_longitude=0))
        fig = plt.figure(figsize=[5,5])

        ax1 = fig.add_subplot(111,projection=ccrs.SouthPolarStereo())
        ax1.set_extent([-180,180,-90,-60],ccrs.PlateCarree())
        # ax1 = fig.add_subplot(111,projection=ccrs.LambertConformal(central_longitude=75,central_latitude=-65,
        #                                                             standard_parallels=(-65,-30),cutoff=-10))
        # ax1.set_extent([30,120,-85,-40],ccrs.PlateCarree())
        # fig.subplots_adjust(bottom=0.2)
        # cax1 = fig.add_axes((0.2,0.12,0.6,0.02))

        #ax2 = fig.add_subplot(312)
        #ax3 = fig.add_subplot(212)

        # levels = np.linspace(0.00001,20000,21)
        # levels = np.linspace(-1,1,21)
        # levels = list(np.linspace(0.01,0.1,10))+list(np.linspace(0.02,0.1,9)*10)+list(np.linspace(0.02,0.1,9)*100)#+list(np.linspace(0.02,0.1,9)*1000)#+list(np.linspace(0.02,0.1,9)*10000)#+list(np.linspace(0.02,0.1,9)*100000)
        # levels = list(np.linspace(0.01,0.1,9)*10)+list(np.linspace(0.02,0.1,9)*100)+list(np.linspace(0.02,0.1,9)*1000)+list(np.linspace(0.02,0.1,9)*10000)#+list(np.linspace(0.02,0.1,9)*100000)
        # print(levels)
        # levels = np.linspace(0,200,21)*len(months)/12
        if Type == 'PREC':        
            # 任意の境界値を指定
            boundaries = [-20,-15,-10,-5,-2.5,-1.0,-0.5,0.5,1.0,2.5,5,10,15,20]  # 区間ごとに色を分ける
            cmap = plt.get_cmap("RdBu", len(boundaries)-1)  # 区間数に合わせた colormap

            # ノルムを作る
            norm = BoundaryNorm(boundaries, cmap.N)

        else:
            max_value = 2
            
        # levels = np.linspace(-5,5,21)*len(months)/12
        # levels = np.linspace(-2500,2500,21)#*len(months)/12
        # levels = np.linspace(0,100,21)*len(months)/12
        # levels = np.linspace(0,20,21)*len(months)/12
        ct_levels = np.arange(500,4000.1,500)
        area_levels = np.linspace(0,1,3)
        pcm = ax1.pcolormesh(
            lon_v, lat_v, slope_map,
            transform=ccrs.PlateCarree(),
            cmap = cmap,
            norm = norm,
            alpha=0.7,
        )
        cbar = plt.colorbar(pcm, ax=ax1, orientation="vertical", pad=0.15, label="Trend [days / year]",shrink=0.7)#,aspect=30)
        cbar.set_label('Trend [{}]'.format(Unit),fontsize=10)
        cbar.set_ticks(boundaries,labels=boundaries,fontsize=7)

        zero_line = ax1.contour(
            lon_v,lat_v,slope_map,
            transform=ccrs.PlateCarree(),
            linewidths=0.2,
            levels=[0],
            colors='k')
        
        plt.rcParams['hatch.linewidth'] = 0.5
        cf = ax1.contourf(lon_v, lat_v, pval_map,
            levels=[0,0.05,1.1],
            colors='none',               # 塗りつぶし色なし
            hatches=['.......',''],#, '\\\\\\\\', '....', 'xxx'],  # 斜線やドット
            transform=ccrs.PlateCarree()
        )

        # ship = ax1.scatter(lon,lat,facecolor='white',edgecolor='b',transform=ccrs.PlateCarree(),s=5,zorder=5)

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
        # ax1.set_title('AR Frequency trend')

        out = '/Users/takahashikazu/Desktop/NIPR/fig/AR_Detection_3d_monthly/3d_frequency/trend/{}'.format(season)
        if not os.path.exists('{}'.format(out)):#,Date.year,Date.month)):
            os.makedirs('{}'.format(out))#,Date.year,Date.month))
        out_dir = '{}/'.format(out)#,Date.year,Date.month)
        if (Type == 'PREC'):
            Type += '{}'.format(t)

        plt.savefig('{}{}_{}_{}.png'.format(out_dir,Type,variable,season),dpi=700)
        plt.close('all')
        
        # plt.show()
        
        
        # print(np.where(corr_map<0))
        # fig, ax = plt.subplots(figsize=(6, 4))  # 図のサイズを指定可能

        # ddy = 0
        # pl = ax.plot(years,ar3d_time_series[:,i,j],linewidth=1,color='red',label='r={:.3f}')#.format(corr_map[i,j]))
        # ax1 = ax.twinx()
        # pl = ax1.plot(years,totaltp_time_series[:,i,j],linewidth=1,color='k')
            
        # ax.grid(linestyle='--', alpha=0.5)
        # ax.legend()
        # # ax.set_yticks(np.arange(0,100.1,10))
        # # ax.set_xticks(np.arange(0,73,6))
        # # ax.set_ylim(0,105)
        # # ax.set_xlim(1,72)
        # # ax.set_xlabel('hours after AR landfall', fontsize=12)
        # # ax.set_ylabel('Prec. contribution', fontsize=12,color='black')

        # plt.show()

