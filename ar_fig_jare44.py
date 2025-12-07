import warnings
warnings.filterwarnings('ignore')

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import glob
import datetime
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import matplotlib.path as mpath
import xarray as xr
import cartopy.crs as ccrs
import cartopy.util as util
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import pandas as pd
import copy
from metpy.units import units
from metpy.constants import earth_avg_radius
import metpy.calc as mpcalc

event = 'extreme'
ar_monthly_dir = '/Volumes/Pegasus32 R8/takahashi/era5/' 
ar_dir = '/Volumes/Pegasus32 R6/takahashi/era5/' 
ar3d_dir = '{}ar_out_3d_monthly_threshold/ar_characteristics/'.format(ar_monthly_dir)
if event == 'AR':
    df = pd.read_csv('{}P_event_ar.csv'.format(ar3d_dir))
elif event == 'extreme':
    df = pd.read_csv('{}ExP_event.csv'.format(ar3d_dir))
ar_event_date = df[df.columns[1]][19:20].reset_index(drop=True)
print(ar_event_date)
# print(type(ar_event_date[0]))

# sensitivity = 0
# year = 2003
# month = 8
# day = 1
# hour = 0
center_lon = 40
# 20  120
# 39 90

regions = ['SH','Fuji'][::1]
variables = ['level','iwv','pr'][::1]#,'sf'
# out = '/Users/takahashikazu/Desktop/NIPR/fig/'
# if not os.path.exists('{}ERA5_40years/JARE64/AR/{}'.format(out,folda)):#,Date.year,Date.month)):
#     os.makedirs('{}ERA5_40years/JARE64/AR/{}'.format(out,folda))#,Date.year,Date.month))
# out_dir = '{}ERA5_40years/JARE64/AR/{}/'.format(out,folda)#,Date.year,Date.month)

# out = '/Users/takahashikazu/Desktop/NIPR/fig/'
# if not os.path.exists('{}eq_analysis/p_coordinate/{}/sensitivity_analysis/{}{:0=2}{:0=2}{:0=2}_{}s{}_2'.format(out,folda,year,month,day,hour,folda,sensitivity)):#,Date.year,Date.month)):
#     os.makedirs('{}eq_analysis/p_coordinate/{}/sensitivity_analysis/{}{:0=2}{:0=2}{:0=2}_{}s{}_2'.format(out,folda,year,month,day,hour,folda,sensitivity))#,Date.year,Date.month))
# out_dir = '{}eq_analysis/p_coordinate/{}/sensitivity_analysis/{}{:0=2}{:0=2}{:0=2}_{}s{}_2/'.format(out,folda,year,month,day,hour,folda,sensitivity)#,Date.year,Date.month)
# out = '/Users/takahashikazu/Desktop/NIPR/fig/'
# if not os.path.exists('{}eq_analysis/p_coordinate/{}2/{}{:0=2}{:0=2}{:0=2}_{}s{}'.format(out,folda,year,month,day,hour,folda,sensitivity)):#,Date.year,Date.month)):
#     os.makedirs('{}eq_analysis/p_coordinate/{}2/{}{:0=2}{:0=2}{:0=2}_{}s{}'.format(out,folda,year,month,day,hour,folda,sensitivity))#,Date.year,Date.month))
# out_dir = '{}eq_analysis/p_coordinate/{}2/{}{:0=2}{:0=2}{:0=2}_{}s{}/'.format(out,folda,year,month,day,hour,folda,sensitivity)#,Date.year,Date.month)


################## LOCATION of Analysis
#print(path_list)
# data_path = '/Volumes/PromisePegasus/takahashi/data/GPS/202'
# path_list = sorted(glob.glob(data_path + '*.txt'))

# days = []
# time=[]
# lat_l=[]
# lon_l=[]
# for i in path_list:
#     #print(i)
#     data = pd.read_table(i,delimiter=',',usecols=[0,1,2,3,4])

#     for j in range(int(len(data))):
#         t = j
#         days.append(data.iloc[t,1].strip()+'-'+data.iloc[t,2].strip())
#         time.append(data.iloc[t,2].strip())
#         lat_l.append(np.where(data.iloc[t,3]>=-90,data.iloc[t,3],np.nan))
#         lon_l.append(np.where(data.iloc[t,4]>=-90,data.iloc[t,4],np.nan))

# ws_path = '/Volumes/PromisePegasus/takahashi/data/WS500/'

# file_list = glob.glob(ws_path + '*' + '202' + '*.csv')
# #print(file_list)
# file = file_list[0]
# filename = os.path.splitext(os.path.basename(file))[0]
# #print(file)

# df_list = []
# lDate = []
# llat = []
# llon = []
# lpres = []
# lwspd = []
# lwdec = []
# ldssd = []
# ldsdr = []

# for f in file_list:
#     df = pd.read_csv(f)#,encoding='cp932')
#     df = df.dropna()
#     llat.append(df[df.columns[13]])
#     llon.append(df[df.columns[16]])
#     lDate.append(df[df.columns[0]])
#     lpres.append(df[df.columns[7]])
#     lwspd.append(df[df.columns[8]])
#     lwdec.append(df[df.columns[9]])
#     ldssd.append(df[df.columns[18]])
#     ldsdr.append(df[df.columns[19]])
# wsDate = pd.concat(lDate,axis=0,sort=True).values
# wslat = pd.concat(llat,axis=0,sort=True).values
# wslon = pd.concat(llon,axis=0,sort=True).values
# wsdssd = pd.concat(ldssd,axis=0,sort=True).values
# wsdsdr = pd.concat(ldsdr,axis=0,sort=True).values
# wsDate = np.where(wsDate=='-',np.nan,wsDate)
# wslat = np.where(wslat=='-',np.nan,wslat).astype(float)
# wslon = np.where(wslon=='-',np.nan,wslon).astype(float)
# wsdssd = np.where(wsdssd=='-',np.nan,wsdssd).astype(float)#df['平均風向(度)']
# wsdsdr = np.where(wsdsdr=='-',np.nan,wsdsdr).astype(float)#df['平均風向(度)']

# wsdr = np.deg2rad(wsdsdr)
# sv = wsdssd*np.cos(wsdr)
# su = wsdssd*np.sin(wsdr)

# print(wsDate)
# def position(year,month,day,hour):
#     if len(np.where(np.array(days)=='{}-{:0=2}-{:0=2}-{:0=2}:00:00'.format(year,month,day,hour))[0]) == 0:
#         if len(np.where(wsDate=='{}-{:0=2}-{:0=2} {:0=2}:00:00'.format(year,month,day,hour))[0]) == 0:
#             lat1 = -65
#             lon1 = 38
#         else:
#             ind1 = np.where(wsDate>='{}-{:0=2}-{:0=2} {:0=2}:00:00'.format(year,month,day,hour))[0][0]

#             lat1 = -1*wslat[ind1]
#             lon1 = wslon[ind1]
#     else:
#         ind1 = np.where((np.array(days)=='{}-{:0=2}-{:0=2}-{:0=2}:00:00'.format(year,month,day,hour)))[0][0]

#         lon1 = lon_l[ind1]
#         lat1 = lat_l[ind1]
#     return lat1,lon1


# lat,lon = position(year,month,day,hour)
# print(lat,lon)
# ind1 = np.where(wsDate>='{}-{:0=2}-{:0=2} {:0=2}:00:00'.format(year,month,day,hour))[0][0]
# ship_u = su[ind1]
# ship_v = sv[ind1]

# iwvcolors=np.array([
#     [255,255,255,1],#white
#     [225,255,255,1],#white
#     [200,255,255,1],#white
#     [175,255,255,1],#white
#     [155,255,255,1],#white
#     [130,255,255,1],#white
#     [100,255,255,1],#white
#     [50,255,255,1],#white
#     [25,255,255,1],#white
#     [0, 255,255,1],#white
#     [0, 225,255,1],#white
#     [0, 200,255,1],#white
#     [0, 130,255,1],#white
#     [0, 100,255,1],#white
#     [0, 75,255,1],#white
#     [0, 50,255,1],#white
#     [0, 0,255,1],#white
#     [0, 50,230,1],#white
#     [0, 100,230,1],#white
#     [0, 130,200,1],#white
#     [0, 155,200,1],#white
#     [0, 180,200,1],#white
#     [0, 200,200,1],#white
#     [0, 255,200,1],#white
#     [0, 255,180,1],#white
#     [0, 255,150,1],#white
#     [0, 255,125,1],#white
#     [0, 255,100,1],#white
#     [0, 255,80,1],#white
#     [0, 255,50,1],#white
#     [80,255,80,1],#white
#     [100,255,100,1],#white
#     [120,255,120,1],#white
#     [150,255,150,1],#white
#     [175,255,150,1],#white
#     [200,255,150,1],#white
#     [225,255,150,1],#white
#     [255,255,150,1],#white
#     [255,255,125,1],#white
#     [255,255,100,1],#white
#     [255,255,80,1],#white
#     [255,255,50,1],#white
#     [255,255,0,1],#white
#     [255,225,0,1],#white
#     [255,200,0,1],#white
#     [255,150,0,1],#white
#     [255,100,0,1],#white
#     [255,80,0,1],#white
#     [255,50,0,1],#white
#     [255,25,0,1],#white
#     [255,0,0,1]#white
# ],dtype=float)
# print(iwvcolors.shape)
# iwvcolors[:,:3]/=256 #RGBを0-1の範囲に変換
# iwvcmap=ListedColormap(iwvcolors)
# iwvcmap2=LinearSegmentedColormap.from_list("gscmap2",colors=iwvcolors)
# level_list = [500]#,600,700,950]#300,350,400,450,500,550,600,650,700,750,800,850,900,950]
# Date = datetime.datetime(year,month,day,hour)

land_mask_file = '{}input_data/lsm_2023-1.nc'.format(ar_dir)
land_mask_data = xr.open_dataset(land_mask_file)['lsm'].sel(latitude=slice(-20,-90)).values[0]

ga = land_mask_data.copy()
zeros = 0*ga
SH = zeros.copy()
SH[40:,:] = 1
day_delta = datetime.timedelta(days=1)

DATE_list = []
for h in range(len(ar_event_date))[::1]:
    D = datetime.datetime.strptime(ar_event_date[h]+' 00:00:00','%Y-%m-%d %H:%M:%S')
    print(type(D))
    Date = datetime.datetime(D.year,D.month,D.day,D.hour)-day_delta

    delta = 72
    for dt in range(int(delta))[::3]:
        try:
            # ここに実行したい処理を書く
            DATE = Date+datetime.timedelta(hours=dt)
            eDate = datetime.datetime(DATE.year,DATE.month,DATE.day,DATE.hour)+datetime.timedelta(hours=24+1)
            print(DATE)
            if DATE in DATE_list:
                print('Already done')
                continue
            DATE_list.append(DATE)
            
            syear = (DATE+datetime.timedelta(hours=1)).year
            smonth = (DATE+datetime.timedelta(hours=1)).month
            sday = (DATE+datetime.timedelta(hours=1)).day
            shour = (DATE+datetime.timedelta(hours=1)).hour
            eyear = eDate.year
            emonth = eDate.month
            eday = eDate.day
            ehour = eDate.hour

            year = DATE.year
            month = DATE.month
            day = DATE.day
            hour = DATE.hour
        
            out = '/Users/takahashikazu/Desktop/NIPR/fig/'
            
            lat,lon = -77.3, 39.7#position(year,month,day,hour)
            lat_syowa,lon_syowa = -69.007, 39.579#position(year,month,day,hour)
            # print(lat,lon)
            # ind1 = np.where(wsDate>='{}-{:0=2}-{:0=2} {:0=2}:00:00'.format(year,month,day,hour))[0][0]
            # ship_u = su[ind1]
            # ship_v = sv[ind1]

            inputdata_dir = '/Volumes/Pegasus32 R6/takahashi/era5/'
            
            if not smonth == emonth:
                pr_file = ['{}data/surface/pr/pr_{}{:0=2}.nc'.format(inputdata_dir,syear,smonth),'{}data/surface/pr/pr_{}{:0=2}.nc'.format(inputdata_dir,eyear,emonth)]
                dataset = xr.open_mfdataset(pr_file)['tp'].sel(latitude=slice(-20,-90))
            else:
                pr_file = '{}data/surface/pr/pr_{}{:0=2}.nc'.format(inputdata_dir,syear,smonth)
                dataset = xr.open_dataset(pr_file)['tp'].sel(latitude=slice(-20,-90))
            pr = dataset.sel(valid_time=slice('{:0=4}-{:0=2}-{:0=2}-{:0=2}'.format(syear,smonth,sday,shour),'{:0=4}-{:0=2}-{:0=2}-{:0=2}'.format(eyear,emonth,eday,ehour))).sum(axis=0).values*1000
            if not smonth == emonth:
                e_file = ['{}data/surface/e/e_{}{:0=2}.nc'.format(inputdata_dir,syear,smonth),'{}data/surface/pr/pr_{}{:0=2}.nc'.format(inputdata_dir,eyear,emonth)]
                dataset = xr.open_mfdataset(e_file)['e'].sel(latitude=slice(-20,-90))
            else:
                e_file = '{}data/surface/e/e_{}{:0=2}.nc'.format(inputdata_dir,syear,smonth)
                dataset = xr.open_dataset(e_file)['e'].sel(latitude=slice(-20,-90))
            e = dataset.sel(valid_time=slice('{:0=4}-{:0=2}-{:0=2}-{:0=2}'.format(syear,smonth,sday,shour),'{:0=4}-{:0=2}-{:0=2}-{:0=2}'.format(eyear,emonth,eday,ehour))).sum(axis=0).values*1000
            file = '{}data/surface/2t/2t_{}{:0=2}.nc'.format(inputdata_dir,syear,smonth)
            dataset = xr.open_dataset(file)['t2m'].sel(latitude=slice(-20,-90))
            t2m = dataset.sel(time='{:0=4}-{:0=2}-{:0=2}-{:0=2}'.format(syear,smonth,sday,shour)).values
            # print(pr.shape)

            mask_inv_dir = '/Volumes/Pegasus32 R8/takahashi/era5/ar_out_3d_monthly_threshold/mask_data/'
            mask_dir = '{}ar_out_era5_SH_20S/ar_ivt_data/'.format(inputdata_dir)
            
            if year >= 2024:    
                file = '{}select_AR_mask-{}{:0=2}.nc'.format(mask_dir,year,month)
            else:
                file = '{}select_AR_mask-{}{:0=2}.nc'.format(mask_dir,year,month)
            file_inv = '{}select_AR_mask_num-{}-{}.nc'.format(mask_inv_dir,year,month)
            dataset = xr.open_dataset(file)
            dataset_inv = xr.open_dataset(file_inv)
            if year >= 2024:
                ar_mask_dataset = dataset['ar_mask'].sel(valid_time='{}-{:0=2}-{:0=2}-{:0=2}'.format(year,month,day,hour)).sel(latitude=slice(-20,-90))
            else:
                ar_mask_dataset = dataset['ar_mask'].sel(time='{}-{:0=2}-{:0=2}-{:0=2}'.format(year,month,day,hour)).sel(latitude=slice(-20,-90))
            ar_inv_mask_dataset = dataset_inv['ar_mask'].sel(time='{}-{:0=2}-{:0=2}-{:0=2}'.format(year,month,day,hour)).sel(latitude=slice(-20,-90))
            ar_inv_mask_dataset.values[ar_inv_mask_dataset.values==np.nan] = 0
            ar_num_mask = ar_inv_mask_dataset.values

            pres = ar_inv_mask_dataset.level
            pres_data = (ar_mask_dataset.values*0+1)*pres.values[:,None,None]
            sp_file = '{}data/surface/sp/sp_{}{:0=2}.nc'.format(inputdata_dir,year,month)
            dataset = xr.open_dataset(sp_file)
            sp = dataset['sp'].sel(latitude=slice(-20,-90)).sel(time='{}-{:0=2}-{:0=2}-{:0=2}'.format(year,month,day,hour)).values/100
            sp_data = (ar_inv_mask_dataset.values*0+1)*sp[None,:,:]-100
            ar_test = ar_inv_mask_dataset.values.copy()
            ar_test[pres_data<sp_data] = 0
            ob_list = np.unique(ar_test)[~(np.unique(ar_test)==0)]
            # print(ob_list)

            masked = np.empty_like(ar_test)

            valid_values = ob_list
            masked = np.where(np.isin(ar_inv_mask_dataset.values, valid_values), ar_inv_mask_dataset.values, 0)
            ar_inv_mask_dataset.values = np.where(masked>0,1,0)

            ar_inv_mask_dataset.values = np.where(ar_inv_mask_dataset.values>0,1,0)
            ar_mask = ar_mask_dataset.values
            ar_inv_mask = ar_inv_mask_dataset.values#.max(axis=0)
            pressure = ar_inv_mask_dataset.level.values
            pressure_arr = (ar_inv_mask*0+1)*pressure[:,None,None]
            # print(ar_inv_mask_dataset.sel(latitude=slice(-20,-90)).values.max())
            longitude = ar_mask_dataset.longitude.values
            latitude = ar_mask_dataset.latitude.values
            # print(ar_mask.max())
            
            mask = (ar_inv_mask > 0.5)  # 例: 0/1 マスク

            numerator = np.sum(pressure_arr * mask, axis=0)
            denominator = np.sum(mask, axis=0)

            # division by zero 対策
            centroid_z = np.where(denominator > 0, numerator / denominator, np.nan)
            
            # file_inv = '{}AR_candidates1_mask-{}-{}.nc'.format(mask_inv_dir,year,month)
            # dataset_inv = xr.open_dataset(file_inv)
            # ar_inv_mask_dataset = dataset_inv['ar_mask'].sel(time='{}-{:0=2}-{:0=2}-{:0=2}'.format(year,month,day,hour))
            # ar_inv_mask_dataset.values[ar_inv_mask_dataset.values==np.nan] = 0
            # ar_can1_mask = ar_inv_mask_dataset.sel(latitude=slice(-20,-90)).values

            # mask = (ar_can1_mask > 0.5)  # 例: 0/1 マスク

            # numerator = np.sum(pressure_arr * mask, axis=0)
            # denominator = np.sum(mask, axis=0)

            # ar_candidate = np.where((ar_can1_mask.max(axis=0)-ar_inv_mask.max(axis=0))>0,1,0)
            # # division by zero 対策
            # centroid_z_can1 = np.where(denominator > 0, numerator / denominator, np.nan)
            # centroid_z_can1[ar_candidate==0] = np.nan



            data_dir = '/Volumes/Pegasus32 R6/takahashi/era5/data/'
            
            # if variable == 'pr':
            file_q = '{}pressure/q/q_{}{:0=2}.nc'.format(data_dir,year,month)
            file_u = '{}pressure/u/u_{}{:0=2}.nc'.format(data_dir,year,month)
            file_v = '{}pressure/v/v_{}{:0=2}.nc'.format(data_dir,year,month)
            
            if year >= 2024:
                q = xr.open_dataset(file_q)['q'].sel(pressure_level=slice(500,300)).sel(valid_time='{}-{}-{}-{:0=2}'.format(year,month,day,hour)).sel(latitude=slice(-20,-90)).values[::-1,:,:]
                u = xr.open_dataset(file_u)['u'].sel(pressure_level=slice(500,300)).sel(valid_time='{}-{}-{}-{:0=2}'.format(year,month,day,hour)).sel(latitude=slice(-20,-90)).values[::-1,:,:]
                v = xr.open_dataset(file_v)['v'].sel(pressure_level=slice(500,300)).sel(valid_time='{}-{}-{}-{:0=2}'.format(year,month,day,hour)).sel(latitude=slice(-20,-90)).values[::-1,:,:]
                plev = xr.open_dataset(file_v)['v'].pressure_level.values[::-1]
            else:
                q = xr.open_dataset(file_q)['q'].sel(level=slice(300,500)).sel(time='{}-{}-{}-{:0=2}'.format(year,month,day,hour)).sel(latitude=slice(-20,-90)).values
                u = xr.open_dataset(file_u)['u'].sel(level=slice(300,500)).sel(time='{}-{}-{}-{:0=2}'.format(year,month,day,hour)).sel(latitude=slice(-20,-90)).values
                v = xr.open_dataset(file_v)['v'].sel(level=slice(300,500)).sel(time='{}-{}-{}-{:0=2}'.format(year,month,day,hour)).sel(latitude=slice(-20,-90)).values
                plev = xr.open_dataset(file_v)['v'].sel(level=slice(300,500)).level.values
                
            pfac = 100
            g = 9.80665 
            nl,ny,nx = q.shape
            y_int =np.zeros((ny,nx))
            y1 = q*u
            y2 = q*v
            
            y1_m = ((np.roll(y1,-1,axis=1) + y1)*0.5)[:-1]
            y2_m = ((np.roll(y2,-1,axis=1) + y2)*0.5)[:-1]
            level_dif = (np.roll(plev,-1) - plev)[:-1]
            # print(nl,ny,nx,level_dif)
            

            y1_int = (((y1_m*level_dif[:,np.newaxis,np.newaxis]).sum(axis=0))/g*pfac)
            y2_int = (((y2_m*level_dif[:,np.newaxis,np.newaxis]).sum(axis=0))/g*pfac)
            
            ivt500 = np.power((np.power(y1_int,2) + np.power(y2_int,2)),1/2)

            # else:            
            file_vivt = '{}input_data/n-ivt-{}-{}.nc'.format(inputdata_dir,year,month)
            file_uivt = '{}input_data/e-ivt-{}-{}.nc'.format(inputdata_dir,year,month)
            file_iwv = '{}input_data/IWV-{}-{}.nc'.format(inputdata_dir,year,month)

            if year >= 2024:
                iwv = xr.open_dataset(file_iwv)['IWV'].sel(valid_time='{}-{}-{}-{:0=2}'.format(year,month,day,hour)).sel(latitude=slice(-20,-90)).values
                vivt = xr.open_dataset(file_vivt)['VIVT'].sel(valid_time='{}-{}-{}-{:0=2}'.format(year,month,day,hour)).sel(latitude=slice(-20,-90)).values
                uivt = xr.open_dataset(file_uivt)['UIVT'].sel(valid_time='{}-{}-{}-{:0=2}'.format(year,month,day,hour)).sel(latitude=slice(-20,-90)).values
            else:
                iwv = xr.open_dataset(file_iwv)['IWV'].sel(time='{}-{}-{}-{:0=2}'.format(year,month,day,hour)).sel(latitude=slice(-20,-90)).values
                vivt = xr.open_dataset(file_vivt)['VIVT'].sel(time='{}-{}-{}-{:0=2}'.format(year,month,day,hour)).sel(latitude=slice(-20,-90)).values
                uivt = xr.open_dataset(file_uivt)['UIVT'].sel(time='{}-{}-{}-{:0=2}'.format(year,month,day,hour)).sel(latitude=slice(-20,-90)).values
            
            # if variable == 'pr':
            uivt500 = y1_int
            vivt500 = y1_int

            # if region == 'SH':
            surface_data = '{}surface/'.format(data_dir)
            msl_file = '{}msl/msl_{}{:0=2}.nc'.format(surface_data,year,month)
            dataset = xr.open_dataset(msl_file)['msl'].sel(time='{}-{:0=2}-{:0=2}-{:0=2}'.format(year,month,day,hour)).sel(latitude=slice(-20,-90))
            msl = dataset.values/100

            dir = '{}pressure/z/z_{}{:0=2}.nc'.format(data_dir,year,month)
            dataset = xr.open_dataset(dir)['z'].sel(level=500).sel(latitude=slice(-20,-90))
            z500 = dataset.sel(time='{}-{:0=2}-{:0=2}-{:0=2}'.format(year,month,day,hour)).values/9.80665
            # dataset = xr.open_dataset(dir)['z'].sel(level=300).sel(latitude=slice(-20,-90))
            # z300 = dataset.sel(time='{}-{:0=2}-{:0=2}-{:0=2}'.format(year,month,day,hour)).values/9.80665

            # mt = ivt
            # mt_mean = mt.mean(axis=1)
            # mt_anom = mt - mt_mean[:,None]
            # mt_max = mt.max(axis=1)
            # mt_max_anom = mt_max - mt_mean
            # mt_fac = mt_anom/mt_max_anom[:,None]

            # south = vivt.copy()
            # south[vivt>0] = 0
            # south[vivt<0] = 1
            ar = ar_mask

            iwv, lon_cyc_v = util.add_cyclic_point(iwv,coord=longitude)
            pr, lon_cyc_v = util.add_cyclic_point(pr,coord=longitude)
            e, lon_cyc_v = util.add_cyclic_point(e,coord=longitude)
            t2, lon_cyc_v = util.add_cyclic_point(t2m,coord=longitude)
            # mt_fac, lon_cyc_v = util.add_cyclic_point(mt_fac,coord=longitude)
            # iwv, lon_cyc_v = util.add_cyclic_point(iwv,coord=longitude)
            # theta_eq, lon_cyc_v = util.add_cyclic_point(theta_eq,coord=longitude)
            # theta_eq_m, lon_cyc_v = util.add_cyclic_point(theta_eq_m,coord=longitude)
            # theta, lon_cyc_v = util.add_cyclic_point(theta,coord=longitude)
            # vivt, lon_cyc_v = util.add_cyclic_point(vivt,coord=longitude)
            # uivt, lon_cyc_v = util.add_cyclic_point(uivt,coord=longitude)
            # t, lon_cyc_v = util.add_cyclic_point(t,coord=longitude)
            # q, lon_cyc_v = util.add_cyclic_point(q,coord=longitude)
            # mt, lon_cyc_v = util.add_cyclic_point(mt,coord=longitude)
            # ivt, lon_cyc_v = util.add_cyclic_point(ivt,coord=longitude)
            ar_mask, lon_cyc_v = util.add_cyclic_point(ar_mask,coord=longitude)
            ar_inv_mask, lon_cyc_v = util.add_cyclic_point(ar_inv_mask.max(axis=0),coord=longitude)
            ar_num_mask, lon_cyc_v = util.add_cyclic_point(ar_num_mask.max(axis=0),coord=longitude)
            # ar_can1_mask, lon_cyc_v = util.add_cyclic_point(ar_candidate,coord=longitude)
            # if region == 'Fuji':
            centroid_z, lon_cyc_v = util.add_cyclic_point(centroid_z,coord=longitude)
            # centroid_z_can1, lon_cyc_v = util.add_cyclic_point(centroid_z_can1,coord=longitude)
            # south, lon_cyc_v = util.add_cyclic_point(south,coord=longitude)
            # if region == 'SH':
            z5, longitude_c = util.add_cyclic_point(z500,coord=longitude)
            # z3, longitude_c = util.add_cyclic_point(z300,coord=longitude)
            msl, longitude_c = util.add_cyclic_point(msl,coord=longitude)
            # w, longitude_c = util.add_cyclic_point(w,coord=longitude)

            center, radius = [0.5,0.5],0.5
            theta = np.linspace(0,2*np.pi,100)
            verts = np.vstack([np.sin(theta),np.cos(theta)]).T
            circle = mpath.Path(verts*radius+center)


                
            # else:
            # ax2 = fig.add_subplot(111,projection=ccrs.AlbersEqualArea(central_longitude=(l1+l2)/2,central_latitude=-65,
            #                                                             standard_parallels=(-65)))
            #ax2.set_extent([-20,80,-70,-30],ccrs.PlateCarree())
            #ax2.set_extent([lon1,lon2,lat1,lat2],ccrs.PlateCarree())
            #ax2.set_extent([lon1,lon2,lat1,lat2],ccrs.PlateCarree())
            # if Date in [datetime.datetime(2022,12,9,2),datetime.datetime(2023,2,22,14)]:
                # ax2.set_extent([60,120,-85,-60],ccrs.PlateCarree())
            # else:
            # ax1.set_extent([center-40,center+40,-80,-30],ccrs.PlateCarree())


            # levels=np.arange(0,1001,20)
            hgt_levels = np.arange(4000,6001,100)
            msl_levels = np.arange(930,1050,10)
            t_levels = np.arange(-50,31,2)
            # cf = ax1.contourf(longitude_c,latitude,ivt,transform=ccrs.PlateCarree(),cmap=iwvcmap,levels=levels,extend='max')
            # cbar = plt.colorbar(cf,ax=ax1,shrink=0.7,orientation='horizontal')
            # cbar.set_label(r'IVT [$kg\,m^{-1}s^{-1}$]',fontsize=10)

            # levels=np.arange(0,31,2)
            # cf = ax1.contourf(longitude_c,latitude,iwv,transform=ccrs.PlateCarree(),cmap='Blues',levels=levels,extend='max')
            # cbar = plt.colorbar(cf,ax=ax1,shrink=0.7,orientation='horizontal')
            # cbar.set_label(r'IWV [$kg\,m^{-2}$]',fontsize=10)

            # levels=np.arange(-1,1.1,0.1)
            # cf = ax1.contourf(longitude_c,latitude,w,transform=ccrs.PlateCarree(),cmap='coolwarm_r',levels=levels,extend='both',alpha=0.8)
            # cbar = plt.colorbar(cf,ax=ax1,shrink=0.7,orientation='horizontal')
            # cbar.set_label(r'Vertical velocity [$Pa\,s^{-1}$]',fontsize=10)

            # cf = ax1.contourf(lon_cyc_v,latitude,centroid_z,transform=ccrs.PlateCarree(),cmap='ocean_r',levels=levels,alpha=0.5,zorder=2)
            region = 'Fuji'
            variable = 'level'

            if event == 'extreme':
                if not os.path.exists('{}AR_Detection_3d_monthly/AR03_fig_monthly/exprec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month)):#,Date.year,Date.month)):
                    os.makedirs('{}AR_Detection_3d_monthly/AR03_fig_monthly/exprec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month))#,Date.year,Date.month))
                out_dir = '{}AR_Detection_3d_monthly/AR03_fig_monthly/exprec_DF/{}_{}/{}{:0=2}/'.format(out,region,variable,year,month)#,Date.year,Date.month)
            elif event == 'AR':
                if not os.path.exists('{}AR_Detection_3d_monthly/AR03_fig_monthly/AR_prec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month)):#,Date.year,Date.month)):
                    os.makedirs('{}AR_Detection_3d_monthly/AR03_fig_monthly/AR_prec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month))#,Date.year,Date.month))
                out_dir = '{}AR_Detection_3d_monthly/AR03_fig_monthly/AR_prec_DF/{}_{}/{}{:0=2}/'.format(out,region,variable,year,month)#,Date.year,Date.month)

            fig = plt.figure(figsize=[8,8])
            ax1 = fig.add_subplot(111,projection=ccrs.LambertConformal(central_longitude=   center_lon,central_latitude=-50,
                                                                        standard_parallels=(-60,-30),cutoff=-10))
            ax1.set_extent([center_lon-45,center_lon+45,-85,-55],ccrs.PlateCarree())
                
            levels=np.arange(300,1001,50)
            cf1 = ax1.pcolormesh(
                lon_cyc_v, latitude, centroid_z,
                transform=ccrs.PlateCarree(),
                cmap='ocean_r',
                shading='auto',
                vmin=min(levels),
                vmax=max(levels),
                alpha=0.7,
                zorder=2
            )
            # cf2 = ax1.pcolormesh(
            #     lon_cyc_v, latitude, centroid_z_can1,
            #     transform=ccrs.PlateCarree(),
            #     cmap='ocean_r',
            #     shading='auto',
            #     vmin=min(levels),
            #     vmax=max(levels),
            #     alpha=0.7,
            #     zorder=2
            # )
            cbar = plt.colorbar(cf1,ax=ax1,shrink=0.7,orientation='horizontal',aspect=30,pad=0.02,anchor=(0.3,1.0))
            cbar.set_label(r'Mean pressure [hPa]',fontsize=10)

            uivt_land = np.where(land_mask_data==1,uivt,np.nan)
            vivt_land = np.where(land_mask_data==1,vivt,np.nan)
            uivt_ocean = np.where(land_mask_data==0,uivt,np.nan)
            vivt_ocean = np.where(land_mask_data==0,vivt,np.nan)
            qv2 = ax1.quiver(longitude[::3],latitude[::3],(uivt_ocean)[::3,::3],(vivt_ocean)[::3,::3],transform=ccrs.PlateCarree(),color='lightgray',width=.003,scale=15000.,alpha=1.0,zorder=4)
            ax1.quiverkey(qv2,0.95,-0.05,500,r'IVT 500 [$kg\,m^{-1}s^{-1}$]',color='lightgray',labelpos='S',labelsep=0.03,fontproperties={'size':8})
            qv3 = ax1.quiver(longitude[::3],latitude[::3],(uivt_land*SH)[::3,::3],(vivt_land*SH)[::3,::3],transform=ccrs.PlateCarree(),color='black',width=.003,scale=800.,alpha=1.0,zorder=4)
            ax1.quiverkey(qv3,0.95,-0.1,25,r'IVT 25 [$kg\,m^{-1}s^{-1}$]',color='k',labelpos='S',labelsep=0.03,fontproperties={'size':8})

            ax1.coastlines(resolution="10m", color="black",alpha=1.0,zorder=1)
            gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle=':', ylocs=range(-90,11,10), xlocs=range(-180,181,30))
            
            linestyle = '-'

            cf3 = ax1.contour(lon_cyc_v,latitude,ar_mask,transform=ccrs.PlateCarree(),linewidths=2,colors='yellow',levels=[0.9,1.0],zorder=3,linestyles=linestyle)
            cf3 = ax1.contour(lon_cyc_v,latitude,ar_inv_mask,transform=ccrs.PlateCarree(),linewidths=2,colors='red',levels=[0.9,1.0],zorder=3,linestyles=linestyle)
            ship = ax1.scatter(lon,lat,color='k',transform=ccrs.PlateCarree(),s=75,zorder=5,marker='*')#facecolor='white',edgecolor='black'                        
            
            gl1.xlocator = ticker.FixedLocator(np.arange(-180,181,30))
            gl1.ylocator = ticker.FixedLocator(np.arange(-90,90,10))

            ax1.set_title('{}-{:0=2}-{:0=2} {:0=2}UTC'.format(year,month,day,hour))

            plt.tight_layout()
            plt.savefig('{}{}{:0=2}{:0=2}_{:0=2}z.png'.format(out_dir,year,month,day,hour),dpi=700)
            plt.close()

            variable = 'pr'

            if event == 'extreme':
                if not os.path.exists('{}AR_Detection_3d_monthly/AR03_fig_monthly/exprec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month)):#,Date.year,Date.month)):
                    os.makedirs('{}AR_Detection_3d_monthly/AR03_fig_monthly/exprec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month))#,Date.year,Date.month))
                out_dir = '{}AR_Detection_3d_monthly/AR03_fig_monthly/exprec_DF/{}_{}/{}{:0=2}/'.format(out,region,variable,year,month)#,Date.year,Date.month)
            elif event == 'AR':
                if not os.path.exists('{}AR_Detection_3d_monthly/AR03_fig_monthly/AR_prec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month)):#,Date.year,Date.month)):
                    os.makedirs('{}AR_Detection_3d_monthly/AR03_fig_monthly/AR_prec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month))#,Date.year,Date.month))
                out_dir = '{}AR_Detection_3d_monthly/AR03_fig_monthly/AR_prec_DF/{}_{}/{}{:0=2}/'.format(out,region,variable,year,month)#,Date.year,Date.month)

            fig = plt.figure(figsize=[8,8])
            ax1 = fig.add_subplot(111,projection=ccrs.LambertConformal(central_longitude=   center_lon,central_latitude=-50,
                                                                        standard_parallels=(-60,-30),cutoff=-10))
            ax1.set_extent([center_lon-45,center_lon+45,-85,-55],ccrs.PlateCarree())

            levels = list(np.linspace(0.01,0.1,10))+list(np.linspace(0.02,0.1,9)*10)+list(np.linspace(0.02,0.1,9)*100)#+list(np.linspace(0.02,0.1,9)*1000)+list(np.linspace(0.02,0.1,9)*10000)+list(np.linspace(0.02,0.1,9)*100000)
            cf1 = ax1.contourf(lon_cyc_v,latitude,pr,transform=ccrs.PlateCarree(),levels=levels,cmap='Blues',extend='max',norm=LogNorm(),alpha=0.9)
            cbar = plt.colorbar(cf1,ax=ax1,shrink=0.7,orientation='horizontal',pad=0.02,anchor=(0.3,1.0),aspect=30)
            cbar.set_label(r'24h Precipitation [mm]',fontsize=10)
            ticks = [0.01,0.1,1,10]
            labels = [r'$10^{-2}$',r'$10^{-1}$',r'$10^{0}$',r'$10^{1}$']
            cbar.set_ticks(ticks,labels=labels)
            
            # ct_levels = np.linspace(0,1,21)
            # ct = ax1.contour(lon_cyc_v,latitude,e,transform=ccrs.PlateCarree(),levels=ct_levels,linewidths=0.5,colors='black')
            # ax1.clabel(ct)
            
            uivt_land = np.where(land_mask_data==1,uivt500,np.nan)
            vivt_land = np.where(land_mask_data==1,vivt500,np.nan)
            uivt_ocean = np.where(land_mask_data==0,uivt500,np.nan)
            vivt_ocean = np.where(land_mask_data==0,vivt500,np.nan)
            qv2 = ax1.quiver(longitude[::3],latitude[::3],(uivt_ocean)[::3,::3],(vivt_ocean)[::3,::3],transform=ccrs.PlateCarree(),color='lightgray',width=.003,scale=300.,alpha=1.0,zorder=4)
            ax1.quiverkey(qv2,0.95,-0.05,20,r'$IVT_{500}$ 20 [$kg\,m^{-1}s^{-1}$]',color='lightgray',labelpos='S',labelsep=0.03,fontproperties={'size':8})
            qv3 = ax1.quiver(longitude[::3],latitude[::3],(uivt_land*SH)[::3,::3],(vivt_land*SH)[::3,::3],transform=ccrs.PlateCarree(),color='black',width=.003,scale=80.,alpha=1.0,zorder=4)
            ax1.quiverkey(qv3,0.95,-0.1,5,r'$IVT_{500}$ 5 [$kg\,m^{-1}s^{-1}$]',color='k',labelpos='S',labelsep=0.03,fontproperties={'size':8})

            ax1.coastlines(resolution="10m", color="black",alpha=1.0,zorder=1)
            gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle=':', ylocs=range(-90,11,10), xlocs=range(-180,181,30))
            
            linestyle = '-'

            cf3 = ax1.contour(lon_cyc_v,latitude,ar_mask,transform=ccrs.PlateCarree(),linewidths=2,colors='yellow',levels=[0.9,1.0],zorder=3,linestyles=linestyle)
            cf3 = ax1.contour(lon_cyc_v,latitude,ar_inv_mask,transform=ccrs.PlateCarree(),linewidths=2,colors='red',levels=[0.9,1.0],zorder=3,linestyles=linestyle)
            ship = ax1.scatter(lon,lat,color='k',transform=ccrs.PlateCarree(),s=75,zorder=5,marker='*')#facecolor='white',edgecolor='black'                        
            
            gl1.xlocator = ticker.FixedLocator(np.arange(-180,181,30))
            gl1.ylocator = ticker.FixedLocator(np.arange(-90,90,10))

            ax1.set_title('{}-{:0=2}-{:0=2} {:0=2}UTC'.format(year,month,day,hour))

            plt.tight_layout()
            plt.savefig('{}{}{:0=2}{:0=2}_{:0=2}z.png'.format(out_dir,year,month,day,hour),dpi=700)
            plt.close()
            
            variable = 'iwv'

            if event == 'extreme':
                if not os.path.exists('{}AR_Detection_3d_monthly/AR03_fig_monthly/exprec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month)):#,Date.year,Date.month)):
                    os.makedirs('{}AR_Detection_3d_monthly/AR03_fig_monthly/exprec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month))#,Date.year,Date.month))
                out_dir = '{}AR_Detection_3d_monthly/AR03_fig_monthly/exprec_DF/{}_{}/{}{:0=2}/'.format(out,region,variable,year,month)#,Date.year,Date.month)
            elif event == 'AR':
                if not os.path.exists('{}AR_Detection_3d_monthly/AR03_fig_monthly/AR_prec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month)):#,Date.year,Date.month)):
                    os.makedirs('{}AR_Detection_3d_monthly/AR03_fig_monthly/AR_prec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month))#,Date.year,Date.month))
                out_dir = '{}AR_Detection_3d_monthly/AR03_fig_monthly/AR_prec_DF/{}_{}/{}{:0=2}/'.format(out,region,variable,year,month)#,Date.year,Date.month)

            fig = plt.figure(figsize=[8,8])
            ax1 = fig.add_subplot(111,projection=ccrs.LambertConformal(central_longitude=   center_lon,central_latitude=-50,
                                                                        standard_parallels=(-60,-30),cutoff=-10))
            ax1.set_extent([center_lon-45,center_lon+45,-85,-55],ccrs.PlateCarree())

            levels = list(np.linspace(0.01,0.1,9)*10)+list(np.linspace(0.02,0.1,9)*100)#+list(np.linspace(0.02,0.1,9)*1000)#+list(np.linspace(0.02,0.1,9)*10000)#+list(np.linspace(0.02,0.1,9)*100000)
            cf1 = ax1.contourf(lon_cyc_v,latitude,iwv,transform=ccrs.PlateCarree(),levels=levels,cmap='Blues',extend='max',norm=LogNorm(),alpha=0.9)
            cbar = plt.colorbar(cf1,ax=ax1,shrink=0.7,orientation='horizontal',pad=0.02,anchor=(0.3,1.0),aspect=30)
            cbar.set_label(r'Integrated Water Vapor [$kg\,m^{-2}$]',fontsize=10)
            ticks = [0.1,1,10]#,100]
            labels = [r'$10^{-1}$',r'$10^{0}$',r'$10^{1}$']#,r'$10^{2}$']
            cbar.set_ticks(ticks,labels=labels)
                                    
            # ct_levels = np.linspace(0,1,21)
            # ct = ax1.contour(lon_cyc_v,latitude,e,transform=ccrs.PlateCarree(),levels=ct_levels,linewidths=0.5,colors='black')
            # ax1.clabel(ct)

            uivt_land = np.where(land_mask_data==1,uivt,np.nan)
            vivt_land = np.where(land_mask_data==1,vivt,np.nan)
            uivt_ocean = np.where(land_mask_data==0,uivt,np.nan)
            vivt_ocean = np.where(land_mask_data==0,vivt,np.nan)
            qv2 = ax1.quiver(longitude[::3],latitude[::3],(uivt_ocean)[::3,::3],(vivt_ocean)[::3,::3],transform=ccrs.PlateCarree(),color='lightgray',width=.003,scale=15000.,alpha=1.0,zorder=4)
            ax1.quiverkey(qv2,0.95,-0.05,500,r'IVT 500 [$kg\,m^{-1}s^{-1}$]',color='lightgray',labelpos='S',labelsep=0.03,fontproperties={'size':8})
            qv3 = ax1.quiver(longitude[::3],latitude[::3],(uivt_land*SH)[::3,::3],(vivt_land*SH)[::3,::3],transform=ccrs.PlateCarree(),color='black',width=.003,scale=800.,alpha=1.0,zorder=4)
            ax1.quiverkey(qv3,0.95,-0.1,25,r'IVT 25 [$kg\,m^{-1}s^{-1}$]',color='k',labelpos='S',labelsep=0.03,fontproperties={'size':8})

            
            ax1.coastlines(resolution="10m", color="black",alpha=1.0,zorder=1)
            gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle=':', ylocs=range(-90,11,10), xlocs=range(-180,181,30))
            
            linestyle = '-'

            cf3 = ax1.contour(lon_cyc_v,latitude,ar_mask,transform=ccrs.PlateCarree(),linewidths=2,colors='yellow',levels=[0.9,1.0],zorder=3,linestyles=linestyle)
            cf3 = ax1.contour(lon_cyc_v,latitude,ar_inv_mask,transform=ccrs.PlateCarree(),linewidths=2,colors='red',levels=[0.9,1.0],zorder=3,linestyles=linestyle)
            ship = ax1.scatter(lon,lat,color='k',transform=ccrs.PlateCarree(),s=75,zorder=5,marker='*')#facecolor='white',edgecolor='black'                        
            
            gl1.xlocator = ticker.FixedLocator(np.arange(-180,181,30))
            gl1.ylocator = ticker.FixedLocator(np.arange(-90,90,10))

            ax1.set_title('{}-{:0=2}-{:0=2} {:0=2}UTC'.format(year,month,day,hour))

            plt.tight_layout()
            plt.savefig('{}{}{:0=2}{:0=2}_{:0=2}z.png'.format(out_dir,year,month,day,hour),dpi=700)
            plt.close()


            variable = 'sf'

            if event == 'extreme':
                if not os.path.exists('{}AR_Detection_3d_monthly/AR03_fig_monthly/exprec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month)):#,Date.year,Date.month)):
                    os.makedirs('{}AR_Detection_3d_monthly/AR03_fig_monthly/exprec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month))#,Date.year,Date.month))
                out_dir = '{}AR_Detection_3d_monthly/AR03_fig_monthly/exprec_DF/{}_{}/{}{:0=2}/'.format(out,region,variable,year,month)#,Date.year,Date.month)
            elif event == 'AR':
                if not os.path.exists('{}AR_Detection_3d_monthly/AR03_fig_monthly/AR_prec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month)):#,Date.year,Date.month)):
                    os.makedirs('{}AR_Detection_3d_monthly/AR03_fig_monthly/AR_prec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month))#,Date.year,Date.month))
                out_dir = '{}AR_Detection_3d_monthly/AR03_fig_monthly/AR_prec_DF/{}_{}/{}{:0=2}/'.format(out,region,variable,year,month)#,Date.year,Date.month)

            fig = plt.figure(figsize=[8,8])
            ax1 = fig.add_subplot(111,projection=ccrs.LambertConformal(central_longitude=   center_lon,central_latitude=-50,
                                                                        standard_parallels=(-60,-30),cutoff=-10))
            ax1.set_extent([center_lon-45,center_lon+45,-85,-55],ccrs.PlateCarree())

            levels = np.arange(190,250.1,0.1)
            cf1 = ax1.contourf(lon_cyc_v,latitude,t2,transform=ccrs.PlateCarree(),levels=levels,cmap='nipy_spectral',extend='both',alpha=0.8)#,norm=LogNorm(),alpha=0.7)
            cbar = plt.colorbar(cf1,ax=ax1,shrink=0.7,orientation='horizontal',pad=0.02,anchor=(0.3,1.0),aspect=30)
            cbar.set_label(r'2m temperature [K]',fontsize=10)
            ticks = np.arange(190,250.1,10)
            # labels = [r'$10^{-1}$',r'$10^{0}$',r'$10^{1}$']
            cbar.set_ticks(ticks,labels=ticks)
                                    
            # ct_levels = np.linspace(0,1,21)
            # ct = ax1.contour(lon_cyc_v,latitude,e,transform=ccrs.PlateCarree(),levels=ct_levels,linewidths=0.5,colors='black')
            # ax1.clabel(ct)

            uivt_land = np.where(land_mask_data==1,uivt,np.nan)
            vivt_land = np.where(land_mask_data==1,vivt,np.nan)
            uivt_ocean = np.where(land_mask_data==0,uivt,np.nan)
            vivt_ocean = np.where(land_mask_data==0,vivt,np.nan)
            qv2 = ax1.quiver(longitude[::3],latitude[::3],(uivt_ocean)[::3,::3],(vivt_ocean)[::3,::3],transform=ccrs.PlateCarree(),color='lightgray',width=.003,scale=15000.,alpha=1.0,zorder=4)
            ax1.quiverkey(qv2,0.95,-0.05,500,r'IVT 500 [$kg\,m^{-1}s^{-1}$]',color='lightgray',labelpos='S',labelsep=0.03,fontproperties={'size':8})
            qv3 = ax1.quiver(longitude[::3],latitude[::3],(uivt_land*SH)[::3,::3],(vivt_land*SH)[::3,::3],transform=ccrs.PlateCarree(),color='black',width=.003,scale=800.,alpha=1.0,zorder=4)
            ax1.quiverkey(qv3,0.95,-0.1,25,r'IVT 25 [$kg\,m^{-1}s^{-1}$]',color='k',labelpos='S',labelsep=0.03,fontproperties={'size':8})

            ax1.coastlines(resolution="10m", color="black",alpha=1.0,zorder=1)
            gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle=':', ylocs=range(-90,11,10), xlocs=range(-180,181,30))
            
            linestyle = '-'

            cf3 = ax1.contour(lon_cyc_v,latitude,ar_mask,transform=ccrs.PlateCarree(),linewidths=2,colors='yellow',levels=[0.9,1.0],zorder=3,linestyles=linestyle)
            cf3 = ax1.contour(lon_cyc_v,latitude,ar_inv_mask,transform=ccrs.PlateCarree(),linewidths=2,colors='red',levels=[0.9,1.0],zorder=3,linestyles=linestyle)
            ship = ax1.scatter(lon,lat,color='k',transform=ccrs.PlateCarree(),s=75,zorder=5,marker='*')#facecolor='white',edgecolor='black'                        
            
            gl1.xlocator = ticker.FixedLocator(np.arange(-180,181,30))
            gl1.ylocator = ticker.FixedLocator(np.arange(-90,90,10))

            ax1.set_title('{}-{:0=2}-{:0=2} {:0=2}UTC'.format(year,month,day,hour))

            plt.tight_layout()
            plt.savefig('{}{}{:0=2}{:0=2}_{:0=2}z.png'.format(out_dir,year,month,day,hour),dpi=700)
            plt.close()

            
            region = 'SH'
            variable = 'pr'
                
            fig = plt.figure(figsize=[8,8])

            # if region == 'SH':
            ax1 = fig.add_subplot(111,projection=ccrs.SouthPolarStereo(central_longitude=270))
            ax1.set_extent([-180,180,-90,-40],ccrs.PlateCarree())

            if event == 'extreme':
                if not os.path.exists('{}AR_Detection_3d_monthly/AR03_fig_monthly/exprec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month)):#,Date.year,Date.month)):
                    os.makedirs('{}AR_Detection_3d_monthly/AR03_fig_monthly/exprec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month))#,Date.year,Date.month))
                out_dir = '{}AR_Detection_3d_monthly/AR03_fig_monthly/exprec_DF/{}_{}/{}{:0=2}/'.format(out,region,variable,year,month)#,Date.year,Date.month)
            elif event == 'AR':
                if not os.path.exists('{}AR_Detection_3d_monthly/AR03_fig_monthly/AR_prec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month)):#,Date.year,Date.month)):
                    os.makedirs('{}AR_Detection_3d_monthly/AR03_fig_monthly/AR_prec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month))#,Date.year,Date.month))
                out_dir = '{}AR_Detection_3d_monthly/AR03_fig_monthly/AR_prec_DF/{}_{}/{}{:0=2}/'.format(out,region,variable,year,month)#,Date.year,Date.month)


            ar_num_mask = np.where(ar_num_mask==0,np.nan,ar_num_mask)
            num = ax1.contourf(lon_cyc_v,latitude,ar_num_mask,transform=ccrs.PlateCarree(),cmap='jet',alpha=0.4)
            
            cfmsl = ax1.contour(lon_cyc_v,latitude,msl,transform=ccrs.PlateCarree(),linewidths=0.5,colors='black',levels=8,zorder=3,linestyles=linestyle)
            ax1.clabel(cfmsl)
            
            ax1.coastlines(resolution="10m", color="black",alpha=1.0,zorder=1)
            gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle=':', ylocs=range(-90,11,10), xlocs=range(-180,181,30))
            
            linestyle = '-'

            cf3 = ax1.contour(lon_cyc_v,latitude,ar_mask,transform=ccrs.PlateCarree(),linewidths=2,colors='yellow',levels=[0.9,1.0],zorder=3,linestyles=linestyle)
            cf3 = ax1.contour(lon_cyc_v,latitude,ar_inv_mask,transform=ccrs.PlateCarree(),linewidths=2,colors='red',levels=[0.9,1.0],zorder=3,linestyles=linestyle)
            ship = ax1.scatter(lon,lat,color='k',transform=ccrs.PlateCarree(),s=75,zorder=5,marker='*')#facecolor='white',edgecolor='black'                        
            
            gl1.xlocator = ticker.FixedLocator(np.arange(-180,181,30))
            gl1.ylocator = ticker.FixedLocator(np.arange(-90,90,10))

            ax1.set_boundary(circle,transform=ax1.transAxes)

            ax1.set_title('{}-{:0=2}-{:0=2} {:0=2}UTC'.format(year,month,day,hour))

            plt.tight_layout()
            plt.savefig('{}{}{:0=2}{:0=2}_{:0=2}z.png'.format(out_dir,year,month,day,hour),dpi=700)
            plt.close()
            
            variable = 'level'
                
            fig = plt.figure(figsize=[8,8])

            # if region == 'SH':
            ax1 = fig.add_subplot(111,projection=ccrs.SouthPolarStereo(central_longitude=270))
            ax1.set_extent([-180,180,-90,-40],ccrs.PlateCarree())

            if event == 'extreme':
                if not os.path.exists('{}AR_Detection_3d_monthly/AR03_fig_monthly/exprec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month)):#,Date.year,Date.month)):
                    os.makedirs('{}AR_Detection_3d_monthly/AR03_fig_monthly/exprec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month))#,Date.year,Date.month))
                out_dir = '{}AR_Detection_3d_monthly/AR03_fig_monthly/exprec_DF/{}_{}/{}{:0=2}/'.format(out,region,variable,year,month)#,Date.year,Date.month)
            elif event == 'AR':
                if not os.path.exists('{}AR_Detection_3d_monthly/AR03_fig_monthly/AR_prec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month)):#,Date.year,Date.month)):
                    os.makedirs('{}AR_Detection_3d_monthly/AR03_fig_monthly/AR_prec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month))#,Date.year,Date.month))
                out_dir = '{}AR_Detection_3d_monthly/AR03_fig_monthly/AR_prec_DF/{}_{}/{}{:0=2}/'.format(out,region,variable,year,month)#,Date.year,Date.month)

            
            levels=np.arange(300,1001,50)
            cf1 = ax1.pcolormesh(
                lon_cyc_v, latitude, centroid_z,
                transform=ccrs.PlateCarree(),
                cmap='winter_r',
                shading='auto',
                vmin=min(levels),
                vmax=max(levels),
                alpha=0.8,
                zorder=2
            )
            cbar = plt.colorbar(cf1,ax=ax1,shrink=0.7,orientation='horizontal',aspect=30,pad=0.15)#,anchor=(0.3,1.0))
            cbar.set_label(r'Mean pressure [hPa]',fontsize=10)
            
            cfz5 = ax1.contour(lon_cyc_v,latitude,z5,transform=ccrs.PlateCarree(),linewidths=0.5,colors='black',levels=8,zorder=3,linestyles=linestyle)
            # cfz3 = ax1.contour(lon_cyc_v,latitude,z3,transform=ccrs.PlateCarree(),linewidths=0.5,colors='black',levels=8,zorder=3,linestyles='--')
            ax1.clabel(cfz5)
            # ax1.clabel(cfz3)

            ax1.coastlines(resolution="10m", color="black",alpha=1.0,zorder=1)
            gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle=':', ylocs=range(-90,11,10), xlocs=range(-180,181,30))
            
            linestyle = '-'

            cf3 = ax1.contour(lon_cyc_v,latitude,ar_mask,transform=ccrs.PlateCarree(),linewidths=2,colors='yellow',levels=[0.9,1.0],zorder=3,linestyles=linestyle)
            cf3 = ax1.contour(lon_cyc_v,latitude,ar_inv_mask,transform=ccrs.PlateCarree(),linewidths=2,colors='red',levels=[0.9,1.0],zorder=3,linestyles=linestyle)
            ship = ax1.scatter(lon,lat,color='k',transform=ccrs.PlateCarree(),s=75,zorder=5,marker='*')#facecolor='white',edgecolor='black'

            gl1.xlocator = ticker.FixedLocator(np.arange(-180,181,30))
            gl1.ylocator = ticker.FixedLocator(np.arange(-90,90,10))

            ax1.set_boundary(circle,transform=ax1.transAxes)

            ax1.set_title('{}-{:0=2}-{:0=2} {:0=2}UTC'.format(year,month,day,hour))

            plt.tight_layout()
            plt.savefig('{}{}{:0=2}{:0=2}_{:0=2}z.png'.format(out_dir,year,month,day,hour),dpi=700)
            plt.close()
                
            variable = 'iwv'
                
            fig = plt.figure(figsize=[8,8])

            # if region == 'SH':
            ax1 = fig.add_subplot(111,projection=ccrs.SouthPolarStereo(central_longitude=270))
            ax1.set_extent([-180,180,-90,-40],ccrs.PlateCarree())


            if event == 'extreme':
                if not os.path.exists('{}AR_Detection_3d_monthly/AR03_fig_monthly/exprec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month)):#,Date.year,Date.month)):
                    os.makedirs('{}AR_Detection_3d_monthly/AR03_fig_monthly/exprec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month))#,Date.year,Date.month))
                out_dir = '{}AR_Detection_3d_monthly/AR03_fig_monthly/exprec_DF/{}_{}/{}{:0=2}/'.format(out,region,variable,year,month)#,Date.year,Date.month)
            elif event == 'AR':
                if not os.path.exists('{}AR_Detection_3d_monthly/AR03_fig_monthly/AR_prec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month)):#,Date.year,Date.month)):
                    os.makedirs('{}AR_Detection_3d_monthly/AR03_fig_monthly/AR_prec_DF/{}_{}/{}{:0=2}'.format(out,region,variable,year,month))#,Date.year,Date.month))
                out_dir = '{}AR_Detection_3d_monthly/AR03_fig_monthly/AR_prec_DF/{}_{}/{}{:0=2}/'.format(out,region,variable,year,month)#,Date.year,Date.month)

            
            levels = list(np.linspace(0.01,0.1,9)*10)+list(np.linspace(0.02,0.1,9)*100)+list(np.linspace(0.02,0.1,9)*1000)#+list(np.linspace(0.02,0.1,9)*10000)#+list(np.linspace(0.02,0.1,9)*100000)
            cf1 = ax1.contourf(lon_cyc_v,latitude,iwv,transform=ccrs.PlateCarree(),levels=levels,cmap='Blues',extend='max',norm=LogNorm(),alpha=0.7)
            cbar = plt.colorbar(cf1,ax=ax1,shrink=0.7,orientation='horizontal',pad=0.02,anchor=(0.3,1.0),aspect=30)
            cbar.set_label(r'Integrated Water Vapor [$kg\,m^{-2}$]',fontsize=10)
            ticks = [0.1,1,10,100]
            labels = [r'$10^{-1}$',r'$10^{0}$',r'$10^{1}$',r'$10^{2}$']
            cbar.set_ticks(ticks,labels=labels)
                                    
            ct_levels = np.linspace(0,1,21)
            ct = ax1.contour(lon_cyc_v,latitude,e,transform=ccrs.PlateCarree(),levels=ct_levels,linewidths=0.5,colors='black')
            ax1.clabel(ct)

            cfz5 = ax1.contour(lon_cyc_v,latitude,z5,transform=ccrs.PlateCarree(),linewidths=0.5,colors='black',levels=8,zorder=3,linestyles=linestyle)
            # cfz3 = ax1.contour(lon_cyc_v,latitude,z3,transform=ccrs.PlateCarree(),linewidths=0.5,colors='black',levels=8,zorder=3,linestyles='--')
            ax1.clabel(cfz5)
            # ax1.clabel(cfz3)

            ax1.coastlines(resolution="10m", color="black",alpha=1.0,zorder=1)
            gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle=':', ylocs=range(-90,11,10), xlocs=range(-180,181,30))
            
            linestyle = '-'

            cf3 = ax1.contour(lon_cyc_v,latitude,ar_mask,transform=ccrs.PlateCarree(),linewidths=2,colors='yellow',levels=[0.9,1.0],zorder=3,linestyles=linestyle)
            cf3 = ax1.contour(lon_cyc_v,latitude,ar_inv_mask,transform=ccrs.PlateCarree(),linewidths=2,colors='red',levels=[0.9,1.0],zorder=3,linestyles=linestyle)
            ship = ax1.scatter(lon,lat,color='k',transform=ccrs.PlateCarree(),s=75,zorder=5,marker='*')#facecolor='white',edgecolor='black'

            gl1.xlocator = ticker.FixedLocator(np.arange(-180,181,30))
            gl1.ylocator = ticker.FixedLocator(np.arange(-90,90,10))

            ax1.set_boundary(circle,transform=ax1.transAxes)

            ax1.set_title('{}-{:0=2}-{:0=2} {:0=2}UTC'.format(year,month,day,hour))

            plt.tight_layout()
            plt.savefig('{}{}{:0=2}{:0=2}_{:0=2}z.png'.format(out_dir,year,month,day,hour),dpi=700)
            plt.close()

            # levels=np.arange(0,0.11,0.01)
            # levels=np.arange(0,0.8,0.1)
            # cf = ax1.contourf(lon_cyc_v,latitude,mt_fac,transform=ccrs.PlateCarree(),cmap='ocean_r',levels=levels,extend='both',alpha=0.5,zorder=2)
            # cbar = plt.colorbar(cf,ax=ax1,shrink=0.7,orientation='horizontal')
            # cbar.set_label(r'Normalized zonal anomaly of IVT',fontsize=10)

            # levels=np.arange(260,331,5)
            # levels=np.arange(-30,31,5)
            # cf = ax1.contourf(longitude_c,latitude,theta_eq,transform=ccrs.PlateCarree(),cmap='turbo',levels=levels,extend='both')
            # cbar = plt.colorbar(cf,ax=ax1,shrink=0.7,orientation='horizontal')
            # cbar.set_label(r'Equivalent potential temperature [K]',fontsize=10)

            # cf4 = ax1.contour(longitude_c,latitude,theta_eq,transform=ccrs.PlateCarree(),levels=levels,linewidths=0.8,colors='k')
            # cf = ax1.contour(longitude_c,latitude,theta_eq,transform=ccrs.PlateCarree(),levels=[320],linewidths=1.2,colors='k')
            # cf4 = ax1.contour(longitude_c,latitude,theta_eq-theta_eq_m,transform=ccrs.PlateCarree(),levels=levels,linewidths=1,colors='k')
            # cf4 = ax1.contour(longitude_c,latitude,theta,transform=ccrs.PlateCarree(),levels=levels,linewidths=0.5,colors='k')
            # ax1.clabel(cf4,fontsize=7)

            # levels=np.arange(0,0.11,0.01)
            # levels=np.arange(-20,21,2.5)
            # cf = ax1.contourf(longitude_c,latitude,1000*mt_fac,transform=ccrs.PlateCarree(),cmap='turbo',levels=levels,extend='both')
            # cbar = plt.colorbar(cf,ax=ax1,shrink=0.7,orientation='horizontal')
            # # cbar.set_label(r'Zhu and Newell factor',fontsize=10)
            # cbar.set_label(r'Anomaly',fontsize=10)


            # levels=np.arange(0,10,0.2)
            # cf = ax1.contourf(longitude_c,latitude,q,transform=ccrs.PlateCarree(),cmap=iwvcmap,levels=levels,extend='max')
            # cbar = plt.colorbar(cf,ax=ax1,shrink=0.7,orientation='horizontal')
            # cbar.set_label(r'Specific Humidity [$g\,kg^{-1}$]',fontsize=10)

            # cf4 = ax1.contour(longitude_c,latitude,msl,transform=ccrs.PlateCarree(),levels=msl_levels,linewidths=0.5,colors='k')
            # ax1.clabel(cf4,fontsize=7)

            # cf4 = ax1.contour(longitude_c,latitude,z,transform=ccrs.PlateCarree(),levels=hgt_levels,linewidths=1,colors='k')
            # ax1.clabel(cf4,fontsize=7)
            # ct = ax1.contour(longitude_c,latitude,t,transform=ccrs.PlateCarree(),levels=t_levels,linewidths=0.8,colors='k')
            # ax1.clabel(ct,fontsize=7)

            # cf3 = ax1.contour(lon_cyc_v,latitude,ar_can1_mask,transform=ccrs.PlateCarree(),linewidths=1,colors='magenta',levels=[0.9,1.0],zorder=3,linestyles=linestyle)
            # cf = ax1.contourf(lon_cyc_v, latitude, ar_mask,
            #     levels=[0,0.9,1.1],
            #     colors='none',               # 塗りつぶし色なし
            #     hatches=['','///'],#, '\\\\\\\\', '....', 'xxx'],  # 斜線やドット
            #     transform=ccrs.PlateCarree()
            # )

            # 線の色を黒に統一（パターンの視認性を高める）
            # for coll in cf.collections:
            #     coll.set_edgecolor('k')
            #     coll.set_linewidth(0.5)
            # cf3 = ax1.contour(longitude_c,latitude,ar_mask*south,transform=ccrs.PlateCarree(),levels=[0.0,1.0],linewidths=2,colors='lime')

            # ship = ax1.quiver(np.array([lon]),np.array([lat]),np.array([ship_u]),np.array([ship_v]),transform=ccrs.PlateCarree(),color='k',width=.006,scale=500.)

        except Exception as e:
            print(f"⚠️ Error{e}")
            continue  # エラーが出たらスキップして次のループへ
