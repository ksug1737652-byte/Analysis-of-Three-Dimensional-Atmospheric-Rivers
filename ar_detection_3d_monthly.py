import warnings
warnings.filterwarnings('ignore')
# Import modules and packages
import cartopy.crs as ccrs
import cartopy.util as util
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import csv
from datetime import date
from dateutil.relativedelta import relativedelta
# from geopy.distance import geodesic  # 精度重視。great_circle でも可。
from geopy.distance import distance  # 速度重視。great_circle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import numpy as np
import operator
import os
import os.path
from scipy.ndimage import binary_opening
from scipy.ndimage import rank_filter
from scipy.ndimage import convolve
from shapely.geometry import Point
# import skimage
from collections import defaultdict
from skimage.segmentation import find_boundaries
from skimage.measure import label, regionprops
import xarray as xr
import pandas as pd
import math
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_dilation
import datetime
from pyproj import Geod
from metpy.units import units
from metpy.constants import earth_avg_radius
import metpy.calc as mpcalc

years = np.arange(1979,2024,1)[::1]
months = np.arange(1,13,1)

g = 9.80665 

threshold_mt = 0.3
threshold_ivt = 0
ivt_filtering = 100
min_length = 2000
min_aspect = 2
timescale = 1
timesperday = 24/timescale
lat_res = 1 #再解析データの南北解像度 deg.
lon_res = 1 #再解析データの東西解像度 deg.
min_size = (20/lat_res)*(3/lon_res) #構造物を構成する格子点の最小値，20lat*3lonの構造を想定，幅の最小値が３度，長さが２０度
min_hits = 8 #横縦方向に構造物がある点を構造の一部とする　一点による連結は別の構造物として認識する
# de_min_hits = 4 #コの字で囲われている点を構造の一部と認識する
threshold_ratio = 4/26 #コの字で囲われている点を構造の一部と認識する

data_dir = '/Volumes/Pegasus32 R6/takahashi/era5/data/'
clim_dir = '/Volumes/Pegasus32 R6/takahashi/era5/data/monthly/pressure/'

out_dir = '/Volumes/Pegasus32 R6/takahashi/era5/ar_out_3d_monthly_threshold_02/' 
#creating output files
if not os.path.exists('{}threshold_data'.format(out_dir)):
    os.makedirs('{}threshold_data'.format(out_dir))
if not os.path.exists('{}mask_data'.format(out_dir)):
    os.makedirs('{}mask_data'.format(out_dir))
if not os.path.exists('{}ar_ivt_data'.format(out_dir)):
    os.makedirs('{}ar_ivt_data'.format(out_dir))
if not os.path.exists('{}ar_characteristics'.format(out_dir)):
    os.makedirs('{}ar_characteristics'.format(out_dir))

ar_dir = '/Volumes/Pegasus32 R6/takahashi/era5/' 
input_data_dir = '/Volumes/Pegasus32 R6/takahashi/era5/input_data/' 
grid_areas_file = '{}input_data/grid_areas.nc'.format(ar_dir)
grid_areas = xr.open_dataset(grid_areas_file)['unknown'].sel(latitude=slice(0,-90))
land_mask_file = '{}input_data/lsm_2023-1.nc'.format(ar_dir)
land_mask_data = xr.open_dataset(land_mask_file)['lsm'].sel(latitude=slice(0,-90))
d2,d3 = grid_areas.values.shape


# print('Calculating climatological mean...')
# q_mean = 0
# u_mean = 0
# v_mean = 0
# n_ivt_mean = np.zeros((12,d2,d3))
# e_ivt_mean = np.zeros((12,d2,d3))
# for yyyy in np.arange(1979,2024,1):
# #     dir = '{}q/q_{}.nc'.format(clim_dir,yyyy)
# #     dataset = xr.open_dataset(dir)['q'][:,::-1].sel(pressure_level=slice(300,1000)).sel(latitude=slice(-20,-90))#70-120, 20-50
# #     q_mean += dataset.values/np.arange(1979,2024,1).shape[0]
# #     dir = '{}u/u_{}.nc'.format(clim_dir,yyyy)
# #     dataset = xr.open_dataset(dir)['u'][:,::-1].sel(pressure_level=slice(300,1000)).sel(latitude=slice(-20,-90))#70-120, 20-50
# #     u_mean += dataset.values/np.arange(1979,2024,1).shape[0]
# #     dir = '{}v/v_{}.nc'.format(clim_dir,yyyy)
# #     dataset = xr.open_dataset(dir)['v'][:,::-1].sel(pressure_level=slice(300,1000)).sel(latitude=slice(-20,-90))#70-120, 20-50
# #     v_mean += dataset.values/np.arange(1979,2024,1).shape[0]
# #     print('Now calculating...',yyyy)
#     for mm in np.arange(1,13,1):
#         n_ivt_dataset = '{}n-ivt-{}-{}.nc'.format(input_data_dir,yyyy,mm)
#         e_ivt_dataset = '{}e-ivt-{}-{}.nc'.format(input_data_dir,yyyy,mm)
#         n_ivt_data = xr.open_dataset(n_ivt_dataset)['VIVT'].sel(latitude=slice(0,-90))
#         e_ivt_data = xr.open_dataset(e_ivt_dataset)['UIVT'].sel(latitude=slice(0,-90))
        
#         n_ivt_mean[mm-1] += n_ivt_data.mean(axis=0)/np.arange(1979,2024,1).shape[0]
#         e_ivt_mean[mm-1] += e_ivt_data.mean(axis=0)/np.arange(1979,2024,1).shape[0]

# np.save('calc_files/era_climatology_n_ivt.npy',n_ivt_mean)
# np.save('calc_files/era_climatology_e_ivt.npy',e_ivt_mean)
# np.save('calc_files/climatology_q.npy',q_mean)
# np.save('calc_files/climatology_u.npy',u_mean)
# np.save('calc_files/climatology_v.npy',v_mean)
# print('Climatological mean done...')

print('Loading climatological datasets...')
# q_mean = np.load('calc_files/climatology_q.npy')
# u_mean = np.load('calc_files/climatology_u.npy')
# v_mean = np.load('calc_files/climatology_v.npy')
n_ivt_mean = np.load('calc_files/era_climatology_n_ivt.npy')
e_ivt_mean = np.load('calc_files/era_climatology_e_ivt.npy')


ga = grid_areas.copy()
zeros = 0*ga.values
SH = zeros.copy()
NH = zeros.copy()

# for year in years:

#     for month in months:

print('Start AR detection')
for year in years:
    # with open('{}ar_characteristics/new_ar_details_{}.csv'.format(out_dir,year),'w',newline='') as f:
    #     thewriter = csv.writer(f)
    #     thewriter.writerow(['{}'.format('year'), '{}'.format('month'), '{}'.format('day'), '{}'.format('hour'), '{}'.format('ar_number'), '{}'.format('length'), '{}'.format('width'), '{}'.format('aspect'), '{}'.format('ar_size')])

    for month in months:
        dir = '{}pressure/q/q_{}{:0=2}.nc'.format(data_dir,year,month)
        dataset = xr.open_dataset(dir)['q'].sel(level=slice(300,1000)).sel(latitude=slice(0,-90))#70-120, 20-50
        pressure = dataset.level.values
        time = dataset.time
        # print(time.values.shape)
        # print(time.values[initial:initial+interval])
        q_data = dataset
        dir = '{}pressure/v/v_{}{:0=2}.nc'.format(data_dir,year,month)
        dataset = xr.open_dataset(dir)['v'].sel(level=slice(300,1000)).sel(latitude=slice(0,-90))#70-120, 20-50
        v_data = dataset
        dir = '{}pressure/u/u_{}{:0=2}.nc'.format(data_dir,year,month)
        dataset = xr.open_dataset(dir)['u'].sel(level=slice(300,1000)).sel(latitude=slice(0,-90))#70-120, 20-50
        u_data = dataset
        # dir = '{}pressure/w/w_{}{:0=2}.nc'.format(data_dir,year,month)
        # dataset = xr.open_dataset(dir)['w'].sel(level=slice(100,1000)).sel(latitude=slice(-20,-90))
        # w = dataset.sel(time='{}-{:0=2}-{:0=2}-{:0=2}'.format(year,month,day,hour))
        # dir = '{}pressure/z/z_{}{:0=2}.nc'.format(data_dir,year,month)
        # dataset = xr.open_dataset(dir)['z'].sel(level=slice(100,1000)).sel(latitude=slice(-20,-90))
        # z = dataset.sel(time='{}-{:0=2}-{:0=2}-{:0=2}'.format(year,month,day,hour))/9.80665
        # dir = '{}/z_202301.nc'.format(data_dir,year,month)
        # dataset = xr.open_dataset(dir)['z'].sel(latitude=slice(-20,-90))
        # sz_data = dataset/9.80665
        sp_file = '{}surface/sp/sp_{}{:0=2}.nc'.format(data_dir,year,month)
        dataset = xr.open_dataset(sp_file).sel(latitude=slice(0,-90))
        sp_data = dataset['sp']/100
        
        n_ivt_dataset = '{}n-ivt-{}-{}.nc'.format(input_data_dir,year,month)
        e_ivt_dataset = '{}e-ivt-{}-{}.nc'.format(input_data_dir,year,month)

        n_ivt_data = xr.open_dataset(n_ivt_dataset)['VIVT'].sel(latitude=slice(0,-90))
        e_ivt_data = xr.open_dataset(e_ivt_dataset)['UIVT'].sel(latitude=slice(0,-90))
    
        p_diff = (np.roll(pressure,-1) - pressure)
        latitude = sp_data.latitude
        longitude = sp_data.longitude

        ############################# for debug
        # time = time[:2]
        ############################## for threshold
        print('calculating threshold...')
        
        # Date = datetime.datetime(year,month,1,0)
        # st_date = Date-datetime.timedelta(days=5)
        # ed_date = Date+relativedelta(months=1)+datetime.timedelta(days=5)
        
        # dir = ['{}pressure/q/q_{}{}.nc'.format(data_dir,st_date.year,st_date.month),'{}pressure/q/q_{}{:0=2}.nc'.format(data_dir,year,month),'{}pressure/q/q_{}{:0=2}.nc'.format(data_dir,ed_date.year,ed_date.month)]
        # dataset = xr.open_dataset(dir)['t'].sel(level=slice(300,1000)).sel(latitude=slice(0,-90)).sel(time=slice('{}-{:0=2}-{:0=2}-{:0=2}'.format(st_date.year,st_date.month,st_date.day,st_date.hour),'{}-{:0=2}-{:0=2}-{:0=2}'.format(ed_date.year,ed_date.month,ed_date.day,ed_date.hour)))
        # q_thresh = dataset
        # dir = ['{}pressure/v/v_{}{}.nc'.format(data_dir,st_date.year,st_date.month),'{}pressure/v/v_{}{:0=2}.nc'.format(data_dir,year,month),'{}pressure/v/v_{}{:0=2}.nc'.format(data_dir,ed_date.year,ed_date.month)]
        # dataset = xr.open_dataset(dir)['v'].sel(level=slice(300,1000)).sel(latitude=slice(0,-90)).sel(time=slice('{}-{:0=2}-{:0=2}-{:0=2}'.format(st_date.year,st_date.month,st_date.day,st_date.hour),'{}-{:0=2}-{:0=2}-{:0=2}'.format(ed_date.year,ed_date.month,ed_date.day,ed_date.hour)))
        # v_thresh = dataset
        # dir = ['{}pressure/u/u_{}{}.nc'.format(data_dir,st_date.year,st_date.month),'{}pressure/u/u_{}{:0=2}.nc'.format(data_dir,year,month),'{}pressure/u/u_{}{:0=2}.nc'.format(data_dir,ed_date.year,ed_date.month)]
        # dataset = xr.open_dataset(dir)['u'].sel(level=slice(300,1000)).sel(latitude=slice(0,-90)).sel(time=slice('{}-{:0=2}-{:0=2}-{:0=2}'.format(st_date.year,st_date.month,st_date.day,st_date.hour),'{}-{:0=2}-{:0=2}-{:0=2}'.format(ed_date.year,ed_date.month,ed_date.day,ed_date.hour)))
        # u_thresh = dataset
            

        mt_for_max = q_data.copy()
        mt_zonal_max = q_data.copy()[0,:,:,0]
        mt_zonal_mean = q_data.copy()[0,:,:,0]

        spressure_data = (q_data.values*0+1)*sp_data.values[:,None,:,:]
        pressure_data = (q_data.values*0+1)*pressure[None,:,None,None]
        mt_for_max.values = np.power((np.power(q_data.values*v_data.values,2)+np.power(q_data.values*u_data.values,2)),0.5)
        mt_for_max.values[pressure_data>spressure_data] = np.nan
        mt_zonal_max.values = np.nanmean(np.nanmax((mt_for_max.values - np.nanmean(mt_for_max.values,axis=3)[:,:,:,None]),axis=3),axis=0)
        mt_zonal_mean.values = np.nanmean(np.nanmean(mt_for_max.values,axis=3),axis=0)

        ###############################
        
        
        umt_list = []
        vmt_list = []
        eq_list = []
        ivt_list = []
        object_list = []
        num_ob_list = []
        lab_ob_list = []
        props_ob_list = []
        for t in range(time.shape[0]):
            print(year,month,'{:0=3} timestep...'.format(t))
            Din = datetime.datetime(year,month,1,0)+datetime.timedelta(hours=t)
            # print(q_mean.shape)
            # q_m = q_mean[Din.month-1]
            # u_m = u_mean[Din.month-1]
            # v_m = v_mean[Din.month-1]
            n_ivt_m = n_ivt_mean[Din.month-1]
            e_ivt_m = e_ivt_mean[Din.month-1]
            n_ivt = n_ivt_data[t]
            e_ivt = e_ivt_data[t]
            # T = t_data[t]
            q = q_data[t]
            u = u_data[t]
            v = v_data[t]
            sp2 = sp_data[t]
            
            # dewpoint = mpcalc.dewpoint_from_specific_humidity(pressure[:,None,None] * units('hPa'), q * units('kg/kg'))
            # eq = np.array(mpcalc.equivalent_potential_temperature(pressure[:,None,None] * units('hPa'), T * units('K'), dewpoint))
            # eq_list.append(eq)

            ivt = np.power(np.power(n_ivt, 2) + np.power(e_ivt, 2), 1/2)
            ivt_m = np.power(np.power(n_ivt_m, 2) + np.power(e_ivt_m, 2), 1/2)
            ivt_list.append(ivt)

            spressure_data = (q.values*0+1)*sp2.values[None,:,:]
            pressure_data = (q.values*0+1)*pressure[:,None,None]
            mt = q.copy()
            # q.values[pressure_data>spressure_data] = np.nan
            # q_for_mean = q.copy()
            # q_zonal_mean = q.copy()[:,:,0]
            # q_zonal_max = q.copy()[:,:,0]
            # q_for_mean.values[pressure_data>spressure_data] = np.nan
            # q_zonal_mean.values = np.nanmean(q_for_mean.values,axis=2)[:,:]
            # q_zonal_max.values = np.nanmax((q.values - np.nanmean(q_for_mean.values,axis=2)[:,:,None]),axis=2)
            mt.values = np.power((np.power(q.values*v.values,2)+np.power(q.values*u.values,2)),0.5)
            mt.values[pressure_data>spressure_data] = np.nan
            # mt_m = np.power((np.power(q_m*v_m,2)+np.power(q_m*u_m,2)),0.5)
            mt_for_mean = mt.copy()
            # mt_zonal_mean = mt.copy()[:,:,0]
            # mt_for_mean.values[pressure_data>spressure_data] = np.nan
            # mt_zonal_mean.values = np.nanmean(mt_for_mean.values,axis=2)[:,:]
            mt_anomaly = (mt - mt_zonal_mean)/mt_zonal_max#(q - q_m)/q_m #(mt - mt_zonal_mean)/mt_zonal_max
            # mt_anomaly = (mt - mt_m)/mt_m#(q - q_m)/q_m #(mt - mt_zonal_mean)/mt_zonal_max
            # q_anomaly = (q - q_m)/q_m
            q_anomaly = (ivt - ivt_m)/ivt_m
            # print(mt_anomaly.median())

            # umt_list.append(q.values*u.values)
            # vmt_list.append(q.values*v.values)
            # print(q.shape)
            freq = q
            q_diff = np.roll(freq,1,axis=0) - freq
            q_diff[0] = 0
            p_data = (freq.values*0+1)*pressure[:,None,None]
            p_data[0] = 1000
            sp2_data = (freq.values*0+1)*sp2.values[None,:,:]
            q_diff.values[p_data>sp2_data] = 0
            inversion = ((np.roll(q_diff,-1,axis=0)>=0)  & (np.roll(p_data,-1,axis=0)<sp2_data)& (np.roll(p_data,-1,axis=0)>300))#& (q_diff<=0)
            pos_anomaly_mt = mt_anomaly>=threshold_mt
            pos_anomaly_q = q_anomaly>=threshold_ivt
            nan = mt != np.nan

            required =  pos_anomaly_mt & nan & pos_anomaly_q #& inversion
            structure = np.ones((3,3,3),dtype=bool)
            structure[1,1,:] = False

            # rank = np.sum(structure) - min_hits

            # フィルタを適用 構造物の突起部分（周囲に構造物が少ない）を丸めて，一格子点による連結を切断する
            # filtered = rank_filter(required, rank=rank, footprint=structure)
            # binary_mask = filtered > 0
            # opened_required = binary_opening(required,structure=structure)

            binary_mask = required
            pad = 10  # 経度方向拡張

            # 元データ binary_mask: shape (level, lat, lon)
            # wrap拡張
            wrapped = np.concatenate([
                binary_mask[..., -pad:],  # 経度末尾
                binary_mask,
                binary_mask[..., :pad]    # 経度先頭
            ], axis=-1)

            # ラベリング（skimage）
            labeled_wrap = label(wrapped, connectivity=3)

            # オリジナル範囲だけ切り出し
            central = labeled_wrap[..., pad:-pad].copy()

            # 経度末尾と先頭の overlap 領域を取得
            left = labeled_wrap[..., pad:pad*2]
            right = labeled_wrap[..., -pad:]

            # ラベルのペアを検出（接触しているラベルを記録）
            connections = defaultdict(set)
            for l_val in np.unique(left):
                if l_val == 0:
                    continue
                overlap = right[left == l_val]
                for r_val in np.unique(overlap):
                    if r_val != 0:
                        connections[l_val].add(r_val)
                        connections[r_val].add(l_val)

            # Union-Findでラベルを統一
            parent = {}

            def find(x):
                while parent.get(x, x) != x:
                    x = parent[x]
                return x

            def union(x, y):
                x_root = find(x)
                y_root = find(y)
                if x_root != y_root:
                    parent[y_root] = x_root

            # 連結ラベルを統合
            for k, vs in connections.items():
                for v in vs:
                    union(k, v)

            # ラベル変換マッピングを作成
            mapping = {}
            for label_val in np.unique(central):
                if label_val == 0:
                    continue
                root = find(label_val)
                mapping[label_val] = root

            # ラベル番号を統一（ベクトル化適用）
            central_flat = central.ravel()
            remap = np.arange(central_flat.max() + 1)
            for old, new in mapping.items():
                remap[old] = new
            central = remap[central_flat].reshape(central.shape)

            labeled_array = central
            props = regionprops(labeled_array)

            # print(len(props))
            
            #  filtered by number of gridpoints within structure
            valid_labels = [r.label for r in props if r.area >= min_size]
            valid_nums = [n for n,r in enumerate(props) if r.area >= min_size]

            # ラベル配列から有効ラベルだけ残すマスクを作成
            # filtered_mask = np.isin(labeled_array, valid_labels)
            # print(filtered_mask.max())

            # de_structure = np.ones((3,3,3),dtype=bool)
            # de_rank = np.sum(de_structure) - de_min_hits

            num_ob = len(valid_labels)
            # print(num_ob)
            
            out_array = labeled_array.copy()#*0
            
            # for n,i in enumerate(valid_labels):
            #     i -= 1 # label number to list number
            #     idx = valid_nums[n]
            #     # print(n,i,idx)
            #     num_labeled_array = np.where(labeled_array == i+1, 1, 0)

            #     # フィルタを適用
            #     # filtered_mask = rank_filter(num_labeled_array, rank=de_rank, footprint=de_structure)
            #     # num_labeled_array = filtered_mask > 0
                
            #     # 3D のバイナリマスク（構造物 = 1, その他 = 0）
            #     mask = num_labeled_array.astype(np.uint8)

            #     # 周囲の範囲を定義（3x3x3 の立方体構造体）
            #     structure = np.ones((3, 3, 3), dtype=np.uint8)

            #     # 自分自身を含めずに周囲の構造物の数を数える
            #     structure[1, 1, 1] = 0  # 中央点を除く

            #     # 周囲の構造物の数を計算
            #     neighbor_count = convolve(mask, structure, mode='constant', cval=0)

            #     # 周囲の構造物割合がしきい値以上の点を拡張対象にする
            #     threshold_count = int(threshold_ratio * structure.sum())

            #     # 新たに構造物として追加する点
            #     grow_mask = (mask == 0) & (neighbor_count >= threshold_count)

            #     # mask を更新（太らせる）
            #     mask_grown = mask.copy()
            #     mask_grown[grow_mask] = 1
                
            #     out_array += mask_grown * (i+1)
            
            object_list.append(out_array)
            num_ob_list.append(valid_nums)
            lab_ob_list.append(valid_labels)
            props_ob_list.append(props)

        print('Computing length...')
        length_list = []
        length_filter = []
        for i in range(time.shape[0]):
            print('computing {}-{}-time step'.format(year,month),'{:0=3}'.format(i+1),'...')
            timely_length_list = []
            Filter = []
            for n,j in enumerate(lab_ob_list[i]):
                j -= 1
                ar_mask = object_list[i].copy()
                ar_mask = np.where(ar_mask == j+1, 1, 0)
                ar_mask_shadow = ar_mask.max(axis=0)

                eroded = binary_erosion(ar_mask_shadow)
                edge = ar_mask_shadow ^ eroded  # XORで外周を取得
                # r = props_ob_list[i][num_ob_list[i][n]]
                # center = r.centroid
                # central_coord = (latitude[int(round(center[1],0))],longitude[int(round(center[2],0))])
                # print(r.area)
                
                coords = np.argwhere(edge)  # (y, x)
                if len(coords) < 40:
                    max_distance = 0
                    timely_length_list.append(max_distance)
                    Filter.append(j+1)
                    continue

                # 緯度経度の抽出
                latlon_points = [
                    (latitude[y], longitude[x])
                    for y, x in coords
                ]
                # print(len(latlon_points))
                # 全点間ペアの geodesic 距離を計算（効率は低めだが高精度）
                # max_distance = 0
                # for k in range(len(latlon_points)):
                #     for l in range(k + 1, len(latlon_points)):
                #         dist = distance(latlon_points[k], latlon_points[l]).km
                #         # dist_center = distance(latlon_points[k], central_coord).km + distance(central_coord, latlon_points[l]).km
                #         if dist > max_distance:
                #             # print(dist)
                #             max_distance = dist
                #             # max_distance_center = dist_center
                            
                # print(max_distance)
                # max_distance = 0
                
                # WGS84 楕円体
                geod = Geod(ellps="WGS84")

                # latlon_points を配列化（[ (lat, lon), ... ]）
                points = np.array(latlon_points)
                lats = points[:, 0]
                lons = points[:, 1]

                # 全組み合わせのインデックス
                i_idx, j_idx = np.triu_indices(len(points), k=1)  # k=1でi<jの組だけ

                # 各組の座標を抽出
                lon1 = lons[i_idx]
                lat1 = lats[i_idx]
                lon2 = lons[j_idx]
                lat2 = lats[j_idx]

                # 一括距離計算（m）
                _, _, dist_m = geod.inv(lon1, lat1, lon2, lat2)

                # 最大距離 (km)
                max_distance = np.max(dist_m) / 1000.0
                # print(max_distance)


                timely_length_list.append(max_distance)

                if max_distance > min_length:
                    length_check = True
                else:
                    length_check = False
                if length_check == False:
                    Filter.append(j+1)
            #timely_width_list.append(min(objective_length_list))
            length_list.append(timely_length_list)
            length_filter.append(Filter)
        #print(len(length_list[0]))


        print('Computing length/width ratio...')
        grid_areas_file = '{}input_data/grid_areas.nc'.format(ar_dir)
        grid_areas = xr.open_dataset(grid_areas_file)['unknown'].sel(latitude=slice(0,-90))
        ar_area_list = []
        ar_iwv_list = []
        for i in range(time.shape[0]):
            timely_area_list = []
            timely_wv_list = []
            for n,j in enumerate(lab_ob_list[i]):
                j -= 1
                ar_mask = object_list[i].copy()
                ar_mask = np.where(ar_mask == j+1, 1, 0)
                ar_grid_area = ar_mask.max(axis=0) * grid_areas.values
                ar_area = ar_grid_area.sum()
                timely_area_list.append(ar_area)
                #calculating magnitude of Water Vapor included in AR
            ar_area_list.append(timely_area_list)


        width_list = []
        for i in range(time.shape[0]):
            widths = []
            for n,j in enumerate(lab_ob_list[i]):
                j -= 1
                if (length_list[i][n] > 0):
                    width = ar_area_list[i][n] / length_list[i][n]
                    widths.append(width)
                else:
                    width = 0
                    widths.append(width)
            width_list.append(widths)

        aspect_filter = []
        aspect_list = []
        for i in range(time.shape[0]):
            Filter = []
            timely_aspect_list = []
            for n,j in enumerate(lab_ob_list[i]):
                j -= 1
                if ((j+1) not in length_filter[i]):
                    narrowness_check = ((length_list[i][n] 
                                        / width_list[i][n]) > min_aspect)
                    aspect = length_list[i][n] / width_list[i][n]
                    timely_aspect_list.append(aspect)
                    if narrowness_check == False:
                        Filter.append(j+1)
                else:
                    timely_aspect_list.append(0)
            aspect_filter.append(Filter)
            aspect_list.append(timely_aspect_list)

        mean_ivt_list = []
        max_ivt_list = []
        for i in range(time.shape[0]):
            NewList_max = []
            for n,j in enumerate(lab_ob_list[i]):
                j -= 1
                ar_mask = object_list[i].copy()
                ar_mask = np.where(ar_mask == j+1, 1, 0)
                ar_ivt = ar_mask.max(axis=0)*ivt_list[i].values
                NewList_max.append(ar_ivt.max())
            max_ivt_list.append(NewList_max)

        ivt_filter = []
        for i in range(time.shape[0]):
            Filter = []
            for n,j in enumerate(lab_ob_list[i]):
                j -= 1
                if ((max_ivt_list[i][n]>=ivt_filtering)):
                    ivt_check = True
                else:
                    ivt_check = False
                if ivt_check == False:
                    Filter.append(j+1)
            ivt_filter.append(Filter)

        print('Computing ar_mask data...')
        ar_mask_data = q_data.values*0
        for i in range(time.shape[0]):
            for n,j in enumerate(lab_ob_list[i]):
                j -= 1
                if (
                    ((j+1) not in length_filter[i])
                    & ((j+1) not in aspect_filter[i])
                    & ((j+1) not in ivt_filter[i])
                    ):
                    ar_mask = object_list[i].copy()
                    ar_mask = np.where(ar_mask == j+1, 1, 0)
                    ar_mask_data[i] += ar_mask
                    # ar_mask_data_2d[i] += ar_mask_list[i][n]                

        ar_without_ivt_filter = q_data.values*0
        for i in range(time.shape[0]):
            for n,j in enumerate(lab_ob_list[i]):
                j -= 1
                if (
                    ((j+1) not in length_filter[i])
                    & ((j+1) not in aspect_filter[i])
                    ):
                    ar_mask = object_list[i].copy()
                    ar_mask = np.where(ar_mask == j+1, 1, 0)
                    ar_without_ivt_filter[i] += ar_mask
                    # ar_mask_data_2d[i] += ar_mask_list[i][n]                


        ar_without_geometric = q_data.values*0
        for i in range(time.shape[0]):
            ar_mask = object_list[i].copy()
            ar_without_geometric[i] = np.where(ar_mask>=1,1,0)

        print('Saving ar characteristics...')

        ar_mask_datasets = q_data.values*0
        for i in range(time.shape[0]):
            dt = i%timesperday
            hours = math.floor(dt*timescale)
            days = math.floor(1+(i-dt)/timesperday)
            Count=0
            for n,j in enumerate(lab_ob_list[i]):
                # print(n,j)
                j -= 1
                if (
                    ((j+1) not in length_filter[i])
                    & ((j+1) not in aspect_filter[i])
                    & ((j+1) not in ivt_filter[i])
                    ):
                    Count += 1
                    
                    #Saving individual ar mask data
                    ind_ar_mask = object_list[i].copy()
                    ind_ar_mask = np.where(ind_ar_mask == j + 1,1,0)
                    # print(ar_list[i][n])
                    ar_mask_datasets[i] += ind_ar_mask*Count#*ar_list[i][n]#*Count
                    # ar_mask_datasets_2d[i] += ar_mask_list[i][n]*ar_list[i][n]#*Count

                    with open('{}ar_characteristics/new_ar_details_{}.csv'.format(out_dir,year),'a',newline='') as f:
                        thewriter = csv.writer(f)
                        thewriter.writerow(['{}'.format(year), '{}'.format(month), '{}'.format(days), '{}'.format(hours), '{}'.format(Count), '{}'.format(length_list[i][n]), '{}'.format(width_list[i][n]), '{}'.format(aspect_list[i][n]), '{}'.format(ar_area_list[i][n])])

        print('Constructing ar projection...')
        
        ar_test = ar_mask_datasets.copy()
        pressure_data = (ar_test) * pressure_data[None,:,:,:]
        spressure_data = ((ar_test) * spressure_data[None,:,:,:]) - 100
        ar_test[pressure_data<=spressure_data] = 0
        ob_list = [np.unique(ar_test[i])[~(np.unique(ar_test[i])==0)] for i in range(ar_test.shape[0])]
        # print(ob_list)

        masked = np.empty_like(ar_test)

        for t in range(ar_test.shape[0]):  # time方向にループ
            # print(t)
            valid_values = ob_list[t]
            masked[t] = np.where(np.isin(ar_mask_datasets[t], valid_values), ar_mask_datasets[t], 0)
        ar_ground_mask_data = np.where(masked>0,1,0)

        print('Saving ar datasets...')

        out_AR_mask = xr.DataArray(
            ar_without_geometric,
            name = 'ar_mask',
            dims = ['time','level','latitude','longitude'],
            coords = (
                time,pressure,latitude,longitude,
            ),
            attrs = {
                'long_name':'ARs Mask Data without geometric requirements',
                'units':'#'
            },
        )
        outfile_ar_mask = '{}mask_data/AR_candidates1_mask-{}-{}.nc'.format(out_dir,year,month)
        out_AR_mask.to_netcdf(outfile_ar_mask)

        out_AR_mask = xr.DataArray(
            ar_without_ivt_filter,
            name = 'ar_mask',
            dims = ['time','level','latitude','longitude'],
            coords = (
                time,pressure,latitude,longitude,
            ),
            attrs = {
                'long_name':'ARs Mask Data without IVT filter',
                'units':'#'
            },
        )
        outfile_ar_mask = '{}mask_data/AR_candidates2_mask-{}-{}.nc'.format(out_dir,year,month)
        out_AR_mask.to_netcdf(outfile_ar_mask)

        out_AR_mask = xr.DataArray(
            ar_ground_mask_data,
            name = 'ar_mask',
            dims = ['time','level','latitude','longitude'],
            coords = (
                time,pressure,latitude,longitude,
            ),
            attrs = {
                'long_name':'ARs Mask Data',
                'units':'#'
            },
        )
        outfile_ar_mask = '{}ar_ivt_data/select_groundAR_mask-{}-{}.nc'.format(out_dir,year,month)
        out_AR_mask.to_netcdf(outfile_ar_mask)

        out_AR_mask = xr.DataArray(
            ar_mask_datasets,
            name = 'ar_mask',
            dims = ['time','level','latitude','longitude'],
            coords = (
                time,pressure,latitude,longitude,
            ),
            attrs = {
                'long_name':'ARs Mask Data',
                'units':'no.'
            },
        )
        outfile_ar_mask = '{}mask_data/select_AR_mask_num-{}-{}.nc'.format(out_dir,year,month)
        out_AR_mask.to_netcdf(outfile_ar_mask)
