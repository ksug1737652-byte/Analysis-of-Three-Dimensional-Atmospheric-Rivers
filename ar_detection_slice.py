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
from scipy.ndimage import binary_dilation

year = 2003
month = 8
day = 1
hour = 0

g = 9.80665
#data_dir = '/media/takahashi/HDPH-UT/Kyoto-U/AR_jra55/AR_detection/AR_detection_Kennet/data/'
#data_dir = '/Volumes/PromisePegasus/takahashi/JRA55/AR-2022-2023/'#,min_length,min_aspect,eqw_lat,plw_lat,dist_lat,distance_between)
inputdata_dir = '/Volumes/Pegasus32 R6/takahashi/era5/'
input_data_dir = '/Volumes/Pegasus32 R6/takahashi/era5/input_data/'
mask_dir = '{}ar_out_era5_SH_20S/ar_ivt_data/'.format(inputdata_dir)
land_mask_file = '{}input_data/lsm_2023-1.nc'.format(inputdata_dir)
lsm = xr.open_dataset(land_mask_file)['lsm'].sel(latitude=slice(-20,-85)).values[0]
print(lsm.shape)

data_dir = '{}ar_out_era5_SH_20S/'.format(inputdata_dir)

z_file = '{}data/z_202301.nc'.format(inputdata_dir)
z_data = xr.open_dataset(z_file)['z'].sel(latitude=slice(-20,-90)).values[0]/g

data_dir = '{}data/surface/'.format(inputdata_dir)
sp_file = '{}sp/sp_{}{:0=2}.nc'.format(data_dir,year,month)
dataset = xr.open_dataset(sp_file)
sp = dataset['sp'].sel(latitude=slice(-20,-85)).sel(
    time='{}-{:0=2}-{:0=2}-{:0=2}'.format(year,month,day,hour)
).values/100

print(sp.shape)
file = '{}select_AR_mask-{}{:0=2}.nc'.format(mask_dir,year,month)
dataset = xr.open_dataset(file)
ar_mask_dataset = dataset['ar_mask'].sel(latitude=slice(-20,-85)).sel(
    time='{}-{:0=2}-{:0=2}-{:0=2}'.format(year,month,day,hour)
)
ar_mask_dataset.values[:,60:] = 0
print(np.unique(ar_mask_dataset.values)[~(np.unique(ar_mask_dataset.values)==0)])
ar_mask_dataset.values = np.where(ar_mask_dataset.values==1,1,0)
time = ar_mask_dataset.shape[0]
lat = ar_mask_dataset.latitude
lon = ar_mask_dataset.longitude
ar_mask = ar_mask_dataset.values

# mask_inv_dir = '{}ar_out_3d_02/ar_ivt_data/'.format(inputdata_dir)
# file = '{}select_groundAR_mask-{}-{}.nc'.format(mask_inv_dir,year,month)
# dataset = xr.open_dataset(file)
# ar_mask_dataset = dataset['ar_mask'].sel(latitude=slice(-20,-80)).sel(
#     time='{}-{:0=2}-{:0=2}-{:0=2}'.format(year,month,day,hour)
# )
# ar_mask_dataset = ar_mask_dataset.fillna(0)   # NaNを0に統一
# time = ar_mask_dataset.shape[0]
# lat = ar_mask_dataset.latitude
# lon = ar_mask_dataset.longitude
# pres = ar_mask_dataset.level.values
# pres_data = (ar_mask_dataset.values*0+1) * pres[:,None,None]
# sp_data = (ar_mask_dataset.values*0+1) * sp[None,:,:]
# land = (pres_data>sp_data).astype(float)

# ar3d_mask = ar_mask_dataset.values.astype(float)
# mask_bottom = ar3d_mask.max(axis=0).astype(float)

mask_inv_dir = '/Volumes/Pegasus32 R8/takahashi/era5/ar_out_3d_monthly_threshold/mask_data/'.format(inputdata_dir)
file = '{}select_AR_mask_num-{}-{}.nc'.format(mask_inv_dir,year,month)
dataset = xr.open_dataset(file)
ar_inv_mask_dataset = dataset['ar_mask'].sel(time='{}-{:0=2}-{:0=2}-{:0=2}'.format(year,month,day,hour))
ar_inv_mask_dataset.values[ar_inv_mask_dataset.values==np.nan] = 0
ar_inv_mask_dataset = ar_inv_mask_dataset.sel(latitude=slice(-20,-85))

pres = ar_inv_mask_dataset.level
print(pres)
pres_data = (ar_inv_mask_dataset.values*0+1)*pres.values[:,None,None]
sp_file = '{}data/surface/sp/sp_{}{:0=2}.nc'.format(inputdata_dir,year,month)
dataset = xr.open_dataset(sp_file)
sp = dataset['sp'].sel(latitude=slice(-20,-85)).sel(time='{}-{:0=2}-{:0=2}-{:0=2}'.format(year,month,day,hour)).values/100
sp_data = (ar_inv_mask_dataset.values*0+1)*sp[None,:,:]
land = (pres_data>sp_data).astype(float)
ar_test = ar_inv_mask_dataset.values.copy()
ar_test[pres_data<(sp_data-100)] = 0
ob_list = np.unique(ar_test)[~(np.unique(ar_test)==0)]
print(ob_list)

masked = np.empty_like(ar_test)

valid_values = ob_list
masked = np.where(np.isin(ar_inv_mask_dataset.values, valid_values), ar_inv_mask_dataset.values, 0)
print(masked.max())
ar_inv_mask_dataset.values = np.where(masked==5,1,0)
ar3d_mask = ar_inv_mask_dataset.values.astype(float)
mask_bottom = ar3d_mask.max(axis=0).astype(float)

# 経度を -180〜180 に変換
lon_v = ((lon.values + 180) % 360) - 180
lat_v = lat.values

# ===== 描画範囲の設定 =====
lon_min, lon_max = -10, 90
lat_min, lat_max = -85, -30

lon_mask = (lon_v >= lon_min) & (lon_v <= lon_max)
lat_mask = (lat_v >= lat_min) & (lat_v <= lat_max)

lon_sel = lon_v[lon_mask]
lat_sel = lat_v[lat_mask]

# データもスライス
land_cut = land[:, lat_mask, :][:, :, lon_mask].astype(float)
ar3d_mask_cut = ar3d_mask[:, lat_mask, :][:, :, lon_mask].astype(float)
pres_data_cut = pres_data[:, lat_mask, :][:, :, lon_mask].astype(float)
mask_bottom_cut = mask_bottom[lat_mask, :][:, lon_mask].astype(float)
sp_cut = sp[lat_mask, :][:, lon_mask].astype(float)
land_cut[-1] = np.nan
ar3d_mask_cut = np.roll(ar3d_mask_cut,10,axis=2)
pres_data_cut = np.roll(pres_data_cut,10,axis=1)
mask_bottom_cut = np.roll(mask_bottom_cut,10,axis=1)
sp_cut = np.roll(sp_cut,10,axis=1)
land_cut = np.roll(land_cut,10,axis=2)
lon_sel = np.roll(lon_sel,10,axis=0)

# ====== 描画 ======
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

norm = mcolors.Normalize(vmin=300, vmax=1000)
cmap = cm.winter_r

fig = plt.figure(figsize=(10,10))
ax2 = fig.add_subplot(111, projection='3d')
ax2.set_facecolor(color='grey')
fig.set_facecolor(color='grey')


# 最下層マスク枠線描画
lon_grid, lat_grid = np.meshgrid(lon_sel, lat_sel)

durf = ax2.plot_wireframe(lon_grid, lat_grid,
                        sp_cut,
                        rstride=1,   # ← ここを小さく
                        cstride=1,  
                        color='black',
                        linewidth=0.3, 
                        zorder=1, # ←低めにしておく
                        alpha=0.8
                        )

for idx, p in enumerate(pres):
    ar3d_slice = (pres_data_cut*ar3d_mask_cut)[idx].astype(float)
    ar3d_slice[ar3d_slice==0] = np.nan

    lon_grid, lat_grid = np.meshgrid(lon_sel, lat_sel)
    p_grid = np.full_like(lon_grid, p, dtype=float)

    facecolors = cmap(norm(ar3d_slice))
    alpha_mask = np.where(np.isnan(ar3d_slice), 0.0, 0.4)
    facecolors[..., -1] = alpha_mask

    ax2.plot_surface(
        lon_grid, lat_grid, p_grid,
        facecolors=facecolors,
        rstride=1, cstride=1,
        linewidth=0, antialiased=False,
        shade=False,
        zorder=2  # ←低めにしておく
    )


ax2.contour(
    lon_grid, lat_grid, mask_bottom_cut.astype(float),
    levels=[0.5],          # 0-1境界
    colors='red',
    linewidths=2,
    offset=1000,
    zorder=999,          # ←最前面に出す
    alpha=1.0            # ←完全に不透明
)



# ax2.contour(
#     lon_grid, lat_grid, lsm.astype(float),
#     levels=[0,0.5,1],          # 0-1境界
#     cmap='Grays',
#     alpha=0.8            # ←完全に不透明
# )

# for idx, p in enumerate(pres):
#     land_mask = land_cut[idx].astype(float)

#     lon_grid, lat_grid = np.meshgrid(lon_sel, lat_sel)

#     ax2.contourf(
#         lon_grid, lat_grid, land_mask.astype(float),
#         levels=[0,0.5,1],          # 0-1境界
#         cmap='Grays',
#         linewidths=2,
#         offset=p,
#         alpha=0.4            # ←完全に不透明
#     )

ax2.set_xlabel("Longitude",fontsize=13)
ax2.set_ylabel("Latitude",fontsize=13)
ax2.set_zlabel("Pressure [hPa]",fontsize=13)
x_ticks = np.arange(0,91,30)
y_ticks = np.arange(-80,-29,10)
z_ticks = np.arange(300,1001,200)
ax2.set_xticks(x_ticks)
ax2.set_yticks(y_ticks)
ax2.set_zticks(z_ticks)
ax2.set_xticklabels(x_ticks,fontsize=12)
ax2.set_yticklabels(y_ticks,fontsize=12)
ax2.set_zticklabels(z_ticks,fontsize=12)
ax2.set_zlim(1000, 300)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax2, shrink=0.5, orientation='horizontal',aspect=40,pad=0.2)
cbar.set_label("Pressure [hPa]")
cbar.ax.invert_yaxis()
ax2.set_xlim(lon_min-1, lon_max+1)
ax2.set_ylim(lat_min, lat_max)
ax2.set_box_aspect([2, 3, 1]) 
ax2.view_init(elev=30, azim=20)

out = '/Users/takahashikazu/Desktop/NIPR/fig/'
out_dir = '{}AR_Detection_3d_monthly/DF_comp/'.format(out)#,Date.year,Date.month)
if not os.path.exists(out_dir):#,Date.year,Date.month)):
    os.makedirs(out_dir)#,Date.year,Date.month))

plt.tight_layout()
plt.savefig('{}AR3d_20_{}{:0=2}{:0=2}{:0=2}_grey.png'.format(out_dir,year,month,day,hour),dpi=700)
plt.show()

ar3d_mask = ((ar_mask_dataset.values*0+1) * ar_mask[None,:,:]).astype(float)
mask_bottom = ar_mask.astype(float)

# ar3d_mask[:,:,89:350] = np.nan
# mask_bottom[:,89:350] = np.nan
# land[:,86:352] = 0

# 経度を -180〜180 に変換
lon_v = ((lon.values + 180) % 360) - 180
lat_v = lat.values

# ===== 描画範囲の設定 =====
lon_min, lon_max = -10, 90
lat_min, lat_max = -85, -30

lon_mask = (lon_v >= lon_min) & (lon_v <= lon_max)
lat_mask = (lat_v >= lat_min) & (lat_v <= lat_max)

lon_sel = lon_v[lon_mask]
lat_sel = lat_v[lat_mask]

# データもスライス
land_cut = land[:, lat_mask, :][:, :, lon_mask].astype(float)
ar3d_mask_cut = ar3d_mask[:, lat_mask, :][:, :, lon_mask].astype(float)
pres_data_cut = pres_data[:, lat_mask, :][:, :, lon_mask].astype(float)
mask_bottom_cut = mask_bottom[lat_mask, :][:, lon_mask].astype(float)
land_cut[-1] = np.nan
ar3d_mask_cut = np.roll(ar3d_mask_cut,10,axis=2)
pres_data_cut = np.roll(pres_data_cut,10,axis=1)
mask_bottom_cut = np.roll(mask_bottom_cut,10,axis=1)
land_cut = np.roll(land_cut,10,axis=2)
lon_sel = np.roll(lon_sel,10,axis=0)

# ====== 描画 ======
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

norm = mcolors.Normalize(vmin=300, vmax=1000)
cmap = cm.winter_r

fig = plt.figure(figsize=(10,10))
ax2 = fig.add_subplot(111, projection='3d')
ax2.set_facecolor(color='grey')
fig.set_facecolor(color='grey')

durf = ax2.plot_wireframe(lon_grid, lat_grid,
                        sp_cut,
                        rstride=1,   # ← ここを小さく
                        cstride=1,  
                        color='black',
                        linewidth=0.3, 
                        zorder=1, # ←低めにしておく
                        alpha=0.8
                        )

for idx, p in enumerate(pres):
    ar3d_slice = (pres_data_cut*ar3d_mask_cut)[idx].astype(float)
    ar3d_slice[ar3d_slice==0] = np.nan

    lon_grid, lat_grid = np.meshgrid(lon_sel, lat_sel)
    p_grid = np.full_like(lon_grid, p, dtype=float)

    facecolors = cmap(norm(ar3d_slice))
    alpha_mask = np.where(np.isnan(ar3d_slice), 0.0, 0.4)
    facecolors[..., -1] = alpha_mask

    ax2.plot_surface(
        lon_grid, lat_grid, p_grid,
        facecolors=facecolors,
        rstride=1, cstride=1,
        linewidth=0, antialiased=False,
        shade=False,
        zorder=1   # ←低めにしておく
    )

# 最下層マスク枠線描画
lon_grid, lat_grid = np.meshgrid(lon_sel, lat_sel)

dilated = binary_dilation(mask_bottom_cut.astype(float))

ax2.contour(
    lon_grid, lat_grid, dilated,
    levels=[0.5],          # 0-1境界
    colors='yellow',
    linewidths=2,
    offset=1000,
    zorder=999,          # ←最前面に出す
    alpha=1.0            # ←完全に不透明
)

# ax2.contour(
#     lon_grid, lat_grid, lsm.astype(float),
#     levels=[0,0.5,1],          # 0-1境界
#     cmap='Grays',
#     alpha=0.8            # ←完全に不透明
# )

# for idx, p in enumerate(pres[-4:-3]):
#     land_mask = land_cut[idx].astype(float)

#     lon_grid, lat_grid = np.meshgrid(lon_sel, lat_sel)

#     ax2.contourf(
#         lon_grid, lat_grid, land_mask.astype(float),
#         levels=[0.5,1.0],          # 0-1境界
#         cmap='gray',
#         linewidths=2,
#         offset=p,
#         alpha=0.4            # ←完全に不透明
#     )

ax2.set_xlabel("Longitude",fontsize=13)
ax2.set_ylabel("Latitude",fontsize=13)
ax2.set_zlabel("Pressure [hPa]",fontsize=13)
x_ticks = np.arange(0,91,30)
y_ticks = np.arange(-80,-29,10)
z_ticks = np.arange(300,1001,200)
ax2.set_xticks(x_ticks)
ax2.set_yticks(y_ticks)
ax2.set_zticks(z_ticks)
ax2.set_xticklabels(x_ticks,fontsize=12)
ax2.set_yticklabels(y_ticks,fontsize=12)
ax2.set_zticklabels(z_ticks,fontsize=12)
ax2.set_zlim(1000, 300)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax2, shrink=0.5, orientation='horizontal',aspect=40,pad=0.2)
cbar.set_label("Pressure [hPa]")
cbar.ax.invert_yaxis()
ax2.set_xlim(lon_min-1, lon_max+1)
ax2.set_ylim(lat_min, lat_max)
ax2.set_box_aspect([2, 3, 1]) 
ax2.view_init(elev=30, azim=20)

out = '/Users/takahashikazu/Desktop/NIPR/fig/'
out_dir = '{}AR_Detection_3d_monthly/DF_comp/'.format(out)#,Date.year,Date.month)

plt.tight_layout()
plt.savefig('{}AR2d_20_{}{:0=2}{:0=2}{:0=2}_grey.png'.format(out_dir,year,month,day,hour),dpi=700)
plt.show()
