import glob
import numpy as np
import os
import datetime
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.axes import Axes
import matplotlib.ticker as ticker

import satpy
from satpy.scene import Scene
from satpy import find_files_and_readers
import pyresample as prs

import cartopy.crs as ccrs
import cartopy.util as util
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

variables = ['FC','TC','IR'][::-1]
sensors = ['terra','aqua']

year = 2003

g = 9.80665

inputdata_dir = '/Volumes/Pegasus32 R6/takahashi/era5/'
z_file = '{}data/z_202301.nc'.format(inputdata_dir)
z_data = xr.open_dataset(z_file)['z'].sel(latitude=slice(-20,-90)).values[0]/g

for sensor in sensors:
    for variable in variables:
        if sensor == 'terra':
            file_name = glob.glob('/Volumes/Backup/MODIS_L1_21K/{}/MOD021KM.A{}*'.format(year,year))
        else:
            file_name = glob.glob('/Volumes/Backup/MODIS_L1_21K/{}/MYD021KM.A{}*'.format(year,year))

        senser_type = []
        Datetime = []    
        for i in file_name:
            
            # print(file_name)
            filename = os.path.splitext(os.path.basename(i))[0][:22]
            senser_type.append(filename[:9])
            Date = datetime.datetime(int(filename[10:14]),1,1,0) + datetime.timedelta(days=int(filename[14:17])-1,hours=int(filename[18:20]))
            Datetime.append(Date)

        ob_list = list(np.unique(Datetime))
        # print(ob_list)

        # file_list = glob.glob(ws_path + '*' + '202' + '*.csv')
        # #print(file_list)
        # file = file_list[0]
        # print(Datetime)
        for i in range(len(ob_list))[::1]:
            file_list = [file_name[idx] for idx,DateTime in enumerate(Datetime) if (DateTime == ob_list[i])]+ [file_name[idx] for idx,DateTime in enumerate(Datetime) if (DateTime == ob_list[i]+datetime.timedelta(hours=1))]
            Date = ob_list[i]
            print(Date)
        
            year = Date.year
            month = Date.month
            day = Date.day
            hour = Date.hour

            hour_0 = hour%10
            hour_1 = int((hour-hour_0)/10)

            lat,lon = -77.3, 39.7#position(year,month,day,hour)
            lat_syowa,lon_syowa = -69.007, 39.579#position(year,month,day,hour)
            # print(days)

            
            if not len(file_list) == 0:

                central_lon = 40
                out = '/Users/takahashikazu/Desktop/'
                if not os.path.exists('{}NIPR/fig/AR_Detection_3d_monthly/MODIS_{}/{}{:0=2}'.format(out,variable,year,month)):
                    os.makedirs('{}NIPR/fig/AR_Detection_3d_monthly/MODIS_{}/{}{:0=2}'.format(out,variable,year,month))
                out_dir = '{}NIPR/fig/AR_Detection_3d_monthly/MODIS_{}/{}{:0=2}/'.format(out,variable,year,month)

                inputdata_dir = '/Volumes/Pegasus32 R6/takahashi/era5/'
                mask_eq_dir = '/Volumes/Pegasus32 R8/takahashi/era5/ar_out_3d_monthly_threshold/ar_ivt_data/'#.format(inputdata_dir)
                mask_dir = '{}ar_out_era5_SH_20S/ar_ivt_data/'.format(inputdata_dir)

                file_eq = '{}select_groundAR_mask-{}-{}.nc'.format(mask_eq_dir,year,month)
                file = '{}select_AR_mask-{}{:0=2}.nc'.format(mask_dir,year,month)
                dataset_eq = xr.open_dataset(file_eq)
                dataset = xr.open_dataset(file)

                ar_mask_eq_dataset = dataset_eq['ar_mask'].sel(time='{}-{:0=2}-{:0=2}-{:0=2}'.format(year,month,day,hour),method='nearest').sel(latitude=slice(-20,-90))
                ar_mask_dataset = dataset['ar_mask'].sel(time='{}-{:0=2}-{:0=2}-{:0=2}'.format(year,month,day,hour),method='nearest').sel(latitude=slice(-20,-90))
                ar_mask_eq = ar_mask_eq_dataset.values
                ar_mask_eq[ar_mask_eq==np.nan] = 0
                ar_mask_eq = ar_mask_eq.max(axis=0)
                ar_mask = ar_mask_dataset.values
                latitude = ar_mask_eq_dataset.latitude.values
                longitude = ar_mask_eq_dataset.longitude.values
                
                surface_data = '{}data/surface/'.format(inputdata_dir)
                msl_file = '{}msl/msl_{}{:0=2}.nc'.format(surface_data,year,month)
                dataset = xr.open_dataset(msl_file)['msl'].sel(time='{}-{:0=2}-{:0=2}-{:0=2}'.format(year,month,day,hour)).sel(latitude=slice(-20,-90))
                msl = dataset.values/100
                
                land_mask_file = '{}input_data/lsm_2023-1.nc'.format(inputdata_dir)
                land_mask_data = xr.open_dataset(land_mask_file)['lsm'].sel(latitude=slice(-20,-90)).values[0]
                # msl[land_mask_data==1] = np.nan


                scn =Scene(filenames=file_list,reader='modis_l1b')

                scn.available_dataset_names()

                composite_ids = ['31','7','2','true_color','dust']
                scn.load(composite_ids, upper_right_corner='NE')

                # from pyresample import get_area_def

                # # area_id = 'sardinia_area_1km'
                # area_id = 'antarctic_area_1km'
                # description = 'Antarctica Polar Stereographic'
                # proj_id = 'polar_stereo_south'
                # projection = (
                #     '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
                # )

                # # 南極全体をカバーするおおよその範囲（単位はメートル）
                # area_extent = (-4200000, -4200000, 4200000, 4200000)  # 左,下,右,上


                # x_size = 5000  # 横画素数（例）
                # y_size = 5000  # 縦画素数（例）

                # areadef = get_area_def(area_id, description, proj_id, projection, x_size, y_size, area_extent)

                from pyproj import Proj, Transformer
                from pyresample import get_area_def

                # 中心経度（表示の中央にあたる経度）
                center_lon = 40

                # 基本設定
                area_id = 'antarctic_lambert'
                description = 'Antarctica with Lambert Conformal projection'
                proj_id = 'lambert_antarctica'

                # Lambert Conformal Conic 投影（標準緯線を-60度と-30度に）
                projection = (
                    '+proj=lcc +lat_1=-60 +lat_2=-30 +lat_0=-50 +lon_0=40 '
                    '+datum=WGS84 +units=m +no_defs'
                )

                proj = Proj(projection)
                transformer = Transformer.from_crs("epsg:4326", projection, always_xy=True)

                    # 緯度経度の矩形
                min_lon, min_lat = central_lon-45, -85
                max_lon, max_lat = central_lon+45, -55

                lons = np.linspace(min_lon, max_lon, 200)
                lats = np.linspace(min_lat, max_lat, 200)
                lon_grid, lat_grid = np.meshgrid(lons, lats)

                # 投影座標に変換
                x, y = transformer.transform(lon_grid, lat_grid)

                # 最小・最大を取る（これなら必ず緯度経度矩形をカバー）
                llx, urx = np.nanmin(x), np.nanmax(x)
                lly, ury = np.nanmin(y), np.nanmax(y)

                # 幅・高さ
                width_m  = urx - llx
                height_m = ury - lly

                # ★ アスペクト比を調整して Cartopy と同じ比率にする
                target_ratio = (max_lon - min_lon) / (max_lat - min_lat)   # 緯度経度での比率
                current_ratio = width_m / height_m

                if current_ratio > target_ratio:
                    # 横に広すぎる → 高さを増やす
                    extra = width_m / target_ratio - height_m
                    lly -= extra / 2
                    ury += extra / 2
                else:
                    # 縦に長すぎる → 横を増やす
                    extra = height_m * target_ratio - width_m
                    llx -= extra / 2
                    urx += extra / 2

                area_extent = (llx, lly, urx, ury)

                # 解像度とサイズを決める
                resolution = 5000
                x_size = int((urx - llx) / resolution)
                y_size = int((ury - lly) / resolution)

                # グリッドサイズ（ピクセル）
                # x_size = 1500
                # y_size = 1200

                # 表示範囲（メートル単位、中心から±範囲）
                # 例として、5000km x 4000km の範囲
                # x_extent = 4_000_000
                # y_extent = 4_000_000
                # area_extent = (-x_extent, -y_extent, x_extent, y_extent)

                # エリア定義の作成
                areadef = get_area_def(
                    area_id,
                    description,
                    proj_id,
                    projection,
                    x_size,
                    y_size,
                    area_extent
                )

                scn_resample_nc = scn.resample(areadef)

                # print(scn)
                if variable == 'IR':
                    image = np.asarray(scn_resample_nc["31"])#.transpose(1,2,0)
                    image = np.nan_to_num(image)
                    # image = np.interp(image, (np.percentile(image,1), np.percentile(image,99)), (0, 1))
                    crs = scn_resample_nc["31"].attrs["area"].to_cartopy_crs()
                elif variable == 'UV':
                    image = np.asarray(scn_resample_nc["7"])#.transpose(1,2,0)
                    image = np.nan_to_num(image)
                    # image = np.interp(image, (np.percentile(image,1), np.percentile(image,99)), (0, 1))
                    crs = scn_resample_nc["7"].attrs["area"].to_cartopy_crs()
                elif variable == 'TC':
                    image = np.asarray(scn_resample_nc["true_color"]).transpose(1,2,0)
                    image = np.nan_to_num(image)
                    image = np.interp(image, (np.percentile(image,1), np.percentile(image,99)), (0, 1))
                    crs = scn_resample_nc["true_color"].attrs["area"].to_cartopy_crs()
                elif variable == 'FC':
                    
                    # --- バンドデータ取得 ---
                    r = np.asarray(scn_resample_nc["7"])#.transpose(1,2,0) # Band 7 → R
                    g = np.asarray(scn_resample_nc["7"])#.transpose(1,2,0)  # Band 7 → G
                    b = np.asarray(scn_resample_nc["2"])#.transpose(1,2,0)  # Band 2 → B

                    # --- 簡単なスケーリング関数 ---
                    def stretch(data, lower=0.0, upper=1.0, minv=None, maxv=None):
                        if minv is None:
                            minv = np.nanpercentile(data, 2)
                        if maxv is None:
                            maxv = np.nanpercentile(data, 98)
                        stretched = (data - minv) / (maxv - minv)
                        return np.clip(stretched, lower, upper)

                    r_st = stretch(r)
                    g_st = stretch(g)
                    b_st = stretch(b)

                    image = np.dstack([r_st, g_st, b_st])
                    image = np.nan_to_num(image)
                    crs = scn_resample_nc["7"].attrs["area"].to_cartopy_crs()

                # ... and use it to generate an axes in our figure with the same CRS
                fig = plt.figure(figsize=[8,8])
                ax = fig.add_subplot(1, 1, 1, projection=crs)
                ax.set_extent([center_lon-39,center_lon+39,-85,-45],ccrs.PlateCarree())


                # ... and a lat/lon grid:
                gl = ax.gridlines(draw_labels=True, linestyle='--', ylocs=range(-90,11,10), xlocs=range(-180,181,30))
                # gl.top_labels=False
                # gl.right_labels=False
                # gl.xformatter=LONGITUDE_FORMATTER
                # gl.yformatter=LATITUDE_FORMATTER
                # gl.xlabel_style={'size':14}
                # gl.ylabel_style={'size':14}
                gl.xlocator = ticker.FixedLocator(np.arange(-180,181,30))
                gl.ylocator = ticker.FixedLocator(np.arange(-90,11,10))

                # In the end, we can plot our image data...
                if variable == 'IR':
                    im = ax.imshow(image, transform=crs, extent=crs.bounds, origin="upper", cmap='nipy_spectral', vmin=190, vmax=260, alpha=0.5)
                    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, label="Brightness Temperature (K)",shrink=0.7,aspect=40)
                    # Now we can add some coastlines...
                    ax.coastlines(resolution="10m", color="white")
                elif variable == 'UV':
                    im = ax.imshow(image, transform=crs, extent=crs.bounds, origin="upper", cmap='nipy_spectral', vmin=190, vmax=250,alpha=0.8)
                    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, label="Brightness Temperature (K)",shrink=0.7,aspect=40)
                    # Now we can add some coastlines...
                    ax.coastlines(resolution="10m", color="white")
                elif variable == 'TC':
                    im = ax.imshow(image, transform=crs, extent=crs.bounds, origin="upper")
                    # Now we can add some coastlines...
                    ax.coastlines(resolution="10m", color="blue")
                elif variable == 'FC':
                    im = ax.imshow(image, transform=crs, extent=crs.bounds, origin="upper")
                    # Now we can add some coastlines...
                    ax.coastlines(resolution="10m", color="blue")
                    
                ship = ax.scatter(lon,lat,color='k',transform=ccrs.PlateCarree(),s=75,zorder=4,marker='*')#facecolor='white',edgecolor='k'
                # ship = ax.scatter(lon_syowa,lat_syowa,facecolor='white',edgecolor='b',transform=ccrs.PlateCarree(),s=50,marker=',',zorder=5)

                ar_mask_eq, lon_cyc_v = util.add_cyclic_point(ar_mask_eq,coord=longitude)
                ar_mask, lon_cyc_v = util.add_cyclic_point(ar_mask,coord=longitude)
                # msl, lon_cyc_v = util.add_cyclic_point(msl,coord=longitude)
                z, lon_cyc_v = util.add_cyclic_point(z_data,coord=longitude)

                cf3 = ax.contour(lon_cyc_v,latitude,ar_mask,transform=ccrs.PlateCarree(),linewidths=2,colors='yellow',levels=[0.9,1.0],zorder=3,linestyles='-')
                cf3 = ax.contour(lon_cyc_v,latitude,ar_mask_eq,transform=ccrs.PlateCarree(),linewidths=2,colors='red',levels=[0.9,1.0],zorder=3,linestyles='-')
                # cf = ax.contourf(lon_cyc_v, latitude, ar_mask,
                #     levels=[0,0.9,1.1],
                #     colors='none',               # 塗りつぶし色なし
                #     hatches=['','///'],#, '\\\\\\\\', '....', 'xxx'],  # 斜線やドット
                #     transform=ccrs.PlateCarree()
                # )
                cf_msl = ax.contour(lon_cyc_v,latitude,z,transform=ccrs.PlateCarree(),linewidths=0.7,colors='black',levels=np.arange(500,4501,500),zorder=3,linestyles='-')
                ax.clabel(cf_msl,fontsize=8)

                # and add a title to our plot
                # plt.title("True color composite recorded by MODIS at " + scn_resample_nc.start_time.strftime("%Y-%m-%d %H:%M"), fontsize=20, pad=20.0)
                # ax.set_title('True color composite {}-{:0=2}-{:0=2} from {:0=2}00 to {:0=2}00 UTC'.format(year,month,day,hour,hour+1))
                if variable == 'IR':
                    ax.set_title('IR(11µm)  {}-{:0=2}-{:0=2} from {:0=2}00 to {:0=2}00 UTC'.format(year,month,day,hour,hour+1))
                    plt.tight_layout()
                    plt.savefig('{}IR_{}_{}{:0=2}{:0=2}.[{:0=2}-{:0=2}].png'.format(out_dir,sensor,year,month,day,hour,hour+1),dpi=700)
                elif variable == 'UV':
                    ax.set_title('UV(2µm)  {}-{:0=2}-{:0=2} from {:0=2}00 to {:0=2}00 UTC'.format(year,month,day,hour,hour+1))
                    plt.tight_layout()
                    plt.savefig('{}UV_{}_{}{:0=2}{:0=2}.[{:0=2}-{:0=2}].png'.format(out_dir,sensor,year,month,day,hour,hour+1),dpi=700)
                elif variable == 'TC':
                    ax.set_title('True color  {}-{:0=2}-{:0=2} from {:0=2}00 to {:0=2}00 UTC'.format(year,month,day,hour,hour+1))
                    plt.tight_layout()
                    plt.savefig('{}TC_{}_{}{:0=2}{:0=2}.[{:0=2}-{:0=2}].png'.format(out_dir,sensor,year,month,day,hour,hour+1),dpi=700)
                elif variable == 'FC':
                    ax.set_title('False color(2.1,2.1,0.85)  {}-{:0=2}-{:0=2} from {:0=2}00 to {:0=2}00 UTC'.format(year,month,day,hour,hour+1))
                    plt.tight_layout()
                    plt.savefig('{}FC_{}_{}{:0=2}{:0=2}.[{:0=2}-{:0=2}].png'.format(out_dir,sensor,year,month,day,hour,hour+1),dpi=700)
                plt.close()