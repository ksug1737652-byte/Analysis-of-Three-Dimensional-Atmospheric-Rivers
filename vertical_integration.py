import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import pandas as pd
import cartopy.crs as ccrs
import cartopy.util as util
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import datetime 
from datetime import date
from dateutil.relativedelta import relativedelta
from scipy import signal
import glob
import copy

years = np.arange(1979,2024,1) # List of years to include
months = [1,2,3,4,5,6,7,8,9,10,11,12] # List of months to include

ro = 997

#dir = '/Volumes/PromisePegasus/takahashi/ERA5/'
#dir_sp = '{}surface_pressure/'.format(dir)
#dir_d2m = '{}dewpoint/'.format(dir)
#dir_t2m = '{}TEMP/'.format(dir)
#dir_u10 = '{}UWND/'.format(dir)
#dir_v10 = '{}VWND/'.format(dir)
#dir_u = '{}UWND/'.format(dir)
#dir_v = '{}VWND/'.format(dir)
#dir_q = '{}SPFH/'.format(dir)

dir = '/Volumes/Pegasus32 R6/takahashi/era5/data/'
dir_sp = '{}surface/sp/'.format(dir)
dir_d2m = '{}surface/td/'.format(dir)
dir_u10 = '{}surface/u/'.format(dir)
dir_v10 = '{}surface/v/'.format(dir)
dir_t2m = '{}surface/2t/'.format(dir)
dir_q = '{}pressure/q/'.format(dir)
dir_u = '{}pressure/u/'.format(dir)
dir_v = '{}pressure/v/'.format(dir)
#dir_t = '{}pressure/t/'.format(dir)

out_dir = '/Volumes/Pegasus32 R6/takahashi/era5/' # Names input data dir

def v_integration(plev,spv,y,ys,pfac=100):
    g = 9.80665 
    nt,ny,nx = ys.shape
    y_int =np.zeros((nt,ny,nx))

    i=0
    for p in plev[0:]:
        i -= 1
        mask = (spv >= p)
        if(i < -1):
            maskF = maskr & mask
        else:
            maskF = copy.copy(mask)
        maskr = np.logical_not(mask)

        y_m = ((np.roll(y,-1,axis=1) + y)*0.5)[:,:i,:]
        level_dif = (np.roll(plev,-1) - plev)[:i]

        y_int[maskF] = (((y_m*level_dif[np.newaxis,:,np.newaxis,np.newaxis]).sum(axis=1)+(spv-p)*(y[:,i,:] + ys)*0.5)/g*pfac)[maskF]
    return y_int

for year in years:
    for month in months:
        print('processing...',year,month)

        file_pres = '{}/sp_{}{:0=2}.nc'.format(dir_sp,year,month)
        file_sfq = '{}/td_{}{:0=2}.nc'.format(dir_d2m,year,month)
        file_sfu = '{}/u_{}{:0=2}.nc'.format(dir_u10,year,month)
        file_sfv = '{}/v_{}{:0=2}.nc'.format(dir_v10,year,month)
        file_q = '{}/q_{}{:0=2}.nc'.format(dir_q,year,month)
        file_u = '{}/u_{}{:0=2}.nc'.format(dir_u,year,month)
        file_v = '{}/v_{}{:0=2}.nc'.format(dir_v,year,month)
        p = xr.open_dataset(file_pres)['sp']/100
        dp = xr.open_dataset(file_sfq)['d2m']
        sfu = xr.open_dataset(file_sfu)['u10']
        sfv = xr.open_dataset(file_sfv)['v10']
        if year >= 2024:
            q = xr.open_dataset(file_q)['q'][:,::-1].sel(level=slice(300,1000))
            u = xr.open_dataset(file_u)['u'][:,::-1].sel(level=slice(300,1000))
            v = xr.open_dataset(file_v)['v'][:,::-1].sel(level=slice(300,1000))
        else:
            q = xr.open_dataset(file_q)['q'].sel(level=slice(300,1000))
            u = xr.open_dataset(file_u)['u'].sel(level=slice(300,1000))
            v = xr.open_dataset(file_v)['v'].sel(level=slice(300,1000))
        dp = dp-273.15
        
        #caluculationg specific humidity on the surface from dew point and surface pressure
        e = 6.1078 * 10**((7.5*(dp))/(dp+237.3))
        q_e = 0.622*e/(p-(1-0.622)*e)
        #print(q_e)
        sfq = q_e


        #Making coords for out as netcdf
        nc = xr.open_dataset(file_sfq)
        axis_lon = nc.variables['longitude']
        axis_lat = nc.variables['latitude']
        axis_time = nc.variables['time']
        print(p.shape)
        hours = p.shape[0]
        #days = int(days)

        plev = q['level'].values
        print(plev)

        print(hours,'days')

        for hour in range(hours):
            print('hours',hour+1)

            Date = date(year,month,1)+datetime.timedelta(hours=hour)
            spv = p[hour].values[np.newaxis,:,:]
            qv = q[hour].values[np.newaxis,:,:,:]
            sqv = sfq[hour].values[np.newaxis,:,:]
            uv = u[hour].values[np.newaxis,:,:,:]
            suv = sfu[hour].values[np.newaxis,:,:]
            vv = v[hour].values[np.newaxis,:,:,:]
            svv = sfv[hour].values[np.newaxis,:,:]

            u_flux = qv*uv
            su_flux = sqv*suv
            v_flux = qv*vv
            sv_flux = sqv*svv

            u_int = v_integration(plev,spv,u_flux,su_flux,pfac=100)
            v_int = v_integration(plev,spv,v_flux,sv_flux,pfac=100)
            q_int = v_integration(plev,spv,qv,sqv,pfac=100)
            
            #specific water's unit is converted from kg to cm
            q_v = q_int
            if hour==0:
                qu_data = u_int
                qv_data = v_int
                q_data = q_v
            else:
                qu_data = np.concatenate([qu_data,u_int])
                qv_data = np.concatenate([qv_data,v_int])
                q_data = np.concatenate([q_data,q_v])

        
        #print('U-Component of Vapor Transport',qu_data.shape)
        #print(qu_data)
        #print('V-Component of Vapor Transport',qv_data.shape)
        #print(qv_data)
        
        out_u = xr.DataArray(
            qu_data,
            name = 'UIVT',
            coords = (
                axis_time,axis_lat,axis_lon,
            ),
            attrs = {
                'long_name':'Zonal Integrated Vapor Transport',
                'units':'kg/m/s'
            },
        )
        outfile_u = '{}/input_data/e-ivt-{}-{}.nc'.format(out_dir,year,month)
        out_u.to_netcdf(outfile_u)
        
        out_v = xr.DataArray(
            qv_data,
            name = 'VIVT',
            coords = (
                axis_time,axis_lat,axis_lon,
            ),
            attrs = {
                'long_name':'Meridional Integrated Vapor Transport',
                'units':'kg/m/s'
            },
        )
        outfile_v = '{}/input_data/n-ivt-{}-{}.nc'.format(out_dir,year,month)
        out_v.to_netcdf(outfile_v)

        out_q = xr.DataArray(
            q_data,
            name = 'IWV',
            coords = (
                axis_time,axis_lat,axis_lon,
            ),
            attrs = {
                'long_name':'Integrated Water Vapor',
                'units':'kg m**-2'
            },
        )
        outfile_q = '{}/input_data/IWV-{}-{}.nc'.format(out_dir,year,month)
        out_q.to_netcdf(outfile_q)


        #fig = plt.figure(figsize=(4,2))
        #ax = fig.add_subplot(111)
        #cf = ax.contourf(qv_data.mean(axis=0)[::-1])
        #plt.colorbar(cf)
        #plt.show()
  





