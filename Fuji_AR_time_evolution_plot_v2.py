import warnings
warnings.filterwarnings('ignore')

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
import pandas as pd
import cartopy.crs as ccrs
import cartopy.util as util
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from metpy.units import units
from metpy.constants import earth_avg_radius
import metpy.calc as mpcalc
import datetime 
from datetime import date
from dateutil.relativedelta import relativedelta
import os
import math
import glob
import copy
from sklearn.linear_model import LinearRegression
from scipy import signal
from scipy import stats as stats
import matplotlib.path as mpath

Lat = -77.3
Lon = 39.7
reanalysis = 'JRA3Q'

for reanalysis in ['ERA5','JRA3Q','MERRA2']:
    print(reanalysis)
    years = []
    months = []
    for yyyy in range(11):
        years += [2003]
        months += [yyyy+2]
        
    years += [2004]
    months += [1]

    if reanalysis == 'ERA5':
        inputdata_dir = '/Volumes/Pegasus32 R8/takahashi/era5/'
        time_res = 1
    elif reanalysis == 'JRA3Q':
        inputdata_dir = '/Volumes/HD-PCGU3-A/takahashi/data/JRA3Q/'
        time_res = 6
    elif reanalysis == 'JRA55':
        inputdata_dir = '/Volumes/HD-PCGU3-A/takahashi/data/JRA55/'
        time_res = 6
    elif reanalysis == 'MERRA2':
        inputdata_dir = '/Volumes/Pegasus32 R6/takahashi/era5/'
        time_res = 1
        # inputdata_dir = '/Volumes/HD-PCGU3-A/takahashi/data/MERRA2/'
        # time_res = 3
        
    resolution = int(24/time_res)

    ar3d_list = []
    # ar_can1_list = []
    # ar_list = []
    time_list = []
    ar_frequency = 0
    era_datetime = []
    ivt_list = []
    for n in range(len(years)):
        year = years[n]
        month = months[n]
        
        n_ivt_dataset = '/Volumes/Pegasus32 R6/takahashi/era5/input_data/n-ivt-{}-{}.nc'.format(year,month)
        e_ivt_dataset = '/Volumes/Pegasus32 R6/takahashi/era5/input_data/e-ivt-{}-{}.nc'.format(year,month)
        n_ivt_data = xr.open_dataset(n_ivt_dataset)['VIVT'].interp(latitude=Lat,longitude=Lon)
        e_ivt_data = xr.open_dataset(e_ivt_dataset)['UIVT'].interp(latitude=Lat,longitude=Lon)
        ivt = np.power(np.power(n_ivt_data.values, 2) + np.power(e_ivt_data.values, 2), 1/2)

        # mask_dir = '{}ar_out_era5_SH_20S/ar_ivt_data/'.format(inputdata_dir)
        # file = '{}select_AR_mask-{}{:0=2}.nc'.format(mask_dir,year,month)
        # dataset = xr.open_dataset(file)
        # ar_mask_dataset = dataset['ar_mask'].interp(latitude=-77.3,longitude=39.7)
        # ar_mask = ar_mask_dataset.values
        # t = ar_mask.shape[0]
        # ar_mask_daily = ar_mask.reshape(int(t/24),24).sum(axis=1)
        # # print(ar_mask)
        # ar_list.append(ar_mask_daily)

        # mask_dir = '{}ar_out_3d/mask_data/'.format(inputdata_dir)
        # file = '{}AR_candidates1_mask-{}-{}.nc'.format(mask_dir,year,month)
        # dataset = xr.open_dataset(file)
        # ar_mask_dataset = dataset['ar_mask'].interp(latitude=-77.3,longitude=39.7)
        # ar_mask_dataset.values[ar_mask_dataset.values==np.nan] = 0
        # ar_mask3d_can1 = ar_mask_dataset.values.max(axis=1)

        mask_dir = '{}ar_out_3d_monthly_threshold/ar_ivt_data/'.format(inputdata_dir)
        file = '{}select_groundAR_mask-{}-{}.nc'.format(mask_dir,year,month)
        dataset = xr.open_dataset(file)
        ar_mask_dataset = dataset['ar_mask'].interp(latitude=Lat,longitude=Lon)
        ar_mask_dataset.values[ar_mask_dataset.values==np.nan] = 0
        ar_mask_dataset.values = np.where(ar_mask_dataset.values>0,1,0)
        ar_mask3d = ar_mask_dataset.values.max(axis=1)
        t = ar_mask3d.shape[0]
        time = ar_mask_dataset.time.values[::resolution]
        
        # ar_can1 = np.where((ar_mask3d_can1-ar_mask3d)<1,0,1)
        
        ar_mask3d_daily = ar_mask3d.reshape(int(t/resolution),resolution).sum(axis=1)
        # ar_can1_daily = ar_can1.reshape(int(t/24),24).sum(axis=1)
        # print(time[2:4])
        # print(ar_mask3d.reshape(int(t/24),24)[2:4])
        # print(time[0],time[0].astype("datetime64[ms]").astype(datetime.datetime))
        ar3d_list.append(ar_mask3d_daily)
        # ar_can1_list.append(ar_can1_daily)
        time_list += [d.astype("datetime64[ms]").astype(datetime.datetime).date() for d in time]
        # print(type(time[0]))
        era_datetime += [d.astype("datetime64[ms]").astype(datetime.datetime) for d in ar_mask_dataset.time.values]
        ivt_list.append(ivt)
            
    # AR = np.concatenate(ar_list,axis=0)
    AR3d = np.concatenate(ar3d_list,axis=0)
    IVT = np.concatenate(ivt_list,axis=0)
    # ARcan1 = np.concatenate(ar_can1_list,axis=0)
    Time = time_list

    ############################### Precipitation JARE44
    dir = '/Users/takahashikazu/Downloads/Isotope_DF_prec_amount.xls'
    file = '{}'.format(dir)
    input_file = pd.ExcelFile(file)
    df = input_file.parse(input_file.sheet_names[0])

    syear = df[df.columns[1]][3:].reset_index(drop=True)
    smonth = df[df.columns[2]][3:].reset_index(drop=True)
    sday = df[df.columns[3]][3:].reset_index(drop=True)
    shour = df[df.columns[5]][3:].reset_index(drop=True)

    eyear = df[df.columns[6]][3:].reset_index(drop=True)
    emonth = df[df.columns[7]][3:].reset_index(drop=True)
    eday = df[df.columns[8]][3:].reset_index(drop=True)
    ehour = df[df.columns[10]][3:].reset_index(drop=True)

    prec = df[df.columns[11]][3:].reset_index(drop=True)
    temp = df[df.columns[15]][3:].reset_index(drop=True)
    # print(syear,smonth,sday,shour,prec)

    Date_ar = Time
    # for t in range(Time.shape[0]):
    #     print(Time[t].astype(datetime.datetime))
    #     Date_ar.append(Time[t].astype(datetime.datetime).datetime.date)

    except_tmp_list = []
    except_t_list = []
    except_d_list = []
    except_p_list = []
    except_era_p_list = []
    tmp_list = []
    t_list = []
    d_list = []
    p_list = []
    era_p_list = []
    ar_p = 0
    arcan1_p = 0
    total_p = 0
    ar_erap = 0
    arcan1_erap = 0
    total_erap = 0
    AR3D_list = []
    erap_event_list = []
    p_event_list = []
    p_event_d_list = []
    p_event_can1_list = []
    p_event_can1_d_list = []
    date_list = []#datetime.date(2003,5,5),datetime.date(2003,5,6),datetime.date(2003,9,30),datetime.date(2003,12,15),datetime.date(2003,12,14)]
    for t in range(len(syear)):
        Datetime = datetime.datetime(syear[t],smonth[t],sday[t],int(shour[t]/100),shour[t]-int(shour[t]/100)*100)
        Datetime_end = datetime.datetime(eyear[t],emonth[t],eday[t],int(ehour[t]/100),ehour[t]-int(ehour[t]/100)*100)
        Date = datetime.date(syear[t],smonth[t],sday[t])
        if not Date in date_list:
            t_list.append(Datetime)
            d_list.append(Date)
            p_list.append(prec[t])
            tmp_list.append(temp[t])
            # print(type(Date_ar[0]),type(Date))
            idx = Date_ar.index(Date)
            idx_1db = Date_ar.index((Date+datetime.timedelta(days=1)))
            idx_1da = Date_ar.index((Date-datetime.timedelta(days=1)))
            idx_2da = Date_ar.index((Date-datetime.timedelta(days=2)))
            # print(prec[t],type(prec[t]),np.isnan(prec[t]))
        
            if reanalysis == 'ERA5':
                ################################## ERA5
                if not smonth[t] == emonth[t]:
                    pr_file = ['/Volumes/HD-PCGU3-A/takahashi/data/ERA5/pr/pr_{}{:0=2}.nc'.format(syear[t],smonth[t]),'/Volumes/HD-PCGU3-A/takahashi/data/ERA5/pr/pr_{}{:0=2}.nc'.format(eyear[t],emonth[t])]
                    dataset = xr.open_mfdataset(pr_file)['tp'].interp(latitude=-77.3,longitude=39.7)
                else:
                    pr_file = '/Volumes/HD-PCGU3-A/takahashi/data/ERA5/pr/pr_{}{:0=2}.nc'.format(syear[t],smonth[t])
                    dataset = xr.open_dataset(pr_file)['tp'].interp(latitude=-77.3,longitude=39.7)
                    
                pr = 1000*dataset.sel(valid_time=slice('{:0=4}-{:0=2}-{:0=2}-{:0=2}'.format(syear[t],smonth[t],sday[t],int(shour[t]/100)+1),'{:0=4}-{:0=2}-{:0=2}-{:0=2}'.format(eyear[t],emonth[t],eday[t],int(ehour[t]/100)))).sum(axis=0).values
                era_p_list.append(pr)

            elif reanalysis == 'MERRA2':
                ##################################### MERRA2
                pr_file1 = '/Volumes/HD-PCGU3-A/takahashi/data/MERRA2/MERRA2_by_month/tp_{}{:0=2}.nc'.format(syear[t],smonth[t])
                pr_file2 =  '/Volumes/HD-PCGU3-A/takahashi/data/MERRA2/MERRA2_by_month/tp_{}{:0=2}.nc'.format(eyear[t],emonth[t])
                dataset1 = xr.open_dataset(pr_file1)['PRECTOT'].interp(lat=Lat,lon=Lon).values*1800
                dataset2 = xr.open_dataset(pr_file2)['PRECTOT'].interp(lat=Lat,lon=Lon).values*1800
                dataset = np.concatenate([dataset1,dataset2],axis=0)
                dataset = dataset + np.roll(dataset,1,axis=0)
                
                st = int((Datetime-datetime.datetime(syear[t],smonth[t],1,0)).total_seconds()/3600)+1
                et = int((Datetime_end-datetime.datetime(syear[t],smonth[t],1,0)).total_seconds()/3600)
                
                pr = dataset[st:et].sum(axis=0)
                era_p_list.append(pr)

            elif reanalysis == 'JRA3Q':
                ################################### JRA3Q
                if not smonth[t] == emonth[t]:
                    pr_file = ['/Volumes/HD-PCGU3-A/takahashi/data/JRA3Q/surface/tp/tp_{}{:0=2}.nc'.format(syear[t],smonth[t]),
                                '/Volumes/HD-PCGU3-A/takahashi/data/JRA3Q/surface/tp/tp_{}{:0=2}.nc'.format(eyear[t],emonth[t])]
                    dataset = xr.open_mfdataset(pr_file)['tprate1have-sfc-fc-ll125'].interp(lat=Lat,lon=Lon)*3600
                else:
                    pr_file = '/Volumes/HD-PCGU3-A/takahashi/data/JRA3Q/surface/tp/tp_{}{:0=2}.nc'.format(syear[t],smonth[t])
                    dataset = xr.open_dataset(pr_file)['tprate1have-sfc-fc-ll125'].interp(lat=Lat,lon=Lon)*3600
                pr = dataset.sel(time=slice('{:0=4}-{:0=2}-{:0=2}-{:0=2}'.format(syear[t],smonth[t],sday[t],int(shour[t]/100)+1),'{:0=4}-{:0=2}-{:0=2}-{:0=2}'.format(eyear[t],emonth[t],eday[t],int(ehour[t]/100)))).sum(axis=0).values
                era_p_list.append(pr)

        else:
            except_t_list.append(Datetime)
            except_d_list.append(Date)
            except_p_list.append(prec[t])
            except_tmp_list.append(temp[t])
            # print(type(Date_ar[0]),type(Date))
            idx = Date_ar.index(Date)
            idx_1db = Date_ar.index((Date+datetime.timedelta(days=1)))
            idx_1da = Date_ar.index((Date-datetime.timedelta(days=1)))
            idx_2da = Date_ar.index((Date-datetime.timedelta(days=2)))
            # print(prec[t],type(prec[t]),np.isnan(prec[t]))
        
            if reanalysis == 'ERA5':
                ################################## ERA5
                if not smonth[t] == emonth[t]:
                    pr_file = ['/Volumes/HD-PCGU3-A/takahashi/data/ERA5/pr/pr_{}{:0=2}.nc'.format(syear[t],smonth[t]),'/Volumes/HD-PCGU3-A/takahashi/data/ERA5/pr/pr_{}{:0=2}.nc'.format(eyear[t],emonth[t])]
                    dataset = xr.open_mfdataset(pr_file)['tp'].interp(latitude=-77.3,longitude=39.7)
                else:
                    pr_file = '/Volumes/HD-PCGU3-A/takahashi/data/ERA5/pr/pr_{}{:0=2}.nc'.format(syear[t],smonth[t])
                    dataset = xr.open_dataset(pr_file)['tp'].interp(latitude=-77.3,longitude=39.7)
                    
                pr = 1000*dataset.sel(valid_time=slice('{:0=4}-{:0=2}-{:0=2}-{:0=2}'.format(syear[t],smonth[t],sday[t],int(shour[t]/100)+1),'{:0=4}-{:0=2}-{:0=2}-{:0=2}'.format(eyear[t],emonth[t],eday[t],int(ehour[t]/100)))).sum(axis=0).values
                except_era_p_list.append(pr)

            elif reanalysis == 'MERRA2':
                ##################################### MERRA2
                pr_file1 = '/Volumes/HD-PCGU3-A/takahashi/data/MERRA2/MERRA2_by_month/tp_{}{:0=2}.nc'.format(syear[t],smonth[t])
                pr_file2 =  '/Volumes/HD-PCGU3-A/takahashi/data/MERRA2/MERRA2_by_month/tp_{}{:0=2}.nc'.format(eyear[t],emonth[t])
                dataset1 = xr.open_dataset(pr_file1)['PRECTOT'].interp(lat=Lat,lon=Lon).values*1800
                dataset2 = xr.open_dataset(pr_file2)['PRECTOT'].interp(lat=Lat,lon=Lon).values*1800
                dataset = np.concatenate([dataset1,dataset2],axis=0)
                dataset = dataset + np.roll(dataset,1,axis=0)
                
                st = int((Datetime-datetime.datetime(syear[t],smonth[t],1,0)).total_seconds()/3600)+1
                et = int((Datetime_end-datetime.datetime(syear[t],smonth[t],1,0)).total_seconds()/3600)
                
                pr = dataset[st:et].sum(axis=0)
                except_era_p_list.append(pr)

            elif reanalysis == 'JRA3Q':
                ################################### JRA3Q
                if not smonth[t] == emonth[t]:
                    pr_file = ['/Volumes/HD-PCGU3-A/takahashi/data/JRA3Q/surface/tp/tp_{}{:0=2}.nc'.format(syear[t],smonth[t]),
                                '/Volumes/HD-PCGU3-A/takahashi/data/JRA3Q/surface/tp/tp_{}{:0=2}.nc'.format(eyear[t],emonth[t])]
                    dataset = xr.open_mfdataset(pr_file)['tprate1have-sfc-fc-ll125'].interp(lat=Lat,lon=Lon)*3600
                else:
                    pr_file = '/Volumes/HD-PCGU3-A/takahashi/data/JRA3Q/surface/tp/tp_{}{:0=2}.nc'.format(syear[t],smonth[t])
                    dataset = xr.open_dataset(pr_file)['tprate1have-sfc-fc-ll125'].interp(lat=Lat,lon=Lon)*3600
                pr = dataset.sel(time=slice('{:0=4}-{:0=2}-{:0=2}-{:0=2}'.format(syear[t],smonth[t],sday[t],int(shour[t]/100)+1),'{:0=4}-{:0=2}-{:0=2}-{:0=2}'.format(eyear[t],emonth[t],eday[t],int(ehour[t]/100)))).sum(axis=0).values
                except_era_p_list.append(pr)

        if not np.isnan(prec[t])==True:
            if ((AR3d[idx] > 0) | (AR3d[idx_1db] > 0)) & (not Date in date_list) :# | (AR3d[idx_1db] > 0):# | (AR3d[idx_2da] > 0):
                # print(AR3d[idx])
                if prec[t] < 25:
                    # print(AR3d[idx])
                    ar_p += prec[t]
                    ar_erap += pr
                    p_event_list.append(prec[t])
                    erap_event_list.append(pr)
                    p_event_d_list.append(Date)
                    if (AR3d[idx] > 0) :
                        AR3D_list.append(AR3d[idx])
                    elif (AR3d[idx_1db] > 0)&(not AR3d[idx] > 0): 
                        AR3D_list.append(AR3d[idx_1db])
                    elif (AR3d[idx_1da] > 0)&(not AR3d[idx_1db] > 0)&(not AR3d[idx] > 0): 
                        AR3D_list.append(AR3d[idx_1da])
                    
            # if (AR3d[idx] > 0) | (AR3d[idx_1da] > 0) | (AR3d[idx_1db] > 0) | (ARcan1[idx] > 0) | (ARcan1[idx_1da] > 0) | (ARcan1[idx_1db] > 0):# | (AR3d[idx_2da] > 0):
            #     # print(AR3d[idx])
            #     arcan1_p += prec[t]
            #     arcan1_erap += pr
            #     p_event_can1_list.append(prec[t])
            #     p_event_can1_d_list.append(Date)
            
            
            if (not Date in date_list) :
                total_p += prec[t]
                total_erap += pr


    TP = np.array(era_p_list).sum()
    print(total_p,ar_p,100*ar_p/total_p)
    # print(total_p,arcan1_p,100*arcan1_p/total_p)

    print(total_erap,ar_erap,100*ar_erap/total_erap)
    print(TP,ar_erap,100*ar_erap/TP)
    # print(total_erap,arcan1_erap,100*arcan1_erap/total_erap)

    zip_lists = zip(p_event_list, p_event_d_list)
    # "reverse=True"を指定すると降順でソート
    zip_sort = sorted(zip_lists, reverse=True)
    # zipを解除
    p_sorted, d_sorted = zip(*zip_sort)
    # print(p_sorted)
    # print(d_sorted)
    # for index, value in enumerate(p_sorted):
    #     print(d_sorted[index],p_sorted[index])
    exsnow_ar = [d_sorted[index] for index, value in enumerate(p_sorted) if value > 0.19]
    indices_ar = [index for index, value in enumerate(p_sorted) if value > 0.19]

    mean_era = np.nanmean(np.array(erap_event_list),axis=0)
    std_era = np.nanstd(np.array(erap_event_list),axis=0)
    zip_lists = zip(erap_event_list, p_event_d_list)
    # "reverse=True"を指定すると降順でソート
    zip_sort = sorted(zip_lists, reverse=True)
    # zipを解除
    p_sorted, d_sorted = zip(*zip_sort)
    era_exsnow_ar = [d_sorted[index] for index, value in enumerate(p_sorted) if value > mean_era+std_era]
    era_indices_ar = [index for index, value in enumerate(p_sorted) if value > mean_era+std_era]
    # print(mean_era+std_era)

    # zip_lists = zip(p_event_can1_list, p_event_can1_d_list)
    # "reverse=True"を指定すると降順でソート
    # zip_sort = sorted(zip_lists, reverse=True)
    # zipを解除
    # p_sorted, d_sorted = zip(*zip_sort)
    # print(p_sorted)
    # print(d_sorted)
    # exsnow_ar_can1 = [d_sorted[index] for index, value in enumerate(p_sorted) if value > 0.19]
    # indices_ar_can1 = [index for index, value in enumerate(p_sorted) if value > 0.19]

    indices_total = [d_list[index] for index, value in enumerate(p_list) if value > 0.19]

    # print(len(p_event_list),len(AR3D_list))
    # time_list = p_event_d_list
    ar_idx_list = [idx for idx, AR in enumerate(AR3d) if AR>0]
    p_duration_list = []
    Dec_event = []
    for d in ar_idx_list:
        if (time_list[d] in exsnow_ar)&(not time_list[d].month == 12):
            if not d == 0:
                if (not time_list[d-1] == time_list[d])|(not time_list[d-1]+datetime.timedelta(days=1) == time_list[d]):
                    idxs = d_list.index(time_list[d])
                    if len(d_list) > idxs+10:
                        p_duration_list.append(np.array(p_list)[idxs:idxs+10])
                        # print(np.array(p_list)[idxs:idxs+10].shape)
            else:
                    idxs = d_list.index(time_list[d])
                    p_duration_list.append(np.array(p_list)[idxs:idxs+10])
                # else:
                #     idxs = d_list.index(time_list[d])
                #     print(idxs)
                #     p_duration_list.append(np.array(p_list)[idxs:])
        elif (time_list[d] in exsnow_ar)&(time_list[d].month == 12):
            if not d == 0:
                if (not time_list[d-1] == time_list[d])|(not time_list[d-1]+datetime.timedelta(days=1) == time_list[d]):
                    idxs = d_list.index(time_list[d])
                    if len(d_list) > idxs+10:
                        Dec_event += [np.array(p_list)[idxs:idxs+10]]
            

    p_duration = np.stack(p_duration_list+Dec_event[:1],axis=0)
    p_duration = p_duration/p_duration.max(axis=1)[:,None]
    # print(p_duration.shape)

    ar_idx_list = [idx for idx, AR in enumerate(AR3d) if AR>0]
    p_duration_list = []
    Dec_event = []
    for d in ar_idx_list:
        if (time_list[d] in era_exsnow_ar)&(not time_list[d].month == 12):
            if not d == 0:
                if (not time_list[d-1] == time_list[d])|(not time_list[d-1]+datetime.timedelta(days=1) == time_list[d]):
                    idxs = d_list.index(time_list[d])
                    if len(d_list) > idxs+10:
                        p_duration_list.append(np.array(p_list)[idxs:idxs+10])
                        # print(np.array(p_list)[idxs:idxs+10].shape)
            else:
                    idxs = d_list.index(time_list[d])
                    p_duration_list.append(np.array(p_list)[idxs:idxs+10])
                # else:
                #     idxs = d_list.index(time_list[d])
                #     print(idxs)
                #     p_duration_list.append(np.array(p_list)[idxs:])
        elif (time_list[d] in era_exsnow_ar)&(time_list[d].month == 12):
            if not d == 0:
                if (not time_list[d-1] == time_list[d])|(not time_list[d-1]+datetime.timedelta(days=1) == time_list[d]):
                    idxs = d_list.index(time_list[d])
                    if len(d_list) > idxs+10:
                        Dec_event += [np.array(p_list)[idxs:idxs+10]]
            

    erap_duration = np.stack(p_duration_list+Dec_event[:1],axis=0)
    erap_duration = erap_duration/erap_duration.max(axis=1)[:,None]
    # print(erap_duration.shape)


    # print(len(indices_ar),len(indices_total),100*len(indices_ar)/len(indices_total))
    # print(len(indices_ar_can1),len(indices_total),100*len(indices_ar_can1)/len(indices_total))
    df_p_event_ar = pd.DataFrame(p_event_d_list)
    print(df_p_event_ar)
    df_exp_event_ar = pd.DataFrame(exsnow_ar)
    print(df_exp_event_ar)
    # df_p_event_ar_can1 = pd.DataFrame(p_event_can1_d_list)
    # print(df_p_event_ar_can1)
    # df_exp_event_ar_can1 = pd.DataFrame(exsnow_ar_can1)
    # print(df_exp_event_ar_can1)
    df_p_event_ar.to_csv('{}ar_out_3d_monthly_threshold/ar_characteristics/P_event_ar.csv'.format(inputdata_dir))
    df_exp_event_ar.to_csv('{}ar_out_3d_monthly_threshold/ar_characteristics/ExP_event_ar.csv'.format(inputdata_dir))
    # df_p_event_ar_can1.to_csv('{}ar_out_3d/ar_characteristics/P_event_ar_candidate1.csv'.format(inputdata_dir))
    # df_exp_event_ar_can1.to_csv('{}ar_out_3d/ar_characteristics/ExP_event_ar_candidate1.csv'.format(inputdata_dir))
    df_exp_event = pd.DataFrame(indices_total)
    print(df_exp_event)
    df_exp_event.to_csv('{}ar_out_3d_monthly_threshold/ar_characteristics/ExP_event.csv'.format(inputdata_dir))
    # print(list(set(indices_total)-set(exsnow_ar_can1)))
    # print(list(set(indices_total)-set(exsnow_ar)))
    
    x = np.array(p_list, dtype=float)
    y = np.array(era_p_list, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]

    sigma = np.array(x).std()
    mean = np.array(x).mean()
    sigma_era = np.array(y).std()
    mean_era = np.array(y).mean()
    
    print(mean+sigma)
    print(mean_era+sigma_era)

    fig, ax = plt.subplots(figsize=(11, 4))  # 図のサイズを指定可能

    # 棒グラフを描画
    # bars1 = ax.bar(Time, AR3d, color='black', edgecolor='black', alpha=0.7)
    # bars3 = ax.bar(Time, ARcan1, color='gray', edgecolor='gray', alpha=0.5)
    AR_shade = np.where(AR3d>0,1,0)

    ax1 = ax.twinx()
    # ax2 = ax.twinx()
    # ax2 = ax.twinx()
    # bars3 = ax2.plot(t_list,tmp_list, color='black', alpha=1.0, linewidth=0.5)

    flags,xaxis,label = AR_shade,np.array(Time),'3D-AR'
    flags_roll = np.roll(flags,1,axis=0)
    f = flags-flags_roll
    starts = np.where(f == 1)[0]
    ends   = np.where(f == -1)[0]

    # 最初が1で始まっている場合、先頭に0を追加
    if len(starts) == 0 or (len(ends) > 0 and starts[0] > ends[0]):
        starts = np.r_[0, starts]

    # 最後が1で終わっている場合、末尾に最後のインデックスを追加
    if len(ends) == 0 or (len(starts) > 0 and starts[-1] > ends[-1]):
        ends = np.r_[ends, len(xaxis)-1]

    for n, (s, e) in enumerate(zip(starts, ends), start=1):
        start, end = xaxis[s], xaxis[e]
        if n == 1:
            shade = ax.axvspan(start, end, label=label, alpha=0.2, color='blue',zorder=1)
        else:
            shade = ax.axvspan(start, end, alpha=0.2, color='blue',zorder=1)
            
    bars2 = ax.bar(t_list, p_list, color='black', edgecolor='black', alpha=1.0,zorder=3)
    bars2 = ax.bar(except_t_list, except_p_list, color='gray', edgecolor='gray', alpha=1.0,zorder=2)
    bars4 = ax1.bar(t_list, era_p_list, color='black', edgecolor='black', alpha=1.0,zorder=3)
    bars4 = ax1.bar(except_t_list, except_era_p_list, color='gray', edgecolor='gray', alpha=1.0,zorder=2)
    # plot = ax2.plot(era_datetime,IVT,linewidth=0.5,color='blue')
    
    sig_obs = ax.axhline(0.19,color='red',linewidth=0.5)
    sig_era = ax1.axhline(0.12,color='red',linewidth=0.5)
    
    # 軸ラベルとタイトル
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Prec. Obs. [mm w.e.]', fontsize=10,color='black')
    ax1.set_ylabel('Prec. {} [mm w.e.]'.format(reanalysis), fontsize=10,color='black',loc='bottom')
    # ax2.set_ylabel(r'IVT [$kg\,m^{-1}s^{-1}$]', fontsize=10,color='blue',loc='top')
    # ax.set_title('Daily AR hours count', fontsize=14)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.01), borderaxespad=0)

    # グリッドを追加（オプション）
    ax.grid(linestyle='--', alpha=0.5)
    ax1.grid(linestyle='--', alpha=0.5)

    # 値を棒の上に表示（オプション）
    # for bar in bars:
    #     height = bar.get_height()
    #     ax.text(bar.get_x() + bar.get_width()/2, height + 0.1, str(height),
    #             ha='center', va='bottom', fontsize=10)

    # 表示
    ax.set_yticks(np.arange(0,2.1,0.25))
    ax1.set_yticks(np.arange(0,0.76,0.25))
    # ax2.set_yticks(np.arange(0,31,5))
    ax.set_yticklabels(np.arange(0,2.1,0.25))#,color='red')
    ax1.set_yticklabels(np.arange(0,0.76,0.25))#,color='red')
    # ax2.set_yticklabels(np.arange(0,31,5),color='blue')
    ax.set_ylim(-0.75,2.125)
    ax1.set_ylim(0,2.875)
    # ax2.set_ylim(-25,32.5)
    ax.set_xticks([datetime.datetime(2003,2,1,0),datetime.datetime(2003,3,1,0),datetime.datetime(2003,4,1,0),datetime.datetime(2003,5,1,0),datetime.datetime(2003,6,1,0),datetime.datetime(2003,7,1,0),
                   datetime.datetime(2003,8,1,0),datetime.datetime(2003,9,1,0),datetime.datetime(2003,10,1,0),datetime.datetime(2003,11,1,0),datetime.datetime(2003,12,1,0),datetime.datetime(2004,1,1,0)])
    ax.set_xticklabels(['2003-Feb','2003-Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','2004-Jan'])
    ax.set_xlim(t_list[0],t_list[-1])
    
    from scipy.stats import pearsonr, linregress

    x = np.array(p_list, dtype=float)
    y = np.array(era_p_list, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]

    # ピアソン相関係数
    r, pval = pearsonr(x, y)

    ax.set_title(r'$r$='+'{:.2f}'.format(r))

    out = '/Users/takahashikazu/Desktop/NIPR/fig/AR_Detection_3d_monthly/DF_comp_no_exception/'
    if not os.path.exists('{}{}'.format(out,reanalysis)):#,Date.year,Date.month)):
        os.makedirs('{}{}'.format(out,reanalysis))#,Date.year,Date.month))
    out_dir = '{}{}/'.format(out,reanalysis)#,Date.year,Date.month)

    plt.savefig('{}AR_time_evolusion_DF_p2_test.png'.format(out_dir),dpi=700)

    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))  # 図のサイズを指定可能

    for i in range(p_duration.shape[0]):
        pl = ax.plot(np.arange(0,10),p_duration[i],linewidth=0.5,color='gray')
        
    pl_dec = ax.plot(np.arange(0,10),p_duration[-1],linewidth=1,color='blue')
    pl_m = ax.plot(np.arange(0,10),np.nanmean(p_duration,axis=0),linewidth=1,color='red')

    ax.grid(linestyle='--', alpha=0.5)
    ax.set_yticks(np.arange(0,1.1,0.2))
    ax.set_ylim(0,1.1)
    ax.set_xlim(0,5)
    ax.set_xlabel('Days after AR landfall', fontsize=12)
    ax.set_ylabel('Normalized prec.', fontsize=12,color='black')


    plt.savefig('{}AR_ExP_time_evolusion_DomeFuji.png'.format(out_dir),dpi=700)

    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))  # 図のサイズを指定可能

    for i in range(erap_duration.shape[0]):
        pl = ax.plot(np.arange(0,10),erap_duration[i],linewidth=0.5,color='gray')
        
    # pl_dec = ax.plot(np.arange(0,10),erap_duration[-1],linewidth=1,color='blue')
    pl_m = ax.plot(np.arange(0,10),np.nanmean(erap_duration,axis=0),linewidth=1,color='red')

    ax.grid(linestyle='--', alpha=0.5)
    ax.set_yticks(np.arange(0,1.1,0.2))
    ax.set_ylim(0,1.1)
    ax.set_xlim(0,5)
    ax.set_xlabel('Days after AR landfall', fontsize=12)
    ax.set_ylabel('Normalized prec.', fontsize=12,color='black')


    plt.savefig('{}AR_eraExP_time_evolusion_DomeFuji.png'.format(out_dir),dpi=700)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))  # 図のサイズを指定可能
    # 例: obs, era が Python list の場合
    obs = np.array(p_list)
    era = np.array(era_p_list)


    # NaN除去
    x = np.array(p_list, dtype=float)
    y = np.array(era_p_list, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]

    # ピアソン相関係数
    r, pval = pearsonr(x, y)
    print('ALL')

    print("相関係数:", r, "p値:", pval)

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    print(f"回帰直線: y = {slope:.3f}x + {intercept:.3f}",(len(x)))

    # 描画
    sc = ax.scatter(x, y, alpha=1,s=3,c='k')
    pl = ax.plot(x, slope*x + intercept, color="red", label=f"y = {slope:.3f}x + {intercept:.3f}")
    ax.legend()
    ax.grid(linestyle='--', alpha=0.5)
    ax.set_yticks(np.arange(0,2.1,0.2))
    ax.set_xticks(np.arange(0,2.1,0.2))
    ax.set_ylim(0,1)
    ax.set_xlim(0,1)
    ax.set_xlabel('Observation', fontsize=12)
    ax.set_ylabel('ERA5', fontsize=12)

    plt.savefig('{}eraExP_scatter_DomeFuji_z.png'.format(out_dir),dpi=700)

    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))  # 図のサイズを指定可能

    mask = ((x<1.1))# & ((not x>0.2)|(not y<0.01)))
    x, y = x[mask], y[mask]

    # ピアソン相関係数
    r, pval = pearsonr(x, y)
    print('Without Dec')
    print("相関係数:", r, "p値:", pval)

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    print(f"回帰直線: y = {slope:.3f}x + {intercept:.3f}",(len(x)))

    # 描画
    sc = ax.scatter(x, y, alpha=1,s=3,c='k')
    pl = ax.plot(x, slope*x + intercept, color="red", label=f"y = {slope:.3f}x + {intercept:.3f}")
    ax.legend()
    ax.grid(linestyle='--', alpha=0.5)
    ax.set_yticks(np.arange(0,2.1,0.2))
    ax.set_xticks(np.arange(0,2.1,0.2))
    ax.set_ylim(0,1)
    ax.set_xlim(0,1)
    ax.set_xlabel('Observation', fontsize=12)
    ax.set_ylabel('ERA5', fontsize=12)

    plt.savefig('{}eraExP_scatter_DomeFuji_without_Dec_z.png'.format(out_dir),dpi=700)

    plt.close()

