# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 11:42:52 2021

@author: Ziemann
"""

import os
import numpy as np
import requests
import pygrib
import h5py
from skimage.transform import resize

'''
CFSR Parameters:
pressfc -- Surface pressure
tmpsfc  -- Surface temperature
t200    -- Temperature @ 200 HPa
t500    -- Temperature @ 500 HPa
t700    -- Temperature @ 700 HPa
t850    -- Temperature @ 850 HPa
t1000   -- Temperature @ 1000 HPa
wnd10m  -- Zonal (u) & meridional velocities @ 10m
wnd200  -- Zonal (u) & meridional velocities @ 200 HPa
wnd500  -- Zonal (u) & meridional velocities @ 500 HPa
wnd700  -- Zonal (u) & meridional velocities @ 700 HPa
wnd850  -- Zonal (u) & meridional velocities @ 850 HPa
wnd1000 -- Zonal (u) & meridional velocities @ 1000 HPa
'''

def download_CFSR():

    '''
    Input:
        main_url - NOAA database root data_url
        params   - parameter names to be downloaded
        months   - array of months to be downloaded, format "MM"
        years    - array of years to be downlaoded, format "YYYY"
        save_dir - root directory to save all files

    Downloads data from NOAA database, as specified by user.
    '''
    ############################################################################
    #-- User Inputs

    # Where to save the data to. Default is Ziemann's local large file storage directory.
    save_dir= "E:/HyPhyESN_Datasets/CFSR/T62"

    # NOAA database url to pull data from
    main_url = "https://www.ncei.noaa.gov/data/climate-forecast-system/access/reanalysis/time-series/"

    # Parameters you want to download
    params = ["pressfc", "tmpsfc", "t200", "t500", "t700", "t850", "t1000",
              "wnd10m", "wnd200", "wnd500", "wnd700", "wnd850", "wnd1000"]

    # Months and years you want to download/process
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    years = ["2005", "2006", "2007", "2008", "2009"]

    ############################################################################

    # Iterate through all years and months to grab all files
    for year in years:
        for month in months:
            # Generate the YYYYMM we're looking at
            tstamp = year+month
            print("Downloading "+str(tstamp)+"...")
            # Generate URL from timestamp
            tstamp_url = main_url+tstamp+"/"
            # Create local folder to save files in, format YYYYMM
            data_dir = save_dir+"/"+tstamp
            os.makedirs(data_dir, exist_ok=True)
            # Iterate through all listed parameters in each year/month combo
            for param in params:
                # Generate the filename to download and its url. .l gets T62 low res
                fname = param+".l.gdas."+tstamp+".grb2"
                data_url = tstamp_url+"/"+fname
                data_path = data_dir+"/"+fname
                # Check if the file has already been downloaded, then download it
                if not os.path.exists(data_path):
                    r = requests.get(data_url)
                    open(data_path, 'wb').write(r.content)
                    r2 = requests.get(data_url+".inv")
                    open(data_path+".inv", 'wb').write(r2.content)
    print("Complete!")


def process_CFSR_data():
    '''
    Take a time interval of CFSR hourly data and put it into standard dataset format
    for BaseESN. Read GRIB into numpy arrays, will need to later convert to .jld.
    (Julia doesn't have support for GRIB files that works on windows)

    JLD will import this with reverse formatting, so use the following here:

    formats:
    u[timestep][height][latitude][longitude]
    v[timestep][height][latitude][longitude]
    p[timestep][height=1][latitude][longitude]
    t[timestep][height][latitude][longitude]
    '''
    ############################################################################
    #-- User inputs

    # Where to load dataset from if processing
    dataset_directory = "E:/HyPhyESN_Datasets/CFSR/T62"

    # Months and years you want to download/process
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    years = ["2005", "2006", "2007", "2008", "2009"]

    # Where to save the data to. Default is Ziemann's local large file storage directory.
    save_dir= "E:/HyPhyESN_Datasets/CFSR/T62"

    # name of the .h5 dataset to save w/ all 4 parameters
    save_name = "CFSR_T62_5year_3height.hdf5"

    # create the .hdf5 file to save all arrays to.
    hdf5_dir = save_dir+"/"+save_name
    hf = h5py.File(hdf5_dir, 'w')

    ############################################################################

    #-- Load, format, and save the data for all parameters.

    #-- Longitudinal and latitudinal velocities.

    #wind_param = ["wnd10m", "wnd200", "wnd500", "wnd700", "wnd850", "wnd1000"]
    wind_param = ["wnd10m", "wnd500", "wnd1000"]
    # temp_param = ["tmpsfc", "t200", "t500", "t700", "t850", "t1000"]
    temp_param = ["tmpsfc", "t500", "t1000"]
    press_param = "pressfc"


    temporal_grid_u = []
    temporal_grid_v = []

    # Iterate altitude
    for param in wind_param:
        print("Parameter: u, v -- "+param)
        # Holds a single altitude's data over all time
        temporal_grid_u_slice = []
        temporal_grid_v_slice = []
        # Iterate years
        for year in years:
            # Iterate months
            for month in months:
                # Generate the YYYYMM we're loading
                tstamp = year+month
                print("    Time: "+str(tstamp))
                # Define the local folder load from
                data_dir = dataset_directory+"/"+tstamp

                # Generate the filename to load
                fname = param+".l.gdas."+tstamp+".grb2"
                data_path = data_dir+"/"+fname

                # Open the grib file
                uv_grbs = pygrib.open(data_path)

                # Pull hours 1-6 (hour0 is spinup data)
                uv_hour1 = uv_grbs.select(forecastTime=1)
                uv_hour2 = uv_grbs.select(forecastTime=2)
                uv_hour3 = uv_grbs.select(forecastTime=3)
                uv_hour4 = uv_grbs.select(forecastTime=4)
                uv_hour5 = uv_grbs.select(forecastTime=5)
                uv_hour6 = uv_grbs.select(forecastTime=6)

                # Iterate through each hour, copy the values into a list.
                for i in range(0,len(uv_hour1),2):
                    temporal_grid_u_slice.append(uv_hour1[i].values)
                    temporal_grid_u_slice.append(uv_hour2[i].values)
                    temporal_grid_u_slice.append(uv_hour3[i].values)
                    temporal_grid_u_slice.append(uv_hour4[i].values)
                    temporal_grid_u_slice.append(uv_hour5[i].values)
                    temporal_grid_u_slice.append(uv_hour6[i].values)
                    temporal_grid_v_slice.append(uv_hour1[i+1].values)
                    temporal_grid_v_slice.append(uv_hour2[i+1].values)
                    temporal_grid_v_slice.append(uv_hour3[i+1].values)
                    temporal_grid_v_slice.append(uv_hour4[i+1].values)
                    temporal_grid_v_slice.append(uv_hour5[i+1].values)
                    temporal_grid_v_slice.append(uv_hour6[i+1].values)

                # Close the file
                uv_grbs.close()

        # Copy the altitude you just pulled into the appropriate index of the master list.
        temporal_grid_u.append(np.asarray(temporal_grid_u_slice))
        temporal_grid_v.append(np.asarray(temporal_grid_v_slice))

    # Resize the lowest level array to match the size of the others. CFSR has a grid change from surface to higher altitudes.
    resize_shape = temporal_grid_u[1].shape
    temporal_grid_u[0] = resize(temporal_grid_u[0], resize_shape)
    temporal_grid_v[0] = resize(temporal_grid_v[0], resize_shape)

    # Convert to np array
    temporal_grid_u = np.asarray(temporal_grid_u)
    temporal_grid_v = np.asarray(temporal_grid_v)

    # Reshape from [height, timestep, latitude, longitude] to [timestep, height, latitude, longitude]
    temporal_grid_u = np.reshape(temporal_grid_u, (temporal_grid_u.shape[1],temporal_grid_u.shape[0],temporal_grid_u.shape[2],temporal_grid_u.shape[3]))
    # Reshape from [height, timestep, latitude, longitude] to [timestep, height, latitude, longitude]
    temporal_grid_v = np.reshape(temporal_grid_v, (temporal_grid_v.shape[1],temporal_grid_v.shape[0],temporal_grid_v.shape[2],temporal_grid_v.shape[3]))

    # save temporal_grid_P to h5 file, then delete it to make room in RAM.
    hf.create_dataset("temporal_grid_u", data=temporal_grid_u)
    hf.create_dataset("temporal_grid_v", data=temporal_grid_v)
    del temporal_grid_u, temporal_grid_u_slice
    del temporal_grid_v, temporal_grid_v_slice
    del uv_hour1, uv_hour2, uv_hour3, uv_hour4, uv_hour5, uv_hour6

    #-- Temperature
    temporal_grid_T = []
    # Iterate altitude
    for param in temp_param:
        print("Parameter: T -- "+param)
        # Holds a single altitude's data over all time
        temporal_grid_T_slice = []
        # Iterate years
        for year in years:
            # Iterate months
            for month in months:
                # Generate the YYYYMM we're loading
                tstamp = year+month
                print("    Time: "+str(tstamp))
                # Define the local folder load from
                data_dir = dataset_directory+"/"+tstamp

                # Generate the filename to load
                fname = param+".l.gdas."+tstamp+".grb2"
                data_path = data_dir+"/"+fname

                # Open the grib file
                T_grbs = pygrib.open(data_path)

                # Pull hours 1-6 (hour0 is spinup data)
                T_hour1 = T_grbs.select(forecastTime=1)
                T_hour2 = T_grbs.select(forecastTime=2)
                T_hour3 = T_grbs.select(forecastTime=3)
                T_hour4 = T_grbs.select(forecastTime=4)
                T_hour5 = T_grbs.select(forecastTime=5)
                T_hour6 = T_grbs.select(forecastTime=6)

                # Iterate through each hour, copy the values into a list.
                for i in range(len(T_hour1)):
                    temporal_grid_T_slice.append(T_hour1[i].values)
                    temporal_grid_T_slice.append(T_hour2[i].values)
                    temporal_grid_T_slice.append(T_hour3[i].values)
                    temporal_grid_T_slice.append(T_hour4[i].values)
                    temporal_grid_T_slice.append(T_hour5[i].values)
                    temporal_grid_T_slice.append(T_hour6[i].values)

                # Close the file
                T_grbs.close()

        # Copy the altitude you just pulled into the appropriate index of the master list.
        temporal_grid_T.append(np.asarray(temporal_grid_T_slice))

    # Resize the lowest level array to match the size of the others. CFSR has a grid change from surface to higher altitudes.
    temporal_grid_T[0] = resize(temporal_grid_T[0], resize_shape)

    # Convert to np array
    temporal_grid_T = np.asarray(temporal_grid_T)

    # Reshape from [height, timestep, latitude, longitude] to [timestep, height, latitude, longitude]
    temporal_grid_T = np.reshape(temporal_grid_T, (temporal_grid_T.shape[1],temporal_grid_T.shape[0],temporal_grid_T.shape[2],temporal_grid_T.shape[3]))

    # save temporal_grid_P to h5 file, then delete it to make room in RAM.
    hf.create_dataset("temporal_grid_T", data=temporal_grid_T)

    del temporal_grid_T, temporal_grid_T_slice
    del T_hour1, T_hour2, T_hour3, T_hour4, T_hour5, T_hour6


    #-- Surface pressure
    temporal_grid_P = []
    print("Parameter: Pressure")
    for year in years:
        for month in months:
            # Generate the YYYYMM we're loading
            tstamp = year+month
            print("    Time: "+str(tstamp))
            # Define the local folder load from
            data_dir = dataset_directory+"/"+tstamp

            # Generate the filename to load
            fname = press_param+".l.gdas."+tstamp+".grb2"
            data_path = data_dir+"/"+fname

            # Open the grib file
            P_grbs = pygrib.open(data_path)

            # Pull hours 1-6 (hour0 is spinup data)
            P_hour1 = P_grbs.select(forecastTime=1)
            P_hour2 = P_grbs.select(forecastTime=2)
            P_hour3 = P_grbs.select(forecastTime=3)
            P_hour4 = P_grbs.select(forecastTime=4)
            P_hour5 = P_grbs.select(forecastTime=5)
            P_hour6 = P_grbs.select(forecastTime=6)

            # Iterate through each hour, copy the values into a list.
            for i in range(len(P_hour1)):
                temporal_grid_P.append(P_hour1[i].values)
                temporal_grid_P.append(P_hour2[i].values)
                temporal_grid_P.append(P_hour3[i].values)
                temporal_grid_P.append(P_hour4[i].values)
                temporal_grid_P.append(P_hour5[i].values)
                temporal_grid_P.append(P_hour6[i].values)

            # Close the file
            P_grbs.close()


    # Convert to np array
    temporal_grid_P = np.asarray(temporal_grid_P)
    # Resize to other grid sizes. CFSR has a grid change from surface to higher altitudes.
    temporal_grid_P = resize(temporal_grid_P, resize_shape)
    # Reshape to appropriate ordering of params
    temporal_grid_P = np.reshape(temporal_grid_P, (temporal_grid_P.shape[0],1,temporal_grid_P.shape[1],temporal_grid_P.shape[2]))

    # save to hdf5 file, then delete it to make room in RAM.
    hf.create_dataset("temporal_grid_P", data=temporal_grid_P)
    del temporal_grid_P
    del P_hour1, P_hour2, P_hour3, P_hour4, P_hour5, P_hour6

    hf.close()
    print("Complete!")


def process_CFS_reforecast():

    '''
    Take a CFS reforecast dataset and put it into standard dataset format
    for BaseESN comparison. Read GRIB into numpy arrays, will need to later
    convert to .jld. (Julia doesn't have support for GRIB files that works on windows)

    JLD will import this with reverse formatting, so we use the following here:

    formats:
    u[timestep][height][latitude][longitude]
    v[timestep][height][latitude][longitude]
    p[timestep][height=1][latitude][longitude]
    t[timestep][height][latitude][longitude]
    '''
    ############################################################################
    #-- User inputs

    # Where to load dataset from if processing
    dataset_directory = "E:/HyPhyESN_Datasets/CFS_Reforecast/45-day/HighPriority"

    # Used to select files. Timestamps in YYYYMMDDHH, where HH is hour of day
    start_time = "2006070112"
    end_time = "2006081512"

    # Where to save the data to. Default is Ziemann's local large file storage directory.
    save_dir= "E:/HyPhyESN_Datasets/CFS_Reforecast/45-day/HighPriority"

    # name of the .h5 dataset to save w/ all 4 parameters
    save_name = "CFS_Reforecast_45day_2006070112.hdf5"

    # create the .hdf5 file to save all arrays to.
    hdf5_dir = save_dir+"/"+save_name
    hf = h5py.File(hdf5_dir, 'w')

    ############################################################################

    #-- Load, format, and save the data for all parameters.

    #-- Longitudinal and latitudinal velocities.

    #wind_param = ["wnd10m", "wnd200", "wnd500", "wnd700", "wnd850", "wnd1000"]
    wind_param = "wnd10m"
    # temp_param = ["tmpsfc", "t200", "t500", "t700", "t850", "t1000"]
    temp_param = "tmpsfc"
    press_param = "pressfc"

    temporal_grid_u = []
    temporal_grid_v = []

    # Generate the filename to load
    wfname = wind_param+"_f.01."+start_time+"."+end_time+"."+start_time+".grb2"
    wdata_path = dataset_directory+"/"+wfname

    # Open the grib file
    uv_grbs = pygrib.open(wdata_path)

    # Pull each component
    u = uv_grbs.select(name="10 metre U wind component")
    v = uv_grbs.select(name="10 metre V wind component")

    # Iterate through time, copy the values into a list.
    for i in range(len(u)):
        temporal_grid_u.append(u[i].values)
        temporal_grid_v.append(v[i].values)

    # Close the file
    uv_grbs.close()

    # Convert list to np array.
    temporal_grid_u = np.asarray(temporal_grid_u)
    temporal_grid_v = np.asarray(temporal_grid_v)

    # Resize to same resolution as CFSR grid
    resize_shape = (temporal_grid_u.shape[0],73,144)
    temporal_grid_u = resize(temporal_grid_u, resize_shape)
    temporal_grid_v = resize(temporal_grid_v, resize_shape)

    # Reshape from [timestep, latitude, longitude] to [timestep, height, latitude, longitude]
    temporal_grid_u = np.reshape(temporal_grid_u, (temporal_grid_u.shape[0],1,temporal_grid_u.shape[1],temporal_grid_u.shape[2]))
    # Reshape from [timestep, latitude, longitude] to [timestep, height, latitude, longitude]
    temporal_grid_v = np.reshape(temporal_grid_v, (temporal_grid_v.shape[0],1,temporal_grid_v.shape[1],temporal_grid_v.shape[2]))

    # save temporal_grid to h5 file, then delete it to make room in RAM.
    hf.create_dataset("temporal_grid_u", data=temporal_grid_u)
    hf.create_dataset("temporal_grid_v", data=temporal_grid_v)
    del temporal_grid_u, temporal_grid_v
    del u,v

    #-- Temperature
    temporal_grid_T = []

    # Generate the filename to load
    Tfname = temp_param+"_f.01."+start_time+"."+end_time+"."+start_time+".grb2"
    Tdata_path = dataset_directory+"/"+Tfname

    # Open the grib file
    T_grbs = pygrib.open(Tdata_path)

    # Pull each component
    T = T_grbs.select(name="Temperature")

    # Iterate through time, copy the values into a list.
    for i in range(len(T)):
        temporal_grid_T.append(T[i].values)

    # Close the file
    T_grbs.close()

    # Convert list to np array.
    temporal_grid_T = np.asarray(temporal_grid_T)
    
    # Resize to same resolution as CFSR grid
    temporal_grid_T = resize(temporal_grid_T, resize_shape)

    # Reshape from [timestep, latitude, longitude] to [timestep, height, latitude, longitude]
    temporal_grid_T = np.reshape(temporal_grid_T, (temporal_grid_T.shape[0],1,temporal_grid_T.shape[1],temporal_grid_T.shape[2]))

    # save temporal_grid to h5 file, then delete it to make room in RAM.
    hf.create_dataset("temporal_grid_T", data=temporal_grid_T)
    del temporal_grid_T
    del T


    #-- Surface pressure
    temporal_grid_P = []

    # Generate the filename to load
    Pfname = press_param+"_f.01."+start_time+"."+end_time+"."+start_time+".grb2"
    Pdata_path = dataset_directory+"/"+Pfname

    # Open the grib file
    P_grbs = pygrib.open(Pdata_path)

    # Pull each component
    P = P_grbs.select(name="Surface pressure")

    # Iterate through time, copy the values into a list.
    for i in range(len(P)):
        temporal_grid_P.append(P[i].values)

    # Close the file
    P_grbs.close()

    # Convert list to np array.
    temporal_grid_P = np.asarray(temporal_grid_P)
    
    # Resize to same resolution as CFSR grid
    temporal_grid_P = resize(temporal_grid_P, resize_shape)

    # Reshape from [timestep, latitude, longitude] to [timestep, height, latitude, longitude]
    temporal_grid_P = np.reshape(temporal_grid_P, (temporal_grid_P.shape[0],1,temporal_grid_P.shape[1],temporal_grid_P.shape[2]))

    # save temporal_grid to h5 file, then delete it to make room in RAM.
    hf.create_dataset("temporal_grid_P", data=temporal_grid_P)
    del temporal_grid_P
    del P

    hf.close()
    print("Complete!")

if __name__ == '__main__':
    process_CFS_reforecast()
