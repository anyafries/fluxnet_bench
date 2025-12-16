import xarray as xr   # for the MSC function to work properly, not all pacakge versions work, recommended xarray version: 0.16.1!!!!!
import numpy as np
import datetime
from pandas.tseries.frequencies import to_offset
import model_metrics

def time_resample_mean(data_in, freq, loffset, label=None, skipna=False):
    # Handle deprecated frequency strings
    if freq == 'Y':
        freq = 'YE'
    elif freq == 'M':
        freq = 'ME'
    
    _data = data_in.resample(time=freq, label=label).mean(skipna=skipna)
    _data["time"] = _data.indexes["time"] + to_offset(loffset)
    return _data

def time_resample_sum(data_in, freq, loffset, label=None, skipna=False):
    if freq == 'Y':
        freq = 'YE'
    elif freq == 'M':
        freq = 'ME'
    
    _data = data_in.resample(time=freq, label=label).sum(skipna=skipna)
    _data["time"] = _data.indexes["time"] + to_offset(loffset)
    return _data

def rename_timeagg(data_in,tempres):
    if tempres == 'd':
        data_in = data_in.rename({'timeagg':'doy'})
    elif tempres == 'w':
        data_in = data_in.rename({'timeagg':'woy'})
    elif tempres == '8d':
        data_in = data_in.rename({'timeagg':'8d'})
    elif tempres == '16d':
        data_in = data_in.rename({'timeagg':'16d'})
    elif tempres == 'm':
        data_in = data_in.rename({'timeagg':'month'})
    return data_in
        
def get_timeaxisinfo(tcoord):
    if int(np.unique(np.diff(tcoord))[0]/10**9) == 86400:
        tempres ='d'
        subtimecoords = tcoord['time.dayofyear'].values
        subtimename = 'doy'
    elif int(np.max(np.unique(np.diff(tcoord)))/10**9) == 604800:
        tempres = 'w'
        subtimecoords = np.array( sum(list(np.array([list(np.arange(0,i)) for i in tcoord.astype('bool').groupby('time.year').sum()])),[]) )
        subtimename = 'woy'
    elif int(np.max(np.unique(np.diff(tcoord)))/10**9) == 691200:
        tempres = '8d'
        subtimecoords = np.reshape(np.array([list(np.arange(0,i)) for i in tcoord.astype('bool').groupby('time.year').sum()]),-1)
        subtimename = '8d'
    elif int(np.max(np.unique(np.diff(tcoord)))/10**9) == 691200*2:
        tempres = '16d'
        subtimecoords = np.array( sum(list(np.array([list(np.arange(0,i)) for i in tcoord.astype('bool').groupby('time.year').sum()])),[]) )
        subtimename = '16d'
    elif int(np.unique(np.diff(tcoord))[0]/10**9) >=2419200:
        tempres = 'm'
        subtimecoords = tcoord['time.month'].values
        subtimename = 'month'
    else:
        NotImplementedError('MSC computation only implemented for daily, 8daily, 16daily, monthly or weekly data')
    
    out = {'tempres':tempres,
            'subtimename': subtimename,
            'subtimecoords':subtimecoords}
    return out


def stack_timeagg(data, subtimecoords):
    time_id = np.arange(data.time.size)
    _stacked = []
    for year in np.unique(data["time.year"]):
        year_inc = data["time.year"]==year
        idx = time_id[year_inc]
        _stacked.append(data.isel(time=idx).assign_coords(year=year))
        _stacked[-1]["time"] = subtimecoords[idx]
        _stacked[-1] = _stacked[-1].\
            assign_coords(orig_time=("time", data.isel(time=idx).time.values)).\
            rename(time="timeagg")
    out = xr.concat(_stacked, dim="year", join="outer", coords="different", compat="equals")
    return out

def get_MSC_xarray(data_in1, mask=None, return_long=True, min_contribution=2, method='median', apply_mask=True, return_mask_outlier=False, z_outlier=None,test_direction=0):
    """
    Get a seasonal cycle of all valid data points
    
    Parameters
    ^^^^^^^^^^
    data_in1 : 
        xarray datarray float with data and time dimension
    mask : xarray datarray of same dimensions as data_in1, boolean
        must be provided if min_contribution<1
    apply_mask: True/False
        True: 
            compute MSC from all valid data points and return MSC only for doys when min_contribution full_filled
        False: 
            compute MSC from all valid data points and return result as is (without plausibility checks)

    return_long : boolean 
        False:
            shall a median seaosnal cycle be returned 
        True: 
          shall a time series of the same length as vi be returned containing the replicated MSC 

    min_contribution : 
        integer>=1: 
            put those values in the MSC to nan where less than n_min data points were available for each doy
        decimal <1:
            put those values in the MSC to nan where good quality data contributed less than min_contribution to the cumulative sum of all data

    method: str
        'median' or 'mean'
    return_mask_outlier: boolean
        shall a mask be returned that sets outlier values to false? 
        outliers are defined as samples that deviate more than z_outlier*std from the MSC per day if method==mean
        or as samples smaller than q25-z_outlier*iqr or higher than q75+z_outlier*iqr if method=='median'

    z_outlier: 
        threshold in terms of std or iqr that is allowed for the deviation in the outlier definition, if None, the defaults are 3 if method=='mean' and 1.5 if method=='median'
    test_direction: integer
          * -1 : test for negative outliers
          * 1 : test for positive outliers
          * 0 : test for both positive and negative outliers
    
    RETURNS
    ^^^^^^^
    xarray data array float with the seasonal cycle of length 366 or of the same length as vi

    """
    # determine temporal resolution
    timeinfo = get_timeaxisinfo(data_in1.time)
    tempres = timeinfo['tempres']
    subtimecoords = timeinfo['subtimecoords']

    # handle mask/ apply mask
    if min_contribution < 1:
        if mask is None:
            ValueError('Boolean mask needs to be provided for the computation of an MSC, because n_min is <1.')
        else:
            #mask = mask.assign_coords({'timeagg':('time',subtimecoords)})
            #mask = mask.assign_coords(year=mask['time.year'])    
            #mask = mask.set_index(time=['year', 'timeagg']).unstack('time') 
            mask = stack_timeagg(mask, subtimecoords)
            if tempres == 'd':
                m1 = mask.sel(timeagg = slice(1,365))
                m2 = mask.sel(timeagg = slice(366,367))
            elif tempres == 'w':
                m1 = mask.sel(timeagg = slice(0,51))
                m2 = mask.sel(timeagg = slice(52,53))
            elif tempres =='16d':
                m1 = mask.sel(timeagg = slice(0,22))
                m2 = mask.sel(timeagg = slice(23,24))
            else:# tempres == 'm':
                m1 = mask

    else:
        if mask is None:
            data_in = data_in1.copy()
        else:
            data_in = data_in1.where(mask)

    data_in = stack_timeagg(data_in, subtimecoords)

    if tempres == 'd': 
        d1 = data_in.sel(timeagg=slice(1,365))
        d2 = data_in.sel(timeagg=slice(366,367))
    elif tempres == 'w':
        d1 = data_in.sel(timeagg=slice(0,51))
        d2 = data_in.sel(timeagg=slice(52,53))
    elif tempres == '16d':
        d1 = data_in.sel(timeagg=slice(0,22))
        d2 = data_in.sel(timeagg=slice(23,24))
    else:# tempres == 'm':
        d1 = data_in
    
    idx_axis_tempres = d1.get_axis_num('year')
    if method == 'median':
        e1 = d1.median(dim='year',skipna=True)
    
    if method == 'mean':
        e1 = d1.mean(dim='year',skipna=True)
    
    if apply_mask:
        if min_contribution >= 1:
            e1n = np.isfinite(d1).sum(dim='year')
            e1  = e1.where(e1n >= min_contribution,np.nan)
            
        else:
            e1n = ((m1*np.abs(d1)).sum(dim='year',skipna=True)/(np.abs(d1)).sum(dim='year',skipna=True))
            e1  = e1.where(e1n >= min_contribution,np.nan)
            
    out = e1

    if tempres in ['d','w', '16d']:
        if method == 'median':
            e2 = d2.median(dim='year',skipna=True)
        
        if method == 'mean':
            e2 = d2.mean(dim='year',skipna=True)
        
        if apply_mask:
            if min_contribution >= 1:
                e2n = np.isfinite(d2).sum(dim='year')
                e2  = e2.where(e2n >= min_contribution,np.nan)
            else:
                e2n =((m2*np.abs(d2)).sum(dim='year',skipna=True)/(np.abs(d2)).sum(dim='year',skipna=True))
                e2  = e2.where(e2n >= min_contribution,np.nan)
        
        out = xr.concat((e1,e2),dim='timeagg')
    
    out = rename_timeagg(out,tempres)
    
    if return_long:
        out = expand_MSC(data_in1, out)
        labdrops = set(list(out.coords.keys()))-set(list(data_in1.coords.keys()))
        out = out.drop(labels=labdrops)

    if return_mask_outlier:
        if method == 'median':
            if tempres in ['d','w','16d']:
                #q75_np = np.column_stack( (np.nanquantile(d1,q=0.75,axis=idx_axis_tempres),np.nanquantile(d2,q=0.75,axis=idx_axis_tempres)))
                #q25_np = np.column_stack( (np.nanquantile(d1,q=0.25,axis=idx_axis_tempres),np.nanquantile(d2,q=0.25,axis=idx_axis_tempres)))
                idx_concate = d1.get_axis_num('timeagg')-1
                q75_np = np.concatenate( (np.nanquantile(d1,q=0.75,axis=idx_axis_tempres),np.nanquantile(d2,q=0.75,axis=idx_axis_tempres)),axis=idx_concate)
                q25_np = np.concatenate( (np.nanquantile(d1,q=0.25,axis=idx_axis_tempres),np.nanquantile(d2,q=0.25,axis=idx_axis_tempres)),axis=idx_concate)
            else:
                q75_np = np.nanquantile(d1,q=0.75,axis=idx_axis_tempres)
                q25_np = np.nanquantile(d1,q=0.25,axis=idx_axis_tempres)
            
            yy = data_in.year.values
            idx_leapyear = np.where(yy%4==0)[0][0]
            q75 = xr.DataArray(data=q75_np, coords=rename_timeagg(data_in,tempres).sel(year=yy[idx_leapyear]).coords) #selected year needs to be leap year
            q25 = xr.DataArray(data=q25_np, coords=rename_timeagg(data_in,tempres).sel(year=yy[idx_leapyear]).coords) #selected year needs to be leap year
            spread = xr.DataArray(data=q75-q25, coords=rename_timeagg(data_in,tempres).sel(year=yy[idx_leapyear]).coords) 
            q1 = expand_MSC(data_in1, q25 )
            q3 = expand_MSC(data_in1, q75 )
            if z_outlier is None:
                z_outlier = 1.5
        
        elif method == 'mean':
            if z_outlier is None:
                z_outlier = 3
            
            spread1 = d1.std(dim='year',skipna=True)
            if tempres in ['d','w','16d']:
                spread2 = d2.std(dim='year',skipna=True)
                spread = rename_timeagg( xr.concat((spread1,spread2),dim='timeagg'), tempres)
            else:
                spread = rename_timeagg( spread1, tempres)
            
            if return_long: 
                MSC = out
            else:
                MSC = expand_MSC(data_in1, out)
        
        spread = expand_MSC(data_in1, spread)
        
        if method=='median':
            upper_thres = q3 + z_outlier*spread 
            lower_thres = q1 - z_outlier*spread 
        elif method=='mean':
            upper_thres = MSC + z_outlier*spread #/ 0.6745
            lower_thres = MSC - z_outlier*spread #/ 0.6745
        
        #mask_outl = xr.DataArray(coords=data_in1.coords).isnull()
        mask_outl = xr.full_like(data_in1,fill_value=1)#.where(np.isfinite(data_in1))
        if test_direction==0:
            mask_outl = mask_outl.where(((data_in1 >= lower_thres) & (data_in1<=upper_thres)))
        elif test_direction==-1:
            mask_outl = mask_outl.where( data_in1 >= lower_thres )
        elif test_direction==1:
            mask_outl = mask_outl.where(data_in1<=upper_thres)
        else:
            raise ValueError('Parameter "test_direction" can only be 1,0, or -1.')
        
        out = [out.assign_attrs(**{**data_in.attrs, **{'z_outlier':z_outlier,'method':method}}), 
                mask_outl.assign_attrs(**{**data_in.attrs, **{'z_outlier':z_outlier,'method':method}}), 
                lower_thres.assign_attrs(**{**data_in.attrs, **{'z_outlier':z_outlier,'method':method}}), 
                upper_thres.assign_attrs(**{**data_in.attrs, **{'z_outlier':z_outlier,'method':method}})]
        
        for o in np.arange(0,len(out)):
            labdrops = set(list(out[o].coords.keys()))-set(list(data_in1.coords.keys()))
            out[o] = out[o].drop(labels=labdrops)
    
    return out


def expand_MSC(data_in1, MSC_short):
        """replicate the MSC (short, one annual cycle) to the length of the record of data_in
        both inputs are xarrays
        """
        if 'doy' in MSC_short.dims:
                subtimecoords = data_in1['time.dayofyear'].values
                MSC_short = MSC_short.rename({'doy':'timeagg'})
                tempres='d'
        elif 'woy' in MSC_short.dims:
                subtimecoords = np.array( sum(list(np.array([list(np.arange(0,i)) for i in data_in1.time.astype('bool').groupby('time.year').sum()])),[]) )
                MSC_short = MSC_short.rename({'woy':'timeagg'})
                tempres='w'
        elif '8d' in MSC_short.dims:
                subtimecoords = np.reshape(np.array([list(np.arange(0,i)) for i in data_in1.time.astype('bool').groupby('time.year').sum()]),-1)
                MSC_short = MSC_short.rename({'8d':'timeagg'})
                tempres='8d'
        elif '16d' in MSC_short.dims:
                subtimecoords = np.array( sum(list(np.array([list(np.arange(0,i)) for i in data_in1.time.astype('bool').groupby('time.year').sum()])),[]) )
                MSC_short = MSC_short.rename({'16d':'timeagg'})
                tempres='16d'
        elif 'month' in MSC_short.dims:
                subtimecoords = data_in1['time.month'].values
                MSC_short = MSC_short.rename({'month':'timeagg'})
                tempres='m'
        
        data_in  = data_in1.copy()
        orig_time = data_in['time']
        data_in = stack_timeagg(data_in, subtimecoords)

        if tempres=='d':
                d1 = data_in.sel(timeagg=slice(1,365))
                d2 = data_in.sel(timeagg=slice(366,367)) 
                msc1 = MSC_short.sel(timeagg=slice(1,365))
                msc2 = MSC_short.sel(timeagg=slice(366,367))
        elif tempres=='w':
                d1 = data_in.sel(timeagg=slice(0,51))
                d2 = data_in.sel(timeagg=slice(52,53))
                msc1 = MSC_short.sel(timeagg=slice(0,51))
                msc2 = MSC_short.sel(timeagg=slice(52,53))
        elif tempres=='16d':
                d1 = data_in.sel(timeagg=slice(0,22))
                d2 = data_in.sel(timeagg=slice(23,24))
                msc1 = MSC_short.sel(timeagg=slice(0,22))
                msc2 = MSC_short.sel(timeagg=slice(23,24))
        else: #if tempres=='m':
                d1 = data_in
                msc1 = MSC_short
                d2 = None

        data1, other1 = xr.broadcast(msc1, d1*np.nan)
        data1 = data1.stack(time=['year','timeagg']).reset_index('time')  # stack dataset into original shape

        if d2 is not None:
            if len(d2.timeagg)>0:
                data2, other2 = xr.broadcast(msc2, d2*np.nan)
                data2 = data2.stack(time=['year','timeagg']).reset_index('time')
                msc_long = xr.concat((data1, data2.where(data2.year % 4 == 0, drop=True)), dim='time').sortby(['year', 'timeagg'])
            else:
                msc_long = data1.sortby(['year', 'timeagg'])
        else:
                msc_long = data1.sortby(['year', 'timeagg'])

        
                
        msc_long['time'] = orig_time
        msc_long = msc_long.drop('year').drop('timeagg')

        return msc_long


def get_movWindow_MSC(x, nyears_window=5, mask=None, apply_mask=True,  min_contribution=2, method='median',return_long=False, return_mask_outlier=False, z_outlier=None,test_direction=0):
    """
    if nyears_window=0 do the normal MSC (ie no moving windows), only then return_long can be True or False
    if nyears_window>0, return_long is always True
    """
    if (((nyears_window>0) & (return_long==False))):
        raise ValueError("A sliding MSC shall be computed over nyears_window="+str(nyears_window)+'years, but return_long is set to False, but needs to be True.')
    
    timeinfo = get_timeaxisinfo(x.time)
    tempres = timeinfo['tempres']
    subtimecoords = timeinfo['subtimecoords']
    timeaggname = timeinfo['subtimename']
    
    if mask is None:
        mask = x.notnull()
    
    if nyears_window>0:
        yy = np.unique(x['time.year'])
        res = None
        res_outl = None
        for iy in yy:
            #print(iy)
            if iy-int(nyears_window/2) < np.min(yy):
                year1 = np.min(yy)
                year2 = np.min(yy)+nyears_window
            elif iy+int(nyears_window/2)+1 > np.max(yy):
                year1 = np.max(yy)-nyears_window
                year2 = np.max(yy)
            else:
                year1 = iy-int(nyears_window/2)
                year2 = iy+int(nyears_window/2)+1
            
            if mask is None:
                maske = mask
            else:
                maske = mask.sel(time=slice(str(year1),str(year2)))
            
            d_res = get_MSC_xarray(x.sel(time=slice(str(year1),str(year2))), mask=maske, 
                                        apply_mask = apply_mask,
                                        return_long=return_long, min_contribution=min_contribution, method=method,
                                        return_mask_outlier = return_mask_outlier, 
                                        z_outlier = z_outlier,
                                        test_direction = test_direction)
                            
            if return_mask_outlier:
                d = d_res[0]
            else:
                d = d_res
            
            if tempres=='d':
                tend = 365 #366
            elif tempres=='w':
                tend = 51 #52
            elif tempres=='8d':
                tend = 46
            elif tempres=='16d':
                tend = 22 #23
            elif tempres=='m':
                tend = 12
            
            if iy%4 == 0:
                if tempres in ['d','w','16d']:
                    tend = tend +1
            
            d1 = d.isel(time=slice(0,tend)).assign_coords({'time':x.sel(time=slice(str(iy),str(iy))).time.values[0:tend]})
            
            if res is None:
                res = d1
                if return_mask_outlier:
                    res_outl = d_res[1].sel(time=slice(str(iy),str(iy)))
                
            else:
                res = xr.concat((res,d1),dim='time')
                if return_mask_outlier:
                    res_outl = xr.concat((res_outl,d_res[1].sel(time=slice(str(iy),str(iy)))),dim='time')
        
        res1  = res
    else:
        res = get_MSC_xarray(x, mask=mask, apply_mask = apply_mask, return_long=return_long, min_contribution=min_contribution, method=method,
                                return_mask_outlier=return_mask_outlier, z_outlier=z_outlier,test_direction=test_direction)
        if return_mask_outlier:
            res_outl = res[1]
            res1  = res[0]
        else:
            res1 = res
    
    if (((return_mask_outlier) & (nyears_window>0))):
        return (res1.assign_attrs(**x.attrs), res_outl.assign_attrs(**x.attrs))
    elif (((return_mask_outlier) & (nyears_window==0))):
        return (res1.assign_attrs(**x.attrs), 
                res_outl.assign_attrs(**x.attrs), 
                res[2].assign_attrs(**{**x.attrs, **{'z_outlier':z_outlier,'method':method}}), 
                res[3].assign_attrs(**{**x.attrs, **{'z_outlier':z_outlier,'method':method}}) )
    else:
        return res1.assign_attrs(**x.attrs)##.drop(labels=labdrops)

def flux_d2w(data_d, mask_d, d2w_min_contribution=0.5, apply_mask=True):
    if d2w_min_contribution < 1:
        frac = time_resample_sum(np.abs(data_d)*mask_d, '7D', loffset='3D') / time_resample_sum(np.abs(data_d), '7D', loffset='3D')
    else:
        frac  = time_resample_sum(mask_d, '7D', loffset='3D')
        
    mask_w  = frac > d2w_min_contribution

    if apply_mask:
        return time_resample_mean(data_d, '7D', loffset='3D', skipna=False).where(mask_w).assign_attrs(units=data_d.units)  # skipna True?
    else:
        return time_resample_mean(data_d, '7D', loffset='3D', skipna=False).assign_attrs(units=data_d.units), mask_w


def flux_d2m(data_d, mask_d, d2m_min_contribution=0.5, apply_mask=True):
    if d2m_min_contribution < 1:
        frac = time_resample_sum(np.abs(data_d)*mask_d, 'M', loffset="15D", label='left') / time_resample_sum(np.abs(data_d), 'M', loffset="15D", label='left')
    else:
        frac  = time_resample_sum(mask_d, 'M', loffset="15D", label='left')
        
    mask_m  = frac > d2m_min_contribution
    
    if apply_mask:
        return time_resample_mean(data_d, 'M', loffset="15D", label='left', skipna=False).where(mask_m).assign_attrs(units=data_d.units)  # skipna True?
    else:
        return time_resample_mean(data_d, 'M', loffset="15D",label='left', skipna=False).assign_attrs(units=data_d.units), mask_m  # skipna True?


def flux_d2y(data_d, mask_d, d2y_min_contribution=0.5, apply_mask=True):
    if d2y_min_contribution < 1:
        frac = time_resample_sum(np.abs(data_d)*mask_d, 'Y', label='left', loffset="182D") / time_resample_sum(np.abs(data_d), 'Y', label='left', loffset="182D")
    else:
        frac  = time_resample_sum(mask_d, 'Y', label='left', loffset="182D")
    
    mask_y = frac >= d2y_min_contribution
    if apply_mask:
        return time_resample_mean(data_d, 'Y', label='left',loffset="182D", skipna=False).where(mask_y).assign_attrs(units=data_d.units)  # skipna True?
    else:
        return time_resample_mean(data_d, 'Y', label='left', loffset="182D", skipna=False).assign_attrs(units=data_d.units), mask_y


def MSC(data, mask, h2d_min_contribution=0.5, MSC_nmin_contributionDOY=2, msc_method='mean', return_long=False, apply_mask=True):
    if ('hour' in data.dims):
        data_d, mask_d = flux_h2d(data, mask, h2d_min_contribution=h2d_min_contribution, apply_mask=False)
    elif (((int(np.unique(np.diff(data.time))[0]/10**9)>=86400) & (int(np.unique(np.diff(data.time))[0]/10**9)<=604800))):
        data_d  = data
        mask_d  = mask
    else:
        NotImplementedError('MSC computation not implemented for data with temporal resolution lower than weekly.')
    
    msc = get_MSC_xarray(data_d, mask_d,return_long=return_long, min_contribution=MSC_nmin_contributionDOY, method=msc_method,apply_mask=apply_mask)
    return msc.assign_attrs(units=data_d.units)

def MSC_anom(data, mask, h2d_min_contribution=0.5, MSC_nmin_contributionDOY=2, msc_method='mean'):
    if ('hour' in data.dims):
        data_d = flux_h2d(data, mask, h2d_min_contribution=h2d_min_contribution, apply_mask=True)
    elif (((np.unique(np.diff(data.time))[0]/10**9>=86400) & (np.unique(np.diff(data.time))[0]/10**9<=604800))):
        data_d  = data
    else:
        NotImplementedError('Anomaly computation not implemented for data with temporal resolution lower than weekly.')
    msc = MSC(data, mask,
              h2d_min_contribution    = h2d_min_contribution,
              MSC_nmin_contributionDOY = MSC_nmin_contributionDOY,
              msc_method  = msc_method,
              return_long = True,
              apply_mask = True)
    out = data_d-msc
    #print('anom:'+str(out.notnull().sum()))
    return out.assign_attrs(units=data_d.units)


def IAV(data, mask, d2y_min_contribution=0.5):
    yearly = flux_d2y(data, mask, d2y_min_contribution=d2y_min_contribution)
    out = yearly-yearly.mean(dim='time', skipna=True)
    #print('iav:'+str(out.notnull().sum()))
    return out.assign_attrs(units=yearly.units)


def site_mean(data, mask, d2y_min_contribution=0.5):
    yearly = flux_d2y(data, mask, d2y_min_contribution=d2y_min_contribution)
    out = yearly.mean(dim="time", skipna=True).assign_attrs(units=yearly.units)
    #print('spatial:'+str(out.notnull().sum()))
    return out


def compute_diagnostics(obs,pred,mask,
                       compute_across        = "sites",
                       h2d_nmin              = 0.5,#10,  # minimum number of valid hours per day to aggregate hourly -> daily
                       d2w_nmin              = 0.5,# 5,  # minimum number of valid days per week to aggregate daily -> weekly
                       d2m_nmin              = 0.5,#10, # minimum number of valid days per month to aggregate daily -> monthly
                       d2y_nmin              = 0.5,#120, # minimum number of valid days per year to aggregate daily -> yearly
                       MDC_nmin              = 0.5,#120, # minimum number of valid days per year in  a given hour to caclucate a yearly mean diurnal cycle
                       msc_method            = 'mean',   # median
                       MSC_nminDOY           = 2,   #minimum number of valid day of year across year to calculate mean seasonal cycle
                       diag_scores = ['NSE', 'rRMSD','RMSD','bias', 'rbias', 'MAD','MrAD','MSE'], #'R2', 'Lambda','rov','pcor',
                       dynamic_scales = ['seasonal','iav','anom','spatial','daily','weekly','monthly']):
    statistics_dict = dict()
    for dynamics in dynamic_scales:
        if dynamics=='seasonal':
            x = MSC(obs, mask,return_long=False,msc_method=msc_method,h2d_min_contribution=h2d_nmin,MSC_nmin_contributionDOY=MSC_nminDOY)
            y = MSC(pred, mask,return_long=False,msc_method=msc_method,h2d_min_contribution=h2d_nmin,MSC_nmin_contributionDOY=MSC_nminDOY)
            bias_dim = 'doy'
        elif dynamics=='iav':
            x = IAV(obs, mask, d2y_min_contribution=d2y_nmin)
            y = IAV(pred, mask, d2y_min_contribution=d2y_nmin)
        elif dynamics=='anom':
            x = MSC_anom(obs, mask,msc_method=msc_method,h2d_min_contribution=h2d_nmin,MSC_nmin_contributionDOY=MSC_nminDOY)
            y = MSC_anom(pred, mask,msc_method=msc_method,h2d_min_contribution=h2d_nmin,MSC_nmin_contributionDOY=MSC_nminDOY)
            bias_dim = 'time'
        elif dynamics=='spatial':
            x = site_mean(obs, mask,d2y_min_contribution=d2y_nmin)
            y = site_mean(pred, mask,d2y_min_contribution=d2y_nmin)
        elif dynamics=='daily':
            x = obs.where(mask)
            y = pred.where(mask)
            bias_dim = 'time'
        elif dynamics=='weekly':
            x = flux_d2w(obs, mask,d2w_min_contribution=d2w_nmin,apply_mask=True)
            y = flux_d2w(pred, mask,d2w_min_contribution=d2w_nmin,apply_mask=True)
            bias_dim='time'
        elif dynamics=='monthly':
            x = flux_d2m(obs, mask,d2m_min_contribution=d2m_nmin,apply_mask=True)
            y = flux_d2m(pred, mask,d2m_min_contribution=d2m_nmin,apply_mask=True)
            	
        stats=[]
        if dynamics != 'spatial':
            q = np.array(x.dims)
            if compute_across=='sites':
                along_dim = q[q != 'site']
            elif compute_across == 'samples':
                along_dim = None
        else:
            along_dim = 'site'
        
        diag_functions = dict(
                                RMSD   = model_metrics.rmsd,
                                rRMSD  = model_metrics.rrmsd,
                                MAD    = model_metrics.MAD,
                                MrAD    = model_metrics.MrAD,
                                MSE    = model_metrics.MSE,
                                NSE    = model_metrics.mef,
                                bias   = model_metrics.bias,
                                rbias  = model_metrics.rbias,
                                # rov    = model_metrics.rov,
                                # pcor   = model_metrics.pcor,
                                # R2     = model_metrics.r2,
                                # Lambda = model_metrics.Glambda
                                )
        for dd in diag_scores:
            stats.append(diag_functions[dd](x,y,along=along_dim))
        
        res = xr.merge(stats)
        statistics_dict[dynamics]  = res
    
    return statistics_dict