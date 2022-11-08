# Creates the zero wind phase calibration file by analyzing chunks of L1.5 data. Developed on Python 3.7.
# The script is at the bottom, which updates an existing set of zero wind determinations if new data is available.
# The script, as currently implemented, only works when run on UCB/SSL computers with access to L1.5 data. If any user
# wants to run this independently, they will need to modify the "get_L15_fn" function to point to the Level 1.5 (known as
# "consolidated") data in their own file structure, and also change the "zero_dir" variable.
# Author: Brian J. Harding, July 2021

import numpy as np
from datetime import datetime, timedelta
import MIGHTI_L2
import pandas as pd
import xarray as xr
import glob
import warnings # to avoid nanmean warning
import gc
from scipy.optimize import minimize

    
    
############### GLOBAL PARAMETERS ###############
__version__ = '1.0.0' # Version of this software (zero_wind.py)
VERS = 'v03' # Which version of the L1.5 (and thus L1) dataset to interrogate
TPREC = pd.to_timedelta('48d')
NC_ENG = 'netcdf4' # This is annoying but was found to be necessary in order to save string coordinates as unicode not bytes.
# The rows listed below are inclusive (i.e., the specified row will be reported in the data)
GREEN_NIGHT_ROW_MIN = 1 # In practice, the lowest green row does not reliably have enough signal
GREEN_NIGHT_ROW_MAX = 9 # In simulations, rows 10 and above do not have reliable zero wind phases
GREEN_DAY_ROW_MIN   = 1 # In practice, the lowest green row does not reliably have enough signal
GREEN_DAY_ROW_MAX   = 50 # This one is TBD, but row 50 = 224 km is a conservative choice. No green line data are useful above here.
RED_NIGHT_ROW_MIN   = 16 # In simulations, rows 0-3 do not have reliable zero wind phases. This maps to L1 rows 0-15.
TSTART = pd.to_datetime('2019-12-06') # Start of science data. The first day data will be processed. Zero phase will be extrapolated back to this day.
#################################################
   
    
def groupbin(df, col='los_wind', describe='50%', colx='slt', coly='lat', nx=24, ny = 20, countthresh = 5):
    '''
    Bin a dataset into 2D bins and return a single descriptive statistic for each bin.
    This is fairly self-explanatory, but effort was taken to ensure that if 
    specific bins are requested, the returned DataFrame has all of those bins, even if they are empty. This 
    is not how pd.groupby normally works.
    
    INPUTS:
    - describe: '50%', '25%', 'mean', 'std', 'count', etc... whatever is returned by groupby(...).describe()
    - nx and ny: ints or arrays
        - ints: number of bins. These will be automatically generated and results trimed to only include bins where there are data
        - arrays: edges of bins. Using this input will force the shape of the returned table to be the size of these arrays (minus 1)
    
    OUTPUTS:
    x, y, p
    x: 1D bin edges
    y: 1D bin edges
    p: 2D values
    '''
    specify_bins = hasattr(nx, '__len__') # If true, use specific bins set by the user
    
    if specify_bins:
        xint = pd.IntervalIndex.from_arrays(nx[:-1], nx[1:]) # This includes "unobserved" columns
        yint = pd.IntervalIndex.from_arrays(ny[:-1], ny[1:])
        
        cut_x, _ = pd.cut(df[colx], bins=xint, retbins=True)
        cut_y, _ = pd.cut(df[coly], bins=yint, retbins=True)
    else:
        cut_x, _ = pd.cut(df[colx], bins=nx, retbins=True)
        cut_y, _ = pd.cut(df[coly], bins=ny, retbins=True)
    
    g = df[col].groupby([cut_x, cut_y])
    c = g.describe()
    i = c['count'] < countthresh
    c.loc[i,:] = np.nan
    p = c.pivot_table(index = c.index.get_level_values(1), columns=c.index.get_level_values(0)) # pivot table containing each variable
    p2 = p[describe]
    
    if specify_bins: # Re-index and re-column so that all requested bins are represented, even if they are empty
        p3 = pd.DataFrame(index=yint, columns=xint, dtype=float) # This is the "full sized" version of p2
        for i, idx in enumerate(yint):
            for j, col in enumerate(xint):
                try:
                    p3.loc[idx,col] = p2.loc[idx,col]
                except Exception as e:
                    pass    
        p2 = p3 # Overwrite p2
        
    # Coordinate axes, of size nx+1 and ny+1 (for plotting with pcolormesh)
    x = np.concatenate(([c.left for c in p2.columns], [p2.columns[-1].right]))
    y = np.concatenate(([c.left for c in p2.index],   [p2.index[-1].right]))
    
    return x, y, p2
    
    
    
    
def L15s_to_dataset(L15_fns, skip=1):
    '''
    Load a series of L1.5 files into a xarray.Dataset
    
    skip = Only load every skip^th timestep, to save mem. Default 1, which means load everything.
    
    '''
    
    fns = L15_fns[:]
    fns.sort()
    
    dss = []
    for fn in fns:
        ds = xr.open_dataset(fn)
        if skip > 1:
            ds = ds.isel(time=slice(None,None,skip))
        dss.append(ds) 
    ds = xr.concat(dss, dim='time')
    gc.collect()
    
    # Compute ascending/descending
    
    asc = (ds.lat.diff('time') > 0).astype(float)
    asc_full = np.nan * np.zeros((asc.shape[0]+1, asc.shape[1]), dtype=asc.dtype)
    try: # This will fail if there is only one time stamp, which really shouldn't happen anyway 
        asc_full[:-1,:] = asc
        asc_full[-1,:] = asc[-2,:]
    except IndexError:
        print('WARNING: ascending calc failed, probably because there is only one timestamp in this set.')
    ds = ds.assign(asc=(('time','row'), asc_full))
    
    # Ensure strings are str (annoying handing back/forth between Python 2/3 and different versions of xarray)
    for var in ['mode', 'sensor', 'color']:
        ds[var] = ds[var].astype(str)

    return ds





def get_L15_fn(t, sensor='A', color='green', vers=3):
    
    '''
    t = pandas datetime
    rev = 'latest' or 0 or 1 or 2, etc.
    sensor = 'A' or 'B'
    color = 'green' or 'red'
    
    Always grab latest rev
    '''
    
    year = t.year
    doy = (t - datetime(year, 1, 1)).days+1
    tstr = t.strftime("%Y-%m-%d")
    
    fns = glob.glob('/disks/data/icon/Repository/Archive/CALIBRATION/MIGHTI/Intermediate/%i/*L1-5_MIGHTI-%s*%s_%s_*%sr???.NC'% (year,\
                                                                                                                sensor, color.capitalize(), tstr, vers))
    
    if len(fns) == 0:
        raise Exception('No files found')
    fns.sort()
    return fns[-1]

    
def find_all_L15_files(vers=None, tmin = None, tmax = None):
    '''
    Return filenames for the given version and time ranges
    
    fnsAg, fnsBg, fnsAr, fnsBr
    '''
    
    if vers is None:
        vers = VERS
        
    if tmin is None:
        tmin = TSTART
    
    if tmax is None:
        tmax = pd.to_datetime(datetime.now().date())
    
    datevec = pd.date_range(tmin, tmax)
    
    fnsAg = []
    fnsAr = []
    fnsBg = []
    fnsBr = []
    for date in datevec:

        try:
            fn = get_L15_fn(date, sensor='A', color='green', vers=vers)
            fnsAg.append(fn)
        except Exception as e:
            pass

        try:
            fn = get_L15_fn(date, sensor='A', color='red', vers=vers)
            fnsAr.append(fn)
        except Exception as e:
            pass

        try:
            fn = get_L15_fn(date, sensor='B', color='green', vers=vers)
            fnsBg.append(fn)
        except Exception as e:
            pass

        try:
            fn = get_L15_fn(date, sensor='B', color='red', vers=vers)
            fnsBr.append(fn)
        except Exception as e:
            pass

    return fnsAg, fnsBg, fnsAr, fnsBr
    
    
      
def run_row(dfA, dfB, gap_matching=False, verbose=False, norm=2, weight=False):
    '''
    For a (long-term) set of MIGHTI-A and -B data from a particular row, find the zero wind phase.
    At least 1 precession cycle is needed for a robust result.
    
    dfA, dfB: DataFrames for single rows
    gap_matching = If True, and if there is a gap, make sure that the LT/lat sampling matches on the ascending/descending orbits.
                   This is a performance hit.
    norm = 2: Use least squares (2-norm minimization) [Default]
           1: Use least absolute deviation (1-norm minimization)
    weight = if True, implemented weighting based on the low signal correction
                   
    Returns z0A, z0B, u, v
    '''
    
    dfAm = dfA.copy()
    dfBm = dfB.copy()
    
    # If there are no (or very few) samples in this dataset, then there's nothing we can do.
    # At least make sure there are both ascending and descending data
    if (dfAm.phase.where(dfAm.asc == 1).count() == 0) | (dfAm.phase.where(dfAm.asc == 0).count() == 0) | \
       (dfBm.phase.where(dfBm.asc == 1).count() == 0) | (dfBm.phase.where(dfBm.asc == 0).count() == 0):
        if verbose:
            print('\tMissing asc and/or desc')
        return np.nan, np.nan, np.nan, np.nan 
    
    # Check for gap matching
    if gap_matching:
        try:
            dtA = (dfAm.index[1:] - dfAm.index[:-1])
            dtB = (dfBm.index[1:] - dfBm.index[:-1])
            if max(dtA.max(), dtB.max()) > pd.to_timedelta('25h'): # Only worry about this if there is a 2-day gap or more
                if verbose:
                    print('\tUsing gap matching')
#                 # TODO: Is there a smarter way to pick these parameters? This choice seems to work well in simulation.
#                 latg = np.linspace(-45, 45, 30)
#                 sltg = np.arange(0,25)
#                 countthresh = 5
                # This version works better on the reduced dataset used for testing
                latg = np.linspace(-45, 45, 15)
                sltg = np.linspace(0,24,14)
                countthresh = 2

                dfAa = dfAm.loc[dfAm.asc == 1]
                dfAd = dfAm.loc[dfAm.asc == 0]

                dlat = dfBm.lat.diff()
                dfBa = dfBm.loc[dfBm.asc == 1]
                dfBd = dfBm.loc[dfBm.asc == 0]

                _, _, cAa = groupbin(dfAa, col='phase', describe='count', colx='slt', coly='lat', nx=sltg, ny=latg, countthresh=0)
                _, _, cAd = groupbin(dfAd, col='phase', describe='count', colx='slt', coly='lat', nx=sltg, ny=latg, countthresh=0)
                _, _, cBa = groupbin(dfBa, col='phase', describe='count', colx='slt', coly='lat', nx=sltg, ny=latg, countthresh=0)
                _, _, cBd = groupbin(dfBd, col='phase', describe='count', colx='slt', coly='lat', nx=sltg, ny=latg, countthresh=0)

                keep = (cAa > countthresh) & (cAd > countthresh) & (cBa > countthresh) & (cBd > countthresh)
                
                nx = sltg
                ny = latg
                colx = 'slt'
                coly = 'lat'
                xint = pd.IntervalIndex.from_arrays(nx[:-1], nx[1:]) # This includes "unobserved" columns (copied from icon_analyze.groupbin())
                yint = pd.IntervalIndex.from_arrays(ny[:-1], ny[1:])

                # Mask the samples that are in lat/LT gaps
                for df in [dfAm, dfBm]:
                    cut_x, _ = pd.cut(df[colx], bins=xint, retbins=True)
                    cut_y, _ = pd.cut(df[coly], bins=yint, retbins=True)
                    k = np.zeros(len(df), dtype=bool)
                    idx = ~(cut_x.isna() | cut_y.isna()) # careful to avoid any samples that are outside the bounds
                    k[idx] = keep.stack().loc[zip(cut_y[idx], cut_x[idx])]
                    df[~k] = np.nan
        except Exception as e:
            print('\tError in gap matching: %s' % e)
            return np.nan, np.nan, np.nan, np.nan


    # Angles to radians
    dfAm['azr'] = dfAm.az * np.pi/180.
    dfBm['azr'] = dfBm.az * np.pi/180.

    # Fit zero wind errors by solving least squares matrix equation assuming avg u and v is a constant
    NA = len(dfAm)
    NB = len(dfBm)

    lhs = np.concatenate((dfAm.phase, dfBm.phase))
    A = np.zeros((NA+NB, 4)) # rhs is [u_ave, v_ave, z0A, z0B]
    A[:NA,0] = -np.sin(dfAm.azr)
    A[:NA,1] = -np.cos(dfAm.azr)
    A[:NA,2] = 1.
    A[NA:,0] = -np.sin(dfBm.azr)
    A[NA:,1] = -np.cos(dfBm.azr)
    A[NA:,3] = 1.
    
    if weight:
        # Apply weights/errors based on low-signal correction. 
        # (Statisical errors average out but low-signal correction does not. Do not put
        #  too much weight on the data points that have been corrected a lot).
        sigAsq = 0.01**2 + dfAm.low_signal_corr**2 # 0.01 minimum error
        sigBsq = 0.01**2 + dfBm.low_signal_corr**2
        wA = 1/sigAsq
        wB = 1/sigBsq
        w = np.concatenate((wA,wB))
        lhs = lhs*w
        A = A*w[:,None]

    # Remove rows (equations) with nans
    idx = np.isfinite(A).all(axis=1) & np.isfinite(lhs)
    A = A[idx,:]
    lhs = lhs[idx]
    if verbose:
        print('\t%i equations' % len(lhs))
        
    # Run the inversion
    z0A_sig = np.nan
    z0B_sig = np.nan
    try:
        x, res, _, _ = np.linalg.lstsq(A, lhs, rcond=None ,)

        if norm == 1: # Run least absolute deviation using the least squares solution as the initial guess
            def cost_function(x):
                yp = A.dot(x)
                return np.sum(np.abs(lhs - yp))

            out = minimize(cost_function, x)
            x = out.x

        z0A = x[2]
        z0B = x[3]
        ustar = x[0]
        vstar = x[1]
        
        
#         # Report uncertainty -- use residual to bootstrap data error estimate # This is commented because it isn't used.
#         resid = A.dot(rhs) - lhs
#         lhs_sig = np.sqrt(np.mean(resid**2)) 
#         cov = lhs_sig**2 * np.linalg.inv(A.T.dot(A))
#         z0A_sig = np.sqrt(cov[0,0])
#         z0B_sig = np.sqrt(cov[1,1])
        
    except Exception as e:
        print('Least squares failed: %s' % e)
        return np.nan, np.nan, np.nan, np.nan

    

    return z0A, z0B, ustar, vstar




def run_case(t, dsA, dsB, verbose=False, gap_matching=False, norm=2, restrict_lt = False, weight=False, smooth48=False):
    '''
    For given xr.Datasets for MTA and MTB (all rows), run the zw for all rows and return a dataset.
    
    This is intended to run all rows from one of four cases:
     - Green Night
     - Green Day
     - Red Night
     - Red Day
     Presumably with the cal lamps either on or off, but not both (but that's up to the user)
     
    t: The center date of the window
    dsA, dsB: xarray.Dataset with many rows
    gap_matching = If True, and if there is a gap, make sure that the LT/lat sampling matches on the ascending/descending orbits.
                   This is a performance hit.
    norm = 2: Use least squares (2-norm minimization)
           1: Use least absolute deviation (1-norm minimization)
    restrict_lt = If True, only use data with 3 hours of noon (day mode) or midnight (night mode) local time
    weight = if True, implemented weighting based on the low signal correction
    smooth48 = if True, ensure that the 48-day averaged phases are smooth as a function of altitude (i.e., the hpf function returns 0)
    '''
    rows = dsA.row
    Nr = len(rows)
    z0A = np.nan * np.zeros(Nr)
    z0B = np.nan * np.zeros(Nr)
    ustar = np.nan * np.zeros(Nr)
    vstar = np.nan * np.zeros(Nr)
    
    if (dsA.phase.count() > 1) & (dsB.phase.count() > 1): # Check if there's any data at all
        color = dsA.color.dropna(dim='time')[0].item()
        mode  = dsA.mode.dropna(dim='time')[0].item()

        for nr in range(Nr):
            row = dsA.isel(row=nr).row
            if verbose:
                print('%i / %i, row=%i, color=%s, mode=%s' % (nr, Nr, row, color, mode))

            # Don't waste time running certain rows if the row isn't required or if there is limited data
            if (color == 'green') & (mode == 'Night') & (row < GREEN_NIGHT_ROW_MIN):
                if verbose: 
                    print('\tSkipping masked row...')
                continue
            if (color == 'green') & (mode == 'Night') & (row > GREEN_NIGHT_ROW_MAX):
                if verbose: 
                    print('\tSkipping masked row...')
                continue
            if (color == 'green') & (mode == 'Day') & (row < GREEN_DAY_ROW_MIN):
                if verbose: 
                    print('\tSkipping masked row...')
                continue
            if (color == 'green') & (mode == 'Day')  & (row > GREEN_DAY_ROW_MAX):
                if verbose: 
                    print('\tSkipping masked row...')
                continue
            if (color == 'red')  & (mode == 'Night') & (row < RED_NIGHT_ROW_MIN):
                if verbose: 
                    print('\tSkipping masked row...')
                continue
            if (dsA.phase.count() < 50) | (dsB.phase.count() < 50):
                if verbose: 
                    print('\tSkipping, too few samples (MTA=%i, MTB=%i)...' % (dsA.phase.count().item(), dsB.phase.count().item()))
                continue
            
            dfA = dsA.isel(row=nr).to_dataframe()
            dfB = dsB.isel(row=nr).to_dataframe()
            if restrict_lt:
                if mode == 'Day':
                    dfA = dfA.loc[abs(dfA.slt - 12) < 3]
                    dfB = dfB.loc[abs(dfB.slt - 12) < 3]
                if mode == 'Night':
                    dfA = dfA.loc[(dfA.slt > 21) | (dfA.slt < 3)]
                    dfB = dfB.loc[(dfB.slt > 21) | (dfB.slt < 3)]
            
            a,b,u,v = run_row(dfA, dfB, gap_matching=gap_matching, verbose=verbose, norm=norm, weight=weight)
            z0A[nr] = a
            z0B[nr] = b
            ustar[nr] = u
            vstar[nr] = v
            
        if smooth48:
            # Compute zero-wind-phase-corrected, 48-day average
            thw = pd.to_timedelta('24d')
            pA = (dsA.phase.sel(time=slice(t-thw,t+thw)) - z0A).median(dim='time')
            pB = (dsB.phase.sel(time=slice(t-thw,t+thw)) - z0B).median(dim='time')
            z0A += hpf(pA) # Adjust zero wind phase with the residual "striations" in 48-day averaged phase
            z0B += hpf(pB) # Adjust zero wind phase with the residual "striations" in 48-day averaged phase
    
    # Save zero wind phase, daily-averaged phase, and window-mean u and v.
    # NOTE: If smooth48 is True, then u and v are not correct. It is too difficult to adjust those in postprocessing.
    tstr = t.strftime('%Y-%m-%d')
    pAraw = dsA.phase.sel(time=tstr).median(dim='time')
    pBraw = dsB.phase.sel(time=tstr).median(dim='time')
    pA0 = pAraw - z0A
    pB0 = pBraw - z0B
    ds = xr.Dataset({
                    'z0A':('row',z0A),
                    'z0B':('row',z0B),
                    'phAraw':('row',pAraw),
                    'phBraw':('row',pBraw),
                    'phA0':('row',pA0),
                    'phB0':('row',pB0),
                    'u':('row',ustar),
                    'v':('row',vstar),
                    },
        coords = {'row':('row',rows)}
    )
    
    
    return ds
    

    
def run_window(t, dsAg, dsBg, dsAr, dsBr, cal_lamps=[0,1], gap_matching=False, verbose=False, normal_lvlh=True, 
               outlier_reject=True, outlier_thresh=3, restrict_lt = False, smooth48=False, weight=False):
    '''
    Run a full window, and sort result into the 8 cases
    
    INPUTS:
    t: The center date of the window
    dsAg, dsBg, dsAr, dsBr: Full xarray.Dataset for the rows to be analyzed. 1 full time window must be specified.
    cal_lamps = list. [0] = run off only, [1] = run on only, [0,1] = run both cases
    gap_matching = If True, and if there is a gap, make sure that the LT/lat sampling matches on the ascending/descending orbits.
                   This is a performance hit.
    normal_lvlh = If True, only use data from normal LVLH
    outlier_reject = If True, reject phase samples > 3 sigma
    outlier_thresh = How many sigmas away from the mean to reject as outliers.
    restrict_lt = If True, only use data with 3 hours of noon (day mode) or midnight (night mode) local time
    smooth48 = if True, ensure that the 48-day averaged phases are smooth as a function of altitude (i.e., the hpf function returns 0)
    weight = if True, implemented weighting based on the low signal correction
                   
    OUTPUT:
    dsz - xarray.Dataset 
            - with dimensions of (row, mode, cal_lamp, color, sensor)
            - with one variable: zero_phase_single
    '''
    # Protect input data (I ran 100 days and determined that this made no difference, but is maintained for safety)
    dsAg = dsAg.copy(deep=True)
    dsBg = dsBg.copy(deep=True)
    dsAr = dsAr.copy(deep=True)
    dsBr = dsBr.copy(deep=True)
    
    if normal_lvlh:
        dsAg['phase'] = dsAg.phase.where((dsAg.az > 15)  & (dsAg.az < 80))
        dsAr['phase'] = dsAr.phase.where((dsAr.az > 15)  & (dsAr.az < 80))
        dsBg['phase'] = dsBg.phase.where((dsBg.az > 280) & (dsBg.az < 345))
        dsBr['phase'] = dsBr.phase.where((dsBr.az > 280) & (dsBr.az < 345))
        
    if outlier_reject:
        
        for ds in [dsAg, dsAr, dsBg, dsBr]:
            mu = ds.phase.mean(dim='time')
            s = ds.phase.std(dim='time')
            ds['phase'] = ds.phase.where(abs(ds.phase - mu) < outlier_thresh*s)
    
    
    # Concatenate along one dimension at a time:
    zsmode = []
    for mode in ['Day','Night']:
        zscal = []
        for cal_lamp in cal_lamps:
            dsAg_case = dsAg.where((dsAg.cal_lamp == cal_lamp) & (dsAg['mode'] == mode))
            dsBg_case = dsBg.where((dsBg.cal_lamp == cal_lamp) & (dsBg['mode'] == mode))
            dsAr_case = dsAr.where((dsAr.cal_lamp == cal_lamp) & (dsAr['mode'] == mode))
            dsBr_case = dsBr.where((dsBr.cal_lamp == cal_lamp) & (dsBr['mode'] == mode))

            # Green
            z = run_case(t, dsAg_case, dsBg_case, gap_matching=gap_matching, verbose=verbose, restrict_lt=restrict_lt, smooth48=smooth48, weight=weight)
            zA = z[['z0A','phAraw','phA0']].assign_coords(time=t, cal_lamp=cal_lamp, mode=mode, color='green', sensor='A')
            zB = z[['z0B','phBraw','phB0']].assign_coords(time=t, cal_lamp=cal_lamp, mode=mode, color='green', sensor='B')
            zA = zA.rename({'z0A':'zero_phase_single','phAraw':'phase_single_raw','phA0':'phase_single'})
            zB = zB.rename({'z0B':'zero_phase_single','phBraw':'phase_single_raw','phB0':'phase_single'})
            zg = xr.concat([zA, zB], dim='sensor')

            # Red
            z = run_case(t, dsAr_case, dsBr_case, gap_matching=gap_matching, verbose=verbose, restrict_lt=restrict_lt, smooth48=smooth48, weight=weight)
            zA = z[['z0A','phAraw','phA0']].assign_coords(time=t, cal_lamp=cal_lamp, mode=mode, color='red', sensor='A')
            zB = z[['z0B','phBraw','phB0']].assign_coords(time=t, cal_lamp=cal_lamp, mode=mode, color='red', sensor='B')
            zA = zA.rename({'z0A':'zero_phase_single','phAraw':'phase_single_raw','phA0':'phase_single'})
            zB = zB.rename({'z0B':'zero_phase_single','phBraw':'phase_single_raw','phB0':'phase_single'})
            zr = xr.concat([zA, zB], dim='sensor')

            zscal.append(xr.concat([zg,zr], dim='color'))
        zsmode.append(xr.concat(zscal, dim='cal_lamp'))
    z = xr.concat(zsmode, dim='mode')
    
    return z
    
    
def save_window(ds, folder):
    '''
    Save a zero phase Dataset from an individual window (i.e., an individual date) to a file
    '''
    ti = pd.to_datetime(ds.time.item())
    fn = folder + '/ICON_L1_MIGHTI_Zero-Phase-Single_%s_%s.NC' % (ti.strftime('%Y-%m-%d'), VERS)
    ds.to_netcdf(fn, engine=NC_ENG)
    return fn
    

    
def combine_smooth_and_save(fns, folder_save, bad_zero_phase_periods=[], smooth=True, smooth48=False):
    '''
    Combine multiple "zero-phase-single" files into a single file for all time. Add a separate variable which is the 48-day smoothed
    version of the zero phase, which is what will actually be used in practice. 
    
    INPUTS:
    fns: Either a list of filenames, or a string to pass to glob, e.g., '/home/bharding/MIGHTI/zero_wind/*Zero-Phase-Single*v02.NC'
    folder_save: Where to save the combined file
    smooth: If True (default), do a precession-cycle smoothing and save the parameter as "zero_phase". If false, just copy the 
            zero_phase_single variable into zero_phase. This should always be True, except for debugging purposes.
    smooth48 = if True, ensure that the 48-day averaged phases are smooth as a function of altitude (i.e., the hpf function returns 0)
    bad_zero_phase_periods = list of tuples (start, stop), or empty list. Periods which the zero-phase-single results are known
                             to be bad, and should be interpolated over.
    
    OUTPUTS:
    xarray.Dataset:
        - zero_phase_single: the individual zero wind phase profiles determined for each day
        - zero_phase:        the smoothed version to be used in practice
    '''
    
    if isinstance(fns, str):
        fns = glob.glob(fns)
        fns.sort()
        
    # Find versions of single files and make sure they are all the same
    vers = [fn.split('.')[0][-3:] for fn in fns]
    assert len(pd.unique(vers)) == 1, "Input files are from different versions"
    ver = vers[0]

    dss = []
    for fn in fns:
        ds = xr.open_dataset(fn)
        dss.append(ds)
    ds = xr.concat(dss, dim='time')
    
    # Mark any potential days as bad
    for tstart, tstop in bad_zero_phase_periods:
        ds['zero_phase_single'] = ds['zero_phase_single'].where((ds.time < tstart) | (ds.time > tstop))
    # Interpolate over gaps
    ds['zero_phase_single'] = ds['zero_phase_single'].interpolate_na(dim='time', method='linear')
        
    if smooth: # This should always be true unless we are in some weird debugging mode.
        # Ensure there is 1 time sample per day
        t0 = pd.to_datetime(ds.time[0].item())
        t1 = pd.to_datetime(ds.time[-1].item())
        dt = t1 - t0
        assert dt > pd.to_timedelta('48d'), 'Need 48 day span in order to run zero-phase smoothing (%.0f day span available)' % (dt.total_seconds()/86400.)
        tnew = pd.date_range(t0, t1)
        ds2 = ds.reindex(time=tnew)

        # 48-day smoothing
        ds2['zero_phase'] = ds2.zero_phase_single.rolling(time=48, center=True, min_periods=1).mean()

        # Find times of applicability for the smoothed version. For the sake of conservatism we will use +/- 25 days
        t0 = pd.to_datetime(fns[0].split('_')[-2])
        t1 = pd.to_datetime(fns[-1].split('_')[-2])
        tstart = t0 + pd.to_timedelta('25d')
        tstop  = t1 - pd.to_timedelta('25d')
        ds2['zero_phase'] = ds2['zero_phase'].where( (ds2.time >= tstart) & (ds2.time <= tstop) )

        # This replaces some extrapolation code that wasn't needed in the end:
        ds3 = ds2
        
        # If smooth48 is enabled, FIRST we need to re-ensure the 48-day profile is smooth over the interpolated gaps above
        # TBD if a similar thing can be done for the early mission
        if smooth48:
            p   = ds3['phase_single_raw'] - ds3['zero_phase'] 
            pm  = p.rolling(time=49, center=True, min_periods=1).mean() # Note 49 for true centering
            # Ensure 48d phases are smooth with a two-pass filter nearly identical to the hpf function.
            # The hpf function is not used here since it was not vectorized.
            pmlo =   pm.rolling(row=5, center=True, min_periods=1).median() 
            pmlo = pmlo.rolling(row=5, center=True, min_periods=1).median()
            pmhi = pm - pmlo
            # Compute and apply new zero phase, but only change it in the gaps (keep the previous result outside of the gaps)
            z0 = ds3['zero_phase'] + pmhi
            # But only apply new zero phase where there are gaps
            for t0, t1 in bad_zero_phase_periods:
                ds3['zero_phase'] = z0.where( (z0.time >= t0) & (z0.time <= t1) , ds3['zero_phase'])
                
        # SECOND, adjust the zero phase (in the 48d mean sense) to ensure extra smoothness above row 14 (~130 km)
        if smooth48:
            p   = ds3['phase_single_raw'] - ds3['zero_phase'] 
            pm  = p.rolling(time=49, center=True, min_periods=1).mean() # Note 49 days for true centering
            pmlo = pm.rolling(row=15, center=True, min_periods=1).mean()
            pmhi = pm - pmlo
            # Only use this correction >= row 14
            pmhi = pmhi.where(pmhi.row >= 14, other = 0*pmhi) # 0 means don't change zero wind
            # Treat rows 10-13 as a linear transition zone
            assert GREEN_NIGHT_ROW_MAX == 9, "It seems like you've changed GREEN_NIGHT_ROW_MIN. You need to also change the smoothing/interpolation"
            pmhi = pmhi.where((pmhi.row <= 9) | (pmhi.row >= 14))
            pmhi = pmhi.interpolate_na(dim='row', method='linear')
            pmhi = pmhi.where(~ds3['zero_phase'].isnull()) # Only keep new result where old result existed and where we are actually changing it
            # Apply new zero phase
            ds3['zero_phase'] = ds3['zero_phase'] + pmhi
        
        # Compute new L1-corrected phases.
        ds3['phase']      = ds3['phase_single_raw'] - ds3['zero_phase'] 
        
        ds = ds3
    else:
        # Just use the original, without smoothing
        ds['zero_phase'] = ds['zero_phase_single']
        ds['phase']      = ds['phase_single']
    
    
    # Dates for filename
    # These define the dates of applicability for the zero_phase.
    # This section of code will reflect the above code's decision of where the "zero_phase" is valid.
    z = ds.zero_phase.dropna(dim='time', how='all')
    t0 = pd.to_datetime(z.time[0].item())
    t1 = pd.to_datetime(z.time[-1].item())
    
    fn_full = folder_save + '/ICON_L1_MIGHTI_Zero-Phase_%s_%s_%s.NC' % (t0.strftime('%Y-%m-%d'), t1.strftime('%Y-%m-%d'), ver)
    
    try:
        ds.to_netcdf(fn_full, engine=NC_ENG)
    except Exception as e:
        print('WARNING: FAILED SAVING. Does file aready exist?\n\t%s' % fn_full)    
    
    
    return fn_full

    
    
    
def fill_phase_gaps(ds, fnsAg, fnsBg, fnsAr, fnsBr, TSTART):
    '''
    From a given "_Zero-Phase_" dataset, find dates with missing phase variables and fill them in with the daily-average phase.
    These could include:
     - reverse LVLH
     - days marked as "bad L1 days"
     - the first few months of the mission, when the zero-phase-single analysis could not be run.
     - any other days that failed the zero-phase-single analysis
     
    Return the dataset in the same form, but with the phase_single_raw variable extrapolated to the early mission and filled in with
    whatever L1.5 data exists.
    '''

    tAg = pd.to_datetime([fn.split('_')[-2] for fn in fnsAg]) # L1.5 times
    tAr = pd.to_datetime([fn.split('_')[-2] for fn in fnsAr]) # L1.5 times
    tBg = pd.to_datetime([fn.split('_')[-2] for fn in fnsBg]) # L1.5 times
    tBr = pd.to_datetime([fn.split('_')[-2] for fn in fnsBr]) # L1.5 times
    tm = tAg.intersection(tAr).intersection(tBg).intersection(tBr) # Only days when all cases are available
    
    tmax = pd.to_datetime(ds.time[-1].item())
    tall = pd.date_range(TSTART, tmax)
    ds2 = ds.reindex(time=tall)

    counts = ds2.phase_single_raw.count(dim=['row','cal_lamp','mode','color','sensor'])
    t_missing_phase = pd.to_datetime(ds2.time.where(counts==0, drop=True).values)
    t_to_fix = tm.intersection(t_missing_phase) # Which dates we can actually fill in. Note this will include rLVLH an early mission


    fnsAg_load = [fn for fn in fnsAg if fn.split('_')[-2] in t_to_fix]
    fnsAr_load = [fn for fn in fnsAr if fn.split('_')[-2] in t_to_fix]
    fnsBg_load = [fn for fn in fnsBg if fn.split('_')[-2] in t_to_fix]
    fnsBr_load = [fn for fn in fnsBr if fn.split('_')[-2] in t_to_fix]

    print('Loading data from (Ag=%i, Ar=%i, Bg=%i, Br=%i) files...' % (len(fnsAg_load), len(fnsAr_load), len(fnsBg_load), len(fnsBr_load)))
    for fn in fnsAg_load:
        print(fn)
    dsAr = L15s_to_dataset(fnsAr_load)
    dsAg = L15s_to_dataset(fnsAg_load)
    dsBr = L15s_to_dataset(fnsBr_load)
    dsBg = L15s_to_dataset(fnsBg_load)
    print('Loaded.')
    
    
    psmode = []
    for mode in ['Day','Night']:
        pscal = []
        for cal_lamp in [0,1]:
            dsAg_case = dsAg.where((dsAg.cal_lamp == cal_lamp) & (dsAg['mode'] == mode))
            dsBg_case = dsBg.where((dsBg.cal_lamp == cal_lamp) & (dsBg['mode'] == mode))
            dsAr_case = dsAr.where((dsAr.cal_lamp == cal_lamp) & (dsAr['mode'] == mode))
            dsBr_case = dsBr.where((dsBr.cal_lamp == cal_lamp) & (dsBr['mode'] == mode))

            # Green
            pA = dsAg_case.phase.groupby(dsAg_case.time.dt.floor('1d')).median().rename({'floor':'time'})
            pA = pA.assign_coords(cal_lamp=cal_lamp, mode=mode, color='green', sensor='A')
            pB = dsBg_case.phase.groupby(dsBg_case.time.dt.floor('1d')).median().rename({'floor':'time'})
            pB = pB.assign_coords(cal_lamp=cal_lamp, mode=mode, color='green', sensor='B')
            pg = xr.concat([pA, pB], dim='sensor')

            # Red
            pA = dsAr_case.phase.groupby(dsAr_case.time.dt.floor('1d')).median().rename({'floor':'time'})
            pA = pA.assign_coords(cal_lamp=cal_lamp, mode=mode, color='red', sensor='A')
            pB = dsBr_case.phase.groupby(dsBr_case.time.dt.floor('1d')).median().rename({'floor':'time'})
            pB = pB.assign_coords(cal_lamp=cal_lamp, mode=mode, color='red', sensor='B')
            pr = xr.concat([pA, pB], dim='sensor')

            pscal.append(xr.concat([pg,pr], dim='color'))
        psmode.append(xr.concat(pscal, dim='cal_lamp'))
    p = xr.concat(psmode, dim='mode')

    p1 = ds2.phase_single_raw
    p2 = p1.where(~p1.time.isin(t_to_fix), other=p.reindex_like(ds2))
    ds2['phase_single_raw'] = p2
    
    return ds2
    
   

def extrap_zero_phase_to_start_and_save(ds, folder_save, smooth48=True):
    '''
    From a given "_Zero-Phase_" dataset, extrapolate the zero phase to the beginning date (presumably the start of science ops, 2019-12-06).
    The "phase_single_raw" variable is expected to exist for the early mission, e.g., by running fill_phase_gaps(...). This code 
    uses the row-averaged linear trend to extrapolate, preserving the 48-day smoothness criterion.
    
    smooth48 = if True, ensure that the 48-day averaged phases are smooth as a function of altitude (i.e., the hpf function returns 0)
    folder_save = where to save the resulting file. The version will be defined by global variable VERS.
    Return the dataset in the same form, but with the zero_phase variable extrapolated to the early mission, and save it.
    '''
    ds3 = ds.copy(deep=True)
    d = ds3.dropna(dim='time', how='all', subset=['zero_phase'])
    tfirst = d.time[0]
    tlast = d.time[-1]

    z0 = ds3.zero_phase.sel(time=slice(tfirst, tfirst+pd.to_timedelta('47d'))) # 47d so the total is 48 days
    zm = z0.mean(dim=['time'])
    zt = z0 - zm # Ensure row-mean is zero
    c = zt.mean(dim='row').polyfit(dim='time', deg=1).polyfit_coefficients # Mean trend (i.e., mean over rows, trend over time)
    z0t = xr.polyval(ds3.time, c) # Eval'd vs time. Note this does not include row-dependence
    znew = zm + z0t # This only applies before tstart
    z = ds3.zero_phase.where(ds3.time >= tfirst, other=znew)
    ds3['zero_phase'] = z

    if smooth48:
        ## Extra smoothing to ensure 48d average is smooth
        p   = ds3['phase_single_raw'] - ds3['zero_phase'] 
        pm  = p.rolling(time=49, center=True, min_periods=1).mean() # Note 49 for true centering
        # Ensure 48d phases are smooth with a two-pass filter nearly identical to the hpf function.
        # The hpf function is not used here since it was not vectorized.
        pmlo =   pm.rolling(row=5, center=True, min_periods=1).median() 
        pmlo = pmlo.rolling(row=5, center=True, min_periods=1).median()
        pmhi = pm - pmlo
        # Compute and apply new zero phase, but only change it in the gaps (keep the previous result outside of the gaps)
        z0 = ds3['zero_phase'] + pmhi
        # But only apply new zero phase to the back-extrapolated data
        ds3['zero_phase'] = z0.where( z0.time < tfirst , ds3['zero_phase'])
    
    ds3['phase']      = ds3['phase_single_raw'] - ds3['zero_phase'] 
    
    ds = ds3
    # Dates for filename
    # These define the dates of applicability for the zero_phase.
    # This section of code will reflect the above code's decision of where the "zero_phase" is valid.
    z = ds.zero_phase.dropna(dim='time', how='all')
    t0 = pd.to_datetime(z.time[0].item())
    t1 = pd.to_datetime(z.time[-1].item())
    
    fn_full = folder_save + '/ICON_L1_MIGHTI_Zero-Phase_%s_%s_%s.NC' % (t0.strftime('%Y-%m-%d'), t1.strftime('%Y-%m-%d'), VERS)
    
    try:
        ds.to_netcdf(fn_full, engine=NC_ENG)
    except Exception as e:
        print('WARNING: FAILED SAVING. Does file aready exist?\n\t%s' % fn_full)    
    
    
    return ds, fn_full
    
    
    
    
    
    
def hpf(y, Npasses=2, hw=2):
    '''
    Return the high-frequency oscillations that should be subtracted from the data. This is approximately a high-pass filter.
    
    y = the array to do the smoothing of
    Npasses = number of passes of running average (2 passes is like a triangular window)
    hw = the half-width of the window (not including the center)
    '''
    hw = 2 # half-width of averaging window
    y_orig = y.copy()
    Ny = len(y)
    for npass in range(Npasses):
        y_lpf = np.zeros(Ny)
        for i in range(Ny):
            i0 = max(i - hw, 0)
            i1 = min(i + hw + 1, Ny)
            window = np.concatenate((y[i0:i], y[i:i1])) # include the center
            # Annoying way to catch Runtime warning about nanmean
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                y_lpf[i] = np.nanmean(window)
        y = y_lpf

    y_hpf = y_orig - y_lpf
    return y_hpf




def dv_gmr(fng, fnr, fnz, invert_green=True, invert_red=True):
    '''
    Given a green and red L1.5 file from the same sensor, compute the daily-averaged delta velocity (green minus red) 
    from a common altitude range (nominally 165-185 km) computed from data within 3 hours of local noon.
    
    fng, fnr = paths to L1.5 files for 1 day
    fnz = path to "zero-phase" file
    
    invert_green = whether to onion-peel the green results before using them. (It may add more error than it accounts for)
    invert_red   = whether to onion-peel the red results before using them.
    '''
    
    rowg = slice(28,35) # 165-185 km
    rowr = slice(5,12)

    dsg = L15s_to_dataset([fng])
    dsr = L15s_to_dataset([fnr])
    dsz = xr.open_dataset(fnz)

    sensorg = dsg.sensor[0].item()
    sensorr = dsr.sensor[0].item()
    assert sensorg==sensorr, "Must specify files from the same sensor"
    sensor = sensorg

    tg = pd.to_datetime(dsg.time[-1].item()).floor(freq='1d')
    tr = pd.to_datetime(dsr.time[-1].item()).floor(freq='1d')
    assert tg==tr, "Must specify files from the same date"

    tstr = tg.strftime('%Y-%m-%d')
    try:
        zt = dsz.sel(time=tstr).zero_phase
    except KeyError as e:
        print('Zero wind phase not determined for this date')
        raise

    # Day mode only, cal lamp off
    dsg = dsg.where((dsg['mode'] == 'Day') & (dsg['cal_lamp'] == 0), drop=True)
    dsr = dsr.where((dsr['mode'] == 'Day') & (dsr['cal_lamp'] == 0), drop=True)

    # Subtract zero wind phase
    dsg['phase'] = dsg['phase'] - zt.sel(sensor=sensor, color='green', cal_lamp=0, mode='Day', drop=True) # "drop=True" removes the single-element dimensions
    dsr['phase'] = dsr['phase'] - zt.sel(sensor=sensor, color='red',   cal_lamp=0, mode='Day', drop=True).dropna(dim='row') # dropna omits red rows above 60

    # Perform approximate inversion
    if invert_green:
        D = MIGHTI_L2.create_observation_matrix(dsg.alt.mean(dim='time').values, 600., top_layer='exp', integration_order=0, H=40)
        Ic = (dsg.Iamp * np.exp(1j*dsg.phase)).fillna(0) # NaNs are treated as zero emission
        Ic2 = np.linalg.solve(D, Ic.T)
        dsg['phase2'] = (['time','row'], np.angle(Ic2.T)) # "phase2" is the result after inversion
    else:
        dsg['phase2'] = dsg['phase']
    
    if invert_red:
        D = MIGHTI_L2.create_observation_matrix(dsr.alt.mean(dim='time').values, 600., top_layer='exp', integration_order=0, H=40)
        Ic = (dsr.Iamp * np.exp(1j*dsr.phase)).fillna(0) # NaNs are treated as zero emission
        Ic2 = np.linalg.solve(D, Ic.T)
        dsr['phase2'] = (['time','row'], np.angle(Ic2.T)) # "phase2" is the result after inversion
    else:
        dsr['phase2'] = dsr['phase']
        
    # Convert to m/s
    dsg['los_wind'] = dsg['phase2']*dsg['phase_to_wind_factor']
    dsr['los_wind'] = dsr['phase2']*dsr['phase_to_wind_factor']

    # Avoid data near the terminators
    idx = abs(dsg.slt.sel(row=30) - 12) < 3 # Within 3 hours of noon
    dsg2 = dsg.where(idx)
    dsr2 = dsr.where(idx)

    # Extract certain rows and take median over rows/time
    vg = dsg2.los_wind.sel(row=rowg).median(dim=['row','time'])
    vr = dsr2.los_wind.sel(row=rowr).median(dim=['row','time'])
    dv = vg - vr

    return dv.item()


def dv_gmr_AB(fnAr, fnAg, fnBr, fnBg, fnz, invert_green=True, invert_red=True):
    '''
    Same as dv_gmr, but run it on both MTA and MTB, and return a DataArray with dims "time" and "sensor".
    '''
    tA = pd.to_datetime(fnAg.split('_')[-2])
    tB = pd.to_datetime(fnBg.split('_')[-2])
    assert tA==tB, "Must specify files from the same date"
    
    dvA = dv_gmr(fnAg, fnAr, fnz, invert_green=invert_green, invert_red=invert_red)
    dvB = dv_gmr(fnBg, fnBr, fnz, invert_green=invert_green, invert_red=invert_red)
    
    dsx = xr.Dataset(data_vars={'dv_single':(['sensor'],[dvA, dvB])}, 
                     coords={'sensor':['A','B']}).assign_coords(time=tA)
    
    return dsx
    


def save_dv_gmr(dsx, folder):
    '''
    Save a single-day result of the "green minus red" analysis to a file in the givenfolder. 
    '''
    ti = pd.to_datetime(dsx.time.item())
    fn = folder + '/ICON_L1_MIGHTI_Notch-Single_%s_%s.NC' % (ti.strftime('%Y-%m-%d'), VERS)
    dsx.to_netcdf(fn, engine=NC_ENG)
    return fn



def combine_smooth_and_save_notches(fns, folder_save, smooth=True, bad_notch_periods = []):
    '''
    Combine multiple "notch-single" files into a single file for all time. Add a separate variable which is the bandpass
    filtered version of "dv_single" which is what will be used in practice
    
    INPUTS:
    fns: Either a list of filenames, or a string to pass to glob, e.g., '/home/bharding/MIGHTI/zero_wind/*Notch-Single*v03.NC'
    folder_save: Where to save the combined file
    bad_notch_periods = list of tuples (start, stop), or empty list. Periods which the "dv" (i.e., notch) results are known
                           to be bad. They will be ignored in the smoothing but retained in the resulting data file.
    
    OUTPUTS:
    xarray.Dataset:
        - dv_single: the individual delta-v (green minus red) determined for each day separately [m/s]
        - dv:        same but smoothed in time (several day running mean) [m/s]
        - dv0:       same as dv but with 48-day mean subtracted (so as not to interfere with zero wind computation) [m/s]
                     (This variable will be used in operational code.)
        - dx:        dv converted from m/s to an estimated notch position in units of [pixels]
        - dx0:       same as dx but with 48-day mean subtracted
    '''
    if isinstance(fns, str):
        fns = glob.glob(fns)
        fns.sort()

    # Find versions of single files and make sure they are all the same
    vers = [fn.split('.')[0][-3:] for fn in fns]
    assert len(pd.unique(vers)) == 1, "Input files are from different versions"
    ver = vers[0]

    # Combine datasets
    dss = []
    for fn in fns:
        ds = xr.open_dataset(fn)
        dss.append(ds)
    ds = xr.concat(dss, dim='time')

    # Ensure there is 1 time sample per day
    t0 = pd.to_datetime(ds.time[0].item())
    t1 = pd.to_datetime(ds.time[-1].item())
    dt = t1 - t0
    assert dt > pd.to_timedelta('48d'), 'Need 48 day span in order to run notch analysis (%.0f day span available)' % (dt.total_seconds()/86400.)
    tnew = pd.date_range(t0, t1)
    ds2 = ds.reindex(time=tnew)
    
    # Mark any potential days as bad
    dv_single = ds2['dv_single']
    for tstart, tstop in bad_notch_periods:
        dv_single = dv_single.where((ds2.time < tstart) | (ds2.time > tstop))
    
    # 48-day bandpass filter:
    #  - Smooth over day-to-day noise
    #  - Don't allow any bulk correction on time scales > precession cycle, since this should be accounted for by zero wind
    # Step 1: smoothing
    ds2['dv'] = dv_single.rolling(time=13, center=True, min_periods=1).mean()
    # Step 2: bulk offset removal. Note this uses backwards-looking 48 day mean, so that we don't get too far behind real time.
    ds2['dv0'] = ds2['dv'] - ds2['dv'].rolling(time=48, min_periods=1).mean()
    
    # Define times of applicability.
    # For end, the 13-day averaging requires a 7 day buffer before dv_single is defined
    # For the beginning, the above analysis works fine. No extra extrapolation or smoothing required.
    # Be careful about which restrictions apply to which variables
    z = ds.dv_single.dropna(dim='time', how='all')
    t0z = pd.to_datetime(z.time[0].item()) # Start/stop times for zero phase
    t1z = pd.to_datetime(z.time[-1].item())
    # Beginning (TODO: Adjust this to TSTART once extrapolation is figured out)
    ds2['dv' ] = ds2['dv' ].where(ds2.time >= t0z)
    ds2['dv0'] = ds2['dv0'].where(ds2.time >= t0z)
    # End
    ds2['dv']  = ds2['dv' ].where(ds2.time <= t1z - pd.to_timedelta('7d'))
    ds2['dv0'] = ds2['dv0'].where(ds2.time <= t1z - pd.to_timedelta('7d'))

    # Convert to notch position
    # These factors come from John Harlander's spreadsheet: Green: 168 m/s/pixel, Red: -98 m/s/pixel
    ds2['dx']  = ds2['dv']/(168.+98.) # dv is green minus red
    ds2['dx0'] = ds2['dv0']/(168.+98.)

    # Dates for filename -- these should apply to dv0 (and dx0) since that is what users should use.
    x = ds2.dv0.dropna(dim='time', how='all')
    t0 = pd.to_datetime(x.time[0].item()) # Start/stop times for dv0 & dx0
    t1 = pd.to_datetime(x.time[-1].item())

    fn_full = folder_save + '/ICON_L1_MIGHTI_Notch_%s_%s_%s.NC' % (t0.strftime('%Y-%m-%d'), t1.strftime('%Y-%m-%d'), ver)
    try:
        ds2.to_netcdf(fn_full, engine=NC_ENG)
    except Exception as e:
        print('WARNING: FAILED SAVING. Does file aready exist?\n\t%s' % fn_full)

    return fn_full



    
def combine_zero_phase_notches_and_save(fnz, fnx):
    '''
    Combine the zero wind and notch datasets. Result will be saved in the same folder as fnz.
    This is the file to be passed to the SDC.
    
    This also adds a "zero_phase_with_notch" variable, which is the phase offset that's actually used in practice
    
    fnz: path to a Zero-Phase file (e.g., ICON_L1_MIGHTI_Zero-Phase_2020-01-24_2021-04-10_v02.NC)
    fnx: path to a Notch file (e.g., ICON_L1_MIGHTI_Notch_2020-01-24_2021-04-10_v02.NC
    '''
    # Load the zero wind and striation datasets
    dsz = xr.open_dataset(fnz)
    dsx = xr.open_dataset(fnx)
#     assert (dsx.time.values == dsz.time.values).all(), "Zero-Phase and Notch datasets do not have identical timestamps"

    ds = xr.merge([dsz, dsx])

    # Compute zero phase with notch
    dvg = 168./(168.+98.) * ds.dv0 # Values from John's spreadsheet
    dvr = -98./(168.+98.) * ds.dv0
    # Convert to phase. This uses the 'phase_to_wind_factor' values. MTA and MTB are very close, so a common value is used
    dphg = dvg/479.5
    dphr = dvr/557.5

    zg = ds.zero_phase.sel(color='green') + dphg # This extrapolates the notch correction to night-mode, cal-lamp on, and other rows
    zr = ds.zero_phase.sel(color='red')   + dphr
    ds['zero_phase_with_notch'] = xr.concat([zg,zr],dim='color').transpose(*ds.zero_phase.dims) # Ensure dimension order is the same
    
    # Apply to L1 phase
    ds['phase_with_notch']     = ds['phase_single_raw'] - ds['zero_phase_with_notch'] 
    
    # Dates for filename -- these should apply to phase_with_notch since that is what this file is all about.
    x = ds.phase_with_notch.dropna(dim='time', how='all')
    t0 = pd.to_datetime(x.time[0].item()) # Start/stop times for phase_with_notch
    t1 = pd.to_datetime(x.time[-1].item())

    # Save as a new file in the same location
    s = fnz.split('/')
    folder = '/'.join(s[:-1])
    # 2022 Sep 20 BJH: Changing filename as per SDC request. Also add checking for revision numbers.
    # e.g., ICON_L1_MIGHTI_Calibration-Zero-Phase-Notch_2020-04-13-to-2022-04-14_v03r000.NC
    verstr = s[-1].split('_')[-1][1:3] # e.g., "03"
    # Find the proper revision number
    rev = 0
    fnbase =  folder + '/ICON_L1_MIGHTI_Calibration-Zero-Phase-Notch_%s-to-%s_v%s' % (t0.strftime('%Y-%m-%d'), t1.strftime('%Y-%m-%d'), verstr)
    fns = glob.glob(fnbase + '*')
    fns.sort()
    if len(fns)==0:
        revstr = '000'
    else:
        fn0 = fns[-1]
        revstr0 = fn0.split('.')[-2][-3:]
        revstr = '%03i' % (int(revstr0) + 1)
    fn = '%sr%s.NC' % (fnbase, revstr)
    
    try:
        ds.to_netcdf(fn, engine=NC_ENG)
    except Exception as e:
        print('FAILED SAVING %s' % fn)
        raise
        
    return fn




########################################################################################### 
##################### SCRIPT TO UPDATE ZERO PHASE AND STRIATION FILE ######################    
###########################################################################################


if __name__ == "__main__":
    
    #### PARAMETERS ####
    zero_dir = '/home/bharding/MIGHTI/zero_wind/OPERATIONAL/' # where to find the zero phase and striation files
    gap_matching = False
    Tprec = pd.to_timedelta('48d')
    t_half_window = Tprec
    normal_lvlh = True
    outlier_reject = True
    outlier_thresh = 3
    restrict_lt = False
    smooth48 = True
    days_max = 600 # Max number of days to try to process at once.
    bad_L1_days = pd.date_range('2021-05-06','2021-05-09') # Specify dates or use empty list
    # Manually list any dates for which the "zero-phase-single" results are known to be bad
    bad_zero_phase_periods = [ # (start, stop)
        (pd.to_datetime('2021-04-26'), pd.to_datetime('2021-08-14')), # Windows that touch the June 2021 outage/rLVLH
    ]
    # Manually list any dates for which the "dv" results are known to be bad
    bad_notch_periods = [ # (start, stop) -- inclusive
        (pd.to_datetime('2022-01-15'), pd.to_datetime('2022-01-16')), # The Tonga event (it is unknown why this affects dv)
        (pd.to_datetime('2021-06-10'), pd.to_datetime('2021-06-10')), # An outlier, especially in MIGHTI-B
    ]
    ####################
    
    if not gap_matching:
        print('\nWARNING: Running with gap_matching = False\n')
        
    print('Starting zero phase and notch processing with the following parameters:')
    print('\n\tzero_dir = %s\n\tgap_matching = %s\n\tVERS = %s\n\tt_half_window = %s\n\tnormal_lvlh = %s\n\toutlier_reject = %s\n\toutlier_thresh = %s' % (zero_dir, gap_matching, VERS, t_half_window, normal_lvlh, outlier_reject, outlier_thresh))
    print('\tsmooth48=%s\n\trestrict_lt = %s\n\tdays_max = %s\n\tbad_L1_days=%s' % (smooth48, restrict_lt, days_max,bad_L1_days))
    print('\tbad_zero_phase_periods=%s\n\tbad_notch_periods=%s' % (bad_zero_phase_periods, bad_notch_periods))
    print('\tTSTART = %s\n\tGREEN_DAY_ROW_MIN = %i\n\tGREEN_DAY_ROW_MAX = %i\n\tGREEN_NIGHT_ROW_MIN = %i\n\tGREEN_NIGHT_ROW_MAX = %i\n\tRED_NIGHT_ROW_MIN = %i\n' % (TSTART,GREEN_DAY_ROW_MIN, GREEN_DAY_ROW_MAX, GREEN_NIGHT_ROW_MIN, GREEN_NIGHT_ROW_MAX, RED_NIGHT_ROW_MIN))
    
    truntime0 = pd.to_datetime(datetime.now())

    fnsAg, fnsBg, fnsAr, fnsBr = find_all_L15_files()

    fnzs = glob.glob(zero_dir + '*_Zero-Phase-Single_*.NC')
    fnzs.sort()
    fnxs = glob.glob(zero_dir + '*_Notch-Single_*.NC')
    fnxs.sort()

    # Determine available times
    tz  = pd.to_datetime([fn.split('_')[-2] for fn in fnzs]) # zero phase times
    tx  = pd.to_datetime([fn.split('_')[-2] for fn in fnxs]) # notch times
    tAg = pd.to_datetime([fn.split('_')[-2] for fn in fnsAg]) # L1.5 times
    tAr = pd.to_datetime([fn.split('_')[-2] for fn in fnsAr]) # L1.5 times
    tBg = pd.to_datetime([fn.split('_')[-2] for fn in fnsBg]) # L1.5 times
    tBr = pd.to_datetime([fn.split('_')[-2] for fn in fnsBr]) # L1.5 times
    tm = tAg.intersection(tAr).intersection(tBg).intersection(tBr) # Only days when all cases are available

    
    #################### STEP 1: Zero wind phase #######################
    # Find days when L1.5 files are available but zero-phase-single files have not been made.
    # Only consider dates where we have the requisite half window
    t_half_window_p1 = t_half_window + pd.to_timedelta('1d') # Use a 1-day buffer for some of the logic to avoid off-by-1 errors
    tstart = tm[0]  + t_half_window_p1 # First date we can run
    tstop  = tm[-1] - t_half_window_p1 # Last date we can run

    # Find which dates to run
    can_run = tm.intersection(pd.date_range(tstart, tstop))
    already_run = can_run.isin(tz)
    to_run = can_run[~already_run]
    print('Identified %i dates to run' % (len(to_run)))
    
    # Trim if necessary for memory reasons
    if len(to_run) > days_max:
        to_run = to_run[:days_max]
        print('Trimmed to %i dates to save memory' % (len(to_run)))
        
    print('Dates being run: %s' % to_run)

    if len(to_run) > 0:

        # Load the L1.5 data from the entire necessary period
        tm_load = pd.date_range(to_run[0] - t_half_window_p1, to_run[-1] + t_half_window_p1)
        # Avoid days which are manually labeled bad
        tm_load = tm_load.drop(bad_L1_days, errors='ignore')
        tm_load_str = tm_load.strftime('%Y-%m-%d')
        fnsAg_load = [fn for fn in fnsAg if fn.split('_')[-2] in tm_load_str]
        fnsAr_load = [fn for fn in fnsAr if fn.split('_')[-2] in tm_load_str]
        fnsBg_load = [fn for fn in fnsBg if fn.split('_')[-2] in tm_load_str]
        fnsBr_load = [fn for fn in fnsBr if fn.split('_')[-2] in tm_load_str]
        print('Loading data from (Ag=%i, Ar=%i, Bg=%i, Br=%i) files...' % (len(fnsAg_load), len(fnsAr_load), len(fnsBg_load), len(fnsBr_load)))
        for fn in fnsAg_load:
            print(fn)
        dsmAr = L15s_to_dataset(fnsAr_load)
        dsmAg = L15s_to_dataset(fnsAg_load)
        dsmBr = L15s_to_dataset(fnsBr_load)
        dsmBg = L15s_to_dataset(fnsBg_load)
        print('Loaded.')
        
        # Run the new zero wind single files
        for t in to_run:
            print('Running zero phase %s' % (t.strftime('%Y-%m-%d')))
            
            dsAg = dsmAg.sel(time=slice(t-t_half_window, t+t_half_window))
            dsBg = dsmBg.sel(time=slice(t-t_half_window, t+t_half_window))
            dsAr = dsmAr.sel(time=slice(t-t_half_window, t+t_half_window))
            dsBr = dsmBr.sel(time=slice(t-t_half_window, t+t_half_window))

            ds = run_window(t, dsAg, dsBg, dsAr, dsBr, gap_matching=gap_matching, verbose=False, normal_lvlh=normal_lvlh, 
                            outlier_reject=outlier_reject, outlier_thresh=outlier_thresh, smooth48=smooth48)
            fn = save_window(ds, zero_dir)
            gc.collect()
    
    print('Combining zero phase files')
    # Technically we don't need to do this if we haven't created any new files, but it's easier to just do.
    fnz0 = combine_smooth_and_save(zero_dir + '*Zero-Phase-Single*.NC', zero_dir, bad_zero_phase_periods=bad_zero_phase_periods, smooth48=smooth48)
    print('\tCreated %s' % fnz0)
    
    # Fill missing phase (and add prints)
    print('Filling dates with missing phase')
    ds = xr.open_dataset(fnz0)
    ds2 = fill_phase_gaps(ds, fnsAg, fnsBg, fnsAr, fnsBr, TSTART)
    
    # Back-propagate to early mission
    print('Propagating zero_phase to early mission')
    ds3, fnz = extrap_zero_phase_to_start_and_save(ds2, folder_save=zero_dir)
    print('\tCreated %s' % fnz)
    
    print('Done with zero phase.\n\n')
    
    
    
    
    #################### STEP 2: Notches #######################
    # For all dates that the recently-created zero phase file covers, make sure the notch analysis is available.
    print('Starting notch analysis...\n')
    tz0 = pd.to_datetime(fnz.split('_')[-3])
    tz1 = pd.to_datetime(fnz.split('_')[-2])

    can_run = tm.intersection(pd.date_range(tz0, tz1))
    already_run = can_run.isin(tx)
    to_run = can_run[~already_run]
    print('Identified %i dates to run' % (len(to_run)))

    # Run the new notch single files
    for t in to_run:
        print('Running notch %s' % (t.strftime('%Y-%m-%d')))    

        tstr = t.strftime('%Y-%m-%d')
        fnAg_list = [fn for fn in fnsAg if fn.split('_')[-2] == tstr]
        fnAr_list = [fn for fn in fnsAr if fn.split('_')[-2] == tstr]
        fnBg_list = [fn for fn in fnsBg if fn.split('_')[-2] == tstr]
        fnBr_list = [fn for fn in fnsBr if fn.split('_')[-2] == tstr]
        # If the zero wind was run for this date then we should definitely have all 4 files. Sanity check:
        assert len(fnAg_list) == 1, 'Missing or multiple Ag file'
        assert len(fnAr_list) == 1, 'Missing or multiple Ar file'
        assert len(fnBg_list) == 1, 'Missing or multiple Bg file'
        assert len(fnBr_list) == 1, 'Missing or multiple Br file'

        # Run it
        dsx = dv_gmr_AB(fnAr_list[0], fnAg_list[0], fnBr_list[0], fnBg_list[0], fnz)
        fnx = save_dv_gmr(dsx, zero_dir)
    
    print('Combining all notch files...')
    # Find all Notch-Single files, including the ones we just made
    fnxs = zero_dir + '*_Notch-Single_*.NC'
    fnx = combine_smooth_and_save_notches(fnxs, zero_dir, bad_notch_periods=bad_notch_periods)
    print('\tCreated %s' % fnx)
    
    # Combine everything into the final file
    print('Combining notch and zero-phase files...')
    fn_tot = combine_zero_phase_notches_and_save(fnz, fnx)
    print('\nDone with everything. New file is:\n\n\t%s\n' % fn_tot)

    truntime1 = pd.to_datetime(datetime.now())
    print('\n\nCompleted in %s\n' % (truntime1 - truntime0))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
