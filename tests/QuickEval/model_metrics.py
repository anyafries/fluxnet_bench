import xarray as xr
import numpy as np

# x obs, y pred
def rmsd(x,y,along=None):
    a=(y-x)**2
    out = (a.mean(dim=along,skipna=True))**(0.5)
    return out.rename('RMSD')

def rrmsd(x,y,along=None):
    mask = y.notnull() & x.notnull()
    a=(y-x)**2
    out = (a.mean(dim=along,skipna=True))**(0.5)/x.where(mask).mean(dim=along,skipna=True)
    return out.rename('rRMSD')

def MAD(x,y,along=None):
    out = abs(y-x).mean(dim=along,skipna=True)
    return out.rename('MAD')

def MrAD(x,y,along=None):
    out = (abs((y-x)/x)).mean(dim=along,skipna=True)
    return out.rename('MrAD')
    
def MSE(x,y,along=None):
    out = ((y-x)*(y-x)).mean(dim=along,skipna=True)
    return out.rename('MSE')

def mef(x,y,along=None):
    """
    x is observation
    y is prediction
    """
    mask = x.notnull() & y.notnull()
    a = (x-y)**2
    b = (x-x.where(mask).mean(dim=along,skipna=True))**2
    out = 1- a.sum(dim=along,skipna=True)/b.where(mask).sum(dim=along,skipna=True)
    return out.rename('NSE')

nse = mef

def bias(x,y,along=None):
    """
    x is observation
    y is prediction
    """
    a = x-y
    out = a.mean(dim=along,skipna=True)
    return out.rename('bias')

def rbias(x,y,along=None):
    """
    x is observation
    y is prediction
    """

    a = x-y
    out = a.mean(dim=along,skipna=True)/x.where(a.notnull()).mean(dim=along,skipna=True)
    return out.rename('rbias')

def rov(x,y,along=None):
    # pred/obs
    mask = x.notnull() & y.notnull()
    out = y.where(mask).std(dim=along,skipna=True)/x.where(mask).std(dim=along,skipna=True)
    return out.rename('rov')

def pcor(x,y,along=None):
    return xr.corr(x,y,dim=along).rename('pcor')

def r2(x,y,along=None):
    out = xr.corr(x,y,dim=along)**2
    return out.rename('R2')

def Glambda(x,y,along=None):
    """
    Lambda = 1 - a1/(a2+a3+a4+k)
    """
    n = (np.isfinite(x) & np.isfinite(y)).sum(dim=along).values
    mask = x.notnull() & y.notnull()
    a1 = ((x-y)**2).sum(dim=along,skipna=True)
    a2 = ((x-x.mean(dim=along,skipna=True))**2).where(mask).sum(dim=along,skipna=True)
    a3 = ((y-y.mean(dim=along,skipna=True))**2).where(mask).sum(dim=along,skipna=True)
    a4 = n*(x.where(mask).mean(dim=along,skipna=True)-y.where(mask).mean(dim=along,skipna=True))**2
    
    pcor = xr.corr(x,y,dim=along).values
    pcor[pcor>=0] = 0
    pcor[pcor<0] = 1
    
    k= pcor * 2 * np.abs(((x-x.where(mask).mean(dim=along,skipna=True))*(y-y.where(mask).mean(dim=along,skipna=True))).sum(dim=along,skipna=True))
    
    out = 1 - a1/(a2+a3+a4+k)
    return out.rename('Lambda')
