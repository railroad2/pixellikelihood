from __future__ import print_function
import os
import sys
import time
import datetime 
sys.path.append('./scripts')

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import corner

from tqdm import tqdm
from scipy.stats import norm
from scipy.optimize import curve_fit

from gbpipe.spectrum import get_spectrum_noise, get_spectrum_camb, get_spectrum_xpol
from gbpipe.utils import cl2dl

from cambfast.cambfast import CAMBfast

from pixelcov.covcut import cutmap
from gen_fg import gen_fg 
from fg_setup import mymask


def merge_dat(dat_a):
    print ("merging the data files")
    print ("assuming that the input parameters are the same except for the input maps")

    files = dat_a[0].files
    pnames = dat_a[0]['pnames']
    dat = {}
    for fi in files:
        dat[fi] = dat_a[0][fi]
    
    for pn in pnames:
        dat[pn] = []
        dat[pn+'_err'] = []
        for d in dat_a:
            dat[pn] += [d[pn]]
            dat[pn+'_err'] += [d[pn+'_err']]

        dat[pn] = np.array(dat[pn]).flatten()
        dat[pn+'_err'] = np.array(dat[pn+'_err']).flatten()

    dat['maps'] = []
    for d in dat_a:
        dat['maps'] += [d['maps']]

    dat['maps'] = np.vstack(dat['maps'])
    
    return dat


def plot_result(vals, errs, pname='fit parameter', fout=None, ax=None):
    if ax is None:
        ax = plt.figure()

    ax.errorbar(np.arange(len(vals)), vals, yerr=errs, fmt='*')
    ax.set_xlabel('nsample')
    ax.set_ylabel(pname)
    if fout is not None:
        plt.savefig(f'./result_pics/{date}/{fout}')

    print (f'Average err for {pname} = {np.average(errs)}')


def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


def plot_hist(vals, pname='fit parameter', val_in=None, fout=None, ax=None, nbin=20):
    if ax is None:
        ax = plt.figure()

    hi = ax.hist(vals, bins=nbin)
    ax.set_xlabel(pname)

    ## histogram Gaussian fit
    mu, sig = norm.fit(vals)
    print (f"norm fit for {pname} hist: ", mu, sig)
    nums, bins, _ = hi
    binc = (bins[:-1]+bins[1:])/2

    p0 = [np.sum(nums)*(binc[1]-binc[0]), np.average(vals), np.std(vals)]

    try:
        coe, _ = curve_fit(gauss, binc, nums, p0=p0) # coe, varmat
    except:
        coe = p0


    xbin = np.linspace(bins[0], bins[-1], nbin*10) 
    #ax.plot(xbin, norm.pdf(xbin, mu, sig)*sum(nums)*(bins[1]-bins[0]), label='Gaussian fit')
    ax.plot(xbin, gauss(xbin, *coe))
    print (f"gauss fit for {pname} hist: ", coe[1], coe[2])

    A, mu, sig = coe

    if val_in is not None:
       y_in = ax.get_ylim()
       x_in = (val_in, val_in)
       plt.plot(x_in, y_in, 'k--', label=f'input {pname}')

    ax.legend()
    ax.set_title(r'$\mu = %.3e, \sigma = %.3e$' % (mu, sig))


def plot_corner(dat):
    print ("plotting corner")
    pnames = dat['pnames']
    data = []

    pns = []
    for pn in pnames:
        print (pn, np.var(dat[pn]))
        if np.var(dat[pn]) > 1e-20:
            data.append(dat[pn])
            pns.append(pn)

    data = np.vstack(data).T
    inits = []
    for pn in pns:
        try:
            inits.append(np.average(dat['cmbinits'][()][pn]))
        except:
            inits.append(0)

    #pnames = [r'$\tau$', r'$r$']
    title_fmt=['.3f', '.1f', '.3f', '.3f']
    title_fmt='.4f'
    inits = [0.05, 91.7, 1.4730, -0.4625]
    print (data)
    corner.corner(data, 
                  labels=pns, 
                  quantiles=[0.16, 0.5, 0.84], 
                  show_titles=True, 
                  color='b',
                  bins=50,
                  smooth=False, #True,
                  smooth1d=False, #True,
                  title_fmt=title_fmt,
                  labelpad=20,)
                  #truths=inits, )

    print ("covariance matrix :")
    cov = np.cov(data.T)
    print (cov)
    print ("correlation :")
    print (np.corrcoef(data.T))

    for i, pname in enumerate(pns):
        if len(pns) > 1:
            print (f'uncertainty for {pname} = {np.sqrt(cov[i,i])}')
        else:
            print (f'uncertainty for {pname} = {np.sqrt(cov)}')


def plot_spectrum_r1(dat, cfname=None, TT=False, downsample=1):
    if cfname is not None:
        cf = CAMBfast()
        cf.load_funcs(cfname)
        get_spectrum = cf.get_spectrum
    else:
        get_spectrum = get_spectrum_camb
    
    inputpars = dat['inputpars'][()]
    nside = inputpars['Nside']
    lmax = 3 * nside - 1 
    cmbinits = dat['cmbinits'][()]
    initnames = list(cmbinits.keys())
    kwargs = {}
    for pn in initnames: 
        kwargs[pn] = cmbinits[pn]
    
    ## the input spectrum
    wp0 = kwargs.pop('wp', False)
    cls0_cmb = get_spectrum(lmax=lmax, CMB_unit='muK', isDl=False, **kwargs)
    cls0 = cls0_cmb.copy()
    if wp0:
        nls0 = get_spectrum_noise(lmax=lmax, wp=wp0, CMB_unit='muK', isDl=False)
        cls0 = cls0 + nls0

    ## spectra for the ensemble fit 
    print ('Creating plots for the fit results')
    pnames = dat['pnames']
    nens = len(dat[pnames[0]])
    ell = np.arange(2, lmax+1)
    for i in tqdm(range(nens)[::downsample]):
        for pn in pnames:
            kwargs[pn] = dat[pn][i]
        wp = kwargs.pop('wp', None)

        cls1 = cf.get_spectrum(lmax=lmax, CMB_unit='muK', isDl=False, **kwargs)

        if wp:
            nls1 = get_spectrum_noise(lmax=lmax, wp=wp, CMB_unit='muK', isDl=False)
            cls1 = cls1 + nls1
        elif wp0:
            cls1 = cls1 + nls0

        if TT:
            plt.loglog(ell, cl2dl(cls1).T[2:, 0], 'g', alpha=0.4)
        plt.loglog(ell, cl2dl(cls1).T[2:, 1], 'c', alpha=0.4)
        plt.loglog(ell, cl2dl(cls1).T[2:, 2], 'm', alpha=0.4)

    ## spectra for the input maps
    print ('Creating plots for the input maps')
    maps = dat['maps'] 
    for m in tqdm(maps[::downsample]):
        cla = hp.anafast(m, lmax=lmax)
        if TT:
            plt.loglog(ell, cl2dl(cla).T[2:,0], 'g*', alpha=0.4, ms=1)
        plt.loglog(ell, cl2dl(cla).T[2:,1], 'b*', alpha=0.5, ms=1)
        plt.loglog(ell, cl2dl(cla).T[2:,2], 'r*', alpha=0.5, ms=1)

    ## drawing the input spectra
    if TT:
        plt.loglog(ell, cl2dl(cls0_cmb).T[2:, 0], 'g:')
    plt.loglog(ell, cl2dl(cls0_cmb).T[2:, 1], 'b:')
    plt.loglog(ell, cl2dl(cls0_cmb).T[2:, 2], 'r:')

    if wp0:
        plt.loglog(ell, cl2dl(nls0).T[2:, 0], 'k:')

    if TT:
        plt.loglog(ell, cl2dl(cls0).T[2:, 0], 'g')
    plt.loglog(ell, cl2dl(cls0).T[2:, 1], 'b')
    plt.loglog(ell, cl2dl(cls0).T[2:, 2], 'r')

    cv = (2/(2*ell+1))**0.5#/np.average(mask)
    
    if TT:
        plt.loglog(ell, cl2dl(cls0[0] + cls0_cmb[0]*cv)[2:], 'g--')
        plt.loglog(ell, cl2dl(cls0[0] - cls0_cmb[0]*cv)[2:], 'g--')
    plt.loglog(ell, cl2dl(cls0[1] + cls0_cmb[1]*cv)[2:], 'b--')
    plt.loglog(ell, cl2dl(cls0[1] - cls0_cmb[1]*cv)[2:], 'b--')
    plt.loglog(ell, cl2dl(cls0[2] + cls0_cmb[2]*cv)[2:], 'r--')
    plt.loglog(ell, cl2dl(cls0[2] - cls0_cmb[2]*cv)[2:], 'r--')

    # legend
    if TT:
        plt.plot([],[], 'g', label='input TT')
        plt.plot([],[], 'g--', label='cosmic variance of TT')
        plt.plot([],[], 'g*', label='input map TT')
        plt.plot([],[], 'y', label='TT fits')

    plt.plot([],[], 'b', label='input EE')
    plt.plot([],[], 'b--', label='cosmic variance of EE')
    plt.plot([],[], 'b*', label='input map EE')
    plt.plot([],[], 'c', label='EE fits')

    plt.plot([],[], 'r', label='input BB')
    plt.plot([],[], 'r--', label='cosmic variance of BB')
    plt.plot([],[], 'r*', label='input map BB')
    plt.plot([],[], 'm', label='BB fits')

    plt.legend()

    plt.xlabel(r'Multipole moment, $l$')
    plt.ylabel(r'$D_l (\mu K^2)$')
    title = 'Ensemble test for ' 
    for pn in pnames[:-1]:
        title += pn + ', '
    
    if len(pnames) > 1:
        title = title[:-2]
        title += ' and '

    title += pnames[-1]
    plt.title(title)

    return 


def plot_spectrum_old(dat, cfname=None, TT=False, downsample=1):
    if cfname is not None:
        cf = CAMBfast()
        cf.load_funcs(cfname)
        get_spectrum = cf.get_spectrum
    else:
        get_spectrum = get_spectrum_camb
    
    inputpars = dat['inputpars'][()]
    nside = inputpars['Nside']
    lmax = 3 * nside - 1 
    cmbinits = dat['cmbinits'][()]
    initnames = list(cmbinits.keys())
    kwargs = {}
    for pn in initnames: 
        kwargs[pn] = cmbinits[pn]
    
    ## the input spectrum
    wp0 = kwargs.pop('wp', False)
    try: 
        len(wp0)
        wp0 = wp0[1]
    except:
        wp0 = wp0

    wp0 = 98.5
    cls0_cmb = get_spectrum(lmax=lmax, CMB_unit='muK', isDl=False, **kwargs)
    cls0 = cls0_cmb.copy()
    if wp0 is not None:
        nls0 = get_spectrum_noise(lmax=lmax, wp=wp0, CMB_unit='muK', isDl=False)
        cls0 = cls0 + nls0

    ## spectra for the ensemble fit 
    print ('Creating plots for the fit results')
    pnames = dat['pnames']
    nens = len(dat[pnames[0]])
    ell = np.arange(lmax+1)

    for i in tqdm(range(nens)[::downsample]):
        for pn in pnames:
            kwargs[pn] = dat[pn][i]

        #if kwargs['r'] > 1e-7: continue

        wp = kwargs.pop('wp', None)

        cls1 = get_spectrum(lmax=lmax, CMB_unit='muK', isDl=False, **kwargs)

        if wp:
            nls1 = get_spectrum_noise(lmax=lmax, wp=wp, CMB_unit='muK', isDl=False)
            cls1 = cls1 + nls1
        elif wp0:
            cls1 = cls1 + nls0

        if TT:
            plt.loglog(ell[2:], cl2dl(cls1).T[2:, 0], 'g', alpha=0.4)
        plt.loglog(ell[2:], cl2dl(cls1).T[2:, 1], 'c', alpha=0.4)
        plt.loglog(ell[2:], cl2dl(cls1).T[2:, 2], 'm', alpha=0.4)

    ## spectra for the input maps
    print ('Creating plots for the input maps')
    mask = dat['inputpars'][()]['mask']
    maps = dat['maps'] 
    rs = dat['r']
    for m in tqdm(maps[::downsample]):
        #if r > 1e-7: continue
        mm = np.array([mask.copy()]*3)
        for i, mi in enumerate(mm):
            mi[mi==1] = m[i]
        cla = hp.anafast(mm, lmax=lmax)
        #cla = np.zeros(cls0_cmb.shape)

        if TT:
            plt.loglog(ell[2:], cl2dl(cla[:4]+nls0).T[2:,0], 'g*', alpha=1, ms=1)
        plt.loglog(ell[2:], cl2dl(cla[:4]+nls0).T[2:,1], 'b*', alpha=1, ms=1)
        plt.loglog(ell[2:], cl2dl(cla[:4]+nls0).T[2:,2], 'r*', alpha=1, ms=1)
        #print (cla)

    ## drawing the input spectra
    if TT:
        plt.loglog(ell[2:], cl2dl(cls0_cmb).T[2:, 0], 'g:')
    plt.loglog(ell[2:], cl2dl(cls0_cmb).T[2:, 1], 'b:')
    plt.loglog(ell[2:], cl2dl(cls0_cmb).T[2:, 2], 'r:')

    if wp0:
        plt.loglog(ell[2:], cl2dl(nls0).T[2:, 1], 'k:')

    if TT:
        plt.loglog(ell[2:], cl2dl(cls0).T[2:, 0], 'g')
    plt.loglog(ell[2:], cl2dl(cls0).T[2:, 1], 'b')
    plt.loglog(ell[2:], cl2dl(cls0).T[2:, 2], 'r')

    print (f'Sky coverage = {np.average(mask)}')
    cv = ((2/(2*ell+1))**0.5)#/np.average(mask)**0.5
    
    if TT:
        plt.loglog(ell[2:], cl2dl(cls0[0] + cls0_cmb[0]*cv)[2:], 'g--')
        plt.loglog(ell[2:], cl2dl(cls0[0] - cls0_cmb[0]*cv)[2:], 'g--')
    plt.loglog(ell[2:], cl2dl(cls0[1] + cls0_cmb[1]*cv)[2:], 'b--')
    plt.loglog(ell[2:], cl2dl(cls0[1] - cls0_cmb[1]*cv)[2:], 'b--')
    plt.loglog(ell[2:], cl2dl(cls0[2] + cls0_cmb[2]*cv)[2:], 'r--')
    plt.loglog(ell[2:], cl2dl(cls0[2] - cls0_cmb[2]*cv)[2:], 'r--')

    # legend
    if TT:
        plt.plot([],[], 'g', label='input TT')
        plt.plot([],[], 'g--', label='cosmic variance of TT')
        plt.plot([],[], 'g*', label='input map TT')
        plt.plot([],[], 'y', label='TT fits')

    plt.plot([],[], 'b', label='input EE')
    plt.plot([],[], 'b--', label='cosmic variance of EE')
    plt.plot([],[], 'b*', label='input map EE')
    plt.plot([],[], 'c', label='EE fits')

    plt.plot([],[], 'r', label='input BB')
    plt.plot([],[], 'r--', label='cosmic variance of BB')
    plt.plot([],[], 'r*', label='input map BB')
    plt.plot([],[], 'm', label='BB fits')

    plt.legend()

    plt.xlabel(r'Multipole moment, $l$')
    plt.ylabel(r'$D_l (\mu K^2)$')
    title = 'Ensemble test for ' 

    for pn in pnames[:-1]:
        title += pn + ', '
    
    if len(pnames) > 1:
        title = title[:-2]
        title += ' and '

    title += pnames[-1]
    plt.title(title)

    return 


def plot_spectrum(dat, cfname=None, TT=False, downsample=1):
    if cfname is not None:
        cf = CAMBfast()
        cf.load_funcs(cfname)
        get_spectrum = cf.get_spectrum
    else:
        get_spectrum = get_spectrum_camb
    
    inputpars = dat['inputpars'][()]
    nside = inputpars['Nside']
    lmax = 3 * nside - 1 
    cmbinits = dat['cmbinits'][()]
    initnames = list(cmbinits.keys())
    kwargs = {}
    for pn in initnames: 
        kwargs[pn] = cmbinits[pn]
    
    ## the input spectrum
    wp0 = kwargs.pop('wp', False)
    cs = kwargs.pop('cs', False)

    if cs is False:
        cs = []
        for k in list(kwargs.keys()):
            if k[0]=='c':
                cs.append(kwargs.pop(k, False))

        cs.append(1-np.sum(cs))
        cs = np.array(cs)

    try: 
        len(wp0)
        wp0 = sum((wp0 * cs)**2)**0.5
    except:
        wp0 = np.array(wp0)

    print (f'cs = {cs}')
    print (f'wp0 = {wp0}')

    cls0_cmb = get_spectrum(lmax=lmax, CMB_unit='muK', isDl=False, **kwargs)
    cls0 = cls0_cmb.copy()

    if wp0 is not None:
        nls0 = get_spectrum_noise(lmax=lmax, wp=wp0, CMB_unit='muK', isDl=False)
        cls0 = cls0 + nls0

    ## spectra for the ensemble fit 
    print ('Creating plots for the fit results')
    pnames = dat['pnames']
    nens = len(dat[pnames[0]])
    ell = np.arange(lmax+1)

    for i in tqdm(range(nens)[::downsample]):
        for pn in pnames:
            kwargs[pn] = dat[pn][i]

        #if kwargs['r'] > 1e-7: continue

        wp = kwargs.pop('wp', None)

        cls1 = get_spectrum(lmax=lmax, CMB_unit='muK', isDl=False, **kwargs)

        if wp:
            nls1 = get_spectrum_noise(lmax=lmax, wp=wp, CMB_unit='muK', isDl=False)
            cls1 = cls1 + nls1
        elif wp0:
            cls1 = cls1 + nls0

        if TT:
            plt.loglog(ell[2:], cl2dl(cls1).T[2:, 0], 'g', alpha=0.4)
        plt.loglog(ell[2:], cl2dl(cls1).T[2:, 1], 'c', alpha=0.4)
        #plt.loglog(ell[2:], cl2dl(cls1).T[2:, 2], 'm', alpha=0.4)

    ## spectra for the input maps
    print ('Creating plots for the input maps')
    mask = dat['inputpars'][()]['mask']
    maps = dat['maps'] 
    
    rs = dat['r']
    
    clas = []
    for m in tqdm(maps[::downsample]):
        #if r > 1e-7: continue
        mm = np.array([mask.copy()]*3)
        ms = []
        for i, mi in enumerate(mm):
            mi[mi==1] = m[i]
            #mi = hp.smoothing(mi, fwhm=np.radians(1))
            ms.append(mi)
        ms = np.array(ms)
        cla = hp.anafast(mm, lmax=lmax) / np.mean(mask)
        #cla = np.abs(get_spectrum_xpol(ms, lmax=lmax+1, mask=mask)[1])
        #cla = np.zeros(cls0_cmb.shape)

        if TT:
            plt.loglog(ell[2:], cl2dl(cla[:4]).T[2:,0], 'g*', alpha=1, ms=1)
        plt.loglog(ell[2:], cl2dl(cla[:4]).T[2:,1], 'bo', alpha=0.3, ms=1, lw=0.5)
        clas.append(cla[:4])
        #plt.loglog(ell[2:], cl2dl(cla[:4]).T[2:,2], 'r*', alpha=1, ms=1)
        #print (cla)

    cla_avg = np.average(clas, axis=0)  
    cla_std = np.std(clas, axis=0)  

    ## drawing the input spectra
    if TT:
        plt.loglog(ell[2:], cl2dl(cls0_cmb).T[2:, 0], 'g:')
    plt.loglog(ell[2:], cl2dl(cls0_cmb).T[2:, 1], 'b:')
    #plt.loglog(ell[2:], cl2dl(cls0_cmb).T[2:, 2], 'r:')

    if wp0:
        plt.loglog(ell[2:], cl2dl(nls0).T[2:, 1], 'k:')

    if TT:
        plt.loglog(ell[2:], cl2dl(cls0).T[2:, 0], 'g')
    plt.loglog(ell[2:], cl2dl(cls0).T[2:, 1], 'b')
    #plt.loglog(ell[2:], cl2dl(cls0).T[2:, 2], 'r')

    print (f'Sky coverage = {np.average(mask)}')
    cv = ((2/(2*ell+1))**0.5)/np.average(mask)**0.5
    
    """
    if TT:
        plt.loglog(ell[2:], cl2dl(cls0[0] + cls0_cmb[0]*cv)[2:], 'g--')
        plt.loglog(ell[2:], cl2dl(cls0[0] - cls0_cmb[0]*cv)[2:], 'g--')
    plt.loglog(ell[2:], cl2dl(cls0[1] + cls0[1]*cv)[2:], 'b--')
    plt.loglog(ell[2:], cl2dl(cls0[1] - cls0[1]*cv)[2:], 'b--')
    plt.loglog(ell[2:], cl2dl(cls0[2] + cls0_cmb[2]*cv)[2:], 'r--')
    plt.loglog(ell[2:], cl2dl(cls0[2] - cls0_cmb[2]*cv)[2:], 'r--')
    """

    plt.errorbar(ell[2:], cl2dl(cla_avg).T[2:,1], yerr=cl2dl(cla_std).T[2:,1], 
                 elinewidth=1, capsize=2, marker='s', ls='', ms=3, color='k')

    # legend
    if TT:
        plt.plot([],[], 'g', label='input TT')
        plt.plot([],[], 'g--', label='cosmic variance of TT')
        plt.plot([],[], 'g*', label='input map TT')
        plt.plot([],[], 'y', label='TT fits')

    plt.plot([],[], 'b', label='input EE')
    #plt.plot([],[], 'b--', label='cosmic variance of EE')
    plt.plot([],[], 'b*', label='input map EE')
    plt.plot([],[], 'c', label='EE fits')
    plt.plot([],[], 'ks', label='input map average')
    plt.plot([],[], 'b:', label='CMB EE theory')
    plt.plot([],[], 'k:', label='Noise')

    """
    plt.plot([],[], 'r', label='input BB')
    plt.plot([],[], 'r--', label='cosmic variance of BB')
    plt.plot([],[], 'r*', label='input map BB')
    plt.plot([],[], 'm', label='BB fits')
    """

    plt.legend()

    plt.xlabel(r'Multipole moment, $l$')
    plt.ylabel(r'$D_l (\mu K^2)$')
    title = 'Ensemble test for ' 

    for pn in pnames[:-1]:
        title += pn + ', '
    
    if len(pnames) > 1:
        title = title[:-2]
        title += ' and '

    title += pnames[-1]
    plt.title(title)

    return 


def rms_fgresidual(fgs, cs):
    fgcleaned = np.einsum('i,ijk->jk', cs, fgs)  
    rms = np.mean(np.abs(fgcleaned[1]+1j*fgcleaned[2])**2)**0.5
    return rms


def check_fgresidual(dat, freqs, nside=8):
    fgs_raw = np.array(gen_fg(freqs, nside)) 
    mask = mymask(None, nside=nside)
    fgs = []
    for fr in fgs_raw:
        fgs.append(cutmap(fr, mask))
    fgs = np.array(fgs)
    
    cnames = list(map(lambda f: 'c' + str(f), freqs))[:-1]
    cs = []
    for cn in cnames:
        cs.append(dat[cn])

    cst = np.sum(cs, axis=0)
    cs.append(1-np.array(cst))
    cs = np.array(cs)
    print (cs.shape)

    rms = []
    for c in cs.T:
        rms.append(rms_fgresidual(fgs, c))

    print (f'mean of rms fgres = {np.mean(rms)}')
    print (f'stdv of rms fgres = {np.std(rms)}')
    

def plot_ensemble(dat):
    date = datetime.datetime.now().isoformat()[:10]
    if not os.path.isdir(f'./result_pics/{date}'):
        os.mkdir(f'./result_pics/{date}')

    parnames = dat['pnames']

    npar = len(parnames)
    nsample = len(dat[parnames[0]])

    print (f"### Number of entries = {nsample}")

    ## check rms fg residual
    if '3bands' in sys.argv[1]: 
        freqs = [145, 220, 30]#, 11, 13, 17, 19]
    elif '4bands' in sys.argv[1]: 
        freqs = [145, 220, 30, 40]#, 11, 13, 17, 19]
    elif '6bands_nocorr_1' in sys.argv[1]:
        freqs = [145, 220, 30, 40, 11, 17]
    elif '6bands_nocorr_2' in sys.argv[1]:
        freqs = [145, 220, 30, 40, 11, 19]
    elif '8bands' in sys.argv[1]:
        freqs = [145, 220, 30, 40, 11, 13, 17, 19]
    else:
        #freqs = [30, 40, 11, 13, 17, 19]
        freqs = [145, 220,]
        
    #check_fgresidual(dat, freqs)
    #return 

    ## result
    fig1 = plt.figure()
    
    for i, pn in enumerate(parnames):
        ax = plt.subplot(npar, 1, i+1)
        plot_result(dat[pn], dat[pn+'_err'], pname=pn, ax=ax)
    
    #plt.savefig(f'./result_pics/{date}/ens_{prefix}_res_tau_wp_ntest{ntest}.png')

    ## hist
    nbin = 30
    fig2 = plt.figure()
    for i, pn in enumerate(parnames):
        ax = plt.subplot(1, npar, i+1)
        try:
            val_in = dat['cmbinits'][()][pn]
        except:
            val_in = 0
        plot_hist(dat[pn], pname=pn, ax=ax, nbin=nbin, val_in=val_in)
   
    fig7 = plt.figure()
    for i, pn in enumerate(parnames):
        ax = plt.subplot(1, npar, i+1)
        plot_hist(dat[pn+'_err'], pname=pn, ax=ax, nbin=nbin)

    fig10 = plt.figure()
    for i, pn in enumerate(parnames):
        ax = plt.subplot(1, npar, i+1)
        try:
            val_in = dat['cmbinits'][()][pn]
        except:
            val_in = 0
        dat_in = dat[pn]
        err_in = dat[pn+'_err']
        dat_in = dat_in[err_in<0.10]
        print (dat_in.shape)
        #dat_in = dat_in[dat_in>0.0050]
        plot_hist(dat_in, pname=pn, ax=ax, nbin=nbin, val_in=val_in)

    #plt.savefig(f'./result_pics/{date}/ens_{prefix}_hist_tau_wp_ntest{ntest}.png')

    ## spectrum plots
    fig3 = plt.figure()
    downsample = max(nsample//100, 1)
    #plot_spectrum(dat, downsample=downsample)
    plot_spectrum(dat, cfname='./cambfast/tau_lmax50_npts400_mintau0017.npz', downsample=downsample)

    #plt.savefig(f'./result_pics/{date}/ens_{prefix}_spec_tau_wp_ntest{ntest}.png')

    ## corner plots
    plot_corner(dat)

    
    return


def singlefile():
    fname = sys.argv[1]
    dat = np.load(fname, allow_pickle=True)
    plot_ensemble(dat)

    plt.show()


def multiplefiles():
    fnames = sys.argv[1:]
    dat_a = []
    for fn in fnames:
        dat_a.append(np.load(fn, allow_pickle=True))
    dat = merge_dat(dat_a)

    plot_ensemble(dat)
    plt.show()


if __name__=='__main__':
    if len(sys.argv) == 2:
        singlefile()
    elif len(sys.argv) > 2:
        multiplefiles()
    else:
        pass


