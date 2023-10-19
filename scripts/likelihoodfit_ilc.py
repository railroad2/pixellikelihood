#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import time
import datetime 

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from scipy.stats import norm
from iminuit import Minuit 
from tqdm import tqdm

from gbpipe.spectrum import get_spectrum_noise, get_spectrum_camb
from gbpipe.utils import cl2dl, set_logger
from gbilc import * 

from pixelLikelihood import computeL
from cambfast.cambfast import CAMBfast
from pixelcov.covmat_ana import gen_Pl_ana, gen_Wls_ana
from pixelcov.covcut import cutpls, cutwls, cutmap
from gen_fg import gen_fg

VLVL = 'DEBUG' # verbose level


def noiselvl_combined(wps, cs):
    wps = np.array(wps)
    cs = np.array(cs)
    return (np.sum(wps**2 * cs**2, axis=0))**0.5


def define_lkfunc(maps_in, parnames, nside, wp0, cf, 
                  pls, wls, mask, lamb, lmax, lmin, 
                  covtype, maptype, covblk, regtype, 
                  fix_wp, cswp, fwhm):

    tau_idx = parnames.index('tau') 
    r_idx = parnames.index('r') 
    wp_idx = parnames.index('wp') 

    
    def lkfunc(*pars):
        tau = pars[tau_idx]
        r = pars[r_idx]
        wp = pars[wp_idx]
        cs = pars[3:]
        cs = [*cs, 1.0 - sum(cs)]
        cs = np.array(cs)

        if fix_wp and cswp:
            wp1 = np.sqrt(np.sum((cs * np.array(wp0))**2, axis=0)) 
        else:
            wp1 = wp

        cls = cf.get_spectrum(lmax=lmax, tau=tau, r=r, CMB_unit='muK', isDl=False)
        nls = get_spectrum_noise(lmax=lmax, wp=wp1, fwhm=fwhm, isDl=False, CMB_unit='muK')
        cls += nls

        maps_arr = np.array(maps_in)
        map_cleaned = np.einsum('i,ijk->jk', cs, maps_arr)
        map_in = np.array(map_cleaned)

        t0 = time.time()
        lk = computeL(map_in, cls, nside, pls=pls, wls=wls, 
                      lmax=lmax, lmin=lmin,
                      covtype=covtype, maptype=maptype, 
                      covblk=covblk, mask=[],
                      regtype=regtype, reglamb=lamb)

        return lk

    return lkfunc


def define_lkfunc_negtau(maps_in, parnames, nside, wp0, cf, 
                  pls, wls, mask, lamb, lmax, lmin, 
                  covtype, maptype, covblk, regtype, 
                  fix_wp, cswp, fwhm):

    tau_idx = parnames.index('tau') 
    r_idx = parnames.index('r') 
    wp_idx = parnames.index('wp') 

    
    def lkfunc(*pars):
        tau = pars[tau_idx]
        r = pars[r_idx]
        wp = pars[wp_idx]
        cs = pars[3:]
        cs = [*cs, 1.0 - sum(cs)]
        cs = np.array(cs)

        if fix_wp and cswp:
            wp1 = np.sqrt(np.sum((cs * np.array(wp0))**2, axis=0)) 
        else:
            wp1 = wp

        if tau < 0:
            cl0 = cf.get_spectrum(lmax=lmax, tau=0, r=r, CMB_unit='muK', isDl=False)
            clp = cf.get_spectrum(lmax=lmax, tau=(-1*tau), r=r, CMB_unit='muK', isDl=False)
            cls = 2*cl0 - clp
        else:
            cls = cf.get_spectrum(lmax=lmax, tau=tau, r=r, CMB_unit='muK', isDl=False)

        nls = get_spectrum_noise(lmax=lmax, wp=wp1, fwhm=fwhm, isDl=False, CMB_unit='muK')
        cls += nls

        maps_arr = np.array(maps_in)
        map_cleaned = np.einsum('i,ijk->jk', cs, maps_arr)
        map_in = np.array(map_cleaned)

        t0 = time.time()
        lk = computeL(map_in, cls, nside, pls=pls, wls=wls, 
                      lmax=lmax, lmin=lmin,
                      covtype=covtype, maptype=maptype, 
                      covblk=covblk, mask=[],
                      regtype=regtype, reglamb=lamb)

        return lk

    return lkfunc


def tau_r_wp_ilc_fit(maps_in, nside=None, mask=[], pls=None, wls=None, lamb=1e-12, 
                     cf=None, minos=False, fwhm=1,
                     parnames=['tau', 'r', 'wp', 'c145', 'c220'],
                     tau0=0.05, r0=0.0, 
                     wp0=[3600, 3600, 7200, 7200, 69, 82, 77, 215], 
                     cs0=[1, 0, 0, 0, 0, 0, 0, 0], 
                     fix_tau=False, fix_r=True, fix_wp=False, fix_cs=False, cswp=False):
    logger = set_logger(level=VLVL)
    logger.debug(f'tau_r_wp_ilc_fit() is called')
    if nside is None:
        print ('nside must be given')
        return 1e10
            
    lmax = 3 * nside - 1
    lmin = 0

    ## prepare and cut pls, wls, and the maps
    if isinstance(pls, str):
        pls = gen_Pl_ana(nside=nside, lmax=lmax, prename=pls)

    if isinstance(wls, str):
        wls = gen_Wls_ana(nside=nside, lmax=lmax, prename=wls)

    if mask != []:
        pls = cutpls(pls, mask)
        wls = cutwls(wls, mask)

        maps_cut = []
        for m in maps_in:
            maps_cut.append(cutmap(m, mask))
    else:
        maps_cut = maps_in
        mask = np.ones(np.shape(maps_in))

    try:
        len(wp0)
    except TypeError:
        wp0 = [wp0] * len(wp0)

    ## ranges of fit parameters
    limit_tau = (0.0017, 0.20)
    #limit_tau = (-0.2, 0.20)
    limit_r = (-0.2, 0.2)
    limit_wp = (0., 2000.0)
    limit_cs = (-30, 30)

    ## covariance setup
    maptype = 'QU'
    covtype = 'ana'
    regtype = 'I'
    covblk = ['TT', 'QQ', 'UU', 'TQ', 'TU', 'QU']
    #covblk = ['TT', 'QQ', 'UU']# 'TQ', 'TU', 'QU']

    ## cambfast
    if cf is None:
        logger.debug(f'Setting up cambfast')
        cf = CAMBfast()
        #cfname = '../cambfast/As_tau_r_lmax40_npts27000.npz'
        cfname = '../cambfast/tau_lmax50_npts400_mintau0.npz'
        cf.load_funcs(cfname)

    ## likelihood function definition -> closure
    logger.info('*** Defining the likelihood function')
    
    #lkfunc =  define_lkfunc(maps_in, parnames, nside, wp0, cf,
    lkfunc =  define_lkfunc_negtau(maps_in, parnames, nside, wp0, cf,
                            pls, wls, mask, lamb, lmax, lmin, 
                            covtype, maptype, covblk, regtype,
                            fix_wp, cswp, fwhm)

    cs0_in = cs0[:-1]
    wp1 = noiselvl_combined(wp0, cs0) 
    logger.info('*** value test')
    t0 = time.time()
    logger.info(lkfunc(tau0, r0, wp1, *cs0_in))
    logger.info('It took {} seconds.'.format(time.time() - t0))

    logger.info('Setting up minuit')
    minuit_kwargs = {'tau':tau0, 'r':r0, 'wp':wp1, 
               'limit_tau':limit_tau, 'fix_tau':fix_tau,
               'limit_r':limit_r, 'fix_r':fix_r,
               'limit_wp':limit_wp, 'fix_wp':fix_wp }

    for pn, ci in zip(parnames[3:], cs0_in):
        minuit_kwargs[pn] = ci
        minuit_kwargs['limit_'+pn] = limit_cs
        minuit_kwargs['fix_'+pn] = fix_cs

    m = Minuit(lkfunc, name=parnames, **minuit_kwargs, errordef=1, print_level=2)

    logger.debug('Running migrad ...')
    t0 = time.time()
    resmig = m.migrad()
    m.print_param()
    logger.info('migrad took {} seconds.'.format(time.time() - t0))

    """
    logger.debug('Running hesse ...')
    t0 = time.time()
    reshes = m.hesse()
    m.print_param()
    logger.info('hesse took {} seconds.'.format(time.time() - t0))
    """

    if minos:
        logger.debug('Running minos ...')
        t0 = time.time()
        resmin = m.minos()
        m.print_param()
        logger.info('minos took {} seconds.'.format(time.time() - t0))
     
    return m


def tau_ensemble_ilc(fgmaps=None, noisemaps=None, nocmb=False, nside=4, ntest=10, lmin=0, ofname=None, 
                     mask=[], ilcmask=[], lamb=1e-12, fitfnc=tau_r_wp_ilc_fit, rseed=0, 
                     tau0=0.05, r0=0.05, 
                     freqs=[145, 220, 11, 13, 17, 19, 30, 40],
                     wp0=[93, 584, 3600, 3600, 7200, 7200, 69, 82],
                     cs0=None, cambini=None, cfname=None,
                     fix_tau=False, fix_r=True, fix_wp=False, fix_cs=False, cswp=True, ilcfirst=False, 
                     ilc_wo_noise=False):

    logger = set_logger(level=VLVL) 
    lmax = 3 * nside - 1

    # input spectra given the initial parameters
    cls0 = get_spectrum_camb(lmax=lmax, tau=tau0, r=r0, CMB_unit='muK', isDl=False, 
                             inifile=cambini)
        
    try:
        len(wp0)
    except TypeError:
        wp0 = [wp0] * len(freqs)

    nls0_arr = []
    for wp0i in wp0:
        nls0_arr.append(get_spectrum_noise(lmax=lmax, wp=wp0i, CMB_unit='muK', isDl=False))

    ## fit for ensemble 
    res_arr = []
    maps_in = []
    nfail = 0

    plsname = f'../precomputed/Pls_new_nside{nside:04d}_lmax40.npz'
    wlsname = f'../precomputed/Wls_new_nside{nside:04d}_lmax40.npz'

    logger.debug(f'Generating pls') 
    t0 = time.time()
    pls = gen_Pl_ana(nside, prename=plsname)
    logger.debug(f'Elapsed time to generate pls = {time.time()-t0} s.') 

    logger.debug(f'Generating wls') 
    t0 = time.time()
    wls = gen_Wls_ana(nside, lmax, prename=wlsname)
    logger.debug(f'Elapsed time to generate wls = {time.time()-t0} s.') 

    logger.debug(f'Preparation for the pls and wls to reduce the computation time')
    t0 = time.time()
    pls = cutpls(pls, mask)
    wls = cutwls(wls, mask)
    logger.debug(f'Elapsed time to prepare = {time.time()-t0} s.') 

    ## cambfast
    logger.debug(f'Setting up cambfast')
    cf = CAMBfast()
    if cfname is None:
        cfname = '../cambfast/tau_lmax50_npts200.npz'
    cf.load_funcs(cfname)

    cnames = list(map(lambda f: 'c' + str(f), freqs))[:-1]
    parnames = ['tau','r','wp'] + cnames

    ## preparing foregrounds 
    if fgmaps is None:
        maps_fg = np.array(gen_fg(freqs, nside))
    else:
        maps_fg = fgmaps

    np.random.seed(rseed)

    i1 = 0
    failidx=[]
    t1 = time.time()
    wp_vals = []
    wp_errs = []
    maps_cleaned = []

    ## Ensemble test
    for i in tqdm(range(ntest)):
        t0 = time.time()

        ## prepare CMB map
        map_cmb = hp.synfast(cls0, nside=nside, new=True, verbose=False)
        if nocmb:
            map_cmb *= 0.0

        ## prepare noise maps
        maps_noise = []
        if noisemaps is None:
            for nls0 in nls0_arr:
                nm = hp.synfast(nls0, nside=nside, new=True, verbose=False)
                maps_noise.append(nm)
            maps_noise = np.array(maps_noise)
        else:
            maps_noise = np.array(noisemaps)

        ### prepare input maps
        maps_in_uncut = map_cmb + maps_fg + maps_noise
        maps_in = []
        for m in maps_in_uncut:
            maps_in.append(cutmap(m, mask))
        maps_in = np.array(maps_in)

        if ilcmask == []:
            ilcmask = mask

        maps_ilc = []
        for m in maps_in_uncut:
            maps_ilc.append(cutmap(m, ilcmask))

        maps_ilc = np.array(maps_ilc)

        ## ILC fit for foreground cleaning in case the cs0 is not given.
        if ilcfirst and fix_cs:
            if ilc_wo_noise:
                maps_ilc1 = []
                for m in (map_cmb + maps_fg):
                    maps_ilc1.append(cutmap(m, ilcmask))
                maps_ilc1 = np.array(maps_ilc1)
                ilcres, _ = ilcmaps(*maps_ilc1)
            else:
                ilcres, _ = ilcmaps(*maps_ilc)
            cs0 = ilcres.x
            cs0 = np.array([*cs0, 1-sum(cs0)])
            logger.info(f'cs0={cs0}') 
        else:
            cs0 = np.array(cs0)

        ## likelihood fit
        try:
            logger.debug(f'Starting a fitting ...') 
            res_arr.append(fitfnc(maps_in, nside=nside, pls=pls, wls=wls, mask=[], lamb=lamb, cf=cf,
                                  tau0=tau0, r0=r0, wp0=wp0, cs0=cs0, parnames=parnames,
                                  fix_tau=fix_tau, fix_r=fix_r, fix_wp=fix_wp, fix_cs=fix_cs, cswp=cswp))
            cs = []
            for cn in cnames:
                cs.append(res_arr[-1].values[cn])

            cs.append(1-sum(cs))
            cs = np.array(cs)

            map_cleaned = np.einsum('i,ijk->jk', cs, maps_in) # cleaned map

            maps_cleaned.append(map_cleaned)

            if fix_wp and cswp:
                cs_fit = cs
                cs_err = []
                for cn in cnames:
                    cs_err.append(res_arr[-1].errors[cn])
                cs_err.append(np.sqrt(np.sum((cs_fit[:-1] * np.array(cs_err))**2)))
                cs_err = np.array(cs_err)
                cs_fit = np.array(cs_fit)
                wp = np.array(wp0)
                wp_val = np.sqrt(np.sum((cs_fit * wp)**2))
                wp_err = np.sqrt(np.sum(wp**4 * cs_fit**2 * cs_err**2 / wp_val**2))
                print (wp_val, '+/-', wp_err)
                wp_vals.append(wp_val)
                wp_errs.append(wp_err)

        except (RuntimeError, KeyboardInterrupt):
            logger.warning('User Interrupt. Finishing the ensemble test')
            wfile = 'n'
            try:
                wfile = input('Do you want to save the data in a file? [y/N]')
            except (KeyboardInterrupt):
                ofname = 'ens_temp'
                break

            if (wfile not in ['y', 'Y', 'yes', 'Yes', 'yeah', 'sure']):
                ofname = 'ens_temp'
            break

        except:
            logger.error('Fit fail: Errors occured during the likelihood fit')
            logger.error(sys.exc_info())
            nfail += 1
            failidx.append(i)

        logger.info('Elapsed time for the one fitting: %f s' % (time.time() - t0))

    logger.info(f'Elapsed time for the ensemble test (ntest={i}, nfail={nfail}): %f s' % (time.time() - t1))

    logger.debug(f'Making parameter arrays')
    parnames = res_arr[0].parameters
    val_arr = {} 
    err_arr = {} 
    for pn in parnames:
        val_arr[pn] = []
        err_arr[pn] = []

    for res in res_arr:
        for pn in parnames:
            val_arr[pn].append(res.values[pn])
            err_arr[pn].append(res.errors[pn])

    for pn in parnames:
        logger.info('********* {}'.format(pn))
        logger.info('average of {} = {}'.format(pn, np.average(val_arr[pn])))
        logger.info('average error of {} = {}'.format(pn, np.average(err_arr[pn])))
        logger.info('ensemble error of {} = {}'.format(pn, np.std(val_arr[pn])))

    ## save arrays
    logger.debug(f'Saving arrays')
    if ofname is not None:

        inputpars = {}
        cmbinits = {}

        inputpars['Nside'] = nside
        inputpars['lmax'] = lmax
        inputpars['mask'] = mask

        cmbinits['tau'] = tau0
        cmbinits['r'] = r0
        cmbinits['wp'] = wp0
        cmbinits['cs'] = cs0
        for i, cn in enumerate(cnames):
            cmbinits[cn] = cs0[i] 

        kwargs = {}
        kwargs['maps'] = maps_cleaned
        kwargs['pnames'] = parnames
        kwargs['inputpars'] = inputpars
        kwargs['cmbinits'] = cmbinits
        kwargs['failidx'] = failidx
        kwargs['ntest'] = ntest

        for pn in parnames:
            kwargs[f'{pn}']=val_arr[pn]
            kwargs[f'{pn}_err']=err_arr[pn]

        if fix_wp and cswp:
            kwargs['wp'] = wp_vals
            kwargs['wp_err'] = wp_errs

        np.savez(ofname, **kwargs)

    logger.info(f'The data has been written in {ofname}')

    return 

    for pn in parnames:
        logger.debug(f'Making likelihood profile for {pn}. Note that it is a time consuming job.')
        plt.figure()
        res_arr[-1].draw_profile(pn)
        plt.savefig(f'lkprofile_{pn}.png')
        plt.close()
        logger.debug(f'Likelihood profile for {pn} is finished.')

    logger.debug(f'Ensemble test finished.')

    return


def test_single_fit():
    from gen_fg import gen_fg
    nside = 8
    lmax = 3 * nside -1

    freqs = [11, 13, 17, 19, 30, 40, 90, 145, 220]
    wp0 = [3600, 3600, 7200, 7200, 69, 82, 77, 77, 215]
    cs0 = [ 1.08171177e-03,  1.60547051e-04, -8.56960005e-04, -2.28852768e-04,
           -2.27093594e-01,  4.14118759e-01,  7.64447455e-01,  2.18440248e-01]
    cl0 = get_spectrum_camb(1024, tau=0.05, r=0.05, As=2.092e-9, isDl=False, CMB_unit='muK', 
                            inifile=cambini)
    cmb = hp.synfast(cl0, nside=nside, verbose=0, new=1)

    map_noise = []
    for i in range(len(wp0)):
        nls0 = (get_spectrum_noise(lmax=lmax, wp=float(wp0[i]), CMB_unit='muK', isDl=False))
        map_noise.append(hp.synfast(nls0, nside=nside, new=True, verbose=False))
    map_noise = np.array(map_noise)

    fgs = gen_fg(freqs, nside)

    map_ins = fgs + cmb + map_noise
    mask = mymask(None, nside1=nside)
    plsname = f'../precomputed/Pls_new_nside{nside:04d}_lmax40.npz'
    wlsname = f'../precomputed/Wls_new_nside{nside:04d}_lmax40.npz'

    cs0 = None
    if cs0 is None:
        ilcres, _ = ilcmaps(*map_ins)
        cs0 = ilcres.x


    print (len(cs0))
    print (cs0)

    tau_r_wp_ilc_fit(map_ins, nside=nside, mask=mask, pls=plsname, wls=wlsname, lamb=1e-12, 
              tau0=0.05, r0=0.05, wp0=wp0, 
              cs0=cs0,
              fwhm=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
              fix_tau=False, fix_r=True, fix_wp=False, fix_cs=True, cswp=True)

     
if __name__=='__main__':
    test_single_fit()


