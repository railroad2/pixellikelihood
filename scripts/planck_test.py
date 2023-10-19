#!/usr/bin/env python3
import time
import numpy as np
import healpy as hp
import pylab as plt

import likelihoodfit_ilc as li
from iminuit import Minuit
from gbpipe.utils import cl2dl, set_logger
from gbpipe.spectrum import get_spectrum_noise, get_spectrum_camb
from pixelcov.covmat_ana import gen_Pl_ana, gen_Wls_ana
from pixelcov.covcut import cutmap, cutcov, cutpls, cutwls, partcov, partmap, Kmat
from cambfast.cambfast import CAMBfast
from pixelLikelihood import computeL
from fg_setup import *

VLVL = 'DEBUG'


def fit_planck_cmbmap(map_in):
    logger = set_logger(level=VLVL)
    logger.debug(f'Calling fit_planck_cmbmap().')

    nside = 8
    lmax = 23
    logger.info(f'Nside = {nside}')

    maptype = 'QU'
    covtype = 'ana'
    regtype = 'I'
    covblk = ['TT', 'QQ', 'UU', 'TQ', 'TU', 'QU']
    lamb = 1e-15

    mask = mymask(nside=8, nomask=True, 
                  gbmask=False, galmask=True, syncmask=False, bwmask=True)

    logger.debug(f'The shape of map_in = {np.shape(map_in)}')

    print ('pls wls')
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

    print ('cambfast')
    logger.debug(f'Setting up cambfast')
    cf = CAMBfast()
    cfname = '../cambfast/tau_lmax50_npts400_mintau0017.npz'
    if cfname is None:
        cfname = '../cambfast/tau_lmax50_npts200.npz'
    cf.load_funcs(cfname)

    map_in_cut = cutmap(map_in, mask)
    pls_cut = cutpls(pls, mask)
    wls_cut = cutwls(wls, mask)
      
    def lkfunc(tau, wp):
        cls = cf.get_spectrum(lmax=lmax, tau=tau, CMB_unit='muK', isDl=False)
        nls = get_spectrum_noise(lmax=lmax, wp=wp, isDl=False, CMB_unit='muK')
        cls += nls

        lk = computeL(map_in_cut, cls, nside, pls=pls_cut, wls=wls_cut, 
                      lmax=lmax, covtype=covtype, maptype=maptype, 
                      covblk=covblk, mask=[],
                      regtype=regtype, reglamb=lamb)

        return lk

    limit_tau = (0.0017, 0.2)
    limit_wp  = (0, 500)
     
    m = Minuit(lkfunc, tau=0.07, wp=220, 
               fix_wp=False,
               limit_tau=limit_tau, limit_wp=limit_wp, print_level=2) 

    print ('migrad')
    res = m.migrad()
    print (res)

    #print ('hesse')
    #resh = m.hesse()
    #print (resh)
    #resm = m.minos()

    return res


def test1_planckmap():
    mappath = '/home/kmlee/cmb/forecast/maps/planck/'
    fname = mappath+'COM_CMB_IQU-nilc_2048_R3.00_full.fits'
    
    """
    map_in_gal = hp.read_map(fname, field=(5,6,7), nest=False) * 1e6  # temperature unit to uK
    print ('rotating map')
    rot = hp.Rotator(coord=['G', 'C'])
    map_in_equ = rot.rotate_map_alms(map_in_gal)
    hp.write_map('planck_CMB_2048_equ.fits', map_in_equ)

    map_in = map_in_equ

    hp.mollview(map_in[1])
    map_in = hp.ud_grade(map_in, nside_out=nside) 
    hp.mollview(map_in[1])
    plt.show()

    hp.write_map('planck_CMB_8_equ.fits', map_in)
    """

    map_in = hp.read_map('planck_CMB_8_equ.fits', field=None)

    fit_planck_cmbmap(map_in)


def test2_simulatedmap():
    print ('cambfast')
    cf = CAMBfast()
    cfname = '../cambfast/tau_lmax50_npts400_mintau0017.npz'
    if cfname is None:
        cfname = '../cambfast/tau_lmax50_npts200.npz'
    cf.load_funcs(cfname)

    cl0 = cf.get_spectrum(lmax=40, tau=0.05, CMB_unit='muK', isDl=False)
    nl0 = get_spectrum_noise(lmax=40, wp=260, CMB_unit='muK', isDl=False)
    map_in = hp.synfast(cl0+nl0, nside=8, new=True)

    fit_planck_cmbmap(map_in)


if __name__=="__main__":
    test2_simulatedmap()


