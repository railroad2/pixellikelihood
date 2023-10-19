#!/usr/bin/env python3
import os
import sys
import time
import datetime 

import numpy as np
import healpy as hp

from fg_setup import *
from likelihoodfit_ilc import *
from read_config import read_config, get_config


def ensemble_test(confname='./example.ini', rseed_in=None):
    cfg = read_config(confname)

    fitfnc = globals()[get_config(cfg, 'fitfnc')]
    ntest = get_config(cfg, 'ntest')
    lamb = get_config(cfg, 'lamb')
    nside = get_config(cfg, 'nside')
    cambini = get_config(cfg, 'cambini')
    tau0 = get_config(cfg, 'tau0')
    r0 = get_config(cfg, 'r0')
    nbands = get_config(cfg, 'nbands')
    freqs = get_config(cfg, 'freqs')
    wp0 = get_config(cfg, 'wp0')
    cs0 = get_config(cfg, 'cs0')
    nomask = get_config(cfg, 'nomask')
    gbmask = get_config(cfg, 'gbmask')
    galmask = get_config(cfg, 'galmask')
    syncmask = get_config(cfg, 'syncmask')
    bwmask = get_config(cfg, 'bwmask')
    ilcmask = get_config(cfg, 'ilcmask')
    maskfname = get_config(cfg, 'maskfname')
    fix_tau = get_config(cfg, 'fix_tau')
    fix_r = get_config(cfg, 'fix_r')
    fix_wp = get_config(cfg, 'fix_wp')
    fix_cs = get_config(cfg, 'fix_cs')
    cswp = get_config(cfg, 'cswp')
    ilcfirst = get_config(cfg, 'ilcfirst')
    fnameprefix = get_config(cfg, 'fnameprefix')
    cfname = get_config(cfg, 'cfname')
    Ndetscale = get_config(cfg, 'ndetscale', 1)
    res145scale = get_config(cfg, 'res145scale', 1)
    res220scale = get_config(cfg, 'res220scale', 1)
    ilc_wo_noise = get_config(cfg, 'ilc_wo_noise')
    nocmb = get_config(cfg, 'nocmb')

    foregroundfnc = get_config(cfg, 'foregroundfnc')
    nullfg = get_config(cfg, 'nullfg')
    GBres = get_config(cfg, 'gbres')

    if foregroundfnc is None:
        fgmaps = np.array(gen_fg(freqs, nside))
        if nullfg:
            fgmaps *= 0.
    else:
        foregroundfnc = globals()[foregroundfnc]
        print (f'foreground function = {foregroundfnc}')
        try:
            fgmaps = foregroundfnc(nside, freqs=freqs, GBres=GBres, 
                                   Ndetscale=Ndetscale, Null=nullfg, 
                                   res145scale=res145scale, res220scale=res220scale)
        except:
            print (f"There's something wrong in your foreground function call... Null foregrounds will be used ...")
            fgmaps = np.zeros((3, 12*nside**2))

    if rseed_in is None:
        rseed = get_config(cfg, 'rseed')
    else:
        rseed = rseed_in

    if fnameprefix is None:
        fnameprefix = 'ensemble' 

    wp_in = np.sum((np.array(wp0) * np.array(cs0))**2)**0.5
    print (f'wp_in = {wp_in}')

    ofname = outfilename(fnameprefix, ntest=ntest, rseed_in=rseed, wp_in=wp_in)

    flistname = confname[:-3]+'filelist'


    mask = mymask(nside=8, nomask=nomask, 
                  gbmask=gbmask, galmask=galmask, syncmask=syncmask, bwmask=bwmask)

    if ilcmask:
        ilcmask = mymask(nside=8, nomask=nomask, 
                     gbmask=gbmask, galmask=False, syncmask=False, bwmask=bwmask)
    else:
        ilcmask = mask
    ## include CMB or not

    tau_ensemble_ilc(fgmaps, nocmb=nocmb, nside=nside, ntest=ntest, ofname=ofname, 
                     mask=mask, ilcmask=ilcmask, lamb=lamb, fitfnc=fitfnc, rseed=rseed, 
                     freqs=freqs, tau0=tau0, r0=r0, wp0=wp0, cs0=cs0, 
                     cambini=cambini, cfname=cfname,
                     fix_tau=fix_tau, fix_r=fix_r, fix_wp=fix_wp, 
                     fix_cs=fix_cs, cswp=cswp, ilcfirst=ilcfirst, ilc_wo_noise=ilc_wo_noise)

    with open(flistname, 'a') as f:
        f.write(ofname+'\n')

    return 


if __name__=='__main__':
    print (f'Running this script started at {datetime.datetime.now().isoformat()}')
    if len(sys.argv) == 2:
        confname = sys.argv[1] 
        ensemble_test(confname)
    elif len(sys.argv) == 3:
        confname = sys.argv[1] 
        rseed = int(sys.argv[2])
        ensemble_test(confname, rseed_in=rseed)
    else:
        ensemble_test()


