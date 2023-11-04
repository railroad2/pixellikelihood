#!/usr/bin/env python3

import os
import datetime

import numpy as np
import healpy as hp
import pylab as plt

from gbpipe.gbmap import makegbmask
from gbpipe.spectrum import get_spectrum_noise
from gen_fg import gen_fg

map_path = '/home/kmlee/works/cmb/forecast/maps/'

## pixel noise level
def pnl(NET, fsky, Ndet, Y, t):
    wp = np.array(NET)**2 * 4*np.pi*(10800/np.pi)**2*np.array(fsky) / (np.array(Ndet)*np.array(Y)*t)
    return wp ** 0.5 


def binmask(m, cut=0.8):
    m1 = m.copy()
    m1[m1>=cut] = 1
    m1[m1<cut] = 0

    return m1


def mymask(maskfname=None, coord='equ', nside=8, 
           tilt=20, latOT=61.7, 
           nomask=0, gbmask=1, galmask=1, syncmask=0, bwmask=1,
          ):
    mmin = latOT - tilt - 10
    mmax = latOT + tilt + 10 
    nside0 = 1024
    if maskfname is None:
        saveflg = False
    else:
        try:
            mask = hp.read_map(maskfname)
            return np.array(mask)
        except:
            print (f'The file {maskfname} does not exist. Creating the mask.')
            saveflg = True

    ## gb mask
    if gbmask:
        mask_gb = makegbmask(nside0, mmin, mmax)
        rot = hp.rotator.Rotator(coord=['c','g'])
        if coord == 'gal':
            mask_gb = binmask(rot.rotate_map_pixel(mask_gb))
    else:
        mask_gb = np.ones(12*nside0*nside0)

    ## galactic mask
    if galmask:
        fn_galmask = f'{map_path}/mask/mask_gal_ns{nside0}_{coord}.fits'
        mask_gal = hp.read_map(fn_galmask, verbose=0, dtype=None)
    else:
        mask_gal = np.ones(12*nside0*nside0) 

    ## synchrotron mask
    if syncmask:
        fn_syncmask = f'{map_path}/mask/mask_sync_ns{nside0}_{coord}.fits'
        mask_sync = hp.read_map(fn_syncmask, verbose=0, dtype=None)
    else:
        mask_sync = np.ones(12*nside0*nside0)

    mask = mask_gb * mask_gal * mask_sync

    if bwmask:
        mask = binmask(mask)

    ## trivial mask
    if nomask:
        mask[:] = 1

    print (f'coverage of mask_gb = ',np.average(mask_gb))
    print (f'coverage of mask_gal = ',np.average(mask_gal))
    print (f'coverage of mask_sync = ',np.average(mask_sync))
    print (f'coverage of mymask at nside:{nside0} = ',np.average(mask))

    mask = binmask(hp.ud_grade(mask, nside_out=nside))
    print (f'coverage of mymask at nside:{nside} = ',np.average(mask))

    """
    hp.mollview(mask_gb, title=f'gb mask coord:{coord}')
    hp.mollview(mask_gal, title=f'gal mask coord:{coord}')
    hp.mollview(mask_sync, title=f'sync mask coord:{coord}')
    hp.mollview(mask, title=f'my mask nside:{nside0} coord:{coord}')
    hp.mollview(mask, title=f'my mask nside:{nside} coord:{coord}')
    """


    if saveflg:
        hp.write_map(maskfname, mask)

    return np.array(mask)


def outfilename(prefix, nside=8, ntest=1000, lamb=1.0e-12, rseed_in=42, wp_in=None, mask_in=None, lmin=None, verbose=0):
    date = datetime.datetime.now().isoformat()[:10]
    if not os.path.isdir(f'../result_npz/{date}'):
        if not os.path.isdir(f"../result_npz"):
            os.mkdir(f'../result_npz/')
        os.mkdir(f'../result_npz/{date}')

    ofname = f'../result_npz/{date}/'
    ofname += prefix
    ofname += f'_ns{nside:04d}'
    ofname += f'_ntest{ntest}'

    if verbose > 2:
        ofname += f'_lamb{lamb}'

    if wp_in is None:
        pass
    else:
        #wpstring = []
        #for d in wp0:
        #    wpstring.append(f'{d:5.3f}')
        #wpstring = ','.join(wpstring)
        #ofname += f'_wp{str(wp0).replace(" ","").replace("[","").replace("]","")}'
        ofname += f'_wp{wp_in:4.2f}'

    if lmin:
        ofname += f'_lmin{lmin}'

    if rseed_in != None:
        ofname += f'_rseed{rseed_in}'

    if mask_in is None:
        pass
    elif not nomask:
        ofname += f'_mask'
        if bwmask:
            ofname += f'bw'
        if gbmask:
            ofname += f'_gb{mmin:4.2f},{mmax:4.2f}'
        if galmask:
            ofname += f'_gal'

    if verbose > 2:
        ofname += '_negr_allowed'

    ofname += '.npz'

    return ofname


def foregroundmaps_pysm(nside, freqs=[], GBres=True, Ndetscale=1, Null=False, res145scale=1, res220scale=1):
    nfreq = len(freqs)
    if nfreq == 0:
        return None

    sfreqs = str(freqs).replace(" ", "")
    sfreqs = sfreqs[1:-1]
    
    tmpfname = f'./temp/fg_nside{nside}_freq{sfreqs}'

    if Null:
        tmpfname += '_Null'
    elif GBres:
        tmpfname += '_GBres'
    
    if res145scale != 1:
        tmpfname += f'res145scale{res145scale}'

    if res220scale != 1:
        tmpfname += f'res145scale{res220scale}'


    tmpfname += '.npz'
    print (tmpfname)

    if os.path.isfile(tmpfname):
        print ('loading the foreground from file')
        dat = np.load(tmpfname)
        fgs = dat['fgs']
        return fgs

    fgs = gen_fg(freqs, nside)

    ## residuals
    #fn_res145cmb = map_path+'/residuals/map145cmb_madam_combined_residue.fits'
    #fn_res220cmb = map_path+'/residuals/map220cmb_madam_combined_residue.fits'
    #fn_res145fg  = map_path+'/residuals/map145fg_madam_combined_residue.fits'
    #fn_res220fg  = map_path+'/residuals/map220fg_madam_combined_residue.fits'
    #fn_res145noi = map_path+'/residuals/residual_noise_145.fits'
    #fn_res220noi = map_path+'/residuals/residual_noise_220.fits'

    fn_res145cmb = map_path+'/residuals/2021-09-01_cmb_145__combined_map_diff.fits'
    fn_res220cmb = map_path+'/residuals/2021-09-01_cmb_220__combined_map_diff.fits'
    fn_res145fg  = map_path+'/residuals/2021-09-01_fg_145__combined_map_diff.fits'
    fn_res220fg  = map_path+'/residuals/2021-09-01_fg_220__combined_map_diff.fits'
    fn_res145noi = map_path+'/residuals/2021-09-01_noise_145__combined_map_diff.fits'
    fn_res220noi = map_path+'/residuals/2021-09-01_noise_220__combined_map_diff.fits'

    map_res145cmb = hp.ud_grade(hp.read_map(fn_res145cmb, field=None, verbose=0), nside_out=nside)
    map_res220cmb = hp.ud_grade(hp.read_map(fn_res220cmb, field=None, verbose=0), nside_out=nside)
    map_res145fg  = hp.ud_grade(hp.read_map(fn_res145fg,  field=None, verbose=0), nside_out=nside)
    map_res220fg  = hp.ud_grade(hp.read_map(fn_res220fg,  field=None, verbose=0), nside_out=nside)
    map_res145noi = hp.ud_grade(hp.read_map(fn_res145noi, field=None, verbose=0), nside_out=nside)
    map_res220noi = hp.ud_grade(hp.read_map(fn_res220noi, field=None, verbose=0), nside_out=nside)

    map_res145cmb[map_res145cmb==hp.UNSEEN] = 0
    map_res220cmb[map_res220cmb==hp.UNSEEN] = 0
    map_res145fg [map_res145fg ==hp.UNSEEN] = 0
    map_res220fg [map_res220fg ==hp.UNSEEN] = 0
    map_res145noi[map_res145noi==hp.UNSEEN] = 0
    map_res220noi[map_res220noi==hp.UNSEEN] = 0

    ## adding residues
    if GBres:
        try:
            idx145 = freqs.index(145)
            fgs[idx145] += map_res145cmb
            fgs[idx145] += map_res145fg
            if Ndetscale:
                fgs[idx145] += map_res145noi / (3*365*0.7)**0.5 / Ndetscale**0.5 * res145scale
        except:
            pass

        try:
            idx220 = freqs.index(220)
            fgs[idx220] += map_res220cmb
            fgs[idx220] += map_res220fg
            if Ndetscale:
                fgs[idx220] += map_res220noi / (3*365*0.7)**0.5 / Ndetscale**0.5 * res220scale
        except:
            pass
    else:
        pass

    if Null:
        fgs = np.array(fgs) * 0.0

    print ('****', tmpfname)
    print ('****', fgs)
    np.savez(tmpfname, fgs=fgs) 

    return fgs


def foregroundmaps2_linearcomb(nside):

    ## sync and dust components individually
    map_sync145 = hp.ud_grade(hp.read_map(map_path+'foregrounds/sync/sync_145_TCMB_fwhm1_nside1024_equ.fits', field=(0,1,2)), nside_out=nside)
    map_sync220 = hp.ud_grade(hp.read_map(map_path+'foregrounds/sync/sync_220_TCMB_fwhm1_nside1024_equ.fits', field=(0,1,2)), nside_out=nside)
    map_dust145 = hp.ud_grade(hp.read_map(map_path+'foregrounds/dust/dust_145_TCMB_fwhm1_nside1024_equ.fits', field=(0,1,2)), nside_out=nside)
    map_dust220 = hp.ud_grade(hp.read_map(map_path+'foregrounds/dust/dust_220_TCMB_fwhm1_nside1024_equ.fits', field=(0,1,2)), nside_out=nside)
    map_fg145   = map_sync145 + 0.1 * map_dust220
    map_fg220   = 0.1 * map_sync145 + map_dust220

    return [map_fg145, map_fg220]


def foregroundmaps2_linearcomb_resmadam(nside):

    ## sync and dust components individually
    map_sync145 = hp.ud_grade(hp.read_map(map_path+'foregrounds/sync/sync_145_TCMB_fwhm1_nside1024_equ.fits', field=(0,1,2)), nside_out=nside)
    map_sync220 = hp.ud_grade(hp.read_map(map_path+'foregrounds/sync/sync_220_TCMB_fwhm1_nside1024_equ.fits', field=(0,1,2)), nside_out=nside)
    map_dust145 = hp.ud_grade(hp.read_map(map_path+'foregrounds/dust/dust_145_TCMB_fwhm1_nside1024_equ.fits', field=(0,1,2)), nside_out=nside)
    map_dust220 = hp.ud_grade(hp.read_map(map_path+'foregrounds/dust/dust_220_TCMB_fwhm1_nside1024_equ.fits', field=(0,1,2)), nside_out=nside)
    map_res145  = hp.ud_grade(hp.read_map(map_path+'madam/combined/map145fg_madam_combined_residue.fits', field=(0,1,2)), nside_out=nside)
    map_res220  = hp.ud_grade(hp.read_map(map_path+'madam/combined/map220fg_madam_combined_residue.fits', field=(0,1,2)), nside_out=nside)
    map_fg145   = map_sync145 + 0.1 * map_dust220 + map_res145
    map_fg220   = 0.1 * map_sync145 + map_dust220 + map_res220

    return [map_fg145, map_fg220]


def foregroundmaps2_cmbfgres(nside):
    map_res145  = hp.ud_grade(hp.read_map(map_path+'madam/combined/map145cmb_madam_combined_residue.fits', field=(0,1,2)), nside_out=nside)
    map_res220  = hp.ud_grade(hp.read_map(map_path+'madam/combined/map220cmb_madam_combined_residue.fits', field=(0,1,2)), nside_out=nside)
    map_res145fg  = hp.ud_grade(hp.read_map(map_path+'madam/combined/map145fg_madam_combined_residue.fits', field=(0,1,2)), nside_out=nside)
    map_res220fg  = hp.ud_grade(hp.read_map(map_path+'madam/combined/map220fg_madam_combined_residue.fits', field=(0,1,2)), nside_out=nside)
    map_fg145   = map_res145 + map_res145fg
    map_fg220   = map_res220 + map_res220fg

    return [map_fg145, map_fg220]


## madam foregrounds
def foregroundmaps2_madam(nside):
    map_fg145   = hp.read_map(map_path+'madam/combined/map145fg_madam_combined_map.fits', field=None, verbose=0, dtype=None)
    map_fg220   = hp.read_map(map_path+'madam/combined/map220fg_madam_combined_map.fits', field=None, verbose=0, dtype=None)
    #map_fg145  = hp.read_map(map_path+'madam/combined/combined/madam_20200727_1fnoise_fg_145__combined_map.fits', field=None, verbose=0, dtype=None)
    #map_fg220  = hp.read_map(map_path+'madam/combined/combined/madam_20200727_1fnoise_fg_220__combined_map.fits', field=None, verbose=0, dtype=None)
    map_fg145[np.where(map_fg145==hp.UNSEEN)] = 0
    map_fg220[np.where(map_fg220==hp.UNSEEN)] = 0
    map_fg145   = hp.ud_grade(map_fg145, nside_out=nside) 
    map_fg220   = hp.ud_grade(map_fg220, nside_out=nside) 
    map_res145  = hp.ud_grade(hp.read_map(map_path+'madam/combined/map145fg_madam_combined_residue.fits', field=(0,1,2)), nside_out=nside)
    map_res220  = hp.ud_grade(hp.read_map(map_path+'madam/combined/map220fg_madam_combined_residue.fits', field=(0,1,2)), nside_out=nside)
    map_fg145  += map_res145
    map_fg220  += map_res220

    return [map_fg145, map_fg220]


def foregroundmaps3_madam(nside):
    map_fg145 = hp.read_map(map_path+'madam/combined/map145fg_madam_combined_map.fits', field=None, verbose=0, dtype=None)
    map_fg220 = hp.read_map(map_path+'madam/combined/map220fg_madam_combined_map.fits', field=None, verbose=0, dtype=None)
    #map_fg145 = hp.read_map(map_path+'madam/combined/combined/madam_20200727_1fnoise_fg_145__combined_map.fits', field=None, verbose=0, dtype=None)
    #map_fg220 = hp.read_map(map_path+'madam/combined/combined/madam_20200727_1fnoise_fg_220__combined_map.fits', field=None, verbose=0, dtype=None)
    map_fg145[np.where(map_fg145==hp.UNSEEN)] = 0
    map_fg220[np.where(map_fg220==hp.UNSEEN)] = 0
    map_fg145 = hp.ud_grade(map_fg145, nside_out=nside) 
    map_fg220 = hp.ud_grade(map_fg220, nside_out=nside) 

    map_sync30  = hp.ud_grade(hp.read_map(map_path+'foregrounds/sync/sync_30_TCMB_fwhm1_nside1024_equ.fits', field=(0,1,2)), nside_out=nside)
    map_dust30  = hp.ud_grade(hp.read_map(map_path+'foregrounds/dust/dust_30_TCMB_fwhm1_nside1024_equ.fits', field=(0,1,2)), nside_out=nside)
    map_fg30    = map_sync30 + map_dust30

    return [map_fg145, map_fg220, map_fg30]


def foregroundmaps3_madam_fg1fres(nside):
    map_fg145 = hp.read_map(map_path+'madam/combined/map145fg_madam_combined_map.fits', field=None, verbose=0, dtype=None)
    map_fg220 = hp.read_map(map_path+'madam/combined/map220fg_madam_combined_map.fits', field=None, verbose=0, dtype=None)

    map_wnoiseGB145 = hp.read_map(map_path+'madam/combined/wnoiseGB145_madam_combined_map.fits', field=None, verbose=0, dtype=None)
    map_wnoiseGB220 = hp.read_map(map_path+'madam/combined/wnoiseGB220_madam_combined_map.fits', field=None, verbose=0, dtype=None)
    map_noise1fGB145 = hp.read_map(map_path+'madam/combined/noise1fGB145_madam_combined_map.fits', field=None, verbose=0, dtype=None)
    map_noise1fGB220 = hp.read_map(map_path+'madam/combined/noise1fGB220_madam_combined_map.fits', field=None, verbose=0, dtype=None)

    res145 = map_noise1fGB145 - map_wnoiseGB145
    res220 = map_noise1fGB220 - map_wnoiseGB220

    map_fg145 += res145
    map_fg220 += res220

    map_fg145[np.where(map_fg145==hp.UNSEEN)] = 0
    map_fg220[np.where(map_fg220==hp.UNSEEN)] = 0

    map_fg145 = hp.ud_grade(map_fg145, nside_out=nside) 
    map_fg220 = hp.ud_grade(map_fg220, nside_out=nside) 

    map_sync30  = hp.ud_grade(hp.read_map(map_path+'foregrounds/sync/sync_30_TCMB_fwhm1_nside1024_equ.fits', field=(0,1,2)), nside_out=nside)
    map_dust30  = hp.ud_grade(hp.read_map(map_path+'foregrounds/dust/dust_30_TCMB_fwhm1_nside1024_equ.fits', field=(0,1,2)), nside_out=nside)
    map_fg30    = map_sync30 + map_dust30

    return [map_fg145, map_fg220, map_fg30]


def foregroundmaps2_madam_1f(nside):
    map_fg145 = hp.read_map(map_path+'madam/combined/combined/madam_20200727_1fnoise_fg_145__combined_map.fits', field=None, verbose=0, dtype=None)
    map_fg220 = hp.read_map(map_path+'madam/combined/combined/madam_20200727_1fnoise_fg_220__combined_map.fits', field=None, verbose=0, dtype=None)
    bmap_noise145 = hp.read_map(map_path+'madam/combined/combined/madam_20200727_1fnoise_noise_145__combined_bmap.fits', field=None, verbose=0, dtype=None)
    bmap_noise220 = hp.read_map(map_path+'madam/combined/combined/madam_20200727_1fnoise_noise_220__combined_bmap.fits', field=None, verbose=0, dtype=None)
    map_noise145 = hp.read_map(map_path+'madam/combined/combined/madam_20200727_1fnoise_noise_145__combined_map.fits', field=None, verbose=0, dtype=None)
    map_noise220 = hp.read_map(map_path+'madam/combined/combined/madam_20200727_1fnoise_noise_220__combined_map.fits', field=None, verbose=0, dtype=None)
    map_fg145[np.where(map_fg145==hp.UNSEEN)] = 0
    map_fg220[np.where(map_fg220==hp.UNSEEN)] = 0
    bmap_noise145[np.where(bmap_noise145==hp.UNSEEN)] = 0
    bmap_noise220[np.where(bmap_noise220==hp.UNSEEN)] = 0
    map_noise145[np.where(map_noise145==hp.UNSEEN)] = 0
    map_noise220[np.where(map_noise220==hp.UNSEEN)] = 0
    dmap_noise145 = bmap_noise145 - map_noise145
    dmap_noise220 = bmap_noise220 - map_noise220
    map_fg145 += dmap_noise145
    map_fg220 += dmap_noise220
    map_fg145 = hp.ud_grade(map_fg145, nside_out=nside) 
    map_fg220 = hp.ud_grade(map_fg220, nside_out=nside) 

    return [map_fg145, map_fg220]


def foregroundmaps2_dustonly(nside):
    map_fg145   = hp.ud_grade(hp.read_map(map_path+'foregrounds/dust/dust_145_TCMB_fwhm1_nside1024_equ.fits', field=(0,1,2)), nside_out=nside)
    map_fg220   = hp.ud_grade(hp.read_map(map_path+'foregrounds/dust/dust_220_TCMB_fwhm1_nside1024_equ.fits', field=(0,1,2)), nside_out=nside)
    map_res145  = hp.ud_grade(hp.read_map(map_path+'madam/combined/map145fg_madam_combined_residue.fits', field=(0,1,2)), nside_out=nside)
    map_res220  = hp.ud_grade(hp.read_map(map_path+'madam/combined/map220fg_madam_combined_residue.fits', field=(0,1,2)), nside_out=nside)
    map_fg145  += map_res145
    map_fg220  += map_res220

    return [map_fg145, map_fg220]


def foregroundmaps2_dustonly_nores(nside):
    map_fg145   = hp.ud_grade(hp.read_map(map_path+'foregrounds/dust/dust_145_TCMB_fwhm1_nside1024_equ.fits', field=(0,1,2)), nside_out=nside)
    map_fg220   = hp.ud_grade(hp.read_map(map_path+'foregrounds/dust/dust_220_TCMB_fwhm1_nside1024_equ.fits', field=(0,1,2)), nside_out=nside)
    map_res145  = hp.ud_grade(hp.read_map(map_path+'madam/combined/map145fg_madam_combined_residue.fits', field=(0,1,2)), nside_out=nside)
    map_res220  = hp.ud_grade(hp.read_map(map_path+'madam/combined/map220fg_madam_combined_residue.fits', field=(0,1,2)), nside_out=nside)
    #map_fg145  += map_res145
    #map_fg220  += map_res220

    return [map_fg145, map_fg220]


def foregroundmaps2_dustsync(nside):
    map_sync145 = hp.ud_grade(hp.read_map(map_path+'foregrounds/sync/sync_145_TCMB_fwhm1_nside1024_equ.fits', field=(0,1,2)), nside_out=nside)
    map_sync220 = hp.ud_grade(hp.read_map(map_path+'foregrounds/sync/sync_220_TCMB_fwhm1_nside1024_equ.fits', field=(0,1,2)), nside_out=nside)
    map_dust145 = hp.ud_grade(hp.read_map(map_path+'foregrounds/dust/dust_145_TCMB_fwhm1_nside1024_equ.fits', field=(0,1,2)), nside_out=nside)
    map_dust220 = hp.ud_grade(hp.read_map(map_path+'foregrounds/dust/dust_220_TCMB_fwhm1_nside1024_equ.fits', field=(0,1,2)), nside_out=nside)
    map_res145  = hp.ud_grade(hp.read_map(map_path+'madam/combined/map145fg_madam_combined_residue.fits', field=(0,1,2)), nside_out=nside)
    map_res220  = hp.ud_grade(hp.read_map(map_path+'madam/combined/map220fg_madam_combined_residue.fits', field=(0,1,2)), nside_out=nside)
    map_fg145   = map_dust145 + map_sync145 + map_res145
    map_fg220   = map_dust220 + map_sync220 + map_res220

    return [map_fg145, map_fg220]


def foregroundmaps2_dustsync_nores(nside):
    map_sync145 = hp.ud_grade(hp.read_map(map_path+'foregrounds/sync/sync_145_TCMB_fwhm1_nside1024_equ.fits', field=(0,1,2)), nside_out=nside)
    map_sync220 = hp.ud_grade(hp.read_map(map_path+'foregrounds/sync/sync_220_TCMB_fwhm1_nside1024_equ.fits', field=(0,1,2)), nside_out=nside)
    map_dust145 = hp.ud_grade(hp.read_map(map_path+'foregrounds/dust/dust_145_TCMB_fwhm1_nside1024_equ.fits', field=(0,1,2)), nside_out=nside)
    map_dust220 = hp.ud_grade(hp.read_map(map_path+'foregrounds/dust/dust_220_TCMB_fwhm1_nside1024_equ.fits', field=(0,1,2)), nside_out=nside)
    map_res145  = hp.ud_grade(hp.read_map(map_path+'madam/combined/map145fg_madam_combined_residue.fits', field=(0,1,2)), nside_out=nside)
    map_res220  = hp.ud_grade(hp.read_map(map_path+'madam/combined/map220fg_madam_combined_residue.fits', field=(0,1,2)), nside_out=nside)
    map_fg145   = map_dust145 + map_sync145 #+ map_res145
    map_fg220   = map_dust220 + map_sync220 #+ map_res220

    return [map_fg145, map_fg220]


def noisemaps_madam(nside):
    map_noi145 = hp.read_map(map_path+'madam/combined/noise145_madam_combined_map.fits', field=None, verbose=0, dtype=None)
    map_noi220 = hp.read_map(map_path+'madam/combined/noise220_madam_combined_map.fits', field=None, verbose=0, dtype=None)
    map_noi145[np.where(map_noi145==hp.UNSEEN)] = 0
    map_noi220[np.where(map_noi220==hp.UNSEEN)] = 0
    map_noi145 = hp.ud_grade(map_noi145, nside_out=nside) 
    map_noi220 = hp.ud_grade(map_noi220, nside_out=nside) 

    return [map_noi145, map_noi220]


def noisemaps_madam_1f(nside, scale=1):
    map_noi145 = hp.read_map(map_path+'madam/combined/combined/madam_20200727_1fnoise_noise_145__combined_map.fits', field=None, verbose=0, dtype=None)
    map_noi220 = hp.read_map(map_path+'madam/combined/combined/madam_20200727_1fnoise_noise_220__combined_map.fits', field=None, verbose=0, dtype=None)
    map_noi145[np.where(map_noi145==hp.UNSEEN)] = 0
    map_noi220[np.where(map_noi220==hp.UNSEEN)] = 0

    map_noi145 = map_noi145 * scale 
    map_noi220 = map_noi220 * scale 

    map_noi145 = hp.ud_grade(map_noi145, nside_out=nside) 
    map_noi220 = hp.ud_grade(map_noi220, nside_out=nside) 

    return [map_noi145, map_noi220]


def noisemaps_madam_white(nside, scale=1):
    map_noi145 = hp.read_map(map_path+'madam/combined/combined/madam_nshort5000_test_wnoise_GBnoise_145__combined_map.fits', field=None, verbose=0, dtype=None)
    map_noi220 = hp.read_map(map_path+'madam/combined/combined/madam_nshort5000_test_wnoise_GBnoise_220__combined_map.fits', field=None, verbose=0, dtype=None)
    map_noi145[np.where(map_noi145==hp.UNSEEN)] = 0
    map_noi220[np.where(map_noi220==hp.UNSEEN)] = 0

    map_noi145 = map_noi145 * scale 
    map_noi220 = map_noi220 * scale

    map_noi145 = hp.ud_grade(map_noi145, nside_out=nside) 
    map_noi220 = hp.ud_grade(map_noi220, nside_out=nside) 

    return [map_noi145, map_noi220]


def noisemaps_madam_iso(nside, scale=1):
    nl145 = get_spectrum_noise(4000, wp0[0], CMB_unit='muK')
    nl220 = get_spectrum_noise(4000, wp0[1], CMB_unit='muK')
    map_noi145 = hp.synfast(nl145, nside=1024, new=True)
    map_noi220 = hp.synfast(nl220, nside=1024, new=True) 

    map_noi145 = map_noi145 * scale
    map_noi220 = map_noi220 * scale 

    map_noi145 = hp.ud_grade(map_noi145, nside_out=nside) 
    map_noi220 = hp.ud_grade(map_noi220, nside_out=nside) 

    return [map_noi145, map_noi220]


if __name__=="__main__":
    mask = mymask(None, 'equ')
    mask = mymask(None, 'gal')
    plt.show()




