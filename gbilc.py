import numpy as np
import healpy as hp
import pylab as plt

from gbpipe.spectrum import get_spectrum_camb, get_spectrum_xpol, get_spectrum_noise
from scipy.optimize import minimize
from gbpipe.gbmap import makegbmask
from scripts.fg_setup import mymask


k = 1.38064852e-23 # m^2 kg s^-2 K^-1
c = 2.99792458e+08 # m/s


def rms(data):
    return np.sqrt(np.mean(np.square(data)))


def RJ2CMB(T_RJ, nu):
    x = nu/56.78
    T_CMB = (np.exp(x) - 1)**2/(x**2 * np.exp(x)) * T_RJ

    return T_CMB


def CMB2RJ(T_CMB, nu):
    x = nu/56.78
    T_RJ = (x**2 * np.exp(x))/(np.exp(x) - 1)**2 * T_CMB

    return T_RJ


def changefreq(map_in, f_in, f_out, beta):
    return (f_out/f_in)**beta * map_in


def masking(maps, mask):
    if mask is None:
        mm = maps
    else:
        mm = []
        mm.append(maps[0][mask==1])
        mm.append(maps[1][mask==1])
        mm.append(maps[2][mask==1])

    mm = np.array(mm)
    return mm


def ilcmaps(*maps, mask=None):
    if mask is None:
        mm = np.array(maps).copy()
    else:
        mm = []
        for m in maps:
            mm = masking(m, mask)

    def fnc(c):
        cs = list(c)
        cs.append(1-np.sum(c))
        cs = np.array(cs)
        tmp = [ct * mt for ct, mt in zip(cs, mm)]
        cleanedmap = np.sum(tmp, axis=0)
        #lk = np.var(cleanedmap[1]) + np.var(cleanedmap[2])
        lk = rms(np.sqrt(cleanedmap[1]**2 + cleanedmap[2]**2))
        return lk

    c0 = [1.0/(len(maps)-1)] * (len(maps)-1)
    res = minimize(fnc, c0)
    print (res)
    cs = list(res.x)
    cs.append(1-np.sum(cs))
    tmp = [ct * mt for ct, mt in zip(cs, maps)]
    cleanedmap = sum(tmp)

    return res, cleanedmap


def show_rms(maps, mapname, mask=None):
    if len(np.shape(maps)) == 2:
        maps = [maps]
        mapname = [mapname]

    for m in maps:
        print ('*'*30, end=' ')

    print ('')
    for m, mn in zip(maps, mapname):
        print (f'* {mn:26} *', end=' ')

    print ('')
    for m in maps:
        mm = masking(m, mask)
        print (f' rms Q  = {rms(mm[1]):12.5E}'+' '*8, end=' ')

    print ('')
    for m in maps:
        mm = masking(m, mask)
        print (f' rms U  = {rms(mm[2]):12.5E}'+' '*8, end=' ')

    print ('')
    for m in maps:
        mm = masking(m, mask)
        print (f' rms Ip = {rms(np.abs(mm[1]+1j*mm[2])):12.5E}'+' '*8, end=' ')

    print ('')
    for m in maps:
        print ('*'*30, end=' ')

    print ('')

    return 


def test_gb2maps_extrapolation():
    nside = 16

    ## cmb
    cl0 = get_spectrum_camb(lmax=100, isDl=False, CMB_unit='muK')
    cmbCMB = hp.synfast(cl0, nside=nside, verbose=0, new=1)
    cmb145RJ = CMB2RJ(cmbCMB, 145)
    cmb220RJ = CMB2RJ(cmbCMB, 220)

    ## foregrounds by pysm
    map_path = '/home/kmlee/cmb/forecast/maps/'
    sync145RJ = hp.ud_grade(hp.read_map(map_path+'fg_sync145.fits', field=(0,1,2), verbose=0), nside_out=nside)
    sync220RJ = hp.ud_grade(hp.read_map(map_path+'fg_sync220.fits', field=(0,1,2), verbose=0), nside_out=nside)
    dust145RJ = hp.ud_grade(hp.read_map(map_path+'fg_dust145.fits', field=(0,1,2), verbose=0), nside_out=nside)
    dust220RJ = hp.ud_grade(hp.read_map(map_path+'fg_dust220.fits', field=(0,1,2), verbose=0), nside_out=nside)

    sync145CMB = RJ2CMB(sync145RJ, 145)
    dust220CMB = RJ2CMB(dust220RJ, 220)
    sync220CMB = RJ2CMB(sync220RJ, 220)
    dust145CMB = RJ2CMB(dust145RJ, 145)
    sync220CMB = RJ2CMB(changefreq(sync145RJ, 145, 220, -3.0), 220)
    dust145CMB = RJ2CMB(changefreq(dust220RJ, 220, 145, 1.6), 145)
    sync220RJ = CMB2RJ(sync220CMB, 220)
    dust145RJ = CMB2RJ(dust145CMB, 145)

    map145CMB = np.zeros(np.shape(cmbCMB))
    map220CMB = np.zeros(np.shape(cmbCMB))

    map145CMB += cmbCMB;     map220CMB += cmbCMB 
    map145CMB += dust145CMB; map220CMB += dust220CMB 
    map145CMB += sync145CMB; map220CMB += sync220CMB 

    ## mask
    maskgb = makegbmask(nside, 41.7, 81.7)
    maskga = hp.read_map(f'{map_path}/COM_Mask_Likelihood-polarization-143_16_equ.fits', verbose=False) 
    maskga = hp.ud_grade(maskga, nside_out=nside)

    masksync = np.sqrt(sync145CMB[1]**2+sync145CMB[2]**2)
    hp.mollview(masksync, max=0.1)
    masksync[masksync>0.1] = 1
    masksync[masksync<=0.1] = 0
    masksync = np.logical_not(masksync)

    mask = maskgb #* maskga
    rot = hp.rotator.Rotator(coord=['C','G'])
    mask = rot.rotate_map_pixel(mask)

    mask = mask*masksync
    mask[mask>0] = 1
    mask[mask<1] = 0

    #mask = None

    ## ILC
    maps = [masking(map145CMB, mask), masking(map220CMB, mask)]
    res = ilcmaps(*maps)

    ## cleaned map
    cs = list(res.x)
    cs.append(1-np.sum(res.x))
    cs = np.array(cs)
    tmp = [ct * mt for ct, mt in zip(cs, (map145CMB, map220CMB))]
    cleanedmap = sum(tmp)
    print(np.shape(cleanedmap))

    cf = changefreq(1.0, 220, 145, 1.6) * CMB2RJ(1, 220) * RJ2CMB(1, 145)
    dustless = (map145CMB - cf * map220CMB) / (1 - cf)

    ## print rms
    show_rms([cmbCMB, cmb145RJ, cmb220RJ], ['cmb (K_CMB)', 'cmb145 (K_RJ)', 'cmb220 (K_RJ)'], mask)
    show_rms([sync145CMB, sync145RJ, sync220CMB, sync220RJ], ['sync145 (K_CMB)', 'sync145 (K_RJ)', 'sync220 (K_CMB)', 'sync220 (K_RJ)'], mask)
    show_rms([dust145CMB, dust145RJ, dust220CMB, dust220RJ], ['dust145 (K_CMB)', 'dust145 (K_RJ)', 'dust220 (K_CMB)', 'dust220 (K_RJ)'], mask)
    show_rms([map145CMB, map220CMB], ['map145 (K_CMB)', 'map220 (K_CMB)'], mask)

    show_rms([cleanedmap, dustless], ['Cleaned ILC (K_CMB)', 'Cleaned dust (K_CMB)'], mask)

    print (f'coeffs (ILC) = {cs}')
    print (f'coeffs (dustless) = {1/(1-cf)}, {-cf/(1-cf)}')
    print (cf)
    print (np.average(mask))

    hp.mollview(cmbCMB[1]*mask, title='CMB Q')
    hp.mollview(cmbCMB[2]*mask, title='CMB U')
    hp.mollview(cleanedmap[1]*mask, title='cleaned Q')
    hp.mollview(cleanedmap[2]*mask, title='cleaned U')
    hp.mollview(dustless[1]*mask, title='dustless Q')
    hp.mollview(dustless[2]*mask, title='dustless U')

    return 


def test_gb2maps_allpysm():
    nside = 8

    ## cmb
    cl0 = get_spectrum_camb(lmax=100, isDl=False, CMB_unit='muK')
    cmbCMB = hp.synfast(cl0, nside=nside, verbose=0, new=1)
    cmb145RJ = CMB2RJ(cmbCMB, 145)
    cmb220RJ = CMB2RJ(cmbCMB, 220)

    ## foregrounds by pysm
    map_path = '/home/kmlee/cmb/forecast/maps/'
    sync145RJ = hp.ud_grade(hp.read_map(map_path+'fg_sync145.fits', field=(0,1,2), verbose=0), nside_out=nside)
    sync220RJ = hp.ud_grade(hp.read_map(map_path+'fg_sync220.fits', field=(0,1,2), verbose=0), nside_out=nside)
    dust145RJ = hp.ud_grade(hp.read_map(map_path+'fg_dust145.fits', field=(0,1,2), verbose=0), nside_out=nside)
    dust220RJ = hp.ud_grade(hp.read_map(map_path+'fg_dust220.fits', field=(0,1,2), verbose=0), nside_out=nside)

    sync145CMB = RJ2CMB(sync145RJ, 145)
    dust220CMB = RJ2CMB(dust220RJ, 220)
    sync220CMB = RJ2CMB(sync220RJ, 220)
    dust145CMB = RJ2CMB(dust145RJ, 145)
    #sync220CMB = RJ2CMB(changefreq(sync145RJ, 145, 220, -3.0), 220)
    #dust145CMB = RJ2CMB(changefreq(dust220RJ, 220, 145, 1.6), 145)
    #sync220RJ = CMB2RJ(sync220CMB, 220)
    #dust145RJ = CMB2RJ(dust145CMB, 145)

    map145CMB = np.zeros(np.shape(cmbCMB))
    map220CMB = np.zeros(np.shape(cmbCMB))

    map145CMB += cmbCMB;     map220CMB += cmbCMB 
    map145CMB += dust145CMB; map220CMB += dust220CMB 
    map145CMB += sync145CMB; map220CMB += sync220CMB 

    ## mask
    maskgb = makegbmask(nside, 41.7, 81.7)
    maskga = hp.read_map(f'{map_path}/COM_Mask_Likelihood-polarization-143_16_equ.fits', verbose=False) 
    maskga = hp.ud_grade(maskga, nside_out=nside)
    mask = maskgb * maskga
    rot = hp.rotator.Rotator(coord=['C','G'])
    mask = rot.rotate_map_pixel(mask)
    mask[mask>0] = 1
    mask[mask<1] = 0

    hp.mollview(mask)
    mask[:]=1

    ## ILC
    maps = [masking(map145CMB, mask), masking(map220CMB, mask)]
    res = ilcmaps(*maps)

    ## cleaned map
    cs = list(res.x)
    cs.append(1-np.sum(res.x))
    cs = np.array(cs)
    tmp = [ct * mt for ct, mt in zip(cs, (map145CMB, map220CMB))]
    cleanedmap = sum(tmp)
    print(np.shape(cleanedmap))

    mpp = RJ2CMB(changefreq(CMB2RJ(map220CMB, 220), 220, 145, 1.6), 145) 
    cf = changefreq(1.0, 220, 145, 1.52) * CMB2RJ(1, 220) * RJ2CMB(1, 145)
    dustless = (map145CMB - cf * map220CMB) / (1 - cf)

    ## print rms
    show_rms([cmbCMB, cmb145RJ, cmb220RJ], ['cmb (K_CMB)', 'cmb145 (K_RJ)', 'cmb220 (K_RJ)'], mask)
    show_rms([sync145CMB, sync145RJ, sync220CMB, sync220RJ], ['sync145 (K_CMB)', 'sync145 (K_RJ)', 'sync220 (K_CMB)', 'sync220 (K_RJ)'], mask)
    show_rms([dust145CMB, dust145RJ, dust220CMB, dust220RJ], ['dust145 (K_CMB)', 'dust145 (K_RJ)', 'dust220 (K_CMB)', 'dust220 (K_RJ)'], mask)
    show_rms([map145CMB, map220CMB], ['map145 (K_CMB)', 'map220 (K_CMB)'], mask)

    show_rms([cleanedmap, dustless], ['Cleaned ILC (K_CMB)', 'Cleaned dust (K_CMB)'], mask)

    print (f'coeffs (ILC) = {cs}')
    print (f'coeffs (dustless) = {1/(1-cf)}, {-cf/(1-cf)}')
    print (cf)

    plt.figure()
    plt.loglog(cl0[1:3].T, 'b-', label='CAMB')
    plt.loglog(np.abs(get_spectrum_xpol(map145CMB,  lmax=23, mask=mask)[1][1:3]).T, 'b:', label='map145')
    plt.loglog(np.abs(get_spectrum_xpol(map220CMB,  lmax=23, mask=mask)[1][1:3]).T, 'r:', label='map220')
    plt.loglog(np.abs(get_spectrum_xpol(sync145CMB, lmax=23, mask=mask)[1][1:3]).T, 'g:', label='sync145')
    plt.loglog(np.abs(get_spectrum_xpol(dust145CMB, lmax=23, mask=mask)[1][1:3]).T, 'y:', label='dust145')
    plt.loglog(np.abs(get_spectrum_xpol(cleanedmap, lmax=23, mask=mask)[1][1:3]).T, 'k-', label='cleaned')
    plt.loglog(np.abs(get_spectrum_xpol(dustless,   lmax=23, mask=mask)[1][1:3]).T, 'k--', label='dustless')
    plt.legend()

    return 


def test_gb2maps_allpysm_betamap():
    nside = 4

    ## cmb
    cl0 = get_spectrum_camb(lmax=100, isDl=False, CMB_unit='muK')
    cmbCMB = hp.synfast(cl0, nside=nside, verbose=0, new=1)
    cmb145RJ = CMB2RJ(cmbCMB, 145)
    cmb220RJ = CMB2RJ(cmbCMB, 220)

    ## foregrounds by pysm
    map_path = '/home/kmlee/cmb/forecast/maps/'
    sync145RJ = hp.ud_grade(hp.read_map(map_path+'fg_sync145.fits', field=(0,1,2), verbose=0), nside_out=nside)
    sync220RJ = hp.ud_grade(hp.read_map(map_path+'fg_sync220.fits', field=(0,1,2), verbose=0), nside_out=nside)
    dust145RJ = hp.ud_grade(hp.read_map(map_path+'fg_dust145.fits', field=(0,1,2), verbose=0), nside_out=nside)
    dust220RJ = hp.ud_grade(hp.read_map(map_path+'fg_dust220.fits', field=(0,1,2), verbose=0), nside_out=nside)
    dustbeta = hp.ud_grade(hp.read_map(map_path+'dust_beta_d1.fits', verbose=0), nside_out=nside)

    sync145CMB = RJ2CMB(sync145RJ, 145)
    dust220CMB = RJ2CMB(dust220RJ, 220)
    sync220CMB = RJ2CMB(sync220RJ, 220)
    dust145CMB = RJ2CMB(dust145RJ, 145)
    #sync220CMB = RJ2CMB(changefreq(sync145RJ, 145, 220, -3.0), 220)
    #dust145CMB = RJ2CMB(changefreq(dust220RJ, 220, 145, 1.6), 145)
    #sync220RJ = CMB2RJ(sync220CMB, 220)
    #dust145RJ = CMB2RJ(dust145CMB, 145)

    map145CMB = np.zeros(np.shape(cmbCMB))
    map220CMB = np.zeros(np.shape(cmbCMB))

    #map145CMB += cmbCMB;     map220CMB += cmbCMB 
    map145CMB += dust145CMB; map220CMB += dust220CMB 
    #map145CMB += sync145CMB; map220CMB += sync220CMB 

    ## mask
    maskgb = makegbmask(nside, 41.7, 81.7)
    maskga = hp.read_map(f'{map_path}/COM_Mask_Likelihood-polarization-143_16_equ.fits', verbose=False) 
    maskga = hp.ud_grade(maskga, nside_out=nside)
    mask = maskgb * maskga
    rot = hp.rotator.Rotator(coord=['C','G'])
    mask = rot.rotate_map_pixel(mask)
    mask[mask>0] = 1
    mask[mask<1] = 0

    #mask = None

    ## ILC
    maps = [masking(map145CMB, mask), masking(map220CMB, mask)]
    res = ilcmaps(*maps)

    ## cleaned map
    cs = list(res.x)
    cs.append(1-np.sum(res.x))
    cs = np.array(cs)
    tmp = [ct * mt for ct, mt in zip(cs, (map145CMB, map220CMB))]
    cleanedmap = sum(tmp)
    print(np.shape(cleanedmap))

    cf = changefreq(1.0, 220, 145, dustbeta) * CMB2RJ(1, 220) * RJ2CMB(1, 145)
    dustless = (map145CMB - cf * map220CMB) / (1 - cf)

    ## print rms
    show_rms([cmbCMB, cmb145RJ, cmb220RJ], ['cmb (K_CMB)', 'cmb145 (K_RJ)', 'cmb220 (K_RJ)'], mask)
    show_rms([sync145CMB, sync145RJ, sync220CMB, sync220RJ], ['sync145 (K_CMB)', 'sync145 (K_RJ)', 'sync220 (K_CMB)', 'sync220 (K_RJ)'], mask)
    show_rms([dust145CMB, dust145RJ, dust220CMB, dust220RJ], ['dust145 (K_CMB)', 'dust145 (K_RJ)', 'dust220 (K_CMB)', 'dust220 (K_RJ)'], mask)
    show_rms([map145CMB, map220CMB], ['map145 (K_CMB)', 'map220 (K_CMB)'], mask)
    show_rms([cleanedmap, dustless], ['Cleaned ILC (K_CMB)', 'Cleaned dust (K_CMB)'], mask)

    print (f'coeffs (ILC) = {cs}')
    print (f'coeffs (dustless) = {1/(1-cf)}, {-cf/(1-cf)}')
    print (cf)

    return 


def test_gb2maps_madam():
    nside = 128
    
    np.random.seed(0)

    ## cmb
    cl0 = get_spectrum_camb(lmax=100, isDl=False, CMB_unit='muK')
    cmbCMB = hp.synfast(cl0, nside=nside, verbose=0, new=1)
    cmb145RJ = CMB2RJ(cmbCMB, 145)
    cmb220RJ = CMB2RJ(cmbCMB, 220)

    ## foregrounds by pysm
    map_path = '/home/kmlee/cmb/forecast/maps/'
    FG145CMB = hp.ud_grade(hp.read_map(map_path+'madam/map145fg_madam_combined_map.fits', field=(0,1,2), verbose=0), nside_out=nside)
    FG220CMB = hp.ud_grade(hp.read_map(map_path+'madam/map220fg_madam_combined_map.fits', field=(0,1,2), verbose=0), nside_out=nside)

    FG145CMB[np.where(FG145CMB==hp.UNSEEN)] = 0
    FG220CMB[np.where(FG220CMB==hp.UNSEEN)] = 0

    map145CMB = np.zeros(np.shape(cmbCMB))
    map220CMB = np.zeros(np.shape(cmbCMB))

    map145CMB += cmbCMB;     map220CMB += cmbCMB 
    map145CMB += FG145CMB;   map220CMB += FG220CMB 

    ## mask
    maskgb = makegbmask(nside, 33, 90)
    maskga = hp.read_map(f'{map_path}/mask/mask_gal_ns1024_equ.fits', verbose=False) 
    maskga = hp.ud_grade(maskga, nside_out=nside)
    masksy = hp.read_map(f'{map_path}/mask/mask_sync_ns1024_equ.fits', verbose=False)
    mask = maskgb * maskga #* masksy
    #rot = hp.rotator.Rotator(coord=['C','G'])
    #mask = rot.rotate_map_pixel(mask)
    mask[mask>0] = 1
    mask[mask<1] = 0

    hp.mollview(mask)
    #mask[:]=1

    ## ILC
    maps = [masking(map145CMB, mask), masking(map220CMB, mask)]
    res, _ = ilcmaps(*maps)

    ## cleaned map
    cs = list(res.x)
    cs.append(1-np.sum(res.x))
    cs = np.array(cs)
    tmp = [ct * mt for ct, mt in zip(cs, (map145CMB*mask, map220CMB*mask))]
    cleanedmap = sum(tmp)
    print(np.shape(cleanedmap))

    mpp = RJ2CMB(changefreq(CMB2RJ(map220CMB, 220), 220, 145, 1.6), 145) 
    cf = changefreq(1.0, 220, 145, 1.52) * CMB2RJ(1, 220) * RJ2CMB(1, 145)
    dustless = (map145CMB - cf * map220CMB) / (1 - cf)

    ## print rms
    show_rms([cmbCMB, cmb145RJ, cmb220RJ], ['cmb (K_CMB)', 'cmb145 (K_RJ)', 'cmb220 (K_RJ)'], mask)
    #show_rms([sync145CMB, sync145RJ, sync220CMB, sync220RJ], ['sync145 (K_CMB)', 'sync145 (K_RJ)', 'sync220 (K_CMB)', 'sync220 (K_RJ)'], mask)
    #show_rms([dust145CMB, dust145RJ, dust220CMB, dust220RJ], ['dust145 (K_CMB)', 'dust145 (K_RJ)', 'dust220 (K_CMB)', 'dust220 (K_RJ)'], mask)
    show_rms([FG145CMB, FG220CMB], ['FG145 (K_CMB)', 'FG220 (K_CMB)'], mask)
    show_rms([map145CMB, map220CMB], ['map145 (K_CMB)', 'map220 (K_CMB)'], mask)

    show_rms([cleanedmap, dustless], ['Cleaned ILC (K_CMB)', 'Cleaned dust (K_CMB)'], mask)

    print (f'coeffs (ILC) = {cs}')
    print (f'coeffs (dustless) = {1/(1-cf)}, {-cf/(1-cf)}')
    print (cf)
    print (f'coverage = {np.average(mask)}')

    dcc = cleanedmap - cmbCMB * mask

    cleanedmap[np.where(cleanedmap==0)] = hp.UNSEEN
    dcc[np.where(dcc==0)] = hp.UNSEEN

    hp.mollview(cleanedmap[0], min=-2.5, max=2.5)
    hp.mollview(cleanedmap[1], min=-2.5, max=2.5)
    hp.mollview(cleanedmap[2], min=-2.5, max=2.5)

    hp.mollview(dcc[0], min=-0.25, max=0.25)
    hp.mollview(dcc[1], min=-0.25, max=0.25)
    hp.mollview(dcc[2], min=-0.25, max=0.25)
    
    """
    plt.figure()
    plt.loglog(cl0[1:3].T, 'b-', label='CAMB')
    plt.loglog(np.abs(get_spectrum_xpol(map145CMB,  lmax=23, mask=mask)[1][1:3]).T, 'b:', label='map145')
    plt.loglog(np.abs(get_spectrum_xpol(map220CMB,  lmax=23, mask=mask)[1][1:3]).T, 'r:', label='map220')
    plt.loglog(np.abs(get_spectrum_xpol(sync145CMB, lmax=23, mask=mask)[1][1:3]).T, 'g:', label='sync145')
    plt.loglog(np.abs(get_spectrum_xpol(dust145CMB, lmax=23, mask=mask)[1][1:3]).T, 'y:', label='dust145')
    plt.loglog(np.abs(get_spectrum_xpol(cleanedmap, lmax=23, mask=mask)[1][1:3]).T, 'k-', label='cleaned')
    plt.loglog(np.abs(get_spectrum_xpol(dustless,   lmax=23, mask=mask)[1][1:3]).T, 'k--', label='dustless')
    plt.legend()
    """

    return 


def test_gb2maps_pysm():
    nside = 128
    
    np.random.seed(0)

    ## cmb
    cl0 = get_spectrum_camb(lmax=100, isDl=False, CMB_unit='muK')
    cmbCMB = hp.synfast(cl0, nside=nside, verbose=0, new=1)
    cmb145RJ = CMB2RJ(cmbCMB, 145)
    cmb220RJ = CMB2RJ(cmbCMB, 220)

    ## foregrounds by pysm
    map_path = '/home/kmlee/cmb/forecast/maps/'
    FG90CMB  = hp.ud_grade(hp.read_map(map_path+f'foregrounds/fg/fg_90_TCMB_nside{nside:04d}_equ.fits', field=(0,1,2), verbose=0), nside_out=nside)
    FG145CMB = hp.ud_grade(hp.read_map(map_path+f'foregrounds/fg/fg_145_TCMB_nside{nside:04d}_equ.fits', field=(0,1,2), verbose=0), nside_out=nside)
    FG220CMB = hp.ud_grade(hp.read_map(map_path+f'foregrounds/fg/fg_220_TCMB_nside{nside:04d}_equ.fits', field=(0,1,2), verbose=0), nside_out=nside)

    FG90CMB[np.where(FG90CMB==hp.UNSEEN)] = 0
    FG145CMB[np.where(FG145CMB==hp.UNSEEN)] = 0
    FG220CMB[np.where(FG220CMB==hp.UNSEEN)] = 0

    map90CMB = np.zeros(np.shape(cmbCMB))
    map145CMB = np.zeros(np.shape(cmbCMB))
    map220CMB = np.zeros(np.shape(cmbCMB))

    #map90CMB  += cmbCMB 
    #map145CMB += cmbCMB
    #map220CMB += cmbCMB 
    map90CMB  += FG90CMB
    map145CMB += FG145CMB
    map220CMB += FG220CMB 

    ## mask
    maskgb = makegbmask(nside, 33, 90)
    maskga = hp.read_map(f'{map_path}/mask/mask_gal_ns1024_equ.fits', verbose=False) 
    maskga = hp.ud_grade(maskga, nside_out=nside)
    masksy = hp.read_map(f'{map_path}/mask/mask_sync_ns1024_equ.fits', verbose=False)
    mask = maskgb * maskga #* masksy
    #rot = hp.rotator.Rotator(coord=['C','G'])
    #mask = rot.rotate_map_pixel(mask)
    mask[mask>0] = 1
    mask[mask<1] = 0

    hp.mollview(mask)
    #mask[:]=1

    ## ILC
    maps = [masking(map145CMB, mask), masking(map220CMB, mask)]
    res, _ = ilcmaps(*maps)

    ## cleaned map
    cs = list(res.x)
    cs.append(1-np.sum(res.x))
    cs = np.array(cs)
    tmp = [ct * mt for ct, mt in zip(cs, (map145CMB*mask, map220CMB*mask))]
    cleanedmap = sum(tmp)
    print(np.shape(cleanedmap))

    mpp = RJ2CMB(changefreq(CMB2RJ(map220CMB, 220), 220, 145, 1.6), 145) 
    cf = changefreq(1.0, 220, 145, 1.52) * CMB2RJ(1, 220) * RJ2CMB(1, 145)
    dustless = (map145CMB - cf * map220CMB) / (1 - cf)

    ## print rms
    show_rms([cmbCMB, cmb145RJ, cmb220RJ], ['cmb (K_CMB)', 'cmb145 (K_RJ)', 'cmb220 (K_RJ)'], mask)
    #show_rms([sync145CMB, sync145RJ, sync220CMB, sync220RJ], ['sync145 (K_CMB)', 'sync145 (K_RJ)', 'sync220 (K_CMB)', 'sync220 (K_RJ)'], mask)
    #show_rms([dust145CMB, dust145RJ, dust220CMB, dust220RJ], ['dust145 (K_CMB)', 'dust145 (K_RJ)', 'dust220 (K_CMB)', 'dust220 (K_RJ)'], mask)
    show_rms([FG145CMB, FG220CMB], ['FG145 (K_CMB)', 'FG220 (K_CMB)'], mask)
    show_rms([map145CMB, map220CMB], ['map145 (K_CMB)', 'map220 (K_CMB)'], mask)

    show_rms([cleanedmap, dustless], ['Cleaned ILC (K_CMB)', 'Cleaned dust (K_CMB)'], mask)

    print (f'coeffs (ILC) = {cs}')
    print (f'coeffs (dustless) = {1/(1-cf)}, {-cf/(1-cf)}')
    print (cf)
    print (f'coverage = {np.average(mask)}')

    dcc = cleanedmap - cmbCMB * mask

    cleanedmap[np.where(cleanedmap==0)] = hp.UNSEEN
    dcc[np.where(dcc==0)] = hp.UNSEEN

    hp.mollview(cleanedmap[0], min=-2.5, max=2.5)
    hp.mollview(cleanedmap[1], min=-2.5, max=2.5)
    hp.mollview(cleanedmap[2], min=-2.5, max=2.5)

    hp.mollview(dcc[0], min=-0.25, max=0.25)
    hp.mollview(dcc[1], min=-0.25, max=0.25)
    hp.mollview(dcc[2], min=-0.25, max=0.25)
    
    """
    plt.figure()
    plt.loglog(cl0[1:3].T, 'b-', label='CAMB')
    plt.loglog(np.abs(get_spectrum_xpol(map145CMB,  lmax=23, mask=mask)[1][1:3]).T, 'b:', label='map145')
    plt.loglog(np.abs(get_spectrum_xpol(map220CMB,  lmax=23, mask=mask)[1][1:3]).T, 'r:', label='map220')
    plt.loglog(np.abs(get_spectrum_xpol(sync145CMB, lmax=23, mask=mask)[1][1:3]).T, 'g:', label='sync145')
    plt.loglog(np.abs(get_spectrum_xpol(dust145CMB, lmax=23, mask=mask)[1][1:3]).T, 'y:', label='dust145')
    plt.loglog(np.abs(get_spectrum_xpol(cleanedmap, lmax=23, mask=mask)[1][1:3]).T, 'k-', label='cleaned')
    plt.loglog(np.abs(get_spectrum_xpol(dustless,   lmax=23, mask=mask)[1][1:3]).T, 'k--', label='dustless')
    plt.legend()
    """

    return 


def test_gb3maps_pysm():
    nside = 128
    
    np.random.seed(0)

    ## cmb
    cl0 = get_spectrum_camb(lmax=100, isDl=False, CMB_unit='muK')
    cmbCMB = hp.synfast(cl0, nside=nside, verbose=0, new=1)
    cmb145RJ = CMB2RJ(cmbCMB, 145)
    cmb220RJ = CMB2RJ(cmbCMB, 220)

    ## foregrounds by pysm
    map_path = '/home/kmlee/cmb/forecast/maps/'
    FG90CMB  = hp.ud_grade(hp.read_map(map_path+f'foregrounds/fg/fg_90_TCMB_nside{nside:04d}_equ.fits', field=(0,1,2), verbose=0), nside_out=nside)
    FG145CMB = hp.ud_grade(hp.read_map(map_path+f'foregrounds/fg/fg_145_TCMB_nside{nside:04d}_equ.fits', field=(0,1,2), verbose=0), nside_out=nside)
    FG220CMB = hp.ud_grade(hp.read_map(map_path+f'foregrounds/fg/fg_220_TCMB_nside{nside:04d}_equ.fits', field=(0,1,2), verbose=0), nside_out=nside)

    FG90CMB[np.where(FG90CMB==hp.UNSEEN)] = 0
    FG145CMB[np.where(FG145CMB==hp.UNSEEN)] = 0
    FG220CMB[np.where(FG220CMB==hp.UNSEEN)] = 0

    map90CMB = np.zeros(np.shape(cmbCMB))
    map145CMB = np.zeros(np.shape(cmbCMB))
    map220CMB = np.zeros(np.shape(cmbCMB))

    #map90CMB  += cmbCMB 
    #map145CMB += cmbCMB
    #map220CMB += cmbCMB 
    map90CMB  += FG90CMB
    map145CMB += FG145CMB
    map220CMB += FG220CMB 

    ## mask
    maskgb = makegbmask(nside, 33, 90)
    maskga = hp.read_map(f'{map_path}/mask/mask_gal_ns1024_equ.fits', verbose=False) 
    maskga = hp.ud_grade(maskga, nside_out=nside)
    masksy = hp.read_map(f'{map_path}/mask/mask_sync_ns1024_equ.fits', verbose=False)
    mask = maskgb * maskga #* masksy
    #rot = hp.rotator.Rotator(coord=['C','G'])
    #mask = rot.rotate_map_pixel(mask)
    mask[mask>0] = 1
    mask[mask<1] = 0

    hp.mollview(mask)
    #mask[:]=1

    ## ILC
    maps = [masking(map90CMB, mask), masking(map145CMB, mask), masking(map220CMB, mask)]
    res, _ = ilcmaps(*maps)

    ## cleaned map
    cs = list(res.x)
    cs.append(1-np.sum(res.x))
    cs = np.array(cs)
    tmp = [ct * mt for ct, mt in zip(cs, (map90CMB*mask, map145CMB*mask, map220CMB*mask))]
    cleanedmap = sum(tmp)
    print(np.shape(cleanedmap))

    mpp = RJ2CMB(changefreq(CMB2RJ(map220CMB, 220), 220, 145, 1.6), 145) 
    cf = changefreq(1.0, 220, 145, 1.52) * CMB2RJ(1, 220) * RJ2CMB(1, 145)
    dustless = (map145CMB - cf * map220CMB) / (1 - cf)

    ## print rms
    show_rms([cmbCMB, cmb145RJ, cmb220RJ], ['cmb (K_CMB)', 'cmb145 (K_RJ)', 'cmb220 (K_RJ)'], mask)
    #show_rms([sync145CMB, sync145RJ, sync220CMB, sync220RJ], ['sync145 (K_CMB)', 'sync145 (K_RJ)', 'sync220 (K_CMB)', 'sync220 (K_RJ)'], mask)
    #show_rms([dust145CMB, dust145RJ, dust220CMB, dust220RJ], ['dust145 (K_CMB)', 'dust145 (K_RJ)', 'dust220 (K_CMB)', 'dust220 (K_RJ)'], mask)
    show_rms([FG145CMB, FG220CMB], ['FG145 (K_CMB)', 'FG220 (K_CMB)'], mask)
    show_rms([map145CMB, map220CMB], ['map145 (K_CMB)', 'map220 (K_CMB)'], mask)

    show_rms([cleanedmap, dustless], ['Cleaned ILC (K_CMB)', 'Cleaned dust (K_CMB)'], mask)

    print (f'coeffs (ILC) = {cs}')
    print (f'coeffs (dustless) = {1/(1-cf)}, {-cf/(1-cf)}')
    print (cf)
    print (f'coverage = {np.average(mask)}')

    dcc = cleanedmap - cmbCMB * mask

    cleanedmap[np.where(cleanedmap==0)] = hp.UNSEEN
    dcc[np.where(dcc==0)] = hp.UNSEEN

    hp.mollview(cleanedmap[0], min=-2.5, max=2.5)
    hp.mollview(cleanedmap[1], min=-2.5, max=2.5)
    hp.mollview(cleanedmap[2], min=-2.5, max=2.5)

    hp.mollview(dcc[0], min=-0.25, max=0.25)
    hp.mollview(dcc[1], min=-0.25, max=0.25)
    hp.mollview(dcc[2], min=-0.25, max=0.25)
    
    """
    plt.figure()
    plt.loglog(cl0[1:3].T, 'b-', label='CAMB')
    plt.loglog(np.abs(get_spectrum_xpol(map145CMB,  lmax=23, mask=mask)[1][1:3]).T, 'b:', label='map145')
    plt.loglog(np.abs(get_spectrum_xpol(map220CMB,  lmax=23, mask=mask)[1][1:3]).T, 'r:', label='map220')
    plt.loglog(np.abs(get_spectrum_xpol(sync145CMB, lmax=23, mask=mask)[1][1:3]).T, 'g:', label='sync145')
    plt.loglog(np.abs(get_spectrum_xpol(dust145CMB, lmax=23, mask=mask)[1][1:3]).T, 'y:', label='dust145')
    plt.loglog(np.abs(get_spectrum_xpol(cleanedmap, lmax=23, mask=mask)[1][1:3]).T, 'k-', label='cleaned')
    plt.loglog(np.abs(get_spectrum_xpol(dustless,   lmax=23, mask=mask)[1][1:3]).T, 'k--', label='dustless')
    plt.legend()
    """

    return 


def test_gb4maps_pysm():
    nside = 128
    
    np.random.seed(0)

    ## cmb
    cl0 = get_spectrum_camb(lmax=100, isDl=False, CMB_unit='muK')
    cmbCMB = hp.synfast(cl0, nside=nside, verbose=0, new=1)
    cmb145RJ = CMB2RJ(cmbCMB, 145)
    cmb220RJ = CMB2RJ(cmbCMB, 220)

    ## foregrounds by pysm
    map_path = '/home/kmlee/cmb/forecast/maps/'
    FG30CMB  = hp.ud_grade(hp.read_map(map_path+f'foregrounds/fg/fg_23_TCMB_nside{nside:04d}_equ.fits', field=(0,1,2), verbose=0), nside_out=nside)
    FG90CMB  = hp.ud_grade(hp.read_map(map_path+f'foregrounds/fg/fg_90_TCMB_nside{nside:04d}_equ.fits', field=(0,1,2), verbose=0), nside_out=nside)
    FG145CMB = hp.ud_grade(hp.read_map(map_path+f'foregrounds/fg/fg_145_TCMB_nside{nside:04d}_equ.fits', field=(0,1,2), verbose=0), nside_out=nside)
    FG220CMB = hp.ud_grade(hp.read_map(map_path+f'foregrounds/fg/fg_220_TCMB_nside{nside:04d}_equ.fits', field=(0,1,2), verbose=0), nside_out=nside)

    FG30CMB[np.where(FG30CMB==hp.UNSEEN)] = 0
    FG90CMB[np.where(FG90CMB==hp.UNSEEN)] = 0
    FG145CMB[np.where(FG145CMB==hp.UNSEEN)] = 0
    FG220CMB[np.where(FG220CMB==hp.UNSEEN)] = 0

    map30CMB = np.zeros(np.shape(cmbCMB))
    map90CMB = np.zeros(np.shape(cmbCMB))
    map145CMB = np.zeros(np.shape(cmbCMB))
    map220CMB = np.zeros(np.shape(cmbCMB))

    #map30CMB  += cmbCMB 
    #map90CMB  += cmbCMB 
    #map145CMB += cmbCMB
    #map220CMB += cmbCMB 
    map30CMB  += FG30CMB
    map90CMB  += FG90CMB
    map145CMB += FG145CMB
    map220CMB += FG220CMB 

    ## mask
    maskgb = makegbmask(nside, 33, 90)
    maskga = hp.read_map(f'{map_path}/mask/mask_gal_ns1024_equ.fits', verbose=False) 
    maskga = hp.ud_grade(maskga, nside_out=nside)
    masksy = hp.read_map(f'{map_path}/mask/mask_sync_ns1024_equ.fits', verbose=False)
    mask = maskgb * maskga #* masksy
    #rot = hp.rotator.Rotator(coord=['C','G'])
    #mask = rot.rotate_map_pixel(mask)
    mask[mask>0] = 1
    mask[mask<1] = 0

    hp.mollview(mask)
    #mask[:]=1

    ## ILC
    maps = [masking(map30CMB, mask), masking(map90CMB, mask), masking(map145CMB, mask), masking(map220CMB, mask)]
    res, _ = ilcmaps(*maps)

    ## cleaned map
    cs = list(res.x)
    cs.append(1-np.sum(res.x))
    cs = np.array(cs)
    tmp = [ct * mt for ct, mt in zip(cs, (map30CMB*mask, map90CMB*mask, map145CMB*mask, map220CMB*mask))]
    cleanedmap = sum(tmp)
    print(np.shape(cleanedmap))

    mpp = RJ2CMB(changefreq(CMB2RJ(map220CMB, 220), 220, 145, 1.6), 145) 
    cf = changefreq(1.0, 220, 145, 1.52) * CMB2RJ(1, 220) * RJ2CMB(1, 145)
    dustless = (map145CMB - cf * map220CMB) / (1 - cf)

    ## print rms
    show_rms([cmbCMB, cmb145RJ, cmb220RJ], ['cmb (K_CMB)', 'cmb145 (K_RJ)', 'cmb220 (K_RJ)'], mask)
    show_rms([FG145CMB, FG220CMB], ['FG145 (K_CMB)', 'FG220 (K_CMB)'], mask)
    show_rms([map145CMB, map220CMB], ['map145 (K_CMB)', 'map220 (K_CMB)'], mask)

    show_rms([cleanedmap, dustless], ['Cleaned ILC (K_CMB)', 'Cleaned dust (K_CMB)'], mask)

    print (f'coeffs (ILC) = {cs}')
    print (f'coeffs (dustless) = {1/(1-cf)}, {-cf/(1-cf)}')
    print (cf)
    print (f'coverage = {np.average(mask)}')

    dcc = cleanedmap - cmbCMB * mask

    cleanedmap[np.where(cleanedmap==0)] = hp.UNSEEN
    dcc[np.where(dcc==0)] = hp.UNSEEN

    hp.mollview(cleanedmap[0], min=-2.5, max=2.5)
    hp.mollview(cleanedmap[1], min=-2.5, max=2.5)
    hp.mollview(cleanedmap[2], min=-2.5, max=2.5)

    hp.mollview(dcc[0], min=-0.25, max=0.25)
    hp.mollview(dcc[1], min=-0.25, max=0.25)
    hp.mollview(dcc[2], min=-0.25, max=0.25)
    
    """
    plt.figure()
    plt.loglog(cl0[1:3].T, 'b-', label='CAMB')
    plt.loglog(np.abs(get_spectrum_xpol(map145CMB,  lmax=23, mask=mask)[1][1:3]).T, 'b:', label='map145')
    plt.loglog(np.abs(get_spectrum_xpol(map220CMB,  lmax=23, mask=mask)[1][1:3]).T, 'r:', label='map220')
    plt.loglog(np.abs(get_spectrum_xpol(sync145CMB, lmax=23, mask=mask)[1][1:3]).T, 'g:', label='sync145')
    plt.loglog(np.abs(get_spectrum_xpol(dust145CMB, lmax=23, mask=mask)[1][1:3]).T, 'y:', label='dust145')
    plt.loglog(np.abs(get_spectrum_xpol(cleanedmap, lmax=23, mask=mask)[1][1:3]).T, 'k-', label='cleaned')
    plt.loglog(np.abs(get_spectrum_xpol(dustless,   lmax=23, mask=mask)[1][1:3]).T, 'k--', label='dustless')
    plt.legend()
    """

    return 


def test_gbilc_pysm(freqs=[23, 30, 90, 145, 220], nside=128, wps=[0, 0, 0, 0, 0], include_cmb=True, include_noise=True, verbose=1, syncmask=False, cs_in=None):
    #np.random.seed(0)

    ## cmb
    cl0 = get_spectrum_camb(lmax=100, tau=0.05, r=0.05, As=2.092e-9, isDl=False, CMB_unit='muK')
    cmb = hp.synfast(cl0, nside=nside, verbose=0, new=1)

    ## foregrounds by pysm
    map_path = '/home/kmlee/cmb/forecast/maps/'
    fgs = [] 
    nms = []
    for f, wp in zip(freqs, wps):
        fgmap = hp.ud_grade(hp.read_map(map_path+f'foregrounds/fg/fg_{f:d}_TCMB_nside{nside:04d}_equ.fits', field=(0,1,2), verbose=0), nside_out=nside)
        fgmap[np.where(fgmap==hp.UNSEEN)] = 0
        fgs.append(fgmap)

        nl0 = get_spectrum_noise(lmax=100, wp=wp, isDl=False, CMB_unit='muK')
        nm = hp.synfast(nl0, nside, new=True, verbose=False)
        nms.append(nm)


    fgs = np.array(fgs)
    maps = fgs.copy()

    for i, _ in enumerate(maps):
        if include_cmb:
            maps[i] += cmb
        if include_noise:
            maps[i] += nms[i]


    ## mask
    mask = mymask(nside1=nside, syncmask=syncmask)

    ## ILC
    res, _ = ilcmaps(*maps)

    ## cleaned map
    if cs_in is not None:
        cs = cs_in
    else:
        cs = list(res.x)
        cs.append(1-np.sum(res.x))

    cs = np.array(cs)
    tmp = [ct * mt for ct, mt in zip(cs, maps)]
    cleanedmap = sum(tmp)

    if verbose:
        ## print rms
        show_rms([cmb], ['cmb (K)'] , mask)
        for i, m in enumerate(maps):
            show_rms([m], [f'map{freqs[i]} (K)'], mask)
        show_rms([cleanedmap], ['Cleaned ILC (K)'], mask)

        print (f'coeffs (ILC) = {cs}')
        print (f'coverage = {np.average(mask)}')

        maptobedrawn = cleanedmap*mask
        maptobedrawn = np.abs(maptobedrawn[1] + 1j*maptobedrawn[2])
        maptobedrawn[maptobedrawn==0] = hp.UNSEEN

        if include_cmb:
            hp.mollview(maptobedrawn, title=f'{freqs} GHz', unit=r'$\mu K$')
            maptobedrawn = cleanedmap*mask
            maptobedrawn -= cmb*mask
            maptobedrawn = np.abs(maptobedrawn[1] + 1j*maptobedrawn[2])
            maptobedrawn[maptobedrawn==0] = hp.UNSEEN
            hp.mollview(maptobedrawn, title=f'difference {freqs} GHz', min=0, max=0.7, unit=r'$\mu K$')
        else:
            hp.mollview(maptobedrawn, title=f'{freqs} GHz', min=0, max=0.7, unit=r'$\mu K$')

        tnoise = np.sqrt(np.sum(np.array(cs)**2 * np.array(wps)**2))
        print (f'total noise level = {tnoise}')
        print ('-'*50)

    return cs


def test():
    include_cmb = False#True#True
    include_noise = False#True#True
    nside = 8
    wp30 = 69
    wp90 =  77
    wp145 = 77
    wp220 = 215

    #test_gbilc_pysm(freqs=[145, 220], nside=nside, wps=[wp145, wp220], include_cmb=include_cmb, include_noise=include_noise, syncmask=True)
    #test_gbilc_pysm(freqs=[145, 220], nside=nside, wps=[wp145, wp220], include_cmb=include_cmb, include_noise=include_noise, syncmask=True, cs_in = [1.381, -0.381])
    #test_gbilc_pysm(freqs=[30, 145, 220], nside=nside, wps=[wp30, wp145, wp220], include_cmb=include_cmb, include_noise=include_noise)
    test_gbilc_pysm(freqs=[145, 220, 30], nside=nside, wps=[wp145, wp220, wp30], include_cmb=include_cmb, include_noise=include_noise, syncmask=False)
    #test_gbilc_pysm(freqs=[90, 145, 220], nside=nside, wps=[wp90, wp145, wp220], include_cmb=include_cmb, include_noise=include_noise)
    #test_gbilc_pysm(freqs=[30, 90, 145, 220], nside=nside, wps=[wp30, wp90, wp145, wp220], include_cmb=include_cmb, include_noise=include_noise)
    test_gbilc_pysm(freqs=[145, 220, 30, 90], nside=nside, wps=[wp145, wp220, wp30, wp90], include_cmb=include_cmb, include_noise=include_noise, syncmask=False)

    plt.show()


def ensemble_3ch():
    include_cmb = 0#True
    include_noise = 0#True
    nside = 128
    wp30 = 85 
    wp90 = 10 
    wp145 = 47
    wp220 = 130

    cs1 = []
    cs2 = []

    ntest = 100
    for i in range(ntest):
        cs1_tmp = test_gbilc_pysm(freqs=[30, 145, 220], nside=nside, wps=[wp30, wp145, wp220], include_cmb=include_cmb, include_noise=include_noise, verbose=0)
        cs2_tmp = test_gbilc_pysm(freqs=[145, 220, 30], nside=nside, wps=[wp145, wp220, wp30], include_cmb=include_cmb, include_noise=include_noise, verbose=0)
        cs1.append(cs1_tmp)
        cs2.append(cs2_tmp)

    cs1avg = np.average(cs1, axis=0)
    cs2avg = np.average(cs2, axis=0)
    cs1sig = np.std(cs1, axis=0)
    cs2sig = np.std(cs2, axis=0)

    wp1 = (np.sum(np.array([wp30, wp145, wp220])**2 * cs1avg**2))**0.5
    wp2 = (np.sum(np.array([wp145, wp220, wp30])**2 * cs2avg**2))**0.5

    print ('-'*50)
    print (f'freqs = [30, 145, 220], avg cs = {cs1avg}, sig cs = {cs1sig}, wp={wp1}')
    print (f'freqs = [145, 220, 30], avg cs = {cs2avg}, sig cs = {cs2sig}, wp={wp2}')
    print ('-'*50)

    plt.show()


if __name__=='__main__':
    #ensemble_3ch() 
    test()


