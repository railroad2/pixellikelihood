import numpy as np
import healpy as hp

import pysm
from pysm.nominal import models


def RJ2CMB(T_RJ, nu):
    x = nu/56.78
    T_CMB = (np.exp(x) - 1)**2/(x**2 * np.exp(x)) * T_RJ

    return T_CMB


def gen_fg(freqs, nside=8):
    try:
        freqs = list(freqs)
    except: 
        freqs = list([freqs])

    dustmodel = 'd1'
    syncmodel = 's1'
    sky_config = {
        'synchrotron' : models(syncmodel, nside),
        'dust'        : models(dustmodel, nside),
        'freefree'    : models('f1', nside),
        #'cmb'         : models('c1', nside),
        'ame'         : models('a1', nside),
    }

    sky = pysm.Sky(sky_config)
    sky.output_unit = "uK_RJ"

    rot = hp.rotator.Rotator(coord=['g','c'])

    fgs = []
    for freq in freqs:
        fwhm = 1 
        fweight = True if (nside > 16) else False 

        fg_gal = sky.signal()(freq)
        fg_equ = rot.rotate_map_alms(fg_gal, use_pixel_weights=fweight)
        fg_equ = RJ2CMB(fg_equ, freq)
        fg_equ_sm = hp.smoothing(fg_equ, fwhm=np.radians(fwhm), verbose=0)
        fgs.append(fg_equ_sm)

    if len(fgs) == 1:
        return fgs[0]
    return fgs

    instrument_delta_bpass = {
        'frequencies' : freqs,
        'channels' : (),
        'beams' : np.ones(3)*70.,
        'sens_I' : np.ones(3),
        'sens_P' : np.ones(3),
        'nside' : nside,
        'noise_seed' : 1234,
        'use_bandpass' : False,
        'add_noise' : False,
        'output_units' : 'uK_RJ',
        'use_smoothing' : True,
        'output_directory' : '.utofs/hive/home/kmlee/cmb/forecast/sim_map/',
        'output_prefix' : 'test',
    }

    instrument = pysm.Instrument(instrument_delta_bpass)
    instrument.observe(sky)

    return

