from __future__ import print_function
import healpy as hp
import numpy as np
import pylab as plt
import sys
import time

from cambfast.cambfast import CAMBfast

from iminuit import Minuit
from pprint import pprint

from gbpipe.spectrum import get_spectrum_camb, get_spectrum_noise

from pixelcov.covmat_ana  import getcov_ana, gen_Pl_ana, gen_Wls_ana, getcov_ana_pol
from pixelcov.covmat_nk   import getcov_nk_pol
from pixelcov.covmat_est  import getcov_est
from pixelcov.covinv      import detinv, detinv_pseudo 
from pixelcov.covreg      import covreg_I, covreg_1, covreg_D, covreg_R
from pixelcov.covreg      import regularize
from pixelcov.utils       import * 
from pixelcov.covcut      import cutmap, cutcov, cutpls, cutwls, partcov, partmap, Kmat


def getcov(cls, nside, lmax=None, lmin=None, pls=[], wls=[], covtype='ana'):
    if lmax is None:
        lmax = 3*nside - 1

    if lmin==None :
        lmin = 0

    if (covtype=='ana'):
        if pls == []:
            pls = gen_Pl_ana(nside, lmax=lmax)

        if wls == []:
            wls = gen_Wls_ana(nside, lmax=lmax)

        cls[:,:lmin] = 0
        cov = getcov_ana_pol(cls, pls=pls, wls=wls, nside=nside, lmax=lmax, isDl=False)

    elif (covtype=='nk'):
        cov = getcov_nk_pol(cls)

    elif (covtype=='est'):
        cov = getcov_est(cls, nside=nside, nsample=10000, isDl=False, pol=True, rseed=42)

    else:
        pass

    return cov


def computeL(map_in, cls, nside, lmax=None, lmin=None, pls=[], wls=[], 
             mask=[], covtype='ana', maptype=None, regtype=None, reglamb=0, covblk=[]):

    cov = getcov(cls, nside, lmax, lmin, pls, wls, covtype)

    x = map_in.copy()
    x = x.flatten()

    # cut map and cov
    if not mask == []:
        x = cutmap(x, mask) 
        cov = cutcov(cov, mask)
    
    # Part cov
    x = partmap(x, maptype)
    cov = partcov(cov, maptype, covblk)

    # Regularize cov
    cov = regularize(cov, reglamb, regtype)

    try:
        logdet, covi = detinv(cov)
    except:
        return 1e30

    k = len(x) 
    n2logL = k*np.log(2*np.pi) + logdet + np.dot(x, np.array(np.dot(covi, x)).flatten())

    return n2logL


def computeL2(map_in, cls, nside, lmax=None, lmin=None, pls=[], wls=[], 
             mask=[], covtype='ana', maptype=None, regtype=None, reglamb=0, covblk=[]):

    cov = getcov(cls, nside, lmax, lmin, pls, wls, covtype)

    x = map_in.copy()
    x = x.flatten()

    # cut map and cov
    if not mask == []:
        x = cutmap(x, mask) 
        cov = cutcov(cov, mask)
        K = Kmat(nside, lmin=lmin)
        K = np.concatenate(np.concatenate(np.array([[K]*3]*3), axis=1), axis=1)
        K = cutcov(K, mask)
    else:
        K = Kmat(nside, 1)

    # Part cov
    x = partmap(x, maptype)
    cov = partcov(cov, maptype, covblk)
    K = partcov(K, maptype, covblk)

    # Regularize cov
    cov = regularize(cov, reglamb, regtype)

    try:
        logdet, covi = detinv(cov*K)
        logdet, covi2 = detinv(cov)
    except:
        return 1e30

    k = len(x) 
    #n2logL = k*np.log(2*np.pi) + logdet + np.dot(x, np.array(np.dot(K*covi, x)).flatten())
    n2logL = k*np.log(2*np.pi) + logdet + np.dot(x, np.array(np.dot(covi, x)).flatten())

    #print (logdet)
    #print (np.dot(x, np.array(np.dot(K*covi, x)).flatten()))

    #plt.matshow(cov)
    #plt.colorbar()
    #plt.matshow(covi)
    #plt.colorbar()
    #plt.matshow(K)
    #plt.show()
    #n2logL = k*np.log(2*np.pi) + logdet + np.linalg.multi_dot([x, covi, x]).flatten()

    return n2logL


def contour_minuit(mm, pnames, fname=None, inputs=None, latex=False):
    if latex:
        from matplotlib import rc
        rc('font', **{'family':'sans-serif', 'sans-serif':['Helvetica'], 'size':15})
        rc('text', usetex=True)

    fig = plt.figure()
    ax = []
    ndim = len(pnames)
    for i in range(len(pnames)):
        for j in range(i+1):
            print (pnames[i], pnames[j])
            k = ndim*i + j
            ax.append(fig.add_subplot(ndim, ndim, k+1))

            if i==j:
                pn = pnames[i]
                p = mm.profile(pn)

                val = mm.values[pn]
                err = mm.errors[pn]

                plt.plot(p[0], np.exp(-p[1]-max(-p[1])))
                if pn == 'tau':
                    dn = r'$\tau$'
                elif pn == 'wp':
                    dn = r'$w_p^{-0.5}$'
                elif pn == 'c0':
                    dn = r'$c_{145}$'
                elif pn == 'c1':
                    dn = r'$c_{220}$'
                else:
                    dn = pn

                ax[-1].set_xlabel(pname_latex(pnames[i]))
                ax[-1].set_ylabel(pname_latex(pnames[i]))
                ax[-1].set_title(f'{dn}$={val:4.3f}\pm{err:4.3f}$')

                ax[-1].plot((val, val), (0, 1), 'k', label=f'{dn} fit')
                ax[-1].plot((val-err, val-err), (0, 1), 'k--')
                ax[-1].plot((val+err, val+err), (0, 1), 'k--')
                plt.xlim((val-2*err, val+2*err))
                plt.ylim((0, 1))

                if inputs is not None:
                    inp = inputs[i]
                    ax[-1].plot((inp, inp), (0, 1), 'r', label=f'{dn} input')
                   
            else:
                try:
                    vx, vy, vz = mm.contour(pnames[j], pnames[i], bound=2, subtract_min=True)
                    v = [mm.errordef * (i + 1) for i in range(4)]
                    plt.contour(vx, vy, vz, v)
                    plt.xlabel(pname_latex(pnames[j]))
                    plt.ylabel(pname_latex(pnames[i]))
                    plt.axhline(mm.values[pnames[i]], color="k", ls="--")
                    plt.axvline(mm.values[pnames[j]], color="k", ls="--")

                    if inputs is not None:
                        inpx = inputs[j]
                        inpy = inputs[i]
                        pnx = pnames[j]
                        pny = pnames[i]
                        valx = mm.values[pnx]
                        valy = mm.values[pny]
                        errx = mm.errors[pnx]
                        erry = mm.errors[pny]
                        ax[-1].plot((inpx, inpx), (valy-2*erry, valy+2*erry), 'r')
                        ax[-1].plot((valx-2*errx, valx+2*errx), (inpy, inpy), 'r')
                        #ax[-1].scatter(inpx, inpy, 'r*')
                        plt.xlim((valx-2*errx, valx+2*errx))
                        plt.ylim((valy-2*erry, valy+2*erry))
                        
                except Exception as e:
                    print (f'An error occured during drawing the contour between {pnames[i]} and {pnames[j]}.')
                    print (e)
                    pass

            if j != 0:
                ax[-1].set_ylabel('')
                ax[-1].set_yticks([])
            if i != len(pnames)-1:
                ax[-1].set_xlabel('')
                ax[-1].set_xticks([])
            
    #plt.tight_layout()
    if type(fname) is str:
        plt.savefig(fname)


    return 


def mncontour_minuit(mm, pnames, fname=None, inputs=None, merr=None):
    from matplotlib import rc
    rc('font', **{'family':'sans-serif', 'sans-serif':['Helvetica'], 'size':15})
    rc('text', usetex=True)
    fig = plt.figure()
    ax = []
    ndim = len(pnames)
    for i in range(len(pnames)):
        for j in range(i+1):
            print (pnames[i], pnames[j])
            k = ndim*i + j
            ax.append(fig.add_subplot(ndim, ndim, k+1))

            if i==j:
                pn = pnames[i]
                ran = set_ranges(pn) 
                p = mm.mnprofile(pn)

                val = mm.values[pn]
                err = mm.errors[pn]
                lerr = merr[(pn, -1.0)] * -1
                uerr = merr[(pn, 1.0)]

                dn = pname_latex(pn)
                plt.plot(p[0], np.exp(-p[1]-max(-p[1])))
                ax[-1].set_xlabel(pname_latex(pnames[i]))
                ax[-1].set_ylabel(pname_latex(pnames[i]))
                ax[-1].set_title(f'{dn}$={val:4.3f}^{{+{uerr:4.3f}}}_{{-{lerr:4.3f}}}$')

                ax[-1].plot((val, val), (0, 1), 'k', label=f'{dn} fit')
                ax[-1].plot((val-lerr, val-lerr), (0, 1), 'k--')
                ax[-1].plot((val+uerr, val+uerr), (0, 1), 'k--')
                #plt.xlim((val-2*err, val+2*err))
                plt.xlim(ran)
                plt.ylim((0, 1))

                if inputs is not None:
                    inp = inputs[i]
                    ax[-1].plot((inp, inp), (0, 1), 'r', label=f'{dn} input')
                   
            else:
                try:
                    mm.draw_mncontour(pnames[j], pnames[i])
                    ax[-1].set_xlabel(pname_latex(pnames[j]))
                    ax[-1].set_ylabel(pname_latex(pnames[i]))
                    pnx = pnames[j]
                    pny = pnames[i]

                    if inputs is not None:
                        inpx = inputs[j]
                        inpy = inputs[i]
                        valx = mm.values[pnx]
                        valy = mm.values[pny]
                        errx = mm.errors[pnx]
                        erry = mm.errors[pny]
                        ax[-1].plot((inpx, inpx), (valy-2*erry, valy+2*erry), 'r')
                        ax[-1].plot((valx-2*errx, valx+2*errx), (inpy, inpy), 'r')
                        #ax[-1].scatter(inpx, inpy, 'r*')

                    #plt.xlim((valx-2*errx, valx+2*errx))
                    #plt.ylim((valy-2*erry, valy+2*erry))
                    xran = set_ranges(pnx)
                    yran = set_ranges(pny)
                    plt.xlim(xran)
                    plt.ylim(yran)
                        
                except Exception as e:
                    print (f'An error occured during drawing the contour between {pnames[i]} and {pnames[j]}.')
                    print (e)
                    pass

            if j != 0:
                ax[-1].set_ylabel('')
                ax[-1].set_yticks([])
            if i != len(pnames)-1:
                ax[-1].set_xlabel('')
                ax[-1].set_xticks([])
            
    #plt.tight_layout()
    if type(fname) is str:
        plt.savefig(fname)

    return 


def pname_latex(pn):
    if pn == 'tau':
        dn = r'$\tau$'
    elif pn == 'wp':
        dn = r'$w_p^{-0.5}$'
    elif pn == 'c0':
        dn = r'$c_{145}$'
    elif pn == 'c1':
        dn = r'$c_{220}$'
    else:
        dn = f'${pn}$'

    return dn 


def set_ranges(pn):
    if pn == 'tau':
        ran = (0.003, 0.110)
    elif pn == 'c0':
        ran = (1.411, 1.485)
    elif pn == 'c1':
        ran = (-0.461, -0.397)
    else:
        ran = None

    return ran 

