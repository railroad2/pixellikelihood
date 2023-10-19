import os 
import sys

import numpy as np
import healpy as hp
import pylab as plt

from pixelcov import covmat_ana, covmat_est, covmat_nk, spectrum
from pixelcov.utils import dl2cl
from pixelcov.vis_covmat import show_cov


class debugcov():
    def __init__(self, dls, nside=2):
        self.dl = dls
        self.nside = nside
        self.npix = 12*self.nside*self.nside
        self.lmax = 3*self.nside-1

        self.cl = dl2cl(self.dl)
        self.ell = np.arange(len(self.dl[0]))
        self.cov_ana = []
        self.cov_est = []
        self.gen_cov_ana()
        self.gen_cov_est()


    def gen_cov_ana(self):
        nside = self.nside
        lmax = self.lmax
        npix = self.npix
        cov = covmat_ana.getcov_ana_pol(self.cl, nside=nside, lmax=lmax, isDl=False)
        self.cov_ana = cov

    
    def gen_cov_est(self):
        nside = self.nside
        lmax = self.lmax 
        npix = self.npix
        cov = covmat_est.getcov_est(self.cl, nside, lmax=lmax, nsample=100000, isDl=False, pol=True)
        self.cov_est = cov


    def compare_cov(self, cov_ana, cov_est):
        print ('Dls:\n',self.dl)

        diag_expect = np.sum((2.*self.ell+1)/4/np.pi*self.cl[0])
        print ('Expected diagonal component( sum_l (2l+1)Cl/(4pi) ):', diag_expect)

        diag_ana = np.average(np.diagonal(cov_ana[:self.npix, :self.npix]))
        print ('average of diagonal terms of analytic cov:', diag_ana)

        diag_est = np.average(np.diagonal(cov_est[:self.npix, :self.npix]))
        print ('average of diagonal terms of estimated cov:', diag_est)

        print (cov_ana)
        print (cov_est)

        show_cov(cov_ana, title='Analytic',   logscale=False)
        show_cov(cov_est, title='Estimation', logscale=False)

        cov_diff = cov_ana - cov_est
        show_cov(cov_diff, title='(ana) - (est)', logscale=False)


    def compare_TT(self):
        cov_ana = self.cov_ana[:self.npix, :self.npix]
        cov_est = self.cov_est[:self.npix, :self.npix]
        self.compare_cov(cov_ana, cov_est)


    def compare_TQTU(self):
        cov_ana = self.cov_ana[:self.npix, self.npix:self.npix*3]
        cov_est = self.cov_est[:self.npix, self.npix:self.npix*3]
        self.compare_cov(cov_ana, cov_est)


    def compare_QQUU(self):
        cov_ana = self.cov_ana[self.npix:self.npix*3, self.npix:self.npix*3]
        cov_est = self.cov_est[self.npix:self.npix*3, self.npix:self.npix*3]
        self.compare_cov(cov_ana, cov_est)


def TT():
    nside = 4
    lmax  = 3 * nside - 1

    dl    = np.zeros(lmax+1)+1
    dl[0] = 0
    dl[1] = 0
    dl    = np.array([dl, dl*1, dl*0, dl*1])
    dl   *= 0
    
    ell       = 3
    dl[0,ell] = 1

    dc = debugcov(dl, nside=nside)

    dc.compare_TT()
    plt.show()
    

def TQTU():
    nside = 4
    lmax  = 3 * nside - 1

    dl    = np.zeros(lmax+1)+1
    dl[0] = 0
    dl[1] = 0
    dl    = np.array([dl, dl*1, dl*0, dl*1])
    dl   *= 0
    
    ell       = 8
    dl[0,ell] = 1
    dl[1,ell] = 1
    dl[3,ell] = 1

    dc = debugcov(dl, nside=nside)

    dc.compare_TQTU()
    plt.show()


def TQTU():
    nside = 1
    lmax  = 3 * nside - 1

    dl    = np.zeros(lmax+1)+1
    dl[0] = 0
    dl[1] = 0
    dl    = np.array([dl, dl*1, dl*0, dl*1])
    dl   *= 0
    
    ell       = 2
    dl[0,ell] = 0
    dl[1,ell] = 1
    dl[3,ell] = 1

    dc = debugcov(dl, nside=nside)

    dc.compare_QQUU()
    plt.show()


if __name__=='__main__':
    TQTU()


