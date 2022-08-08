"""
This code was done to determine the structural parameters for Y2Q1 DG (or cluster) candidates using MCMC approach.
It uses a list of input parameters (center, limits for exponential radius and PA), which can be changed iteratively. See line 46.
Starting from the input list, the program opens each cluster file named name[i] (see line 49)
from the DES catalogue.
I have problems using s_ml and s0 units as stars/arcmin^2. Please, do not change these units (they are in stars/deg^2).
Written and implemented by Adriano Pieres since 2015/07/18.
"""
import numpy as np
import matplotlib.pyplot as plt
import emcee
import astropy.io.fits as fits
import corner


def lnprior(theta):
    re, ell, th, bg, xc, yc = theta
    if 1. < re < 20. and 0.0 < ell < 0.3 and -1.8 < th < 1.8 and 0. < bg < 500. and -3. < xc < 3. and -3. < yc < 3.:
        return 0
    return -np.inf


def lnlike(theta, RA, DEC):
    re, ell, th, bg, xc, yc = theta
    dX = (RA-(ra0+xc/60.))*np.cos(np.radians(dec0))
    dY = (DEC-(dec0+yc/60.))
    ri = 60.*np.sqrt(((dX*np.cos(np.radians(100*th))-dY*np.sin(np.radians(100*th)))/(1.-ell))
                     ** 2.+(dX*np.sin(np.radians(100*th))+dY*np.cos(np.radians(100*th)))**2.)
    s0 = (len(RA)-bg*3600.)/((1-ell)*2.*np.pi*re**2)
    if not np.isfinite(2.*np.sum(np.log(bg+s0*np.exp(-ri/re)))):
        return -np.inf
    return 2.*np.sum(np.log(bg+s0*np.exp(-ri/re)))


def lnprob(theta, RA, DEC):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, RA, DEC)


# Some definitions:
R0 = 1.0
# ra0, dec0, reinf, resup, thmin, thmax = 67.3156, -23.8368, 0.1, 10., 0.0, 180.
HPX_, ra0_, dec0_, re_kick_, ell_kick_, pa_kick_ = np.loadtxt(
    'star_clusters_simulated.dat', usecols=(0, 6, 7, 8, 9, 10), unpack=True)

hdu = fits.open('des_mockcat_for_detection.fits', memmap=True)
RA = hdu[1].data.field('ra')
DEC = hdu[1].data.field('dec')
MAGG = hdu[1].data.field('mag_g_with_err')
MAGGERR = hdu[1].data.field('magerr_g')
MAGR = hdu[1].data.field('mag_g_with_err')
MAGRERR = hdu[1].data.field('magerr_r')
hdu.close()

for i in range(0,2): #len(ra0_)):
    reinf = 0.5 * re_kick_[i]
    resup = 1.5 * re_kick_[i]
    thmin = pa_kick_[i] - 45.
    thmax = pa_kick_[i] + 45.
    ra0 = ra0_[i]
    dec0 = dec0_[i]

    R = 60.*np.sqrt(((np.cos(np.radians(dec0))*(ra0-RA))**2.)+(DEC-dec0)**2)
    '''
    plt.scatter(RA[R < 10.], DEC[R < 10.], s=1.)
    plt.xlabel('RA')
    plt.ylabel('DEC')
    plt.show()

    plt.scatter(MAGG[R < 2.]-MAGR[R < 2.], MAGG[R < 2.])
    plt.show()
    '''
    re_ml = (reinf + resup)/2.
    ell_ml = 0.1
    th_ml = (thmin + thmax)/200.
    idbg = (R > R0)
    robg = float(len(RA[idbg]))/(3600.-np.pi*(R0)**2)  # stars/arcmin^2
    N_star = (len(RA)-robg*3600.)
    s_ml = N_star/(2.*np.pi*re_ml**2)
    xc_ml = 0.0
    yc_ml = 0.0
    result = re_ml, ell_ml, th_ml, robg, xc_ml, yc_ml
    print(re_ml, ell_ml, th_ml, robg, xc_ml, yc_ml)

    ndim, nwalkers = 6, 300  # 300
    pos = [result + 1e-4*np.random.randn(ndim) for j in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(RA, DEC))
    sampler.run_mcmc(pos, 500)  # 500
    samples = sampler.chain[:, 100:, :].reshape((-1, ndim))

    samples[:, 2] = 100.*(samples[:, 2])
    re_mcmc, ell_mcmc, th_mcmc, robg_mcmc, xc_mcmc, yc_mcmc = map(lambda v: (
        v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))

    print(HPX_[i], re_mcmc[0], ell_mcmc[0], th_mcmc[0], robg_mcmc[0], xc_mcmc[0], yc_mcmc[0])
# Plotting data
    fig = corner.corner(samples, labels=["$re$", "$ell$", "$th$", r"$\rho_{bg}$", r"$\Delta\alpha$", r"$\Delta\delta$"], truths=[
                        re_mcmc[0], ell_mcmc[0], th_mcmc[0], robg_mcmc[0], xc_mcmc[0], yc_mcmc[0]], quantiles=[0.16, 0.5, 0.84], show_titles=True, plot_contours=True)
    plt.savefig(str(HPX_[i]) + '_plus.png')
    plt.close()

    L = np.zeros(len(RA))
    dX = (RA-(ra0+xc_mcmc[0]/60.))*np.cos(np.radians(dec0))
    dY = (DEC-(dec0+yc_mcmc[0]/60.))
    ri = 60.*np.sqrt(((dX*np.cos(np.radians(100*th_mcmc[0]))-dY*np.sin(np.radians(100*th_mcmc[0])))/(
        1.-ell_mcmc[0]))**2.+(dX*np.sin(np.radians(100*th_mcmc[0]))+dY*np.cos(np.radians(100*th_mcmc[0])))**2.)
    L = (np.exp(-ri/re_mcmc[0]))

#a = open('J0429_stars_list_profile.dat','w')
# for j in range(len(RA)):
#    if (L[j]<0.):
#        L[j]=0.
#    print >> a, RA[j], DEC[j], MAGG[j], MAGR[j], ERRG[j], ERRR[j], SM[j], L[j]
# a.close
