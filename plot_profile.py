import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt

HPX_, Nstar, ra0_, dec0_, re_kick_, ell_kick_, pa_kick_, dist = np.loadtxt(
    'star_clusters_simulated.dat', usecols=(0, 1, 6, 7, 8, 9, 10, 12), unpack=True)

re_kick_arcmin = 60. * (180. / np.pi) * np.arctan(re_kick_ / dist)

hdu = fits.open('des_mockcat_for_detection.fits', memmap=True)
RA = hdu[1].data.field('ra')
DEC = hdu[1].data.field('dec')
MAGG = hdu[1].data.field('mag_g_with_err')
MAGGERR = hdu[1].data.field('magerr_g')
MAGR = hdu[1].data.field('mag_g_with_err')
MAGRERR = hdu[1].data.field('magerr_r')
hdu.close()

# HPX re(0,+,-) ell(0,+,-) pa(0,+,-) robg(0,+,-) shift_ra(0,+,-) shift dec(0,+,-)
re_fit, ell_fit, pa_fit, robg_fit, Nstar_fit = np.loadtxt('results.dat', usecols=(1, 4, 7, 10, 19), unpack=True)

for i in range(len(HPX_)):
    dX = (RA-(ra0_[i]))*np.cos(np.radians(dec0_[i]))
    dY = (DEC-(dec0_[i]))
    radius = np.logspace(-1, 0.602, 40)
    area_annulus = [np.pi * (j ** 2 - a **2) for a,j in zip(radius[0:-1], radius[1::])]
    radius_mean = [0.5 * (j + a) for a,j in zip(radius[0:-1], radius[1::])]
    S0 = Nstar[i] / (2*np.pi * (1-ell_kick_[i]) * re_kick_arcmin[i] ** 2 )
    S0_fit = Nstar_fit[i] / (2*np.pi * (1-ell_fit[i]) * re_fit[i] ** 2 )

    profile = S0 * np.exp(-radius/re_kick_arcmin[i])
    fit_profile = S0_fit * np.exp(-radius/re_fit[i])
    ri = 60.*np.sqrt(((dX*np.cos(np.radians(pa_kick_[i]))-dY*np.sin(np.radians(pa_kick_[i])))/(1.-ell_kick_[i]))**2.+ (dX*np.sin(np.radians(pa_kick_[i]))+dY*np.cos(np.radians(pa_kick_[i])))**2.)
    # L = np.exp(-ri/re_kick_[i])
    print(re_kick_arcmin[i], ra0_[i], dec0_[i])
    # a = plt.hist(ri, bins=40, range=(0., 1.))
    #density = a[0][0:-1] / (area_annulus)
    n_star = np.zeros(len(radius_mean))
    for j in range(len(radius_mean)):
        n_star[j] = len(ri[(ri > radius[j])&(ri < radius[j+1])])
    density = np.divide(n_star, area_annulus)
    density_err = np.divide(np.sqrt(n_star), area_annulus)
    plt.plot(radius_mean, density[-1] + profile[:-1], color='k', label='Truth table')
    plt.plot(radius_mean, robg_fit[i] + fit_profile[:-1], color='r', label='Fit')
    plt.errorbar(radius[:-1], density, yerr=density_err, fmt='o', label='Data')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('r (arcmin)')
    plt.ylabel(r'$\rho\ (stars/arcmin^2)$')
    plt.legend()
    plt.savefig('{:d}_profile.png'.format(int(HPX_[i])))
    plt.close()

