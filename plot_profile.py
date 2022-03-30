import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt

HPX_, ra0_, dec0_, re_kick_, ell_kick_, pa_kick_, dist = np.loadtxt(
    'star_clusters_simulated.dat', usecols=(0, 6, 7, 8, 9, 10, 12), unpack=True)

re_kick_arcmin = 60. * (180. / np.pi) * np.arctan(re_kick_ / dist)

hdu = fits.open('des_mockcat_for_detection.fits', memmap=True)
RA = hdu[1].data.field('ra')
DEC = hdu[1].data.field('dec')
MAGG = hdu[1].data.field('mag_g_with_err')
MAGGERR = hdu[1].data.field('magerr_g')
MAGR = hdu[1].data.field('mag_g_with_err')
MAGRERR = hdu[1].data.field('magerr_r')
hdu.close()

for i in range(len(HPX_)):
    dX = (RA-(ra0_[i]))*np.cos(np.radians(dec0_[i]))
    dY = (DEC-(dec0_[i]))
    radius = np.linspace(0., 2., 20)
    L_radius = (np.exp(-radius/re_kick_arcmin[i]))
    ri = 60.*np.sqrt(((dX*np.cos(np.radians(pa_kick_[i]))-dY*np.sin(np.radians(pa_kick_[i])))/(1.-ell_kick_[i]))**2.+ (dX*np.sin(np.radians(pa_kick_[i]))+dY*np.cos(np.radians(pa_kick_[i])))**2.)
    L = (np.exp(-ri/re_kick_[i]))
    print(re_kick_arcmin[i], ra0_[i], dec0_[i])
    a = plt.hist(ri, bins=40, range=(0,2.))
    plt.plot(radius, np.max(a[0]) * L_radius)
    plt.show()

