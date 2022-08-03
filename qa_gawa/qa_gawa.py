import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as fits
import matplotlib as mpl
from astropy import units as u
import astropy.coordinates as coord
# from astropy.coordinates import SkyCoord


def radec2GCdist(ra, dec, dist_kpc):
    """
    Return Galactocentric distance from ra, dec, D_sun_kpc.

    Parameters
    ----------
    ra, dec : float or list
        Coordinates of the objects (in deg)
    dist_kpc : float or list
        Distance in kpc of the objects

    Returns
    -------
    float of list
        the Galactocentric distance to the object[s]
    """

    c1 = coord.SkyCoord(
        ra=ra * u.degree, dec=dec * u.degree, distance=dist_kpc * u.kpc, frame="icrs"
    )
    x, y, z = (
        c1.transform_to(coord.Galactocentric).x.value,
        c1.transform_to(coord.Galactocentric).y.value,
        c1.transform_to(coord.Galactocentric).z.value,
    )

    return np.sqrt(x * x + y * y + z * z)


def read_real_cat(cat_DG = "catalogs/objects_in_ref.dat", cat_GC = "catalogs/Harris_updated.dat"):

    ra_DG, dec_DG, dist_kpc_DG, Mv_DG, rhl_pc_DG, FeH_DG = np.loadtxt(
        cat_DG, usecols=(0, 1, 4, 8, 10, 11), unpack=True
    )

    name_DG = np.loadtxt(
        cat_DG, dtype=str, usecols=(2), unpack=True
    )

    #  Catalogo Harris_updated.dat
    # 0-Name 1-L 2-B 3-R_gc 4-Fe/H 5-M-M 6-Mv 7-rhl arcmin
    R_MW_GC, FeH_GC, mM_GC, Mv_GC, rhl_arcmin_GC = np.loadtxt(
        cat_GC, usecols=(3, 4, 5, 6, 7), unpack=True
    )
    
    dist_kpc_GC = 10 ** ((mM_GC / 5) - 2)
    
    rhl_pc_GC = 1000 * dist_kpc_GC * np.tan(rhl_arcmin_GC / (60 * 180 / np.pi))
    
    name_GC = np.loadtxt(
        cat_GC, dtype=str, usecols=(0), unpack=True
    )

    return name_DG, ra_DG, dec_DG, dist_kpc_DG, Mv_DG, rhl_pc_DG, FeH_DG, name_GC, R_MW_GC, FeH_GC, mM_GC, Mv_GC, rhl_pc_GC, dist_kpc_GC, rhl_arcmin_GC


def plots_ang_size(star_clusters_simulated, unmatch_file, FeH_iso):
    """Plots to analyze the simulated clusters."""

    cmap = mpl.cm.get_cmap("inferno")
    cmap.set_under("dimgray")
    cmap.set_bad("black")
    
    # TODO: Variaveis Instanciadas e não usadas
    hp_sample_un, NSTARS, MAG_ABS_V, NSTARS_CLEAN, MAG_ABS_V_CLEAN, RA_pix, DEC_pix, r_exp, ell, pa, mass, dist = np.loadtxt(star_clusters_simulated, usecols=(0, 1, 2, 4, 5, 9, 10, 11, 12, 13, 14, 15), unpack=True)
    
    name_DG, ra_DG, dec_DG, dist_kpc_DG, Mv_DG, rhl_pc_DG, FeH_DG, name_GC, R_MW_GC, FeH_GC, mM_GC, Mv_GC, rhl_pc_GC, dist_kpc_GC, rhl_arcmin_GC = read_real_cat()
    
    hp_sample_un, NSTARS, MAG_ABS_V, NSTARS_CLEAN, MAG_ABS_V_CLEAN, RA_pix, DEC_pix, r_exp, ell, pa, mass, dist = np.loadtxt(unmatch_file, usecols=(0, 1, 2, 4, 5, 9, 10, 11, 12, 13, 14, 15), unpack=True)
    
    ang_size_DG = 60. * (180. / np.pi) * np.arctan(rhl_pc_DG / (1000. * dist_kpc_DG))
    ang_size = 60 * np.rad2deg(np.arctan(1.7 * r_exp / dist))
    
    RHL_PC_SIM = 1.7 * r_exp

    MW_center_distance_DG_kpc = radec2GCdist(ra_DG, dec_DG, dist_kpc_DG)

    f, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(5, 2, figsize=(15, 23))

    ax1.hist(dist_kpc_DG, bins=np.linspace(0, 2. * np.max(dist) / 1000, 20), label='DG', color='b', alpha=0.5, histtype='stepfilled')
    ax1.hist(dist_kpc_GC, bins=np.linspace(0, 2. * np.max(dist) / 1000, 20), label='GC', color='k', alpha=0.5, lw=2, histtype='step')
    ax1.hist(dist / 1000, bins=np.linspace(0, 2. * np.max(dist) / 1000, 20), label='Sim', color='r', alpha=0.5)
    ax1.legend()
    ax1.set_xlabel("Distance (kpc)")
    ax1.set_ylabel("N objects")
    ax1.set_title('Histogram of distances (linear scale)')
    ax1.set_xlim([0, 2. * np.max(dist) / 1000])

    ax2.hist(dist_kpc_DG, bins=np.linspace(0, 2. * np.max(dist) / 1000, 20), label='DG', color='b', alpha=0.5, histtype='stepfilled')
    ax2.hist(dist_kpc_GC, bins=np.linspace(0, 2. * np.max(dist) / 1000, 20), label='GC', color='k', alpha=0.5, lw=2, histtype='step')
    ax2.hist(dist / 1000, bins=np.linspace(0, 2. * np.max(dist) / 1000, 20), label='Sim', color='r', alpha=0.5)
    ax2.legend()
    ax2.set_title('Histogram of distances (log scale)')
    ax2.set_xlabel("Distance (kpc)")
    ax2.set_ylabel("N objects")
    ax2.set_yscale('log')
    ax2.set_xlim([0, 2. * np.max(dist) / 1000])
    
    ax3.hist(ang_size_DG, bins=np.linspace(np.min(ang_size) / 2, 2. * np.max(ang_size), 20), label='DG', color='b', alpha=0.5, histtype='stepfilled')
    ax3.hist(rhl_arcmin_GC, bins=np.linspace(np.min(ang_size) / 2, 2. * np.max(ang_size), 20), label='GC', color='k', alpha=0.5, lw=2, histtype='step')
    ax3.hist(ang_size, bins=np.linspace(np.min(ang_size) / 2, 2. * np.max(ang_size), 20), label='Sim', color='r', alpha=0.5)
    ax3.legend()
    ax3.set_xlim([np.min(ang_size) / 2, 2. * np.max(ang_size)])
    ax3.set_xlabel(r"$r_{1/2}$ (arcmin)")
    ax3.set_ylabel("N objects")
    ax3.set_title('Histogram of angular sizes (linear scale)')

    ax4.hist(ang_size_DG, bins=np.linspace(np.min(ang_size) / 2, 2. * np.max(ang_size), 20), label='DG', color='b', alpha=0.5, histtype='stepfilled')
    ax4.hist(rhl_arcmin_GC, bins=np.linspace(np.min(ang_size) / 2, 2. * np.max(ang_size), 20), label='GC', color='k', alpha=0.5, lw=2, histtype='step')
    ax4.hist(ang_size, bins=np.linspace(np.min(ang_size) / 2, 2. * np.max(ang_size), 20), label='Sim', color='r', alpha=0.5)
    ax4.legend()
    ax4.set_xlim([np.min(ang_size) / 2, 2. * np.max(ang_size)])
    ax4.set_yscale('log')
    ax4.set_xlabel(r"$r_{1/2}$ (arcmin)")
    ax4.set_ylabel("N objects")
    ax4.set_title('Histogram of angular sizes (log scale)')

    ax5.scatter(dist / 1000, ang_size, label='Sim', color='r')
    ax5.scatter(dist_kpc_DG, ang_size_DG, label='DG', color='b')
    ax5.scatter(dist_kpc_GC, rhl_arcmin_GC, label='GC', color='k')
    ax5.set_xlabel("Distance (kpc)")
    ax5.set_ylabel(r"$r_{1/2}$ (arcmin)")
    ax5.set_yscale('log')
    ax5.legend()
    ax5.set_title('Distances X Angular sizes')
    
    for i, j in enumerate(mass):
        if MAG_ABS_V[i] < 0.0:
            ax6.plot([mass[i], mass[i]], [NSTARS[i], NSTARS_CLEAN[i]], color='darkred', lw=0.2)
    ax6.scatter(mass, NSTARS, label='Sim', color='r')
    ax6.scatter(mass, NSTARS_CLEAN, label='Sim filt', color='darkred')
    ax6.set_xlabel("MASS(MSun)")
    ax6.set_ylabel("N stars")
    ax6.legend()
    ax6.set_title('Visible Mass X Star counts')

    ax7.hist(Mv_DG, bins=20, range=(-16, 0.0), histtype="stepfilled", label="DG", color="b", alpha=0.5)
    ax7.hist(Mv_GC, bins=20, range=(-16, 0.0), histtype="step", label="GC", color="k")
    ax7.hist(MAG_ABS_V, bins=20, range=(-16, 0.0), histtype="step", label="Sim", color="r", ls="--", alpha=0.5)
    ax7.hist(MAG_ABS_V_CLEAN, bins=20, range=(-16, 0.0), histtype="stepfilled", label="Sim filt", color="darkred", ls="--", alpha=0.5)
    ax7.set_xlabel(r"$M_V$")
    ax7.set_ylabel("N")
    ax7.legend(loc=2)
    ax7.set_title('Histogram of Absolute Magnitude (V band)')

    ax8.hist(rhl_pc_DG, bins=20, histtype="stepfilled", range=(10, 2400), label="DG", color="b", alpha=0.5)
    ax8.hist(rhl_pc_GC, bins=20, histtype="step", range=(10, 2400), label="GC", color="k")
    ax8.hist(RHL_PC_SIM, bins=20, histtype="stepfilled", range=(10, 2400), label="Sim", color="r", ls="--", alpha=0.5)
    ax8.set_xlabel(r"$r_{1/2}$[pc]")
    ax8.legend(loc=1)
    # ax8.set_xscale('log')
    ax8.set_yscale('log')
    ax8.set_title(r'Histogram of $r_{1/2}$ (parsecs)')

    ax9.hist(np.repeat(FeH_iso, len(MAG_ABS_V)), bins=20, range=(-3, 1.0), histtype="stepfilled", label="Sim", color="r", ls="--", alpha=0.5)
    ax9.hist(FeH_DG, bins=20, range=(-3, 1.0), histtype="stepfilled", label="DG", color="b", alpha=0.5)
    ax9.hist(FeH_GC, bins=20, range=(-3, 1.0), histtype="step", label="GC", color="k")
    ax9.set_xlabel("[Fe/H]")
    ax9.legend(loc=1)
    ax9.set_title('Absolute Magnitude (V band) X Metalicity')
    
    ax10.scatter(dist / 1000, np.repeat(FeH_iso, len(dist)), label="Sim", color="r", marker="x", lw=1.0)
    ax10.scatter(MW_center_distance_DG_kpc, FeH_DG, label="DG", color="b")
    ax10.scatter(R_MW_GC, FeH_GC, label="GC", color="k")
    ax10.set_xlabel("Distance to the Galactic center (kpc)")
    ax10.set_ylabel("[Fe/H]")
    ax10.set_ylim([-3.5, 0])
    ax10.legend()
    ax10.grid()
    ax10.set_title('Galactocentric distances X Metalicity')

    # plt.savefig(output_plots + '/hist_mass.png')
    plt.suptitle("Physical features of 58 Dwarf Gal + 152 GC + " + str(len(hp_sample_un)) + " Simulations", fontsize=16)
    f.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


def general_plots(star_clusters_simulated, unmatch_clusters_file):
    
    name_DG, ra_DG, dec_DG, dist_kpc_DG, Mv_DG, rhl_pc_DG, FeH_DG, name_GC, R_MW_GC, FeH_GC, mM_GC, Mv_GC, rhl_pc_GC, dist_kpc_GC, rhl_arcmin_GC = read_real_cat()

    PIX_sim_un, NSTARS_un, MAG_ABS_V_un, NSTARS_CLEAN_un, MAG_ABS_V_CLEAN_un, RA_un, DEC_un, R_EXP_un, ELL_un, PA_un, MASS_un, DIST_un = np.loadtxt(unmatch_clusters_file, usecols=(0, 1, 2, 4, 5, 9, 10, 11, 12, 13, 14, 15), unpack=True)
    PIX_sim, NSTARS, MAG_ABS_V, NSTARS_CLEAN, MAG_ABS_V_CLEAN, RA, DEC, R_EXP, ELL, PA, MASS, DIST = np.loadtxt(star_clusters_simulated, usecols=(0, 1, 2, 4, 5, 9, 10, 11, 12, 13, 14, 15), unpack=True)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 5))
    ax1.scatter(1.7 * R_EXP[MAG_ABS_V < 0.0], MAG_ABS_V[MAG_ABS_V < 0.0], color='r', label='Sim', alpha=0.2)
    ax1.scatter(1.7 * R_EXP[MAG_ABS_V < 0.0], MAG_ABS_V_CLEAN[MAG_ABS_V < 0.0], color='darkred', label='Sim filt', alpha=0.2)
    ax1.scatter(1.7 * R_EXP_un[MAG_ABS_V_un < 0.0], MAG_ABS_V_CLEAN_un[MAG_ABS_V_un < 0.0], color='olive', label='Undetected')
    ax1.scatter(rhl_pc_DG, Mv_DG, color='b', marker='x', label='DG')
    ax1.scatter(rhl_pc_GC, Mv_GC, color='k', marker='x', label='GC')
    for i, j in enumerate(R_EXP):
        if MAG_ABS_V[i] < 0.0:
            ax1.plot([1.7 * R_EXP[i], 1.7 * R_EXP[i]],
                     [MAG_ABS_V[i], MAG_ABS_V_CLEAN[i]], color='darkred', lw=0.1)
    for i, j in enumerate(rhl_pc_DG):
        ax1.annotate(name_DG[i], (rhl_pc_DG[i], Mv_DG[i]))
    for i, j in enumerate(rhl_pc_GC):
        ax1.annotate(name_GC[i], (rhl_pc_GC[i], Mv_GC[i]))
    ax1.set_ylabel("M(V)")
    ax1.set_xlabel(r"$r_{1/2}$ (pc))")
    ax1.set_xlim([np.min(1.7 * R_EXP[MAG_ABS_V < 0.0]) - 0.1, np.max(1.7 * R_EXP[MAG_ABS_V < 0.0]) + 0.1])
    ax1.set_ylim([np.max(MAG_ABS_V_CLEAN[MAG_ABS_V < 0.0]) + 0.1, np.min(MAG_ABS_V[MAG_ABS_V < 0.0]) - 0.1])
    ax1.set_xscale("log")
    ax1.legend()

    ax2.scatter(1.7 * R_EXP[MAG_ABS_V < 0.0], MAG_ABS_V[MAG_ABS_V < 0.0], color='r', label='Sim', alpha=0.2)
    ax2.scatter(1.7 * R_EXP[MAG_ABS_V < 0.0], MAG_ABS_V_CLEAN[MAG_ABS_V < 0.0], color='darkred', label='Sim filt', alpha=0.2)
    ax2.scatter(1.7 * R_EXP_un[MAG_ABS_V_un < 0.0], MAG_ABS_V_CLEAN_un[MAG_ABS_V_un < 0.0], color='olive', label='Undetected')
    ax2.scatter(rhl_pc_DG, Mv_DG, color='b', marker='x', label='DG')
    ax2.scatter(rhl_pc_GC, Mv_GC, color='k', marker='x', label='GC')
    #for i, j in enumerate(rhl_pc_DG):
    #    ax2.annotate(name_DG[i], (np.log10(rhl_pc_DG[i]), Mv_DG[i]))
    #for i, j in enumerate(rhl_pc_GC):
    #    ax2.annotate(name_GC[i], (np.log10(rhl_pc_GC[i]), Mv_GC[i]))
    ax2.set_xlabel(r"$r_{1/2}$ (pc))")
    ax2.legend()
    ax2.plot(np.logspace(np.log10(1.8), np.log10(1800), 10, endpoint=True),
        np.linspace(1, -14, 10, endpoint=True), color="b", ls=":")
    ax2.plot(np.logspace(np.log10(4.2), np.log10(4200), 10, endpoint=True),
        np.linspace(1, -14, 10, endpoint=True), color="b", ls=":")
    ax2.plot(np.logspace(np.log10(11), np.log10(11000), 10, endpoint=True),
        np.linspace(1, -14, 10, endpoint=True), color="b", ls=":")
    ax2.plot(np.logspace(np.log10(28), np.log10(28000), 10, endpoint=True),
        np.linspace(1, -14, 10, endpoint=True), color="b", ls=":")
    ax2.text(300, -7.9, r"$\mu_V=27\ mag/arcsec$", rotation=45)
    ax2.text(400, -4.2, r"$\mu_V=31\ mag/arcsec$", rotation=45)
    ax2.set_xscale("log")
    ax2.set_xlim([0.4, 4000])
    ax2.set_ylim([1, -14])

    ax3.scatter(MASS, MAG_ABS_V, label='Sim', color='r', alpha=0.2)
    ax3.scatter(MASS, MAG_ABS_V_CLEAN, label='Sim filt', color='darkred', alpha=0.2)
    ax3.scatter(MASS_un, MAG_ABS_V_CLEAN_un, label='Undetected', color='olive')
    for i, j in enumerate(MASS):
        if MAG_ABS_V[i] < 0.0:
            ax3.plot([MASS[i], MASS[i]],
                     [MAG_ABS_V[i], MAG_ABS_V_CLEAN[i]], color='darkred', lw=0.2)
    ax3.set_xlabel("mass(Msun)")
    ax3.set_ylim([np.max(MAG_ABS_V_CLEAN[MAG_ABS_V < 0.0]) + 0.1, np.min(MAG_ABS_V[MAG_ABS_V < 0.0]) - 0.1])
    ax3.legend()
    plt.show()
    plt.close()



def plot_clus_position(unmatch_file, ra_str, dec_str, star_cats_path):

    PIX_sim_un, NSTARS_un, MAG_ABS_V_un, NSTARS_CLEAN_un, MAG_ABS_V_CLEAN_un, RA_un, DEC_un, R_EXP_un, ELL_un, PA_un, MASS_un, DIST_un = np.loadtxt(unmatch_file, usecols=(0, 1, 2, 4, 5, 9, 10, 11, 12, 13, 14, 15), unpack=True)
    
    len_ipix = len(PIX_sim_un)

    # ipix = [int((i.split('/')[-1]).split('.')[0]) for i in ipix_cats]

    ra_cen, dec_cen = RA_un, DEC_un#hp.pix2ang(nside, ipix, nest=True, lonlat=True)

    for i in range(len_ipix):
        
        data = fits.getdata(star_cats_path + '/' + str(int(PIX_sim_un[i])) + '.fits')
        RA_orig = data[ra_str]
        DEC_orig = data[dec_str]
        GC_orig = data['GC']
        st_line_arcsec = 10.
        half_size_plot_dec = 7 * st_line_arcsec / 3600.
        half_size_plot_ra = half_size_plot_dec / np.cos(np.deg2rad(dec_cen[i]))
        
        if len(RA_orig[(RA_orig < ra_cen[i] + half_size_plot_ra) & (RA_orig > ra_cen[i] - half_size_plot_ra) & (DEC_orig < dec_cen[i] + half_size_plot_dec) & (DEC_orig > dec_cen[i] - half_size_plot_dec)]) > 10.:
            
            fig, ax = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
  
            ax[1].set_yticks([])
            ax[2].set_yticks([])

            st_line_arcsec = 10.
            half_size_plot_dec = 7 * st_line_arcsec / 3600.
            half_size_plot_ra = half_size_plot_dec / np.cos(np.deg2rad(dec_cen[i]))

            data = fits.getdata(star_cats_path + '/' + str(int(PIX_sim_un[i])) + '.fits')
            RA = data[ra_str]
            DEC = data[dec_str]
            GC = data['GC']
            col = 0
            ax[col].scatter(RA_orig[(GC_orig == 0)], DEC_orig[(GC_orig == 0)], edgecolor='b', color='None', s=20, label='MW stars')
            ax[col].scatter(
                RA[(GC == 1)], DEC[(GC == 1)], edgecolor='r', color='None', s=20, label='Cl stars')
            ax[col].set_xlim(
                [ra_cen[i] + half_size_plot_ra, ra_cen[i] - half_size_plot_ra])
            ax[col].set_ylim(
                [dec_cen[i] - half_size_plot_dec, dec_cen[i] + half_size_plot_dec])
            ax[col].set_title('HPX {:d}'.format(int(PIX_sim_un[i])), y= 0.9, pad=8, backgroundcolor='w') #{x=ra_cen[i], y=dec_cen[i], pad=8)
            ax[col].legend(loc=3)
            ax[col].scatter(ra_cen[i], dec_cen[i], color='k', s=100, marker='+', label='Cluster center')
            ax[col].set_xlabel('RA (deg)')
            ax[col].set_ylabel('DEC (deg)')
            ax[col].text(ra_cen[i] - half_size_plot_ra + 2. * st_line_arcsec / (np.cos(np.deg2rad(dec_cen[i]))*3600), dec_cen[i] - 0.96 * half_size_plot_dec, '{:d} arcsec'.format(int(st_line_arcsec)), fontsize=8.)
            ax[col].plot([ra_cen[i] - half_size_plot_ra + st_line_arcsec / (np.cos(np.deg2rad(dec_cen[i]))*3600), ra_cen[i] - half_size_plot_ra + 2. * st_line_arcsec / (np.cos(np.deg2rad(dec_cen[i]))*3600)], [dec_cen[i] - 0.9 * half_size_plot_dec, dec_cen[i] - 0.9 * half_size_plot_dec], color='k', lw=1)
            
            col = 1
            st_line_arcsec = 20.
            half_size_plot_dec = 7 * st_line_arcsec / 3600.
            half_size_plot_ra = half_size_plot_dec / np.cos(np.deg2rad(dec_cen[i]))
            ax[col].scatter(RA_orig[(GC_orig == 0)], DEC_orig[(GC_orig == 0)], edgecolor='b', color='None',     s=20, label='MW stars')
            ax[col].scatter(
                RA[(GC == 1)], DEC[(GC == 1)], edgecolor='r', color='None', s=20, label='Cl stars')
            ax[col].set_xlim(
                [ra_cen[i] + half_size_plot_ra, ra_cen[i] - half_size_plot_ra])
            ax[col].set_ylim(
                [dec_cen[i] - half_size_plot_dec, dec_cen[i] + half_size_plot_dec])
            ax[col].set_title('HPX {:d}'.format(int(PIX_sim_un[i])), y= 0.9, pad=8, backgroundcolor='w') #{x=ra_cen[i],     y=dec_cen[i], pad=8)
            ax[col].legend(loc=3)
            ax[col].scatter(ra_cen[i], dec_cen[i], color='k', s=100, marker='+', label='Cluster center')
            ax[col].set_xlabel('RA (deg)')
            ax[col].text(ra_cen[i] - half_size_plot_ra + 2. * st_line_arcsec / (np.cos(np.deg2rad(dec_cen[i]))*3600), dec_cen[i] - 0.96 * half_size_plot_dec, '{:d} arcsec'.format(int(st_line_arcsec)), fontsize=8.)
            ax[col].plot([ra_cen[i] - half_size_plot_ra + st_line_arcsec / (np.cos(np.deg2rad(dec_cen[i]))*3600), ra_cen[i] - half_size_plot_ra + 2. * st_line_arcsec / (np.cos(np.deg2rad(dec_cen[i]))*3600)], [dec_cen[i] - 0.9 * half_size_plot_dec, dec_cen[i] - 0.9 * half_size_plot_dec], color='k', lw=1)

            col = 2
            st_line_arcsec = 30.
            half_size_plot_dec = 7 * st_line_arcsec / 3600.
            half_size_plot_ra = half_size_plot_dec / np.cos(np.deg2rad(dec_cen[i]))
            ax[col].scatter(RA_orig[(GC_orig == 0)], DEC_orig[(GC_orig == 0)], edgecolor='b', color='None',         s=20, label='MW stars')
            ax[col].scatter(
                RA[(GC == 1)], DEC[(GC == 1)], edgecolor='r', color='None', s=20, label='Cl stars')
            ax[col].set_xlim(
                [ra_cen[i] + half_size_plot_ra, ra_cen[i] - half_size_plot_ra])
            ax[col].set_ylim(
                [dec_cen[i] - half_size_plot_dec, dec_cen[i] + half_size_plot_dec])
            ax[col].set_title('HPX {:d}'.format(int(PIX_sim_un[i])), y= 0.9, pad=8, backgroundcolor='w') #{x=ra_cen[i],         y=dec_cen[i], pad=8)
            ax[col].legend(loc=3)
            ax[col].scatter(ra_cen[i], dec_cen[i], color='k', s=100, marker='+', label='Cluster center')
            ax[col].set_xlabel('RA (deg)')
            ax[col].text(ra_cen[i] - half_size_plot_ra + 2. * st_line_arcsec / (np.cos(np.deg2rad(dec_cen[i]))*3600), dec_cen[i] - 0.96 * half_size_plot_dec, '{:d} arcsec'.format(int(st_line_arcsec)), fontsize=8.)
            ax[col].plot([ra_cen[i] - half_size_plot_ra + st_line_arcsec / (np.cos(np.deg2rad(dec_cen[i]))*3600), ra_cen[i] - half_size_plot_ra + 2. * st_line_arcsec / (np.cos(np.deg2rad(dec_cen[i]))*3600)], [dec_cen[i] - 0.9 * half_size_plot_dec, dec_cen[i] - 0.9 * half_size_plot_dec], color='k', lw=1)


            plt.subplots_adjust(wspace=0, hspace=0)
            # plt.savefig(output_dir + '/clusters_with_and_without_crowded_stars.png')
            plt.show()
            # plt.close()


def calc_comp_hist(Mv_sim, Mv_det, log10_rad_sim, log10_rad_det):

    Mmin, Mmax, r_log_min, r_log_max = -11, 2, 1, 3.1
    
    n_bins=13

    H_sim = np.histogram2d(Mv_sim, log10_rad_sim, bins=[n_bins, n_bins],
                range=[[Mmin, Mmax], [r_log_min, r_log_max]])
    H_det = np.histogram2d(Mv_det, log10_rad_det, bins=[n_bins, n_bins],
                range=[[Mmin, Mmax], [r_log_min, r_log_max]])
    H_comp = H_det[0] / H_sim[0]
    return H_comp
    

def full_completeness_distances(Mv_sim, Mv_det, radius_sim, radius_det, dist_sim, dist_sim_det):

    cmap = plt.cm.inferno_r
    cmap.set_bad('lightgray', 1.)
    
    Mmin, Mmax, r_log_min, r_log_max = -11, 2, 1, 3.1

    name_DG, ra_DG, dec_DG, dist_kpc_DG, Mv_DG, rhl_pc_DG, FeH_DG, name_GC, R_MW_GC, FeH_GC, mM_GC, Mv_GC, rhl_pc_GC, dist_kpc_GC, rhl_arcmin_GC = read_real_cat()
    mM_DG = 5 * np.log10(100*dist_kpc_DG)

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16), dpi=100)
    
    mM_sim = 5. * np.log10(dist_sim) - 5.
    mM_det = 5. * np.log10(dist_sim_det) - 5.
        
    cond_sim = (mM_sim > 10.)&(mM_sim < 15.)
    cond_det = (mM_det > 10.)&(mM_det < 15.)
    cond_DG = (mM_DG > 10.)&(mM_DG < 15.)
    cond_GC = (mM_GC > 10.)&(mM_GC < 15.)

    H = calc_comp_hist(Mv_sim[cond_sim], Mv_det[cond_det], np.log10(radius_sim[cond_sim]),
                        np.log10(radius_det[cond_det]))
    ax1.set_title(r'10<$(m-M)_0$<15')
    ax1.set_xlim([Mmin, Mmax])
    ax1.set_ylim([r_log_min, r_log_max])
    ax1.set_xlabel(r'$M_V$')
    ax1.set_ylabel(r'$log_{10}(r_{1/2}[pc])$')
    ax1.grid(True, lw=0.2)
    im1 = ax1.imshow(H.T, extent=[Mmin, Mmax, r_log_min, r_log_max], aspect='auto',
                vmin=0., vmax=1.00, interpolation='None', cmap=cmap)
    ax1.scatter(Mv_GC[cond_GC], np.log10(rhl_pc_GC[cond_GC]), marker='x', color='k')
    ax1.scatter(Mv_DG[cond_DG], np.log10(rhl_pc_DG[cond_DG]), marker='x', color='b')
    for i, j in enumerate(rhl_pc_DG):
        if cond_DG[i]:
            ax1.annotate(name_DG[i], (Mv_DG[i], np.log10(rhl_pc_DG[i])), color='darkmagenta')
    for i, j in enumerate(rhl_pc_GC):
        if cond_GC[i]:
            ax1.annotate(name_GC[i], (Mv_GC[i], np.log10(rhl_pc_GC[i])), color='darkmagenta')


    cond_sim = (mM_sim > 15.)&(mM_sim < 20.)
    cond_det = (mM_det > 15.)&(mM_det < 20.)
    cond_DG = (mM_DG > 15.)&(mM_DG < 20.)
    cond_GC = (mM_GC > 15.)&(mM_GC < 20.)

    H = calc_comp_hist(Mv_sim[cond_sim], Mv_det[cond_det], np.log10(radius_sim[cond_sim]),
                        np.log10(radius_det[cond_det]))
    ax2.set_title(r'15<$(m-M)_0$<20')
    ax2.set_xlim([Mmin, Mmax])
    ax2.set_ylim([r_log_min, r_log_max])
    ax2.set_xlabel(r'$M_V$')
    ax2.set_ylabel(r'$log_{10}(r_{1/2}[pc])$')
    ax2.grid(True, lw=0.2)
    im2 = ax2.imshow(H.T, extent=[Mmin, Mmax, r_log_min, r_log_max], aspect='auto',
                vmin=0., vmax=1.00, interpolation='None', cmap=cmap)
    ax2.scatter(Mv_GC[cond_GC], np.log10(rhl_pc_GC[cond_GC]), marker='x', color='k')
    ax2.scatter(Mv_DG[cond_DG], np.log10(rhl_pc_DG[cond_DG]), marker='x', color='b')
    for i, j in enumerate(rhl_pc_DG):
        if cond_DG[i]:
            ax2.annotate(name_DG[i], (Mv_DG[i], np.log10(rhl_pc_DG[i])), color='darkmagenta')
    for i, j in enumerate(rhl_pc_GC):
        if cond_GC[i]:
            ax2.annotate(name_GC[i], (Mv_GC[i], np.log10(rhl_pc_GC[i])), color='darkmagenta')

    cond_sim = (mM_sim > 20.)&(mM_sim < 25.)
    cond_det = (mM_det > 20.)&(mM_det < 25.)
    cond_DG = (mM_DG > 20.)&(mM_DG < 25.)
    cond_GC = (mM_GC > 20.)&(mM_GC < 25.)

    H = calc_comp_hist(Mv_sim[cond_sim], Mv_det[cond_det], np.log10(radius_sim[cond_sim]),
                        np.log10(radius_det[cond_det]))
    ax3.set_title(r'20<$(m-M)_0$<25')
    ax3.set_xlim([Mmin, Mmax])
    ax3.set_ylim([r_log_min, r_log_max])
    ax3.set_xlabel(r'$M_V$')
    ax3.set_ylabel(r'$log_{10}(r_{1/2}[pc])$')
    ax3.grid(True, lw=0.2)
    im3 = ax3.imshow(H.T, extent=[Mmin, Mmax, r_log_min, r_log_max], aspect='auto',
                vmin=0., vmax=1.00, interpolation='None', cmap=cmap)

    ax3.scatter(Mv_GC[cond_GC], np.log10(rhl_pc_GC[cond_GC]), marker='x', color='k')
    ax3.scatter(Mv_DG[cond_DG], np.log10(rhl_pc_DG[cond_DG]), marker='x', color='b')
    for i, j in enumerate(rhl_pc_DG):
        if cond_DG[i]:
            ax3.annotate(name_DG[i], (Mv_DG[i], np.log10(rhl_pc_DG[i])), color='darkmagenta')
    for i, j in enumerate(rhl_pc_GC):
        if cond_GC[i]:
            ax3.annotate(name_GC[i], (Mv_GC[i], np.log10(rhl_pc_GC[i])), color='darkmagenta')
    cond_sim = (mM_sim > 25.)&(mM_sim < 30.)
    cond_det = (mM_det > 25.)&(mM_det < 30.)
    cond_DG = (mM_DG > 25.)&(mM_DG < 30.)
    cond_GC = (mM_GC > 25.)&(mM_GC < 30.)

    H = calc_comp_hist(Mv_sim[cond_sim], Mv_det[cond_det], np.log10(radius_sim[cond_sim]),
                        np.log10(radius_det[cond_det]))
    ax4.set_title(r'25<$(m-M)_0$<30')
    ax4.set_xlim([Mmin, Mmax])
    ax4.set_ylim([r_log_min, r_log_max])
    ax4.set_xlabel(r'$M_V$')
    ax4.set_ylabel(r'$log_{10}(r_{1/2}[pc])$')
    ax4.grid(True, lw=0.2)
    im4 = ax4.imshow(H.T, extent=[Mmin, Mmax, r_log_min, r_log_max], aspect='auto',
                vmin=0., vmax=1.00, interpolation='None', cmap=cmap)
    ax4.scatter(Mv_GC[cond_GC], np.log10(rhl_pc_GC[cond_GC]), marker='x', color='k')
    ax4.scatter(Mv_DG[cond_DG], np.log10(rhl_pc_DG[cond_DG]), marker='x', color='b')
    for i, j in enumerate(rhl_pc_DG):
        if cond_DG[i]:
            ax4.annotate(name_DG[i], (Mv_DG[i], np.log10(rhl_pc_DG[i])), color='darkmagenta')
    for i, j in enumerate(rhl_pc_GC):
        if cond_GC[i]:
            ax4.annotate(name_GC[i], (Mv_GC[i], np.log10(rhl_pc_GC[i])), color='darkmagenta')


    cbaxes = f.add_axes([0.90, 0.126, 0.01, 0.755])
    cbar = f.colorbar(im3, cax=cbaxes, cmap=cmap, orientation='vertical', label='Completeness')
    plt.subplots_adjust(wspace=0.2)
    plt.show()




def plot_pure(arg_all, arg_conf, label, title, bins=20):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
    over = (max(np.max(arg_all), np.max(arg_conf)) - min(np.min(arg_all), np.min(arg_conf))) * 0.1
    min_ = min(np.min(arg_all), np.min(arg_conf)) - over
    max_ = max(np.max(arg_all), np.max(arg_conf)) + over
    try:
        A = ax1.hist(arg_all, bins=bins, range=(bins[0], bins[-1]), histtype='step', lw=2, label='All detections')
        B = ax1.hist(arg_conf, bins=bins, range=(bins[0], bins[-1]), histtype='stepfilled', lw=2, label='True clusters')
        pureness = B[0] / A[0]
        ax1.set_xlabel(label)
        ax1.set_ylabel('Number of clusters detected')
        ax1.set_xlim([min_, max_])
        ax1.legend(loc=2)
    
        ax2.step(bins, np.append(pureness[0], pureness), 'r', lw=2, label='Data')
        # ax2.step(A[1][0:-1],pureness, label='Data', color='k')
        ax2.set_xlabel(label)
        ax2.set_ylabel('Purity')
        ax2.set_ylim([0,1.2])
        ax2.set_xlim([min_, max_])
        ax2.legend()
        fig.suptitle(title)
        plt.show()
    except:
        A = ax1.hist(arg_all, bins=bins, range=(min_, max_), histtype='step', lw=2, label='All detections')
        B = ax1.hist(arg_conf, bins=bins, range=(min_, max_), histtype='stepfilled', lw=2, label='True clusters')
        pureness = B[0] / A[0]
        ax1.set_xlabel(label)
        ax1.set_ylabel('Number of clusters detected')
        ax1.set_xlim([min_, max_])
        ax1.legend(loc=2)

        ax2.step(A[1][0:-1], np.nan_to_num(pureness), 'r', lw=2, label='Data')
        ax2.set_xlabel(label)
        ax2.set_ylabel('Purity')
        ax2.set_ylim([0,1.2])
        ax2.set_xlim([min_, max_])
        ax2.legend()
        fig.suptitle(title)
        plt.show()
        
        
def plot_comp(arg, idxs, label, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
    bins = 20
    step = (np.max(arg) - np.min(arg)) / bins
    A = ax1.hist(arg[idxs], bins=bins, range=(np.min(arg), np.max(arg)), histtype='step', \
                 color = "r", label='Detections')
    B = ax1.hist(arg, bins=bins, range=(np.min(arg), np.max(arg)), histtype='step', \
                 color = "mediumblue", label='Simulated clusters')
    completeness = A[0] / B[0]
    completeness[completeness >= 1.] = 1.
    # Only to set steps equal to zero where the completeness does not have results.
    # Warning: the values replaced by zero are those ones where the completeness in undetermined.
    compl = np.append(0., np.nan_to_num(completeness))
    ax1.set_xlabel(label)
    ax1.set_ylabel( '# Detected Clusters')
    ax1.legend()

    ax2.step(np.append(A[1][0] - step, A[1]), np.append(compl, 0), 'k', label='Data', where='mid')
    ax2.set_xlabel(label)
    ax2.set_ylabel('Completeness')
    ax2.set_ylim([0,1.1])
    ax2.legend()
    fig.suptitle(title)
    plt.show()
