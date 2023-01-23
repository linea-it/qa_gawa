import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as fits
from astropy.io.fits import getdata
import matplotlib as mpl
from astropy import units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.io import ascii
import tabulate


def calc_N_hist(Mv_sim, log10_rad_sim):
    """Calculates completeness 2D histogram regarding absolute magnitude and half-light radii.

    Parameters
    ----------
    Mv_sim : list
        Absolute magnitude of simulated systems in V band.
    log10_rad_sim : list
        10-log of half-light radii of simulated systems, in parsecs.

    Returns
    -------
    array-like
        2d histogram of simulations in plane absolute magnitude x half-light radii.
    """
    Mmin, Mmax, r_log_min, r_log_max = -11, 2, 1, 3.1

    n_bins = 13

    H_sim = np.histogram2d(Mv_sim, log10_rad_sim, bins=[n_bins, n_bins],
                           range=[[Mmin, Mmax], [r_log_min, r_log_max]])

    return H_sim[0]


def full_N_distances(Mv_sim, radius_sim, dist_sim):
    """Calculates and show the completeness of detections (in 2D histogram)
    in four bins of distance.

    Parameters
    ----------
    Mv_sim : list
        Absolute magnitude of simulations.
    radius_sim : list
        Half-light radius of simulations.
    dist_sim : list
        Distance of simulations.
    """
    cmap = plt.cm.Blues
    cmap.set_bad('lightgray', 0.0)
    cmap.set_under('lightgray', .01)

    Mmin, Mmax, r_log_min, r_log_max = -11, 2, 1, 3.1

    name_DG, ra_DG, dec_DG, dist_kpc_DG, Mv_DG, rhl_pc_DG, FeH_DG, name_GC, R_MW_GC, FeH_GC, mM_GC, Mv_GC, rhl_pc_GC, dist_kpc_GC, rhl_arcmin_GC = read_real_cat()
    mM_DG = 5 * np.log10(100*dist_kpc_DG)

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16), dpi=100)

    mM_sim = 5. * np.log10(dist_sim) - 5.

    cond_sim = (mM_sim > 10.) & (mM_sim < 15.)
    H1 = calc_N_hist(Mv_sim[cond_sim], np.log10(radius_sim[cond_sim]))

    cond_sim = (mM_sim > 15.) & (mM_sim < 20.)
    H2 = calc_N_hist(Mv_sim[cond_sim], np.log10(radius_sim[cond_sim]))
    
    cond_sim = (mM_sim > 20.) & (mM_sim < 25.)
    H3 = calc_N_hist(Mv_sim[cond_sim], np.log10(radius_sim[cond_sim]))
    
    cond_sim = (mM_sim > 25.) & (mM_sim < 30.)
    H4 = calc_N_hist(Mv_sim[cond_sim], np.log10(radius_sim[cond_sim]))

    N_max_1 = np.max(np.maximum(H1, H2))
    N_max_2 = np.max(np.maximum(H3, H4))
    N_max = max(N_max_1, N_max_2)

    cond_sim = (mM_sim > 10.) & (mM_sim < 15.)
    cond_DG = (mM_DG > 10.) & (mM_DG < 15.)
    cond_GC = (mM_GC > 10.) & (mM_GC < 15.)

    ax1.set_title(r'10<$(m-M)_0$<15')
    ax1.set_xlim([Mmin, Mmax])
    ax1.set_ylim([r_log_min, r_log_max])
    ax1.set_xlabel(r'$M_V$')
    ax1.set_ylabel(r'$log_{10}(r_{1/2}[pc])$')
    ax1.grid(True, lw=0.2)
    im1 = ax1.imshow(np.flipud(H1.T), extent=[Mmin, Mmax, r_log_min, r_log_max], aspect='auto',
                     vmin=0., interpolation='None', cmap=cmap)
    ax1.scatter(Mv_GC[cond_GC], np.log10(
        rhl_pc_GC[cond_GC]), marker='x', color='k', label='GC')
    ax1.scatter(Mv_DG[cond_DG], np.log10(
        rhl_pc_DG[cond_DG]), marker='x', color='b', label='DG')
    for i, j in enumerate(rhl_pc_DG):
        if cond_DG[i]:
            ax1.annotate(name_DG[i], (Mv_DG[i], np.log10(
                rhl_pc_DG[i])), color='darkmagenta')
    for i, j in enumerate(rhl_pc_GC):
        if cond_GC[i]:
            ax1.annotate(name_GC[i], (Mv_GC[i], np.log10(
                rhl_pc_GC[i])), color='darkmagenta')

    cond_sim = (mM_sim > 15.) & (mM_sim < 20.)
    cond_DG = (mM_DG > 15.) & (mM_DG < 20.)
    cond_GC = (mM_GC > 15.) & (mM_GC < 20.)

    ax2.set_title(r'15<$(m-M)_0$<20')
    ax2.set_xlim([Mmin, Mmax])
    ax2.set_ylim([r_log_min, r_log_max])
    ax2.set_xlabel(r'$M_V$')
    ax2.set_ylabel(r'$log_{10}(r_{1/2}[pc])$')
    ax2.grid(True, lw=0.2)
    im2 = ax2.imshow(np.flipud(H2.T), extent=[Mmin, Mmax, r_log_min, r_log_max], aspect='auto',
                     vmin=0., vmax=N_max, interpolation='None', cmap=cmap)
    ax2.scatter(Mv_GC[cond_GC], np.log10(
        rhl_pc_GC[cond_GC]), marker='x', color='k', label='GC')
    ax2.scatter(Mv_DG[cond_DG], np.log10(
        rhl_pc_DG[cond_DG]), marker='x', color='b', label='DG')
    for i, j in enumerate(rhl_pc_DG):
        if cond_DG[i]:
            ax2.annotate(name_DG[i], (Mv_DG[i], np.log10(
                rhl_pc_DG[i])), color='darkmagenta')
    for i, j in enumerate(rhl_pc_GC):
        if cond_GC[i]:
            ax2.annotate(name_GC[i], (Mv_GC[i], np.log10(
                rhl_pc_GC[i])), color='darkmagenta')

    cond_sim = (mM_sim > 20.) & (mM_sim < 25.)
    cond_DG = (mM_DG > 20.) & (mM_DG < 25.)
    cond_GC = (mM_GC > 20.) & (mM_GC < 25.)

    ax3.set_title(r'20<$(m-M)_0$<25')
    ax3.set_xlim([Mmin, Mmax])
    ax3.set_ylim([r_log_min, r_log_max])
    ax3.set_xlabel(r'$M_V$')
    ax3.set_ylabel(r'$log_{10}(r_{1/2}[pc])$')
    ax3.grid(True, lw=0.2)
    im3 = ax3.imshow(np.flipud(H3.T), extent=[Mmin, Mmax, r_log_min, r_log_max], aspect='auto',
                     vmin=0., vmax=N_max, interpolation='None', cmap=cmap)

    ax3.scatter(Mv_GC[cond_GC], np.log10(
        rhl_pc_GC[cond_GC]), marker='x', color='k', label='GC')
    ax3.scatter(Mv_DG[cond_DG], np.log10(
        rhl_pc_DG[cond_DG]), marker='x', color='b', label='DG')
    for i, j in enumerate(rhl_pc_DG):
        if cond_DG[i]:
            ax3.annotate(name_DG[i], (Mv_DG[i], np.log10(
                rhl_pc_DG[i])), color='darkmagenta')
    for i, j in enumerate(rhl_pc_GC):
        if cond_GC[i]:
            ax3.annotate(name_GC[i], (Mv_GC[i], np.log10(
                rhl_pc_GC[i])), color='darkmagenta')
    cond_sim = (mM_sim > 25.) & (mM_sim < 30.)
    cond_DG = (mM_DG > 25.) & (mM_DG < 30.)
    cond_GC = (mM_GC > 25.) & (mM_GC < 30.)

    ax4.set_title(r'25<$(m-M)_0$<30')
    ax4.set_xlim([Mmin, Mmax])
    ax4.set_ylim([r_log_min, r_log_max])
    ax4.set_xlabel(r'$M_V$')
    ax4.set_ylabel(r'$log_{10}(r_{1/2}[pc])$')
    ax4.grid(True, lw=0.2)
    im4 = ax4.imshow(np.flipud(H4.T), extent=[Mmin, Mmax, r_log_min, r_log_max], aspect='auto',
                     vmin=0., vmax=N_max, interpolation='None', cmap=cmap)
    ax4.scatter(Mv_GC[cond_GC], np.log10(
        rhl_pc_GC[cond_GC]), marker='x', color='k', label='GC')
    ax4.scatter(Mv_DG[cond_DG], np.log10(
        rhl_pc_DG[cond_DG]), marker='x', color='b', label='DG')
    for i, j in enumerate(rhl_pc_DG):
        if cond_DG[i]:
            ax4.annotate(name_DG[i], (Mv_DG[i], np.log10(
                rhl_pc_DG[i])), color='darkmagenta')
    for i, j in enumerate(rhl_pc_GC):
        if cond_GC[i]:
            ax4.annotate(name_GC[i], (Mv_GC[i], np.log10(
                rhl_pc_GC[i])), color='darkmagenta')

    cbaxes = f.add_axes([0.90, 0.126, 0.01, 0.755])
    cbar = f.colorbar(im3, cax=cbaxes, cmap=cmap,
                      orientation='vertical', label='Sample of simulations')
    plt.suptitle('Clusters simulated')
    plt.subplots_adjust(wspace=0.2)
    plt.show()


def undet_cmds(unmatch_file, mask_file, input_simulation_path, input_detection_path, param2):
    """This function creates plots for the candidates that were not detected.

    Parameters
    ----------
    unmatch_file : str
        File name of the undetected clusters.
    mask_file : str
        File name of the isochronal masks.
    input_simulation_path : str
        Path to the file with the simulations table.
    input_detection_path : str
        Path to the file with the detection table.
    param2 : dict
        Dictionary with the parameters for the detection.
    """
    HPX64, N = np.loadtxt(unmatch_file, usecols=(0, 1), unpack=True)

    n_col = 5
    n_row = int(len(N) / n_col)
    n_reminder = len(N) % n_col

    gr, g = np.loadtxt(mask_file, usecols=(0, 1), unpack=True)

    # fig, axs = plt.subplots(n_row, n_col, figsize=(16, 4 * n_row), dpi=100.)

    slices_file = input_detection_path + '/dslices.fits'
    data_sl = getdata(slices_file)
    d_slices_pc = data_sl["dist_pc"]
    mM_slices = 5 * np.log10(d_slices_pc) - 5.
    to_be_shown = 20

    for i in range(n_row):

        if i != n_row:

            for j in range(n_col):

                idx_obj = i * n_col + j
        
                fig, axx = plt.subplots(n_col, 1, figsize=(20, 4), dpi=150.)

                for ax in axx.flat:
                    ax.set_title('HPX: {:d}'.format(int(HPX64[idx_obj])))
                    data = fits.getdata(input_simulation_path +
                                '/hpx_cats_clean/' + str(int(HPX64[idx_obj])) + '.fits')
                    MAGG = data['mag_g_with_err']
                    MAGR = data['mag_r_with_err']
                    GC = data['GC']
                    ax.scatter(MAGG[GC == 0] - MAGR[GC == 0], MAGG[GC == 0],
                               color='lightgrey', label='MW', s=0.1)
                    ax.scatter(MAGG[GC == 1] - MAGR[GC == 1], MAGG[GC == 1],
                               color='r', label='Cluster', s=0.4)

                    for ii in range(len(mM_slices)):
                        ax.plot(gr, g + mM_slices[ii], label='m-M={:.2f}'.format(mM_slices[ii]), lw=1)
                    ax.set_xlim(param2['isochrone_masks'][param2['survey']]['mask_color_min'],
                            param2['isochrone_masks'][param2['survey']]['mask_color_max'])
                    ax.set_ylim(param2['isochrone_masks'][param2['survey']]['mask_mag_max'],
                            param2['isochrone_masks'][param2['survey']]['mask_mag_min'])
                    ax.set_xlabel(r'$g_0-r_0$')
                    ax.set_ylabel(r'$g_0$')
                    plt.tight_layout()
                    if i < to_be_shown:
                        plt.show()
                    else:
                        # plt.savefig(str(HPX64[idx_obj]) + '.png')
                        # plt.close()
                        pass

        else:
            for j in n_reminder:

                idx_obj = i * n_row + j
        
                fig, axx = plt.subplots(n_reminder, 1, figsize=(20, 4), dpi=150.)

                for ax in axx.flat:
                    ax.set_title('HPX: {:d}'.format(int(HPX64[idx_obj])))
                    data = fits.getdata(input_simulation_path +
                                '/hpx_cats_clean/' + str(int(HPX64[idx_obj])) + '.fits')
                    MAGG = data['mag_g_with_err']
                    MAGR = data['mag_r_with_err']
                    GC = data['GC']
                    ax.scatter(MAGG[GC == 0] - MAGR[GC == 0], MAGG[GC == 0],
                               color='lightgrey', label='MW', s=0.1)
                    ax.scatter(MAGG[GC == 1] - MAGR[GC == 1], MAGG[GC == 1],
                               color='r', label='Cluster', s=0.4)

                    for ii in range(len(mM_slices)):
                        ax.plot(gr, g + mM_slices[ii], label='m-M={:.2f}'.format(mM_slices[ii]), lw=1)
                    ax.set_xlim(param2['isochrone_masks'][param2['survey']]['mask_color_min'],
                            param2['isochrone_masks'][param2['survey']]['mask_color_max'])
                    ax.set_ylim(param2['isochrone_masks'][param2['survey']]['mask_mag_max'],
                            param2['isochrone_masks'][param2['survey']]['mask_mag_min'])
                    ax.set_xlabel(r'$g_0-r_0$')
                    ax.set_ylabel(r'$g_0$')
                    plt.tight_layout()
                    if i < to_be_shown:
                        plt.show()
                    else:
                        # plt.savefig(str(HPX64[idx_obj]) + '.png')
                        # plt.close()
                        pass



def print_undet_table(unmatch_file, n):
    """This function plots the table of undetected clusters as an html table.

    Parameters
    ----------
    unmatch_file : str
        File name of the undetected clusters.

    Returns
    -------
    object
        Html table
    """
    with open(unmatch_file) as f:
        first_line = f.readline()

    HPX64 = np.loadtxt(unmatch_file, usecols=(0), dtype=int, unpack=True)

    table = tabulate.tabulate(np.loadtxt(unmatch_file)[:][0:n],
                              tablefmt='html',
                              headers=(first_line[1:].split()))

    print('Total of clusters not detected: {:d}\n'.format(len(HPX64)))

    return table


def plot_pure_SNR(match_file, SNR_min):
    """This function plots the purity wrt SNR.

    Parameters
    ----------
    match_file : str
        File name of the detected clusters.
    """

    SNR_det, SNR_sim, det = np.loadtxt(match_file, usecols=(12, 28, 38), unpack=True)

    SNR_sim = SNR_sim[SNR_det > SNR_min]
    SNR_det = SNR_det[SNR_det > SNR_min]
    # SNR_sim = SNR_sim[SNR_det > SNR_min]

    true_positive = (det == 1.)

    plot_pure(SNR_det, [SNR_det[i] for i in true_positive if true_positive],
              ' SNR from detections',
              'Purity wrt SNR (detections) > {:.2f}'.format(SNR_min))


def plot_pure_mM(input_detection_path, match_file):
    """_summary_

    Parameters
    ----------
    input_detection_path : str
        Path to the file with the detection table.
    match_file : str
       File name of the detected clusters.
    """

    SNR_sim, det = np.loadtxt(match_file, usecols=(28, 38), unpack=True)

    slices_file = input_detection_path + '/dslices.fits'
    data_sl = getdata(slices_file)
    d_slices_pc = data_sl["dist_pc"]
    mM_slices = 5 * np.log10(d_slices_pc) - 5.

    bin_size_mM = mM_slices[1] - mM_slices[0]
    bins_mM = np.linspace(mM_slices[0] - bin_size_mM / 2, mM_slices[-1] +
                          bin_size_mM / 2, len(mM_slices) + 1, endpoint=True)

    true_positive = (det == 1.)

    dist_kpc_det = np.loadtxt(match_file, usecols=(5), unpack=True)

    m_M_det = 5 * np.log10(dist_kpc_det) + 10.

    plot_pure(m_M_det, m_M_det[true_positive], 'Detection distance module',
              'Purity wrt Distance Modulus (detection)', bins_mM)


def puri_comp(input_detection_path, input_simulation_path, match_file, unmatch_file):
    """This function calculates and plots the completeness and purity regargind SNR.

    Parameters
    ----------
    input_simulation_path : str
        Path to the file with the simulations table.
    match_file : str
       File name of the detected clusters.
    """
    SNR_det, SNR_sim, det = np.loadtxt(match_file, usecols=(12, 28, 38), unpack=True)

    SNR_sim_all, ra_sim = np.loadtxt(input_simulation_path + '/star_clusters_simulated.dat',
                        usecols=(6, 9), unpack=True)

    true_positive = (det == 1.)

    SNR_range = np.arange(0., np.max(SNR_sim), 1.)

    comp_wrt_SNR = np.zeros(len(SNR_range)-1)
    pur_wrt_SNR = np.zeros(len(SNR_range)-1)

    for k, i, j in zip(range(len(SNR_range)-1), SNR_range[0:-1], SNR_range[1:]):
        # put here a constrint to minimum and maximum
        if len(SNR_det[(SNR_det > i) & (SNR_det < j)]) > 0:
            pur_wrt_SNR[k] = len(SNR_det[(true_positive) & (
            SNR_det > i) & (SNR_det < j)]) / len(SNR_det[(SNR_det > i) & (SNR_det < j)])
        if len(SNR_sim[(SNR_sim > i)&(SNR_sim < j)]) > 0:
            comp_wrt_SNR[k] = len(SNR_sim[(true_positive) & (
            SNR_det > i) & (SNR_det < j)]) / len(SNR_sim_all[(SNR_sim_all > i)&(SNR_sim_all < j)])

    comp_wrt_SNR[comp_wrt_SNR > 1.] = 1.

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(SNR_range[:-1], comp_wrt_SNR, label='Completeness', color='r', lw=2)
    ax.plot(SNR_range[:-1], pur_wrt_SNR, label='Purity', color='b', lw=2)
    ax.set_xlim([0., 1.1 * np.max(SNR_range)])
    ax.set_ylim([0, 1.1])
    ax.set_title('Purity/Completeness versus SNR')
    ax.set_xlabel('SNR')
    ax.set_ylabel('Completeness / Purity')
    ax.legend()
    plt.show()
    '''
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(SNR_range, comp_wrt_SNR, label='Completeness', color='r', lw=2)
    ax.plot(SNR_range, pur_wrt_SNR, label='Purity', color='b', lw=2)
    ax.set_xlim([3., 10.])
    ax.set_ylim([0.9, 1.05])
    ax.set_title('Purity/Completeness versus SNR')
    ax.set_xlabel('SNR from detections')
    ax.set_ylabel('Completeness / Purity')
    ax.legend()
    plt.show()
    '''


def plot_comp_all(input_simulation_path, match_file, idx_sim):
    """_summary_

    Parameters
    ----------
    input_simulation_path : str
        Path to the file with the simulations table.
    match_file : str
        File name of the detected clusters.
    idx_sim : list
        List with indexes of the simulated clusters that match detections.
    """

    M_V_sim_det = np.loadtxt(match_file, usecols=(24), unpack=True)

    M_V, SNR, r_exp_pc, dist = np.loadtxt(input_simulation_path + '/star_clusters_simulated.dat',
                                          usecols=(5, 6, 11, 15), unpack=True)

    plot_comp(M_V, idx_sim, 'M_V', 'Absolute Magnitude in V band')
    plot_comp(dist, idx_sim, 'd (pc) simulated',
              'Completeness wrt Distance (simulations)')
    plot_comp(SNR, idx_sim, 'SNR from simulations', 'Completeness wrt SNR from simulations')
    mM_sim = 5 * np.log10(dist) - 5.

    plot_comp(mM_sim, idx_sim, 'm-M', 'Completeness wrt Distance modulus')
    exp_rad_sim_det, M_V_sim_det, dist_sim_det = np.loadtxt(
        match_file, usecols=(33, 27, 37), unpack=True)
    
    full_N_distances(M_V, 1.7 * r_exp_pc, dist)
    full_completeness_distances(
        M_V, M_V_sim_det, 1.7 * r_exp_pc, 1.7 * exp_rad_sim_det, dist, dist_sim_det)


def SNR_SNR(match_file):
    """Plots SNR of detections versus SNR from simulations.

    Parameters
    ----------
    match_file : str
       File name of the detected clusters.
    """

    SNR_det, SNR_sim = np.loadtxt(match_file, usecols=(12, 28), unpack=True)

    fig = plt.figure(figsize=(16, 10))
    plt.scatter(SNR_sim, SNR_det)
    plt.plot(np.linspace(0., 1.1 * max(np.max(SNR_sim), np.max(SNR_det)), 4),
             np.linspace(0., 1.1 * max(np.max(SNR_sim), np.max(SNR_det)), 4), color='r')
    plt.xlabel('SNR (simulations)')
    plt.ylabel('SNR (detections)')
    plt.xlim([0.1, 1.05 * max(np.max(SNR_sim), np.max(SNR_det))])
    plt.ylim([0.1, 1.05 * max(np.max(SNR_sim), np.max(SNR_det))])
    plt.show()


def dist_dist(match_file):
    """Plots distance of detections versus distance of simulations.

    Parameters
    ----------
    match_file : str
       File name of the detected clusters.
    """

    dist_init_kpc_det, dist_err_kpc_det, SNR_sim, dist_sim, det = np.loadtxt(match_file,
                                                                        usecols=(
                                                                            5, 6, 28, 37, 38),
                                                                        unpack=True)
    true_positive = (det == 1.)

    dist_sim_kpc = dist_sim / 1000
    fig = plt.figure(figsize=(16, 10))
    # plt.errorbar(dist_sim_kpc[true_positive], dist_init_kpc_det[true_positive],
    #              yerr=dist_err_kpc_det[true_positive], xerr=None,  fmt='o', c='k')
    plt.plot(np.linspace(0.8 * min(np.min(dist_sim_kpc), np.min(dist_init_kpc_det)), max(np.max(dist_sim_kpc), np.max(dist_init_kpc_det)), 4),
             np.linspace(0.8 * min(np.min(dist_sim_kpc), np.min(dist_init_kpc_det)), max(np.max(dist_sim_kpc), np.max(dist_init_kpc_det)), 4), color='r')
    dist_sim_kpc_bin = np.linspace(np.min(dist_sim_kpc), np.max(dist_sim_kpc), 6, endpoint=True)
    for ii in range(len(dist_sim_kpc_bin)-1):
        plt.violinplot(dist_init_kpc_det[(dist_sim_kpc > dist_sim_kpc_bin[ii])&(dist_init_kpc_det < dist_sim_kpc_bin[ii+1])], [(dist_sim_kpc_bin[ii] + dist_sim_kpc_bin[ii+1]) / 2.], points=100, widths=100., showmeans=True, showextrema=True, showmedians=True, quantiles=[0.05, 0.25, 0.75, 0.95], bw_method=0.5)
    plt.xlim([0.8 * min(np.min(dist_sim_kpc), np.min(dist_init_kpc_det)),
             max(np.max(dist_sim_kpc), np.max(dist_init_kpc_det))])
    plt.ylim([0.8 * min(np.min(dist_sim_kpc), np.min(dist_init_kpc_det)),
             max(np.max(dist_sim_kpc), np.max(dist_init_kpc_det))])
    plt.title('Comparing recovery distances (0.05 / 0.25 / 0.75 / 0.95)')
    plt.xlabel('Distances (kpc) from simulations')
    plt.ylabel('Distances (kpc) from detections')
    plt.show()


def det_sky(input_simulation_path, match_file, unmatch_file):
    """Plots the detections into the sky, color-coded by SNR. Also shows the undetected
    clusters as empty circles.

    Parameters
    ----------
    input_simulation_path : str
        Path to the file with the simulations table.
    match_file : str
       File name of the detected clusters.
    unmatch_file : str
        File name of the undetected clusters.
    """
    ra_det, dec_det, SNR_det, HPX64, SNR_sim, detected = np.loadtxt(
        match_file, usecols=(1, 2, 12, 22, 28, 38), unpack=True)

    SNR_sim_all, ra_sim, dec_sim = np.loadtxt(
        input_simulation_path + '/star_clusters_simulated.dat', usecols=(6, 9, 10), unpack=True)

    SNR_undet, ra_undet, dec_undet = np.loadtxt(
        unmatch_file, usecols=(6, 9, 10), unpack=True)

    true_positive = (detected == 1)
    false_positive = (detected == 0)

    cm = plt.cm.get_cmap('copper_r')
    '''
    fig = plt.figure(figsize=(30, 12))
    plt.scatter(ra_sim, dec_sim, c=SNR_sim_all, vmin=0, vmax=np.max(SNR_det), cmap=cm, s=10.0, marker='^',
                label='Simulations: ({:d})'.format(len(ra_sim)))
    sc = plt.scatter(ra_det[true_positive], dec_det[true_positive], c=SNR_det[true_positive], vmin=0, vmax=np.max(SNR_det), marker='x', s=10.,
                     cmap=cm, label='True Positives: ({:d})'.format(len(np.unique(HPX64[true_positive]))))
    plt.scatter(ra_det[false_positive], dec_det[false_positive], c=SNR_det[false_positive], s=100.0, cmap=cm,
                lw=2, alpha=0.75, label='Not matched: ({:d})'.format(len(ra_det[false_positive])))
    plt.scatter(ra_undet, dec_undet, color='None', edgecolor='k', s=200.0,
                lw=2, alpha=0.75, label='Not detected: ({:d})'.format(len(ra_undet)))
    plt.colorbar(sc, label='SNR detection')
    plt.xlim(np.max(ra_sim)+0.5, np.min(ra_sim)-0.5)
    plt.ylim(np.min(dec_sim)-0.5, np.max(dec_sim)+1.0)
    plt.xlabel('RA')
    plt.ylabel('DEC')
    plt.title('Spatial distribution of clusters (SN > 3.)')
    plt.legend(loc=1)
    plt.show()
    '''
    fig = plt.figure(figsize=(30, 12))
    plt.scatter(ra_sim, dec_sim, c=SNR_sim_all, vmin=0, vmax=np.max(SNR_det), cmap=cm, s=10.0, marker='^',
                label='Simulations: ({:d})'.format(len(ra_sim)))
    cond = (SNR_det > 5)
    sc = plt.scatter(ra_det[(true_positive) & (cond)], dec_det[(true_positive) & (cond)],
                     c=SNR_det[(true_positive) & (cond)], vmin=0, vmax=np.max(SNR_det), marker='x', s=10.,
                     cmap=cm, label='True Positives: ({:d})'.format(len(np.unique(HPX64[true_positive]))))
    plt.scatter(ra_det[(false_positive) & (cond)], dec_det[(false_positive) & (cond)],
                c=SNR_det[(false_positive) & (cond)], s=100.0, cmap=cm,
                lw=2, alpha=0.75, label='Not matched: ({:d})'.format(len(ra_det[(false_positive) & (cond)])))
    plt.scatter(ra_undet, dec_undet, color='None', edgecolor='k', s=200.0,
                lw=2, alpha=0.75, label='Not detected: ({:d})'.format(len(ra_undet)))
    plt.colorbar(sc, label='SNR detection')
    plt.xlim(np.max(ra_sim)+0.5, np.min(ra_sim)-0.5)
    plt.ylim(np.min(dec_sim)-0.5, np.max(dec_sim)+1.0)
    plt.xlabel('RA')
    plt.ylabel('DEC')
    plt.title('Spatial distribution of clusters with SNR > 5')
    plt.legend(loc=1)
    plt.show()

    fig = plt.figure(figsize=(30, 12))
    plt.scatter(ra_sim, dec_sim, c=SNR_sim_all, vmin=0, vmax=np.max(SNR_det), cmap=cm, s=10.0, marker='^',
                label='Simulations: ({:d})'.format(len(ra_sim)))
    cond = (SNR_det > 10)
    sc = plt.scatter(ra_det[(true_positive) & (cond)], dec_det[(true_positive) & (cond)],
                     c=SNR_det[(true_positive) & (cond)], vmin=0, vmax=np.max(SNR_det), marker='x', s=10.,
                     cmap=cm, label='True Positives: ({:d})'.format(len(np.unique(HPX64[true_positive]))))
    plt.scatter(ra_det[(false_positive) & (cond)], dec_det[(false_positive) & (cond)],
                c=SNR_det[(false_positive) & (cond)], s=100.0, cmap=cm,
                lw=2, alpha=0.75, label='Not matched: ({:d})'.format(len(ra_det[(false_positive) & (cond)])))
    plt.scatter(ra_undet, dec_undet, color='None', edgecolor='k', s=200.0,
                lw=2, alpha=0.75, label='Not detected: ({:d})'.format(len(ra_undet)))
    plt.colorbar(sc, label='SNR detection')
    plt.xlim(np.max(ra_sim) + 0.5, np.min(ra_sim)-0.5)
    plt.ylim(np.min(dec_sim) - 0.5, np.max(dec_sim)+1.0)
    plt.xlabel('RA')
    plt.ylabel('DEC')
    plt.title('Spatial distribution of clusters with SNR > 10')
    plt.legend(loc=1)
    plt.show()

    fig = plt.figure(figsize=(30, 12))
    # sc = plt.scatter(ra_undet, dec_undet, c=SNR_undet, vmin=0, vmax=np.max(SNR_undet), marker='x', s=200.,
    #                 cmap=cm, label='Undetected clusters: ({:d})'.format(len(ra_undet)))
    plt.scatter(ra_undet, dec_undet, color='None', edgecolor='k', s=200.0,
               lw=2, alpha=0.75, label='Not detected: ({:d})'.format(len(ra_undet)))
    # plt.colorbar(sc, label='SNR simulation')
    plt.xlim(np.max(ra_sim) + 0.5, np.min(ra_sim)-0.5)
    plt.ylim(np.min(dec_sim) - 0.5, np.max(dec_sim)+1.0)
    plt.xlabel('RA')
    plt.ylabel('DEC')
    plt.title('Spatial distribution of undetected clusters')
    plt.legend(loc=1)
    plt.show()

def dist_hist(match_file):
    """Plots histogram of distances.

    Parameters
    ----------
    match_file : str
       File name of the detected clusters.
    """
    dist_init_kpc_det, SNR_sim = np.loadtxt(
        match_file, usecols=(5, 28), unpack=True)

    true_positive = (SNR_sim > 0.)
    false_positive = (SNR_sim <= 0.)

    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 6))
    ax1.hist(dist_init_kpc_det, bins=20, range=(np.min(dist_init_kpc_det) - 1., np.max(dist_init_kpc_det) + 1),
             histtype='step', color="r", lw=4, label='Distances (kpc)')
    ax1.hist(dist_init_kpc_det[true_positive], bins=20, range=(np.min(dist_init_kpc_det) - 1, np.max(dist_init_kpc_det) + 1),
             histtype='step', color="mediumblue", lw=2,
             label='Distances [kpc] (true positives)')
    ax1.hist(dist_init_kpc_det[false_positive], bins=20, range=(np.min(dist_init_kpc_det) - 1, np.max(dist_init_kpc_det) + 1),
             histtype='step', color="maroon", lw=3,
             label='Distances [kpc] (false positives)')
    ax1.set_xlabel('Distances [kpc]')
    ax1.set_ylabel('# Clusters')
    ax1.legend()

    fig.suptitle('Distances Histogram (detections)')
    plt.show()


def SNR_hist(match_file, unmatch_file):
    """Plots histogram of SNR.

    Parameters
    ----------
    match_file : str
       File name of the detected clusters.
    """

    SNR_det, SNR_sim, det = np.loadtxt(match_file, usecols=(12, 28, 38), unpack=True)
    SNR_undet = np.loadtxt(unmatch_file, usecols=(6), unpack=True)

    true_positive = (det == 1.)
    false_positive = (det == 0.)

    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 6))
    ax1.hist(SNR_sim, bins=20, range=(np.min(SNR_det) - 1., np.max(SNR_det) + 1), histtype='step',
             color="k", lw=4, label='SNR simulations (all)')
    ax1.hist(SNR_det, bins=20, range=(np.min(SNR_det) - 1., np.max(SNR_det) + 1), histtype='step',
             color="r", lw=4, label='SNR detections (all)')
    ax1.hist(SNR_det[true_positive], bins=20, range=(np.min(SNR_det) - 1, np.max(SNR_det) + 1), histtype='step', color="mediumblue", lw=2, label='SNR detections (true positives)')
    ax1.hist(SNR_det[false_positive], bins=20, range=(np.min(SNR_det) - 1, np.max(SNR_det) + 1), histtype='step', color="maroon", lw=3, label='SNR detections (false positives)')
    ax1.hist(SNR_undet, bins=20, range=(np.min(SNR_det) - 1, np.max(SNR_det) + 1), histtype='step', color="teal", lw=4, label='SNR undetected clusters')
    ax1.set_xlabel('SNR')
    ax1.set_ylabel('# Clusters')
    ax1.set_yscale('log')
    ax1.legend()

    fig.suptitle('SNR Histogram (detections)')
    plt.show()


def write_det_numbers(input_simulation_path, match_file, unmatch_file):
    """Writes a few numbers informing the user about the detection counts, simulations, etc.

    Parameters
    ----------
    input_simulation_path : str
        Path to the file with the simulations table.
    match_file : str
       File name of the detected clusters.
    unmatch_file : str
        File name of the undetected clusters.
    """

    SNR_det, HPX64, SNR_sim = np.loadtxt(
        match_file, usecols=(12, 22, 28), unpack=True)

    ra_sim = np.loadtxt(input_simulation_path +
                        '/star_clusters_simulated.dat', usecols=(9), unpack=True)

    ra_undet = np.loadtxt(unmatch_file, usecols=(9), unpack=True)

    true_positive = (SNR_sim > 0.)

    print('Total of clusters simulated: {:d}.'.format(len(ra_sim)))
    print('Total of clusters detected: {:d} (True Positives).'.format(
        len(np.unique(HPX64[true_positive]))))
    print('Minimum SNR detected: {:.4f}'.format(np.min(SNR_det)))
    print('Total of clusters detected with SNR > 3: {:d}.'.format(
        len(np.unique(HPX64[(true_positive) & (SNR_det > 3.)]))))
    print('Total of clusters detected with SNR > 5: {:d}.'.format(
        len(np.unique(HPX64[(true_positive) & (SNR_det > 5.)]))))
    print('Total of clusters detected with SNR > 10: {:d}.'.format(
        len(np.unique(HPX64[(true_positive) & (SNR_det > 10.)]))))
    print('Total of clusters undetected: {:d}.'.format(len(ra_undet)))


def matching_sim_det(sim_file, det_file, match_file, unmatch_file, dist2match_arcmin):
    """Matches the detections and simulations, and writes matched and unmatched files.

    Parameters
    ----------
    sim_file : str
        File name of the simulation table.
    det_file : str
        File name of the detections table.
    match_file : str
       File name of the detected clusters.
    unmatch_file : str
        File name of the undetected clusters.
    dist2match_arcmin : float
        Angular distance in arcmin to match detections and simulations.

    Returns
    -------
    list
        Indexes of simulations and detections.
    """

    data_det = getdata(det_file)
    ra_det = data_det["ra"]
    dec_det = data_det["dec"]

    data_sim = ascii.read(sim_file)
    ra_sim = data_sim["9-ra"]
    dec_sim = data_sim["10-dec"]

    C_sim = SkyCoord(ra=ra_sim*u.degree, dec=dec_sim*u.degree)
    C_det = SkyCoord(ra=ra_det*u.degree, dec=dec_det*u.degree)

    idx_sim, idx_det, d2d, d3d = C_det.search_around_sky(
        C_sim, dist2match_arcmin*u.arcmin)

    idx_det_outliers = [i for i in range(len(data_det)) if i not in idx_det]

    file_match = open(match_file, 'w')
    print('#0-peak_id 1-ra 2-dec 3-iobj 4-jobj 5-dist_init_kpc 6-dist_err_kpc 7-dist_min_kpc 8-dist_max_kpc 9-coverfrac 10-coverfrac_bkg 11-wradius_arcmin 12-snr 13-Naper 14-Naper_tot 15-NWaper_tot 16-Naper_bkg 17-icyl 18-tile 19-slice 20-id_in_tile 21-id 22-HPX64 23-N 24-MV 25-SNR 26-N_f 27-MV_f 28-SNR_f 29-L 30-B 31-ra 32-dec 33-r_exp 34-ell 35-pa 36-mass 37-dist 38-1_matched-0_not_matched', file=file_match)

    for i, j in zip(idx_sim, idx_det):
        print(*data_det[:][j], *data_sim[i], '1', 
              sep=' ', file=file_match, end='\n')

    for i in (idx_det_outliers):
        print(*data_det[i], ' -99.999 ' * len(data_sim[1]), '0',
              sep=' ', file=file_match, end='\n')

    file_match.close()

    idx_not_det = [i for i in range(len(data_sim)) if i not in idx_sim]

    file_unmatch2 = open(unmatch_file[:-4] + '_unsort.dat', 'w')
    print('#0-HPX64 1-N 2-MV 3-SNR 4-N_f 5-MV_f 6-SNR_f 7-L 8-B 9-ra 10-dec 11-r_exp 12-ell 13-pa 14-mass 15-dist', file=file_unmatch2)

    for i in idx_not_det:
        print(*data_sim[i], sep=' ', file=file_unmatch2, end='\n')
    file_unmatch2.close()

    SNR_f = np.loadtxt(unmatch_file[:-4] + '_unsort.dat', usecols=(6), unpack=True)
    idx_SNR_f = np.argsort(SNR_f)[::-1]

    file_unmatch2 = open(unmatch_file[:-4] + '_unsort.dat', 'r')
    lines = file_unmatch2.readlines()

    line = [lines[i+1] for i in idx_SNR_f]

    file_unmatch = open(unmatch_file, 'w')
    print('#0-HPX64 1-N 2-MV 3-SNR 4-N_f 5-MV_f 6-SNR_f 7-L 8-B 9-ra 10-dec 11-r_exp 12-ell 13-pa 14-mass 15-dist', file=file_unmatch)

    for i in line:
        print(i, sep=' ', file=file_unmatch, end='')
    file_unmatch.close()

    return idx_sim, idx_det


def plot_masks(input_detection_path, mask_file, param2):
    """Plots all the masks in order to evaluate possible areas in CMD not covered by
    isochronal masks.

    Parameters
    ----------
    input_detection_path : str
        Path to the file with the detection table.
    mask_file : str
        File name of the isochronal masks.
    param2 : dict
        Dictionary with the parameters for the detection.
    """
    slices_file = input_detection_path + '/dslices.fits'
    data_sl = getdata(slices_file)
    d_slices_pc = data_sl["dist_pc"]
    mM_slices = 5 * np.log10(d_slices_pc) - 5.

    gr, g = np.loadtxt(mask_file, usecols=(0, 1), unpack=True)

    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 6))
    for i in range(len(mM_slices)):
        ax1.plot(gr, g + mM_slices[i], label='m-M={:.2f}'.format(mM_slices[i]))
    ax1.set_xlim(param2['isochrone_masks'][param2['survey']]['mask_color_min'],
                 param2['isochrone_masks'][param2['survey']]['mask_color_max'])
    ax1.set_ylim(param2['isochrone_masks'][param2['survey']]['mask_mag_max'],
                 param2['isochrone_masks'][param2['survey']]['mask_mag_min'])
    ax1.set_xlabel(r'$g_0-r_0$')
    ax1.set_ylabel(r'$g_0$')
    ax1.set_title('Masks applied to detection')
    ax1.legend()
    plt.show()


def recursive_print_dict(d, indent=0):
    """Show yaml or json table as an html table.

    Parameters
    ----------
    d : dict
        Dictionary with yaml or json format.
    indent : int, optional
        Identation.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            print("    " * indent, f"{k}:")
            recursive_print_dict(v, indent+1)
        else:
            print("    " * indent, f"{k}:{v}")


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


def read_real_cat(cat_DG="catalogs/objects_in_ref.dat", cat_GC="catalogs/Harris_updated.dat"):
    """Reads two catalogs with real objects: one with dwarf galaxies and other catalog
    with the list of Galactic globular clusters (Harris 2010).

    Parameters
    ----------
    cat_DG : str, optional
        Relative path to the catalog, by default "catalogs/objects_in_ref.dat"
    cat_GC : str, optional
        Relative path to the catalog, by default "catalogs/Harris_updated.dat"

    Returns
    -------
    array-like
        Array with many features of real objects.
    """
    ra_DG, dec_DG, dist_kpc_DG, Mv_DG, rhl_pc_DG, FeH_DG = np.loadtxt(
        cat_DG, usecols=(0, 1, 4, 8, 10, 11), unpack=True)

    name_DG = np.loadtxt(cat_DG, dtype=str, usecols=(2), unpack=True)

    #  Catalogo Harris_updated.dat
    # 0-Name 1-L 2-B 3-R_gc 4-Fe/H 5-M-M 6-Mv 7-rhl arcmin
    R_MW_GC, FeH_GC, mM_GC, Mv_GC, rhl_arcmin_GC = np.loadtxt(
        cat_GC, usecols=(3, 4, 5, 6, 7), unpack=True)

    dist_kpc_GC = 10 ** ((mM_GC / 5) - 2)

    rhl_pc_GC = 1000 * dist_kpc_GC * np.tan(rhl_arcmin_GC / (60 * 180 / np.pi))

    name_GC = np.loadtxt(cat_GC, dtype=str, usecols=(0), unpack=True)

    return name_DG, ra_DG, dec_DG, dist_kpc_DG, Mv_DG, rhl_pc_DG, FeH_DG, name_GC, R_MW_GC, FeH_GC, mM_GC, Mv_GC, rhl_pc_GC, dist_kpc_GC, rhl_arcmin_GC


def plots_ang_size(star_clusters_simulated, unmatch_file, FeH_iso):
    """Generates many plots comparing features of the detections, simulations and undetected clusters.

    Parameters
    ----------
    star_clusters_simulated : str
        Path to the file with the simulations table.
    unmatch_file : str
        File name of the undetected clusters.
    FeH_iso : float
        [Fe/H] of the isochrone model used to simulate clusters.
    """

    cmap = mpl.cm.get_cmap("inferno")
    cmap.set_under("dimgray")
    cmap.set_bad("black")

    hp_sample, NSTARS, MAG_ABS_V, NSTARS_CLEAN, MAG_ABS_V_CLEAN, r_exp, mass, dist = np.loadtxt(
        star_clusters_simulated, usecols=(0, 1, 2, 4, 5, 11, 14, 15), unpack=True)

    name_DG, ra_DG, dec_DG, dist_kpc_DG, Mv_DG, rhl_pc_DG, FeH_DG, name_GC, R_MW_GC, FeH_GC, mM_GC, Mv_GC, rhl_pc_GC, dist_kpc_GC, rhl_arcmin_GC = read_real_cat()

    hp_sample_un, MAG_ABS_V_un, NSTARS_CLEAN_un, MAG_ABS_V_CLEAN_un, r_exp_un, mass_un, dist_un = np.loadtxt(
        unmatch_file, usecols=(0, 2, 4, 5, 11, 14, 15), unpack=True)

    ang_size_DG = 60. * (180. / np.pi) * \
        np.arctan(rhl_pc_DG / (1000. * dist_kpc_DG))
    ang_size = 60 * np.rad2deg(np.arctan(1.7 * r_exp / dist))

    RHL_PC_SIM = 1.7 * r_exp

    ang_size_un = 60 * np.rad2deg(np.arctan(1.7 * r_exp_un / dist_un))

    RHL_PC_SIM_un = 1.7 * r_exp_un

    MW_center_distance_DG_kpc = radec2GCdist(ra_DG, dec_DG, dist_kpc_DG)

    f, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8),
        (ax9, ax10)) = plt.subplots(5, 2, figsize=(15, 23))

    ax1.hist(dist_kpc_DG, bins=np.linspace(0, 2. * np.max(dist) / 1000,
             20), label='DG', color='b', alpha=0.5, histtype='stepfilled')
    ax1.hist(dist_kpc_GC, bins=np.linspace(0, 2. * np.max(dist) / 1000,
             20), label='GC', color='k', alpha=0.5, lw=2, histtype='step')
    ax1.hist(dist / 1000, bins=np.linspace(0, 2. * np.max(dist) /
             1000, 20), label='Sim', color='r', alpha=0.5)
    ax1.hist(dist_un / 1000, bins=np.linspace(0, 2. *
             np.max(dist) / 1000, 20), label='Undet', color='darkgreen')
    ax1.legend()
    ax1.set_xlabel("Distance (kpc)")
    ax1.set_ylabel("N objects")
    ax1.set_title('Histogram of distances (linear scale)')
    ax1.set_xlim([0, 2. * np.max(dist) / 1000])

    ax2.hist(dist_kpc_DG, bins=np.linspace(0, 2. * np.max(dist) / 1000,
             20), label='DG', color='b', alpha=0.5, histtype='stepfilled')
    ax2.hist(dist_kpc_GC, bins=np.linspace(0, 2. * np.max(dist) / 1000,
             20), label='GC', color='k', alpha=0.5, lw=2, histtype='step')
    ax2.hist(dist / 1000, bins=np.linspace(0, 2. * np.max(dist) /
             1000, 20), label='Sim', color='r', alpha=0.5)
    ax2.hist(dist_un / 1000, bins=np.linspace(0, 2. *
             np.max(dist) / 1000, 20), label='Undet', color='darkgreen')
    ax2.legend()
    ax2.set_title('Histogram of distances (log scale)')
    ax2.set_xlabel("Distance (kpc)")
    ax2.set_ylabel("N objects")
    ax2.set_yscale('log')
    ax2.set_xlim([0, 2. * np.max(dist) / 1000])

    ax3.hist(ang_size_DG, bins=np.linspace(np.min(ang_size) / 2, 2. *
             np.max(ang_size), 20), label='DG', color='b', alpha=0.5, histtype='stepfilled')
    ax3.hist(rhl_arcmin_GC, bins=np.linspace(np.min(ang_size) / 2, 2. *
             np.max(ang_size), 20), label='GC', color='k', alpha=0.5, lw=2, histtype='step')
    ax3.hist(ang_size, bins=np.linspace(np.min(ang_size) / 2, 2. *
             np.max(ang_size), 20), label='Sim', color='r', alpha=0.5)
    ax3.hist(ang_size_un, bins=np.linspace(np.min(ang_size) / 2,
             2. * np.max(ang_size), 20), label='Undet', color='darkgreen')
    ax3.legend()
    ax3.set_xlim([np.min(ang_size) / 2, 2. * np.max(ang_size)])
    ax3.set_xlabel(r"$r_{1/2}$ (arcmin)")
    ax3.set_ylabel("N objects")
    ax3.set_title('Histogram of angular sizes (linear scale)')

    ax4.hist(ang_size_DG, bins=np.linspace(np.min(ang_size) / 2, 2. *
             np.max(ang_size), 20), label='DG', color='b', alpha=0.5, histtype='stepfilled')
    ax4.hist(rhl_arcmin_GC, bins=np.linspace(np.min(ang_size) / 2, 2. *
             np.max(ang_size), 20), label='GC', color='k', alpha=0.5, lw=2, histtype='step')
    ax4.hist(ang_size, bins=np.linspace(np.min(ang_size) / 2, 2. *
             np.max(ang_size), 20), label='Sim', color='r', alpha=0.5)
    ax4.hist(ang_size_un, bins=np.linspace(np.min(ang_size) / 2,
             2. * np.max(ang_size), 20), label='Undet', color='darkgreen')
    ax4.legend()
    ax4.set_xlim([np.min(ang_size) / 2, 2. * np.max(ang_size)])
    ax4.set_yscale('log')
    ax4.set_xlabel(r"$r_{1/2}$ (arcmin)")
    ax4.set_ylabel("N objects")
    ax4.set_title('Histogram of angular sizes (log scale)')

    ax5.scatter(dist / 1000, ang_size, label='Sim', color='r')
    ax5.scatter(dist_un / 1000, ang_size_un, label='Undet', color='darkgreen')
    ax5.scatter(dist_kpc_DG, ang_size_DG, label='DG', color='b')
    ax5.scatter(dist_kpc_GC, rhl_arcmin_GC, label='GC', color='k')
    ax5.set_xlabel("Distance (kpc)")
    ax5.set_ylabel(r"$r_{1/2}$ (arcmin)")
    ax5.set_yscale('log')
    ax5.legend()
    ax5.set_title('Distances X Angular sizes')

    for i, j in enumerate(mass):
        if MAG_ABS_V[i] < 0.0:
            ax6.plot([mass[i], mass[i]], [NSTARS[i], NSTARS_CLEAN[i]],
                     color='darkred', lw=0.2)
    ax6.scatter(mass, NSTARS, label='Sim', color='r')
    ax6.scatter(mass, NSTARS_CLEAN, label='Sim filt', color='darkred')
    ax6.scatter(mass_un, NSTARS_CLEAN_un, label='Undet', color='darkgreen')
    ax6.set_xlabel("MASS(MSun)")
    ax6.set_ylabel("N stars")
    ax6.legend()
    ax6.set_title('Visible Mass X Star counts')

    ax7.hist(Mv_DG, bins=20, range=(-16, 0.0),
             histtype="stepfilled", label="DG", color="b", alpha=0.5)
    ax7.hist(Mv_GC, bins=20, range=(-16, 0.0),
             histtype="step", label="GC", color="k")
    ax7.hist(MAG_ABS_V, bins=20, range=(-16, 0.0), histtype="step",
             label="Sim", color="r", ls="--", alpha=0.5)
    ax7.hist(MAG_ABS_V_CLEAN, bins=20, range=(-16, 0.0), histtype="stepfilled",
             label="Sim filt", color="darkred", ls="--", alpha=0.5)
    ax7.hist(MAG_ABS_V_CLEAN_un, bins=20, range=(-16, 0.0),
             histtype="stepfilled", label="Undet", color="darkgreen", ls="--", lw=2)
    ax7.set_xlabel(r"$M_V$")
    ax7.set_ylabel("N")
    ax7.legend(loc=2)
    ax7.set_title('Histogram of Absolute Magnitude (V band)')

    ax8.hist(rhl_pc_DG, bins=20, histtype="stepfilled",
             range=(10, 2400), label="DG", color="b", alpha=0.5)
    ax8.hist(rhl_pc_GC, bins=20, histtype="step",
             range=(10, 2400), label="GC", color="k")
    ax8.hist(RHL_PC_SIM, bins=20, histtype="stepfilled", range=(
        10, 2400), label="Sim", color="r", ls="--", alpha=0.5)
    ax8.hist(RHL_PC_SIM_un, bins=20, histtype="stepfilled", range=(
        10, 2400), label="Undet", color="darkgreen", ls="--", lw=2)
    ax8.set_xlabel(r"$r_{1/2}$[pc]")
    ax8.legend(loc=1)
    # ax8.set_xscale('log')
    ax8.set_yscale('log')
    ax8.set_title(r'Histogram of $r_{1/2}$ (parsecs)')

    ax9.hist(np.repeat(FeH_iso, len(MAG_ABS_V)), bins=20, range=(-3, 1.0),
             histtype="stepfilled", label="Sim", color="r", ls="--", alpha=0.5)
    ax9.hist(np.repeat(FeH_iso, len(MAG_ABS_V_un)), bins=20, range=(-3, 1.0),
             histtype="stepfilled", label="Sim", color="darkgreen", ls="--", lw=2)
    ax9.hist(FeH_DG, bins=20, range=(-3, 1.0),
             histtype="stepfilled", label="DG", color="b", alpha=0.5)
    ax9.hist(FeH_GC, bins=20, range=(-3, 1.0),
             histtype="step", label="GC", color="k")
    ax9.set_xlabel("[Fe/H]")
    ax9.legend(loc=1)
    ax9.set_title('Absolute Magnitude (V band) X Metalicity')

    ax10.scatter(dist / 1000, np.repeat(FeH_iso, len(dist)),
                 label="Sim", color="r", marker="x", lw=1.0)
    ax10.scatter(dist_un / 1000, np.repeat(FeH_iso, len(dist_un)),
                 label="Undet", color="darkgreen", marker="x", lw=2.0)
    ax10.scatter(MW_center_distance_DG_kpc, FeH_DG, label="DG", color="b")
    ax10.scatter(R_MW_GC, FeH_GC, label="GC", color="k")
    ax10.set_xlabel("Distance to the Galactic center (kpc)")
    ax10.set_ylabel("[Fe/H]")
    ax10.set_ylim([-3.5, 0])
    ax10.legend()
    ax10.grid()
    ax10.set_title('Galactocentric distances X Metalicity')

    plt.suptitle("Physical features of 58 Dwarf Gal + 152 GC + " + str(len(hp_sample)) +
                 " Simulations + " + str(len(hp_sample_un)) + " Undetected", fontsize=16)
    f.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


def general_plots(star_clusters_simulated, unmatch_clusters_file):
    """Generates plots mainly comparing abolute magnitude, half-light radii and
    masses of simulations, detections, real systems and undetected clusters.

    Parameters
    ----------
    star_clusters_simulated : str
        Path to the file with the simulations table.
    unmatch_clusters_file : str
        File name of the undetected clusters.
    """

    name_DG, ra_DG, dec_DG, dist_kpc_DG, Mv_DG, rhl_pc_DG, FeH_DG, name_GC, R_MW_GC, FeH_GC, mM_GC, Mv_GC, rhl_pc_GC, dist_kpc_GC, rhl_arcmin_GC = read_real_cat()

    MAG_ABS_V_un, MAG_ABS_V_CLEAN_un, R_EXP_un, MASS_un = np.loadtxt(
        unmatch_clusters_file, usecols=(2, 5, 11, 14), unpack=True)
    MAG_ABS_V, MAG_ABS_V_CLEAN, R_EXP, MASS = np.loadtxt(
        star_clusters_simulated, usecols=(2, 5, 11, 14), unpack=True)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 5))
    ax1.scatter(1.7 * R_EXP[MAG_ABS_V < 0.0],
                MAG_ABS_V[MAG_ABS_V < 0.0], color='r', label='Sim', alpha=0.2)
    ax1.scatter(1.7 * R_EXP[MAG_ABS_V < 0.0], MAG_ABS_V_CLEAN[MAG_ABS_V <
                0.0], color='darkred', label='Sim filt', alpha=0.2)
    ax1.scatter(1.7 * R_EXP_un[MAG_ABS_V_un < 0.0],
                MAG_ABS_V_CLEAN_un[MAG_ABS_V_un < 0.0], color='darkgreen', label='Undetected')
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
    ax1.set_xlim([np.min(1.7 * R_EXP[MAG_ABS_V < 0.0]) - 0.1,
                 np.max(1.7 * R_EXP[MAG_ABS_V < 0.0]) + 0.1])
    ax1.set_ylim([np.max(MAG_ABS_V_CLEAN[MAG_ABS_V < 0.0]) +
                 0.1, np.min(MAG_ABS_V[MAG_ABS_V < 0.0]) - 0.1])
    ax1.set_xscale("log")
    ax1.legend()

    ax2.scatter(1.7 * R_EXP[MAG_ABS_V < 0.0],
                MAG_ABS_V[MAG_ABS_V < 0.0], color='r', label='Sim', alpha=0.2)
    ax2.scatter(1.7 * R_EXP[MAG_ABS_V < 0.0], MAG_ABS_V_CLEAN[MAG_ABS_V <
                0.0], color='darkred', label='Sim filt', alpha=0.2)
    ax2.scatter(1.7 * R_EXP_un[MAG_ABS_V_un < 0.0],
                MAG_ABS_V_CLEAN_un[MAG_ABS_V_un < 0.0], color='darkgreen', label='Undetected')
    ax2.scatter(rhl_pc_DG, Mv_DG, color='b', marker='x', label='DG')
    ax2.scatter(rhl_pc_GC, Mv_GC, color='k', marker='x', label='GC')
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
    ax3.scatter(MASS, MAG_ABS_V_CLEAN, label='Sim filt',
                color='darkred', alpha=0.2)
    ax3.scatter(MASS_un, MAG_ABS_V_CLEAN_un, label='Undetected', color='darkgreen')
    for i, j in enumerate(MASS):
        if MAG_ABS_V[i] < 0.0:
            ax3.plot([MASS[i], MASS[i]],
                     [MAG_ABS_V[i], MAG_ABS_V_CLEAN[i]], color='darkred', lw=0.2)
    ax3.set_xlabel("mass(Msun)")
    ax3.set_ylim([np.max(MAG_ABS_V_CLEAN[MAG_ABS_V < 0.0]) +
                 0.1, np.min(MAG_ABS_V[MAG_ABS_V < 0.0]) - 0.1])
    ax3.legend()
    plt.show()
    plt.close()


def plot_clus_position(unmatch_file, ra_str, dec_str, star_cats_path):
    """Generates scatter plots showing the position of stars in each simulated
    system. The figures have distinct angular sizes.

    Parameters
    ----------
    unmatch_file : str
        File name of the undetected clusters.
    ra_str : str
        Label of RA column in FITS file.
    dec_str : str
        Label of RA column in FITS file.
    star_cats_path : str
        Path to the FITS files with the position of stars in each simulated cluster.
    """
    PIX_sim_un, N_f, RA_un, DEC_un, rexp_un, ell, pa, dist_un = np.loadtxt(
        unmatch_file, usecols=(0, 4, 9, 10, 11, 12, 13, 15), unpack=True)

    len_ipix = len(PIX_sim_un)

    ra_cen, dec_cen = RA_un, DEC_un

    rexp_un_arcsec = 3600 * (180. / np.pi) * np.arctan(rexp_un / dist_un) 

    for i in range(20):

        data = fits.getdata(star_cats_path + '/' +
                            str(int(PIX_sim_un[i])) + '.fits')
        RA_orig = data[ra_str]
        DEC_orig = data[dec_str]
        GC_orig = data['GC']
        st_line_arcsec = rexp_un_arcsec[i]
        half_size_plot_dec = 7 * st_line_arcsec / 3600.
        half_size_plot_ra = half_size_plot_dec / np.cos(np.deg2rad(dec_cen[i]))

        if len(RA_orig[(RA_orig < ra_cen[i] + half_size_plot_ra) & (RA_orig > ra_cen[i] - half_size_plot_ra) & (DEC_orig < dec_cen[i] + half_size_plot_dec) & (DEC_orig > dec_cen[i] - half_size_plot_dec)]) > 10.:

            fig, ax = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

            t = np.linspace(0, 2*np.pi, 100)
            a = rexp_un_arcsec[i] / 3600
            b = a * (1. - ell[i])
            ell_deg = np.array([a*np.cos(t) , b*np.sin(t)])
            r_rot = np.array([[np.cos(np.deg2rad(pa[i])) , -np.sin(np.deg2rad(pa[i]))], [np.sin(np.deg2rad(pa[i])) , np.cos(np.deg2rad(pa[i]))]])
            ell_rot_deg = np.zeros((2, ell_deg.shape[1]))

            for ii in range(ell_deg.shape[1]):
                ell_rot_deg[:,ii] = np.dot(r_rot, ell_deg[:,ii])

            ax[1].set_yticks([])
            ax[2].set_yticks([])

            st_line_arcsec = rexp_un_arcsec[i]
            half_size_plot_dec = 3 * st_line_arcsec / 3600.
            half_size_plot_ra = half_size_plot_dec / \
                np.cos(np.deg2rad(dec_cen[i]))

            data = fits.getdata(star_cats_path + '/' +
                                str(int(PIX_sim_un[i])) + '.fits')
            RA = data[ra_str]
            DEC = data[dec_str]
            GC = data['GC']
            col = 0
            ax[col].scatter(RA_orig[(GC_orig == 0)], DEC_orig[(
                GC_orig == 0)], edgecolor='b', color='None', s=20, label='MW stars')
            ax[col].scatter(
                RA[(GC == 1)], DEC[(GC == 1)], edgecolor='r', color='None', s=20, label='Cl stars')
            ax[col].set_xlim(
                [ra_cen[i] + half_size_plot_ra, ra_cen[i] - half_size_plot_ra])
            ax[col].set_ylim(
                [dec_cen[i] - half_size_plot_dec, dec_cen[i] + half_size_plot_dec])
            ax[col].set_title('HPX {:d}'.format(
                int(PIX_sim_un[i])), y=0.9, pad=8, backgroundcolor='w')
            # {x=ra_cen[i], y=dec_cen[i], pad=8)
            # ax[col].set_title('HPX {:d} ({:d} cl stars)'.format(
            #     int(PIX_sim_un[i]), int(N_f[i])), y=0.9, pad=8, backgroundcolor='w')
            ax[col].legend(loc=3)
            ax[col].scatter(ra_cen[i], dec_cen[i], color='k',
                            s=100, marker='+', label='Cluster center')
            ax[col].set_xlabel('RA (deg)')
            ax[col].set_ylabel('DEC (deg)')
            ax[col].text(ra_cen[i] - half_size_plot_ra + 2. * st_line_arcsec / (np.cos(np.deg2rad(dec_cen[i]))*3600),
                         dec_cen[i] - 0.96 * half_size_plot_dec, '{:d} arcsec'.format(int(st_line_arcsec)), fontsize=8.)
            ax[col].plot([ra_cen[i] - half_size_plot_ra + st_line_arcsec / (np.cos(np.deg2rad(dec_cen[i]))*3600), ra_cen[i] - half_size_plot_ra + 2. * st_line_arcsec /
                         (np.cos(np.deg2rad(dec_cen[i]))*3600)], [dec_cen[i] - 0.9 * half_size_plot_dec, dec_cen[i] - 0.9 * half_size_plot_dec], color='k', lw=1)
            ax[col].plot(ra_cen[i]+ell_rot_deg[1,:] / np.cos(np.deg2rad(dec_cen[i])), dec_cen[i]+ell_rot_deg[0,:], ls='--', color='darkorange', label='exp radius')
            ax[col].legend(loc=3)

            col = 1
            # st_line_arcsec = 20.
            half_size_plot_dec = 10 * st_line_arcsec / 3600.
            half_size_plot_ra = half_size_plot_dec / \
                np.cos(np.deg2rad(dec_cen[i]))
            ax[col].scatter(RA_orig[(GC_orig == 0)], DEC_orig[(
                GC_orig == 0)], edgecolor='b', color='None',     s=20, label='MW stars')
            ax[col].scatter(
                RA[(GC == 1)], DEC[(GC == 1)], edgecolor='r', color='None', s=20, label='Cl stars')
            ax[col].set_xlim(
                [ra_cen[i] + half_size_plot_ra, ra_cen[i] - half_size_plot_ra])
            ax[col].set_ylim(
                [dec_cen[i] - half_size_plot_dec, dec_cen[i] + half_size_plot_dec])
            # {x=ra_cen[i],     y=dec_cen[i], pad=8)
            ax[col].set_title('HPX {:d}'.format(
                int(PIX_sim_un[i])), y=0.9, pad=8, backgroundcolor='w')
            ax[col].scatter(ra_cen[i], dec_cen[i], color='k',
                            s=100, marker='+', label='Cluster center')
            ax[col].set_xlabel('RA (deg)')
            ax[col].text(ra_cen[i] - half_size_plot_ra + 2. * st_line_arcsec / (np.cos(np.deg2rad(dec_cen[i]))*3600),
                         dec_cen[i] - 0.96 * half_size_plot_dec, '{:d} arcsec'.format(int(st_line_arcsec)), fontsize=8.)
            ax[col].plot([ra_cen[i] - half_size_plot_ra + st_line_arcsec / (np.cos(np.deg2rad(dec_cen[i]))*3600), ra_cen[i] - half_size_plot_ra + 2. * st_line_arcsec /
                         (np.cos(np.deg2rad(dec_cen[i]))*3600)], [dec_cen[i] - 0.9 * half_size_plot_dec, dec_cen[i] - 0.9 * half_size_plot_dec], color='k', lw=1)

            ax[col].plot(ra_cen[i]+ell_rot_deg[1,:] / np.cos(np.deg2rad(dec_cen[i])), dec_cen[i]+ell_rot_deg[0,:], ls='--', color='darkorange', label='exp radius')
            ax[col].legend(loc=3)

            col = 2
            # st_line_arcsec = 30.
            half_size_plot_dec = 20 * st_line_arcsec / 3600.
            half_size_plot_ra = half_size_plot_dec / \
                np.cos(np.deg2rad(dec_cen[i]))
            ax[col].scatter(RA_orig[(GC_orig == 0)], DEC_orig[(
                GC_orig == 0)], edgecolor='b', color='None',         s=20, label='MW stars')
            ax[col].scatter(
                RA[(GC == 1)], DEC[(GC == 1)], edgecolor='r', color='None', s=20, label='Cl stars')
            ax[col].set_xlim(
                [ra_cen[i] + half_size_plot_ra, ra_cen[i] - half_size_plot_ra])
            ax[col].set_ylim(
                [dec_cen[i] - half_size_plot_dec, dec_cen[i] + half_size_plot_dec])
            # {x=ra_cen[i],         y=dec_cen[i], pad=8)
            ax[col].set_title('HPX {:d}'.format(
                int(PIX_sim_un[i])), y=0.9, pad=8, backgroundcolor='w')
            ax[col].scatter(ra_cen[i], dec_cen[i], color='k',
                            s=100, marker='+', label='Cluster center')
            ax[col].set_xlabel('RA (deg)')
            ax[col].text(ra_cen[i] - half_size_plot_ra + 2. * st_line_arcsec / (np.cos(np.deg2rad(dec_cen[i]))*3600),
                         dec_cen[i] - 0.96 * half_size_plot_dec, '{:d} arcsec'.format(int(st_line_arcsec)), fontsize=8.)
            ax[col].plot([ra_cen[i] - half_size_plot_ra + st_line_arcsec / (np.cos(np.deg2rad(dec_cen[i]))*3600), ra_cen[i] - half_size_plot_ra + 2. * st_line_arcsec /
                         (np.cos(np.deg2rad(dec_cen[i]))*3600)], [dec_cen[i] - 0.9 * half_size_plot_dec, dec_cen[i] - 0.9 * half_size_plot_dec], color='k', lw=1)

            ax[col].plot(ra_cen[i]+ell_rot_deg[1,:] / np.cos(np.deg2rad(dec_cen[i])), dec_cen[i]+ell_rot_deg[0,:], ls='--', color='darkorange', label='exp radius')
            ax[col].legend(loc=3)
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.show()


def calc_comp_hist(Mv_sim, Mv_det, log10_rad_sim, log10_rad_det):
    """Calculates completeness 2D histogram regarding absolute magnitude and half-light radii.

    Parameters
    ----------
    Mv_sim : list
        Absolute magnitude of simulated systems in V band.
    Mv_det : list
        Absolute magnitude of detected systems in V band.
    log10_rad_sim : list
        10-log of half-light radii of simulated systems, in parsecs.
    log10_rad_det : list
        10-log of half-light radii of detected systems, in parsecs.

    Returns
    -------
    array-like
        2d histogram of detections in plane absolute magnitude x half-light radii.
    """
    Mmin, Mmax, r_log_min, r_log_max = -11, 2, 1, 3.1

    n_bins = 13

    H_sim = np.histogram2d(Mv_sim, log10_rad_sim, bins=[n_bins, n_bins],
                           range=[[Mmin, Mmax], [r_log_min, r_log_max]])
    H_det = np.histogram2d(Mv_det, log10_rad_det, bins=[n_bins, n_bins],
                           range=[[Mmin, Mmax], [r_log_min, r_log_max]])
    H_comp = H_det[0] / H_sim[0]
    return H_comp


def full_completeness_distances(Mv_sim, Mv_det, radius_sim, radius_det, dist_sim, dist_sim_det):
    """Calculates and show the completeness of detections (in 2D histogram)
    in four bins of distance.

    Parameters
    ----------
    Mv_sim : list
        Absolute magnitude of simulations.
    Mv_det : list
        Absolute magnitude of detections.
    radius_sim : list
        Half-light radius of simulations.
    radius_det : list
        Half-light radius of detections.
    dist_sim : list
        Distance of simulations.
    dist_sim_det : list
        Distances of simulated clusters that are detected.
    """
    cmap = plt.cm.Blues
    cmap.set_bad('lightgray', 1.)

    Mmin, Mmax, r_log_min, r_log_max = -11, 2, 1, 3.1

    name_DG, ra_DG, dec_DG, dist_kpc_DG, Mv_DG, rhl_pc_DG, FeH_DG, name_GC, R_MW_GC, FeH_GC, mM_GC, Mv_GC, rhl_pc_GC, dist_kpc_GC, rhl_arcmin_GC = read_real_cat()
    mM_DG = 5 * np.log10(100*dist_kpc_DG)

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16), dpi=100)

    mM_sim = 5. * np.log10(dist_sim) - 5.
    mM_det = 5. * np.log10(dist_sim_det) - 5.

    cond_sim = (mM_sim > 10.) & (mM_sim < 15.)
    cond_det = (mM_det > 10.) & (mM_det < 15.)
    cond_DG = (mM_DG > 10.) & (mM_DG < 15.)
    cond_GC = (mM_GC > 10.) & (mM_GC < 15.)

    H = calc_comp_hist(Mv_sim[cond_sim], Mv_det[cond_det], np.log10(radius_sim[cond_sim]),
                       np.log10(radius_det[cond_det]))
    ax1.set_title(r'10<$(m-M)_0$<15')
    ax1.set_xlim([Mmin, Mmax])
    ax1.set_ylim([r_log_min, r_log_max])
    ax1.set_xlabel(r'$M_V$')
    ax1.set_ylabel(r'$log_{10}(r_{1/2}[pc])$')
    ax1.grid(True, lw=0.2)
    im1 = ax1.imshow(np.flipud(H.T), extent=[Mmin, Mmax, r_log_min, r_log_max], aspect='auto',
                     vmin=0., vmax=1.00, interpolation='None', cmap=cmap)
    ax1.scatter(Mv_GC[cond_GC], np.log10(
        rhl_pc_GC[cond_GC]), marker='x', color='k', label='GC')
    ax1.scatter(Mv_DG[cond_DG], np.log10(
        rhl_pc_DG[cond_DG]), marker='x', color='b', label='DG')
    for i, j in enumerate(rhl_pc_DG):
        if cond_DG[i]:
            ax1.annotate(name_DG[i], (Mv_DG[i], np.log10(
                rhl_pc_DG[i])), color='darkmagenta')
    for i, j in enumerate(rhl_pc_GC):
        if cond_GC[i]:
            ax1.annotate(name_GC[i], (Mv_GC[i], np.log10(
                rhl_pc_GC[i])), color='darkmagenta')

    cond_sim = (mM_sim > 15.) & (mM_sim < 20.)
    cond_det = (mM_det > 15.) & (mM_det < 20.)
    cond_DG = (mM_DG > 15.) & (mM_DG < 20.)
    cond_GC = (mM_GC > 15.) & (mM_GC < 20.)

    H = calc_comp_hist(Mv_sim[cond_sim], Mv_det[cond_det], np.log10(radius_sim[cond_sim]),
                       np.log10(radius_det[cond_det]))
    ax2.set_title(r'15<$(m-M)_0$<20')
    ax2.set_xlim([Mmin, Mmax])
    ax2.set_ylim([r_log_min, r_log_max])
    ax2.set_xlabel(r'$M_V$')
    ax2.set_ylabel(r'$log_{10}(r_{1/2}[pc])$')
    ax2.grid(True, lw=0.2)
    im2 = ax2.imshow(np.flipud(H.T), extent=[Mmin, Mmax, r_log_min, r_log_max], aspect='auto',
                     vmin=0., vmax=1.00, interpolation='None', cmap=cmap)
    ax2.scatter(Mv_GC[cond_GC], np.log10(
        rhl_pc_GC[cond_GC]), marker='x', color='k', label='GC')
    ax2.scatter(Mv_DG[cond_DG], np.log10(
        rhl_pc_DG[cond_DG]), marker='x', color='b', label='DG')
    for i, j in enumerate(rhl_pc_DG):
        if cond_DG[i]:
            ax2.annotate(name_DG[i], (Mv_DG[i], np.log10(
                rhl_pc_DG[i])), color='darkmagenta')
    for i, j in enumerate(rhl_pc_GC):
        if cond_GC[i]:
            ax2.annotate(name_GC[i], (Mv_GC[i], np.log10(
                rhl_pc_GC[i])), color='darkmagenta')

    cond_sim = (mM_sim > 20.) & (mM_sim < 25.)
    cond_det = (mM_det > 20.) & (mM_det < 25.)
    cond_DG = (mM_DG > 20.) & (mM_DG < 25.)
    cond_GC = (mM_GC > 20.) & (mM_GC < 25.)

    H = calc_comp_hist(Mv_sim[cond_sim], Mv_det[cond_det], np.log10(radius_sim[cond_sim]),
                       np.log10(radius_det[cond_det]))
    ax3.set_title(r'20<$(m-M)_0$<25')
    ax3.set_xlim([Mmin, Mmax])
    ax3.set_ylim([r_log_min, r_log_max])
    ax3.set_xlabel(r'$M_V$')
    ax3.set_ylabel(r'$log_{10}(r_{1/2}[pc])$')
    ax3.grid(True, lw=0.2)
    im3 = ax3.imshow(np.flipud(H.T), extent=[Mmin, Mmax, r_log_min, r_log_max], aspect='auto',
                     vmin=0., vmax=1.00, interpolation='None', cmap=cmap)

    ax3.scatter(Mv_GC[cond_GC], np.log10(
        rhl_pc_GC[cond_GC]), marker='x', color='k', label='GC')
    ax3.scatter(Mv_DG[cond_DG], np.log10(
        rhl_pc_DG[cond_DG]), marker='x', color='b', label='DG')
    for i, j in enumerate(rhl_pc_DG):
        if cond_DG[i]:
            ax3.annotate(name_DG[i], (Mv_DG[i], np.log10(
                rhl_pc_DG[i])), color='darkmagenta')
    for i, j in enumerate(rhl_pc_GC):
        if cond_GC[i]:
            ax3.annotate(name_GC[i], (Mv_GC[i], np.log10(
                rhl_pc_GC[i])), color='darkmagenta')
    cond_sim = (mM_sim > 25.) & (mM_sim < 30.)
    cond_det = (mM_det > 25.) & (mM_det < 30.)
    cond_DG = (mM_DG > 25.) & (mM_DG < 30.)
    cond_GC = (mM_GC > 25.) & (mM_GC < 30.)

    H = calc_comp_hist(Mv_sim[cond_sim], Mv_det[cond_det], np.log10(radius_sim[cond_sim]),
                       np.log10(radius_det[cond_det]))
    ax4.set_title(r'25<$(m-M)_0$<30')
    ax4.set_xlim([Mmin, Mmax])
    ax4.set_ylim([r_log_min, r_log_max])
    ax4.set_xlabel(r'$M_V$')
    ax4.set_ylabel(r'$log_{10}(r_{1/2}[pc])$')
    ax4.grid(True, lw=0.2)
    im4 = ax4.imshow(np.flipud(H.T), extent=[Mmin, Mmax, r_log_min, r_log_max], aspect='auto',
                     vmin=0., vmax=1.00, interpolation='None', cmap=cmap)
    ax4.scatter(Mv_GC[cond_GC], np.log10(
        rhl_pc_GC[cond_GC]), marker='x', color='k', label='GC')
    ax4.scatter(Mv_DG[cond_DG], np.log10(
        rhl_pc_DG[cond_DG]), marker='x', color='b', label='DG')
    for i, j in enumerate(rhl_pc_DG):
        if cond_DG[i]:
            ax4.annotate(name_DG[i], (Mv_DG[i], np.log10(
                rhl_pc_DG[i])), color='darkmagenta')
    for i, j in enumerate(rhl_pc_GC):
        if cond_GC[i]:
            ax4.annotate(name_GC[i], (Mv_GC[i], np.log10(
                rhl_pc_GC[i])), color='darkmagenta')

    cbaxes = f.add_axes([0.90, 0.126, 0.01, 0.755])
    cbar = f.colorbar(im3, cax=cbaxes, cmap=cmap,
                      orientation='vertical', label='Completeness')
    plt.suptitle('Completeness of detections')
    plt.subplots_adjust(wspace=0.2)
    plt.show()


def plot_pure(arg_all, arg_conf, label, title, bins=20):
    """Plots purity of a sample.

    Parameters
    ----------
    arg_all : list
        Indexes of all the detections.
    arg_conf : list
        Indexes of all the true positives.
    label : str
        Label of x-axis of generated plot.
    title : str
        Title of plot.
    bins : int, optional
        Bins of the purity, by default 20
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    over = (max(np.max(arg_all), np.max(arg_conf)) -
            min(np.min(arg_all), np.min(arg_conf))) * 0.1
    min_ = min(np.min(arg_all), np.min(arg_conf)) - over
    max_ = max(np.max(arg_all), np.max(arg_conf)) + over
    try:
        A = ax1.hist(arg_all, bins=bins, range=(
            bins[0], bins[-1]), histtype='step', lw=2, label='All detections')
        B = ax1.hist(arg_conf, bins=bins, range=(
            bins[0], bins[-1]), histtype='stepfilled', lw=2, label='True clusters')
        pureness = B[0] / A[0]
        ax1.set_xlabel(label)
        ax1.set_ylabel('Number of clusters detected')
        ax1.set_xlim([min_, max_])
        ax1.legend(loc=2)

        ax2.step(bins, np.append(
            pureness[0], pureness), 'b', lw=2, label='Data')
        # ax2.step(A[1][0:-1],pureness, label='Data', color='k')
        ax2.set_xlabel(label)
        ax2.set_ylabel('Purity')
        ax2.set_ylim([0, 1.2])
        ax2.set_xlim([min_, max_])
        ax2.legend()
        fig.suptitle(title)
        plt.show()
    except:
        A = ax1.hist(arg_all, bins=bins, range=(min_, max_),
                     histtype='step', color='b', lw=2, label='All detections')
        B = ax1.hist(arg_conf, bins=bins, range=(min_, max_),
                     histtype='stepfilled', color='orange', lw=2, label='True clusters')
        pureness = B[0] / A[0]
        ax1.set_xlabel(label)
        ax1.set_ylabel('Number of clusters detected')
        ax1.set_xlim([min_, max_])
        ax1.legend(loc=2)

        ax2.step(A[1][0:-1], np.nan_to_num(pureness), 'b', lw=2, label='Data')
        ax2.set_xlabel(label)
        ax2.set_ylabel('Purity')
        ax2.set_ylim([0, 1.2])
        ax2.set_xlim([min_, max_])
        ax2.legend()
        fig.suptitle(title)
        plt.show()


def plot_comp(arg, idxs, label, title):
    """Plots completeness of a sample.

    Parameters
    ----------
    arg : list
        List of values to generate plots.
    idxs : list
        Index of detected objects.
    label : str
        Label of x-axis.
    title : str
        Title of plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    bins = 20
    step = (np.max(arg) - np.min(arg)) / bins
    A = ax1.hist(arg[idxs], bins=bins, range=(np.min(arg), np.max(arg)), histtype='step',
                 lw=2, color="b", label='All detections')
    B = ax1.hist(arg, bins=bins, range=(np.min(arg), np.max(arg)), histtype='stepfilled',
                 lw=2, color="orange", label='True clusters')
    completeness = A[0] / B[0]
    completeness[completeness >= 1.] = 1.
    # Only to set steps equal to zero where the completeness does not have results.
    # Warning: the values replaced by zero are those ones where the completeness in undetermined.
    compl = np.append(0., np.nan_to_num(completeness))
    ax1.set_xlabel(label)
    ax1.set_ylabel('# Detected Clusters')
    ax1.legend()

    ax2.step(np.append(A[1][0] - step, A[1]),
             np.append(compl, 0), 'r', label='Data', where='mid')
    ax2.set_xlabel(label)
    ax2.set_ylabel('Completeness')
    ax2.set_ylim([0, 1.1])
    ax2.legend()
    fig.suptitle(title)
    plt.show()
