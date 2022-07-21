import matplotlib.pyplot as plt
import numpy as np


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

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16), dpi=100)
    
    mM_sim = 5. * np.log10(dist_sim) - 5.
    mM_det = 5. * np.log10(dist_sim_det) - 5.
        
    cond_sim = (mM_sim > 10.)&(mM_sim < 15.)
    cond_det = (mM_det > 10.)&(mM_det < 15.)
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

    cond_sim = (mM_sim > 15.)&(mM_sim < 20.)
    cond_det = (mM_det > 15.)&(mM_det < 20.)
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

    cond_sim = (mM_sim > 20.)&(mM_sim < 25.)
    cond_det = (mM_det > 20.)&(mM_det < 25.)
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

    cond_sim = (mM_sim > 25.)&(mM_sim < 30.)
    cond_det = (mM_det > 25.)&(mM_det < 30.)
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
