import matplotlib.pyplot as plt
import numpy as np

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
        ax1.set_ylabel( 'Number of clusters detected')
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
        ax1.set_ylabel( 'Number of clusters detected')
        ax1.set_xlim([min_, max_])
        ax1.legend(loc=2)

        ax2.step(A[1][0:-1], pureness, 'r', lw=2, label='Data')
        ax2.set_xlabel(label)
        ax2.set_ylabel('Purity')
        ax2.set_ylim([0,1.2])
        ax2.set_xlim([min_, max_])
        ax2.legend()
        fig.suptitle(title)
        plt.show()
        
        
def plot_comp(arg, idxs, label, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
    A = ax1.hist(arg[idxs], bins=20, range=(np.min(arg), np.max(arg)), histtype='step', \
                 color = "r", label='Detections')
    B = ax1.hist(arg, bins=20, range=(np.min(arg), np.max(arg)), histtype='step', \
                 color = "mediumblue", label='Simulated clusters')
    completeness = A[0] / B[0]
    ax1.set_xlabel(label)
    ax1.set_ylabel( '# Detected Clusters')
    ax1.legend()

    ax2.step(A[1][0:-1], np.nan_to_num(completeness), 'k', label='Data')
    ax2.set_xlabel(label)
    ax2.set_ylabel('Completeness')
    ax2.set_ylim([0,1.1])
    ax2.legend()
    fig.suptitle(title)
    plt.show()