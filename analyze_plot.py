import numpy as np
from numpy import fft
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy import stats, interpolate
import pickle
from pathlib import Path, PurePath

from Map import Map


with open('parameters_dic', 'rb') as _file:
    parameters_dic = pickle.load(_file)

num_sample = parameters_dic['num_sample']
x_max = parameters_dic['x_max']
y_max = parameters_dic['y_max']
sig_amp = parameters_dic['sig_amp']
noise_sigma2 = parameters_dic['noise_sigma2']
num_pix_x = parameters_dic['num_pix_x']
num_pix_y = parameters_dic['num_pix_y']
crosslink = parameters_dic['crosslink']
num_iter = max_iter = parameters_dic['num_iter']
results_dir = parameters_dic['results_dir']
figs_dir = parameters_dic['figs_dir']
cache_dir = parameters_dic['cache_dir']
seed = parameters_dic['seed']
f_scan_list = parameters_dic['f_scan_list']
num_eta_arr = parameters_dic['num_eta_arr']
f_sample_knee_apo_arr = parameters_dic['f_sample_knee_apo_arr']
offsets = parameters_dic['offsets']
comps = parameters_dic['comps']
f_sample_list = parameters_dic['f_sample_list']
f_knee_list = parameters_dic['f_knee_list']


with open('data_list', 'rb') as _file:
    data_list = pickle.load(_file)   # data for each map


# loop over all data_list
for data_dic in  data_list:
    relative_dir = data_dic['relative_dir']
    plot_dir = figs_dir/relative_dir/('num_iter={:d}'.format(num_iter))
    plot_dir.mkdir(parents=True, exist_ok=True)

    # load info
    f = data_dic['frequencies']
    noise_power_spectrum = data_dic['noise_power_spectrum']
    f_scan = data_dic['f_scan']
    f_sample = data_dic['f_sample']
    f_knee = data_dic['f_knee']
    f_apo = data_dic['f_apo']
    scan_info = data_dic['scan_info']
    scan_info_latex = data_dic['scan_info_latex']
    noiseless_map = data_dic['noiseless_map']
    condition_number = data_dic['condition_number']
    chi2_min = data_dic['chi2_min']


    results_list = []
    description_list = []
    description_latex_list = []

    # CG with simple preconditioner
    description_list.append(data_dic['CG_SP_description'])
    description_latex_list.append(data_dic['CG_SP_description_latex'])
    with open (data_dic['CG_SP_file'], 'rb') as _file:
        CG_SP_result = pickle.load(_file)
        results_list.append(CG_SP_result)

    # CG with eta
    description_list.append(data_dic['CG_eta_description'])
    description_latex_list.append(data_dic['CG_eta_description_latex'])
    with open (data_dic['CG_eta_file'], 'rb') as _file:
        CG_eta_result = pickle.load(_file)
        results_list.append(CG_eta_result)

    # CG manual eta
    for num_eta in num_eta_arr:
        description_list.append(
            data_dic['CG_manual_ln_{:d}_eta_description'.format(num_eta)]
        )
        description_latex_list.append(
            data_dic['CG_manual_ln_{:d}_eta_description_latex'\
                .format(num_eta)]
        )
        with open (data_dic['CG_manual_ln_{:d}_eta_file'.format(num_eta)],
                'rb') as _file:
            CG_eta_result = pickle.load(_file)
            results_list.append(CG_eta_result)

    # CG exact eta
    description_list.append(data_dic['CG_exact_eta_description'])
    description_latex_list.append(data_dic['CG_exact_eta_description_latex'])
    with open (data_dic['CG_exact_eta_file'], 'rb') as _file:
        CG_exact_eta_result = pickle.load(_file)
        results_list.append(CG_exact_eta_result)

    # MF iteration
    for num_eta in num_eta_arr:
        description_list.append(
            data_dic['MF_ln_{:d}_eta_description'.format(num_eta)]
        )
        description_latex_list.append(
            data_dic['MF_ln_{:d}_eta_description_latex'\
                .format(num_eta)]
        )
        with open (data_dic['MF_ln_{:d}_eta_file'.format(num_eta)],
                'rb') as _file:
            MF_result = pickle.load(_file)
            results_list.append(MF_result)



    # power spectrum
    plt.figure(figsize=(12,9))
    plt.title('diag($N$)')
    plt.plot(f[1:], noise_power_spectrum[1:])
    plt.xlabel('$f$')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(plot_dir/'power_spectrum.pdf')
    plt.close()


    # plot map result
    titles_noiseless_signal = [
        "noiseless signal I",
        "noiseless signal Q",
        "noiseless signal U"
        ]

    for result, description, latex in zip(results_list, description_list,
            description_latex_list):
        fig,axs = plt.subplots(3,3, figsize=(12,12))
        fig.suptitle('{}\n{}'
            .format(scan_info_latex, latex))
        for i in range(3):
            # noiseless signal
            axs[0,i].imshow(
                noiseless_map[:,:,i],
                cmap=plt.cm.bwr, vmin=-sig_amp, vmax=sig_amp
            )
            axs[0,i].set_title(titles_noiseless_signal[i])
            
            # method results
            axs[1,i].imshow(
                result['m_hist'][-1,:,:,i],
                cmap=plt.cm.bwr, vmin=-sig_amp, vmax=sig_amp
            )
            axs[1,i].set_title('result')

            # residual map
            axs[2,i].imshow(
                result['r_hist'][-1,:,:,i],
                cmap=plt.cm.bwr, vmin=-sig_amp, vmax=sig_amp
            )
            axs[2,i].set_title('residual')
        plt.savefig(plot_dir/'{}.pdf'.format(description))
        plt.close()


    # plot norm(r)
    lines = [i['r_2norm_hist'] for i in results_list]
    plt.figure(figsize=(12,9))
    plt.title('{}\n$\kappa = {:.1e}$'.format(scan_info_latex, condition_number))
    plt.xlabel('num of iteration')
    plt.yscale('log')
    plt.ylabel('$||r||_2$') 
    for i in range(len(lines)):
        plt.plot(lines[i], '-', label=description_latex_list[i])
    plt.legend()
    plt.grid()
    plt.savefig(plot_dir/'r_2norm.pdf')
    plt.close()
    

    # plot Χ² log scale
    lines = [i['chi2_hist'] for i in results_list]
    plt.figure(figsize=(12,9))
    plt.title('{}\n$\kappa = {:.1e}$'.format(scan_info_latex, condition_number))
    plt.xlabel('num of iteration')
    plt.yscale('log')
    plt.ylabel('$\chi^2$') 
    for i in range(len(lines)):
        plt.plot(lines[i], '-', label=description_latex_list[i])
    plt.legend()
    plt.grid()
    plt.savefig(plot_dir/'chi2.pdf')
    plt.close()

    # plot (Χ²-min)/(ini-min)
    chi2_hist_list = [i['chi2_hist'] for i in results_list]
    plt.figure(figsize=(12,9))
    plt.title('{}\n$\kappa = {:.1e}$'.format(scan_info_latex, condition_number))
    plt.xlabel('num of iteration')
    plt.yscale('log')
    plt.ylabel(r'$\frac{\chi^2 - \chi^2_{min}}{\chi^2_{ini} - \chi^2_{min}}$') 
    for i,chi2_hist in enumerate(chi2_hist_list):
        line = (chi2_hist - chi2_min)/(chi2_hist[0] - chi2_min)
        plt.plot(line, '-', label=description_latex_list[i])
    plt.legend()
    plt.grid()
    plt.savefig(plot_dir/'chi2_min_value.pdf')
    plt.close()

    # plot η vs iter
    lines = [i['etas_iter'] for i in results_list]
    plt.figure(figsize=(12,9))
    plt.title(scan_info_latex)
    plt.xlabel('num of iteration')
    plt.ylabel('$\eta$') 
    plt.yscale('log')
    for i in range(len(lines)):
        plt.plot(np.arange(num_iter+1), lines[i], 
            '-', label=description_latex_list[i]
        )
    plt.legend()
    plt.grid()
    plt.savefig(plot_dir/'eta_iter.pdf')
    plt.close()

    # plot Χ²(m(η),η) 
    (eta_arr, chi2_arr) = data_dic['chi2_vs_eta']
    fig,ax = plt.subplots(figsize=(12,9))
    plt.title(scan_info_latex)
    ax.plot(eta_arr, chi2_arr)
    ax.set_yscale('log')
    ax.set_xlabel('$\eta$')
    ax.set_xscale('log')
    ax.set_ylabel('$\chi^2(\hat{m}(\eta),\eta)$') 
    ax.set_xticks(ticks=data_dic['etas_arr'], minor=True)
    ax.set_xticks(ticks=np.logspace(-10,0,11,base=10), minor=False)
    ax.set_xticklabels([],minor=True) # clear minor xticks
    ax.tick_params(axis='x', which='minor', length=5, width=2.5, color='r')
    ax.grid(axis='x', which='minor', color='r', alpha=0.3)
    ax.set_xlim(eta_arr.min(),1)
    plt.savefig(plot_dir/'chi2_vs_eta.pdf')
    plt.close()


    # plot -δΧ²(m, η)/Χ²(m(η), η)
    for result in results_list:
        try:
            dchi2_arr = result['dchi2_eta_hist'][1:]  # 0th element is 0
            eta_arr = result['etas_iter'][1:]
            (etas, chi2) = data_dic['chi2_vs_eta']
            log_chi2 = interpolate.interp1d(np.log(etas), np.log(chi2))
            dchi2_chi2 = np.zeros(dchi2_arr.shape)
            for i in range(len(eta_arr)):
                eta = eta_arr[i]
                chi2 = np.exp(log_chi2(np.log(eta)))
                dchi2_chi2[i] = -dchi2_arr[i]/chi2
            fig,ax = plt.subplots(figsize=(12,9))
            plt.title(scan_info_latex)
            ax.plot(dchi2_chi2)
            ax.set_yscale('log')
            ax.set_xlabel('num of iteration')
            ax.set_ylabel(r'$-\frac{\delta\chi^2(m,\eta)}{\chi^2(\hat{m}(\eta),\eta)}$') 
            plt.savefig(plot_dir/'dchi2_chi2.pdf')
            plt.close()
        except KeyError:
            pass

plt.close('all')





