import numpy as np
from numpy import fft
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy import stats, interpolate
import pickle
from pathlib import Path, PurePath
#import tikzplotlib

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
#num_snapshots = parameters_dic['num_snapshots']
#max_iter = parameters_dic['max_iter']
num_iter = max_iter = parameters_dic['num_iter']
results_dir = parameters_dic['results_dir']
figs_dir = parameters_dic['figs_dir']
cache_dir = parameters_dic['cache_dir']
seed = parameters_dic['seed']
f_scan_list = parameters_dic['f_scan_list']
#condition_number_arr = parameters_dic['condition_number_arr']
#num_eta_iter_per_eta = parameters_dic['num_eta_iter_per_eta']
num_eta_arr = parameters_dic['num_eta_arr']
f_sample_knee_apo_arr = parameters_dic['f_sample_knee_apo_arr']
offsets = parameters_dic['offsets']
comps = parameters_dic['comps']
f_sample_list = parameters_dic['f_sample_list']
f_knee_list = parameters_dic['f_knee_list']


with open('data_list', 'rb') as _file:
    data_list = pickle.load(_file)   # data for each map


# loop over all data_list
#max_condition_num = -np.inf
#max_condition_num_wo_f0 = -np.inf
#CG_PT_collection_list = []   # collect all CG method with perterbation info
for data_dic in  data_list:
    relative_dir = data_dic['relative_dir']
    plot_dir = figs_dir/relative_dir/('num_iter={:d}'.format(num_iter))
    plot_dir.mkdir(parents=True, exist_ok=True)

    # load info
    #max_condition_num = max(data_dic['condition_number'], max_condition_num)
    #max_condition_num_wo_f0 = \
    #    max(data_dic['condition_number_wo_f0'], max_condition_num_wo_f0)
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
    #for num_eta in num_eta_arr:
    #    description_list.append(
    #        data_dic['MF_ln_{:d}_eta_description'.format(num_eta)]
    #    )
    #    description_latex_list.append(
    #        data_dic['MF_ln_{:d}_eta_description_latex'\
    #            .format(num_eta)]
    #    )
    #    with open (data_dic['MF_ln_{:d}_eta_file'.format(num_eta)],
    #            'rb') as _file:
    #        MF_result = pickle.load(_file)
    #        results_list.append(MF_result)



    # power spectrum
    plt.figure(figsize=(12,9))
    plt.title('diag($N$)')
    plt.plot(f[1:], noise_power_spectrum[1:])
    plt.xlabel('$f$')
    plt.xscale('log')
    plt.yscale('log')
    #plt.savefig(plot_dir/'power_spectrum.jpeg')
    plt.savefig(plot_dir/'power_spectrum.pdf')
    #tikzplotlib.clean_figure()
    #tikzplotlib.save(plot_dir/'power_spectrum.tex')
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
        #plt.savefig(plot_dir/'{}.jpeg'.format(description))
        plt.savefig(plot_dir/'{}.pdf'.format(description))
        #tikzplotlib.save(plot_dir/'{}.tex'.format(description))
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
    #plt.savefig(plot_dir/'r_2norm.jpeg')
    plt.savefig(plot_dir/'r_2norm.pdf')
    #tikzplotlib.clean_figure()
    #tikzplotlib.save(plot_dir/'r_2norm.tex')
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
    #plt.savefig(plot_dir/'chi2.jpeg')
    plt.savefig(plot_dir/'chi2.pdf')
    #tikzplotlib.clean_figure()
    #tikzplotlib.save(plot_dir/'chi2.tex')
    plt.close()

    ## plot Χ²(m,η) log scale
    #lines = [i['chi2_eta_hist'] for i in results_list]
    #plt.figure(figsize=(12,9))
    #plt.title('{}\n$\kappa = {:.1e}$'.format(scan_info_latex, condition_number))
    #plt.xlabel('num of iteration')
    #plt.yscale('log')
    #plt.ylabel('$\chi^2(m,\eta)$') 
    #for i in range(len(lines)):
    #    plt.plot(lines[i], '-', label=description_latex_list[i])
    #plt.legend()
    #plt.grid()
    ##plt.savefig(plot_dir/'chi2.jpeg')
    #plt.savefig(plot_dir/'chi2_eta.pdf')
    ##tikzplotlib.clean_figure()
    ##tikzplotlib.save(plot_dir/'chi2_eta.tex')
    #plt.close()

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
    #plt.savefig(plot_dir/'chi2_final_value.jpeg')
    plt.savefig(plot_dir/'chi2_min_value.pdf')
    #tikzplotlib.clean_figure()
    #tikzplotlib.save(plot_dir/'chi2_min_value.tex')
    plt.close()


    # Χ² per freq mode
    #bins = 20
    #linewidth = 1
    #lines = [i['chi2_f_hist'] for i in results_list]
    #_min = np.array(lines).min()
    #_max = np.array(lines).max()
    #_f = f.copy()
    #_f[0] = _f[1] # avoid -inf when take log
    #colors = list(mcolors.TABLEAU_COLORS.keys())
    #for i,line in enumerate(lines):
    #    plt.figure(figsize=(12,9))
    #    plt.title(scan_info_latex)
    #    bin_means, bin_edges, binnumber = stats.binned_statistic(
    #        np.log10(_f), line, statistic='mean', bins=bins)
    #    bin_edges = 10**bin_edges
    #    for j,n_step in enumerate(results_list[i]['snapshots_index']):
    #        plt.hlines(bin_means[j,:], bin_edges[:-1], bin_edges[1:],
    #            color=colors[j%len(colors)], lw=linewidth,
    #            label='step={:d}'.format(n_step))
    #    plt.plot(_f, line[-1,:], '-', lw=linewidth, color='grey', alpha=0.3,
    #        zorder=-1)
    #    plt.title('{}\n{}'.format(scan_info_latex, description_latex_list[i]))
    #    plt.yscale('log')
    #    plt.ylabel('$\chi^2$') 
    #    plt.ylim(_min*0.8, _max*1.2)
    #    plt.xscale('log')
    #    plt.xlabel('$f$')
    #    plt.legend(loc='center left', bbox_to_anchor=(1.01,0.5))
    #    plt.grid()
    #    #plt.savefig(plot_dir/('chi2_f_{}.jpeg'.format(description_list[i])),
    #    #    bbox_inches='tight')
    #    plt.savefig(plot_dir/('chi2_f_{}.pdf'.format(description_list[i])),
    #        bbox_inches='tight')
    #    #tikzplotlib.clean_figure()
    #    tikzplotlib.save(plot_dir/('chi2_f_{}.tex'.format(description_list[i])))
    #    plt.close()


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
    #tikzplotlib.clean_figure()
    #tikzplotlib.save(plot_dir/'eta_iter.tex')
    plt.close()

    # plot Χ²(m(η),η) 
    (eta_arr, chi2_arr) = data_dic['chi2_vs_eta']
    fig,ax = plt.subplots(figsize=(12,9))
    plt.title(scan_info_latex)
    ax.plot(eta_arr, chi2_arr)
    #ax.set_ylim(1e14, 1e20)
    ax.set_yscale('log')
    ax.set_xlabel('$\eta$')
    ax.set_xscale('log')
    ax.set_ylabel('$\chi^2(\hat{m}(\eta),\eta)$') 
    ax.set_xticks(ticks=data_dic['etas'], minor=True)
    ax.set_xticks(ticks=np.logspace(-10,0,11,base=10), minor=False)
    ax.set_xticklabels([],minor=True) # clear minor xticks
    ax.tick_params(axis='x', which='minor', length=5, width=2.5, color='r')
    #ax.set_yticks(ticks=[10*chi2_min], minor=True)
    #ax.tick_params(axis='y', which='minor')
    ax.grid(axis='x', which='minor', color='r', alpha=0.3)
    #ax.grid(axis='y', which='minor')
    ax.set_xlim(eta_arr.min(),1)
    plt.savefig(plot_dir/'chi2_vs_eta.pdf')
    #tikzplotlib.clean_figure()
    #tikzplotlib.save(plot_dir/'chi2_vs_eta.tex')
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


    # find out number of iteration Χ² such that
    # log(Χ²_ini/Χ²_final - 1) < chi2_ratio * log(Χ²_ini/Χ²_final - 1)
    # where final Χ² is the minimum value among all methods with same
    # frequencies.
    chi2_ratio = 1e-3
    stop_point_list = []
    for result in results_list:
        #argmax will stop at the first True 
        stop_point = np.argmax(
            result['chi2_hist']/chi2_min-1
            < chi2_ratio*(result['chi2_hist'][0]/chi2_min-1) 
            )
        #print(stop_point)
        stop_point_list.append(stop_point)
    data_dic['stop_point_list'] = stop_point_list


plt.close('all')


# plot stop point vs condition number
# loop over data_list with fixed f_sample, f_knee
colors = list(mcolors.TABLEAU_COLORS.keys())
markers = ['x', '+', '1', '2', '3', '4']
#methods_list = ['simple preconditioner without lambda']
#methods_latex_list = ['simple preconditioner without $\lambda$']
#methods_list + ['simple preconditioner with {:d}x{:d}'
#    .format(i,j) for i,j in num_lamb_iter_per_lamb]
#methods_latex_list.append(['simple preconditioner with ${:d}x{:d}$'
#    .format(i,j) for i,j in num_lamb_iter_per_lamb])
for i,[description,description_latex] in enumerate(zip(description_list, 
        description_latex_list)):
    fig,ax = plt.subplots(figsize=(12,9))
    plt.title(description_latex)
    fig.suptitle('number iteration $\chi^2/\chi^2_{{final}} - 1 '
        r'< {:.1e}\times(\chi^2_{{initial}}/\chi^2_{{final}} - 1) $'
        .format(chi2_ratio))
    legend_elements = []
    for j,f_sample in enumerate(f_sample_list):
        label = '$f_{{sample}}={:g}$'.format(f_sample)
        legend_elements.append(Line2D([0], [0],
            color=colors[j%len(colors)], label=label
            ))
    for j, f_knee in enumerate(f_knee_list):
        label = '$f_{{knee}}={:g}$'.format(f_knee)
        legend_elements.append(Line2D([0], [0], 
            marker=markers[j%len(markers)], color='black',
            markerfacecolor='black', linewidth=0, label=label 
            ))

    for j, f_sample in enumerate(f_sample_list):
        color = colors[j%len(colors)]
        for k, f_knee in enumerate(f_knee_list):
            marker = markers[k%len(markers)]
            for data_dic in data_list:
                if (data_dic['f_sample'] == f_sample
                        and data_dic['f_knee'] == f_knee):
                    # add random number [-1,1) to vertical value so 
                    # so points won't overlap
                    plt.plot(
                        data_dic['condition_number'],
                        data_dic['stop_point_list'][i]+2*np.random.rand()-1,
                        marker, color=color, markersize=10)
    ax.set_xscale('log')
    ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(
        base=10, subs='all', numticks=25))
    ax.set_xlabel('condition number')
    #ax.set_xlim(1, condition_number_arr.max()*1.2)
    plt.ylim(-1, max_iter+1)
    plt.grid(True)
    plt.legend(handles=legend_elements, 
        loc='center left', bbox_to_anchor=(1.01,0.5))
    #plt.savefig(figs_dir/(description+'.jpeg'), 
    #    bbox_inches='tight')
    plt.savefig(figs_dir/(description+'.pdf'), 
        bbox_inches='tight')
    #tikzplotlib.clean_figure()
    #tikzplotlib.save(figs_dir/(description+'.tex') )
    plt.close()




