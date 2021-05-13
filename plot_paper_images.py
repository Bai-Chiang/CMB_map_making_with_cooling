import numpy as np
from numpy import fft
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy import stats, interpolate
import pickle
from pathlib import Path, PurePath

#from Map import Map

figs_dir = Path('images/paper/').expanduser()
figs_dir.mkdir(parents=True, exist_ok=True)

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
num_snapshots = parameters_dic['num_snapshots']
#max_iter = parameters_dic['max_iter']
num_iter = max_iter = parameters_dic['num_iter']
results_dir = parameters_dic['results_dir']
cache_dir = parameters_dic['cache_dir']
seed = parameters_dic['seed']
f_scan_list = parameters_dic['f_scan_list']
condition_number_arr = parameters_dic['condition_number_arr']
#num_eta_iter_per_eta = parameters_dic['num_eta_iter_per_eta']
num_eta_arr = parameters_dic['num_eta_arr']
f_sample_knee_apo_arr = parameters_dic['f_sample_knee_apo_arr']
offsets = parameters_dic['offsets']
comps = parameters_dic['comps']
f_sample_list = parameters_dic['f_sample_list']
f_knee_list = parameters_dic['f_knee_list']


with open('data_list', 'rb') as _file:
    data_list = pickle.load(_file)   # data for each map

P_f = np.zeros(shape=(condition_number_arr.shape[0],
    data_list[0]['frequencies'].shape[0]))
condi_num_list = []
f_scan_results_list = []
results_list = []
for data_dic in  data_list:
    relative_dir = data_dic['relative_dir']

    # load info
    f = data_dic['frequencies']
    noise_power_spectrum = data_dic['noise_power_spectrum']
    f_scan = data_dic['f_scan']
    f_scan_results_list.append(f_scan)
    f_sample = data_dic['f_sample']
    f_knee = data_dic['f_knee']
    f_apo = data_dic['f_apo']
    scan_info = data_dic['scan_info']
    scan_info_latex = data_dic['scan_info_latex']
    noiseless_map = data_dic['noiseless_map']
    condition_number = data_dic['condition_number']
    condi_num_list.append(condition_number)
    print('condition number = {:.5e}, f_apo = {:.5e}, f_scan = {:.5e}'.format(
        condition_number, f_apo, f_scan))
    if condition_number  < 1e3:
        #plot_dir = figs_dir/('{}'.format(f_scan))/'small_condition_num'
        #plot_dir.mkdir(parents=True, exist_ok=True)
        P_f[0,:] = noise_power_spectrum
    elif condition_number > 1e5 and condition_number < 1e7:
        #plot_dir = figs_dir/('{}'.format(f_scan))/'medium_condition_num'
        #plot_dir.mkdir(parents=True, exist_ok=True)
        P_f[1,:] = noise_power_spectrum
    elif condition_number > 1e11:
        #plot_dir = figs_dir/('{}'.format(f_scan))/'large_condition_num'
        #plot_dir.mkdir(parents=True, exist_ok=True)
        P_f[2,:] = noise_power_spectrum
    else:
        print('Error! Check initialize_params.py using default params')
        break
    chi2_min = data_dic['chi2_min']


    #results_list = []
    #description_list = []
    description_latex_list = []
    chi2_list = []

    # CG with simple preconditioner
    description_latex_list.append('CG')
    with open (data_dic['CG_SP_file'], 'rb') as _file:
        CG_SP_result = pickle.load(_file)
        chi2 = CG_SP_result['chi2_hist']
        chi2_list.append((chi2-chi2_min)/(chi2[0]-chi2_min))

    # CG with perturbative auto eta
    description_latex_list.append('CG with $\eta$')
    with open (data_dic['CG_PT_auto_eta_file'], 'rb') as _file:
        CG_PT_result = pickle.load(_file)
        chi2 = CG_PT_result['chi2_hist']
        chi2_list.append((chi2-chi2_min)/(chi2[0]-chi2_min))

    # CG perturbation manual eta
    for num_eta in num_eta_arr:
        description_latex_list.append(
            'CG with $n_{{\eta}} = {:d}$'.format(num_eta)
        )
        with open (data_dic['CG_PT_manual_ln_{:d}_eta_file'.format(num_eta)],
                'rb') as _file:
            CG_PT_result = pickle.load(_file)
            chi2 = CG_PT_result['chi2_hist']
            chi2_list.append((chi2-chi2_min)/(chi2[0]-chi2_min))

    results_list.append({'chi2':chi2_list,'description':description_latex_list})




# power spectrum
plt.figure(figsize=(3,3))
plt.plot(f[1:], P_f[0,1:], label=r'$\kappa = 10^2$')
plt.plot(f[1:], P_f[1,1:], label=r'$\kappa = 10^6$')
plt.plot(f[1:], P_f[2,1:], label=r'$\kappa = 10^{12}$')
plt.xlabel('$f$')
plt.xscale('log')
plt.ylim(1,2e11)
plt.yscale('log')
plt.ylabel('$P(f)$')
plt.legend()
plt.savefig(figs_dir/'P_f.pdf', bbox_inches="tight")
#plt.show()
plt.close()


# plot (Χ²-min)/(min - ini)
for f_scan in f_scan_list:
    fig = plt.figure(figsize=(9,3))
    gs = fig.add_gridspec(1,3, wspace=0)
    axs = gs.subplots(sharey='row')
    for i in range(len(results_list)):
        if f_scan_results_list[i] == f_scan:
            condition_number = condi_num_list[i]
            if condition_number  < 1e3:
                fig_index = 0
                title = r'$\kappa=10^2$'
            elif condition_number > 1e5 and condition_number < 1e7:
                fig_index = 1
                title = r'$\kappa=10^6$'
            elif condition_number > 1e11:
                fig_index = 2
                title = r'$\kappa=10^{12}$'
            for chi2, description in zip(results_list[i]['chi2'], results_list[i]['description']):
                axs[fig_index].plot(chi2, label=description)
            axs[fig_index].set_title(title)
            axs[fig_index].set_yscale('log')
            axs[fig_index].set_ylabel(r'$\frac{\chi^2-\chi^2_{ini}} {\chi^2_{final}- \chi^2_{ini}}$') 
            axs[fig_index].grid()
    axs[1].set_xlabel('number of iteration')
    axs[2].legend(bbox_to_anchor=(1,1), loc="upper left")
    for ax in axs:
        ax.label_outer()
    #plt.show()
    plt.savefig(figs_dir/('f_scan={}.pdf'.format(f_scan)), bbox_inches="tight")
    plt.close()

# plot (Χ²-min)/(min - ini)
f_scan = 0.1
fig = plt.figure(figsize=(9,3))
gs = fig.add_gridspec(1,3, wspace=0)
axs = gs.subplots(sharey='row')
for i in range(len(results_list)):
    if f_scan_results_list[i] == f_scan:
        condition_number = condi_num_list[i]
        if condition_number  < 1e3:
            fig_index = 0
            title = r'$\kappa=10^2$'
        elif condition_number > 1e5 and condition_number < 1e7:
            fig_index = 1
            title = r'$\kappa=10^6$'
        elif condition_number > 1e11:
            fig_index = 2
            title = r'$\kappa=10^{12}$'
        for j in range(2):
            chi2 = results_list[i]['chi2'][j]
            description = results_list[i]['description'][j]
            axs[fig_index].plot(chi2, label=description)
        axs[fig_index].set_title(title)
        axs[fig_index].set_yscale('log')
        axs[fig_index].set_ylabel(r'$\frac{\chi^2-\chi^2_{ini}} {\chi^2_{final}- \chi^2_{ini}}$') 
        axs[fig_index].grid()
axs[1].set_xlabel('number of iteration')
axs[2].legend(bbox_to_anchor=(1,1), loc="upper left")
for ax in axs:
    ax.label_outer()
#plt.show()
plt.savefig(figs_dir/('f_scan={}_CG.pdf'.format(f_scan)), bbox_inches="tight")
plt.close()

