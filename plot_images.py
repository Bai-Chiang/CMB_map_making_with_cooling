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

figs_dir = Path('images/').expanduser()

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


for data_dic in  data_list:
    relative_dir = data_dic['relative_dir']

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
    print('condition number = {:.5e}, f_apo = {:.5e}, f_scan = {:.5e}'.format(
        condition_number, f_apo, f_scan))
    if condition_number  < 1e3:
        plot_dir = figs_dir/('{}'.format(f_scan))/'small_condition_num'
        plot_dir.mkdir(parents=True, exist_ok=True)
    elif condition_number > 1e5 and condition_number < 1e7:
        plot_dir = figs_dir/('{}'.format(f_scan))/'medium_condition_num'
        plot_dir.mkdir(parents=True, exist_ok=True)
    elif condition_number > 1e11:
        plot_dir = figs_dir/('{}'.format(f_scan))/'large_condition_num'
        plot_dir.mkdir(parents=True, exist_ok=True)
    else:
        print('Error! Check initialize_params.py using default params')
        break
    chi2_min = data_dic['chi2_min']


    results_list = []
    description_list = []
    description_latex_list = []

    # CG with simple preconditioner
    description_latex_list.append('CG')
    with open (data_dic['CG_SP_file'], 'rb') as _file:
        CG_SP_result = pickle.load(_file)
        results_list.append(CG_SP_result)

    # CG with perturbative auto eta
    description_latex_list.append('CG with $\eta$')
    with open (data_dic['CG_PT_auto_eta_file'], 'rb') as _file:
        CG_PT_result = pickle.load(_file)
        results_list.append(CG_PT_result)

    # CG perturbation manual eta
    for num_eta in num_eta_arr:
        description_latex_list.append(
            'CG with $n_{{\eta}} = {:d}$'.format(num_eta)
        )
        with open (data_dic['CG_PT_manual_ln_{:d}_eta_file'.format(num_eta)],
                'rb') as _file:
            CG_PT_result = pickle.load(_file)
            results_list.append(CG_PT_result)



    # power spectrum
    plt.figure(figsize=(3,3))
    plt.plot(f[1:], noise_power_spectrum[1:], 'k')
    plt.xlabel('$f$')
    plt.xscale('log')
    plt.ylim(1,2e11)
    plt.yscale('log')
    plt.ylabel('$P(f)$')
    plt.savefig(plot_dir/'P_f.pdf', bbox_inches="tight")
    #tikzplotlib.clean_figure()
    #tikzplotlib.save(plot_dir/'P_f.tex')
    plt.close()


    # plot Χ²/min - 1
    lines = [i['chi2_hist'] for i in results_list]
    plt.figure(figsize=(3,3))
    plt.xlabel('number of iteration')
    plt.yscale('log')
    plt.ylabel('$\chi^2/\chi^2_{final} - 1$') 
    for i in range(len(lines)):
        plt.plot(lines[i]/chi2_min - 1, '-', label=description_latex_list[i])
    #plt.legend(loc='upper right')
    plt.grid()
    plt.savefig(plot_dir/'chi2.pdf', bbox_inches="tight")
    #tikzplotlib.clean_figure()
    #tikzplotlib.save(
    #    plot_dir/'chi2.tex',
    #    axis_width='2in',
    #    axis_height='2in',
    #)
    plt.close()


    # plot η 
    lines = [i['etas_iter'] for i in results_list]
    plt.figure(figsize=(3,3))
    plt.xlabel('number of iteration')
    plt.ylabel('$\eta$') 
    plt.yscale('log')
    for i in range(len(lines)):
        plt.plot(np.arange(num_iter+1), lines[i], 
            '-', label=description_latex_list[i]
        )
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(plot_dir/'eta.pdf', bbox_inches="tight")
    #tikzplotlib.clean_figure()
    #tikzplotlib.save(plot_dir/'eta.tex')
    plt.close()


    if f_scan == 0.1:
        results_list = []
        description_list = []
        description_latex_list = []
        # CG with simple preconditioner
        description_latex_list.append('CG')
        with open (data_dic['CG_SP_file'], 'rb') as _file:
            CG_SP_result = pickle.load(_file)
            results_list.append(CG_SP_result)

        # CG with perturbative auto eta
        description_latex_list.append('CG with $\eta$')
        with open (data_dic['CG_PT_auto_eta_file'], 'rb') as _file:
            CG_PT_result = pickle.load(_file)
            results_list.append(CG_PT_result)

        # plot Χ²/min - 1
        lines = [i['chi2_hist'] for i in results_list]
        plt.figure(figsize=(3,3))
        plt.xlabel('number of iteration')
        plt.yscale('log')
        plt.ylabel('$\chi^2/\chi^2_{final} - 1$') 
        for i in range(len(lines)):
            plt.plot(lines[i]/chi2_min - 1, '-', label=description_latex_list[i])
        plt.legend(loc='upper right')
        plt.grid()
        plt.savefig(plot_dir/'chi2_CG.pdf', bbox_inches="tight")
        #tikzplotlib.clean_figure()
        #tikzplotlib.save(plot_dir/'chi2_CG.tex')
        plt.close()


        # plot η 
        lines = [i['etas_iter'] for i in results_list]
        plt.figure(figsize=(3,3))
        plt.xlabel('number of iteration')
        plt.ylabel('$\eta$') 
        plt.yscale('log')
        for i in range(len(lines)):
            plt.plot(np.arange(num_iter+1), lines[i], 
                '-', label=description_latex_list[i]
            )
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig(plot_dir/'eta_CG.pdf', bbox_inches="tight")
        #tikzplotlib.clean_figure()
        #tikzplotlib.save(plot_dir/'eta_CG.tex')
        plt.close()




plt.close('all')


