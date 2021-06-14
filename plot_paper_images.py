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

figs_dir = Path('figs/paper/').expanduser()
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
#num_snapshots = parameters_dic['num_snapshots']
#max_iter = parameters_dic['max_iter']
num_iter = max_iter = parameters_dic['num_iter']
results_dir = parameters_dic['results_dir']
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


# plot power spectrum
plt.figure(figsize=(3,3))
f = np.linspace(100./2**20, (100./2**20)*(2**19+1), 2**19)
plt.plot(f, 10*(1+(10**3+0**3)/(f**3+0**3)), 
    label=r'$f_{knee} = 10$ $f_{apo} = 0$'
)
plt.plot(f, 10*(1+(10**3+0.1**3)/(f**3+0.1**3)), 
    label=r'$f_{knee}=10$ $f_{apo}=0.1$'
)
plt.plot(f, 10*(1+(10**3+1**3)/(f**3+1**3)), 
    label=r'$f_{knee}=10$ $f_{apo}=1$'
)
plt.plot(f, 10*(1+(0.1**3+0**3)/(f**3+0**3)), ':',
    label=r'$f_{knee} = 0.1$ $f_{apo} = 0$'
)
plt.plot(f, 10*(1+(0.1**3+0.001**3)/(f**3+0.001**3)), ':',
    label=r'$f_{knee}=0.1$ $f_{apo}=0.001$'
)
plt.plot(f, 10*(1+(0.1**3+0.01**3)/(f**3+0.01**3)), ':',
    label=r'$f_{knee}=0.1$ $f_{apo}=0.01$'
)
plt.axvline(x=0.1, color="black", linestyle="-")
plt.xlabel('$f$')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('$P(f)$ ($\mu K^2$)')
plt.legend(bbox_to_anchor=(1,1), loc="upper left")
#plt.show()
plt.savefig(figs_dir/'P_f.pdf', bbox_inches="tight")
plt.close()


with open('data_list', 'rb') as _file:
    data_list = pickle.load(_file)   # data for each map


# plot 1/f results
fig = plt.figure(figsize=(9,3))
gs = fig.add_gridspec(1,3, wspace=0)
axs = gs.subplots(sharey='row')
for data_dic in data_list:
    f_scan = data_dic['f_scan']
    f_knee = data_dic['f_knee']
    f_apo = data_dic['f_apo']
    chi2_min = data_dic['chi2_min']
    if f_knee == 0.1 and f_apo == 0:
        fig_index = 0
        axs[0].set_title(r'$f_{knee}=0.1$')
    elif f_knee == 0.5 and f_apo == 0:
        fig_index = 1
        axs[1].set_title(r'$f_{knee}=0.5$')
    elif f_knee == 1 and f_apo == 0:
        fig_index = 2
        axs[2].set_title(r'$f_{knee}=1.0$')
    else:
        continue
    with open (data_dic['CG_SP_file'], 'rb') as _file:
        CG_SP_result = pickle.load(_file)
        chi2 = CG_SP_result['chi2_hist']
        axs[fig_index].plot(
            (chi2-chi2_min)/(chi2[0]-chi2_min),
            label='CG'
        )
    with open (data_dic['CG_eta_file'], 'rb') as _file:
        CG_eta_result = pickle.load(_file)
        chi2 = CG_eta_result['chi2_hist']
        axs[fig_index].plot(
            (chi2-chi2_min)/(chi2[0]-chi2_min),
            label='CG with $\eta$'
        )
    axs[fig_index].set_yscale('log')
    axs[fig_index].set_ylim(1e-12,1)
    axs[fig_index].grid()
axs[0].set_ylabel(r'$\frac{\chi^2-\chi^2_{min}} {\chi^2_{ini}- \chi^2_{min}}$') 
axs[1].set_xlabel('number of iteration')
axs[2].legend(bbox_to_anchor=(1,1), loc="upper left")
for ax in axs:
    ax.label_outer()
#plt.show()
plt.savefig(figs_dir/'pink_noise_chi2.pdf', bbox_inches="tight")
plt.close()


# plot apodized noise results
fig = plt.figure(figsize=(9,3))
gs = fig.add_gridspec(1,3, wspace=0)
axs = gs.subplots(sharey='row')
for data_dic in data_list:
    f_scan = data_dic['f_scan']
    f_knee = data_dic['f_knee']
    f_apo = data_dic['f_apo']
    chi2_min = data_dic['chi2_min']
    if f_knee == 10 and f_apo == 0:
        fig_index = 0
        axs[0].set_title(r'$f_{apo}=0$')
    elif f_knee == 10 and f_apo == 0.1:
        fig_index = 1
        axs[1].set_title(r'$f_{apo}=0.1$')
    elif f_knee == 10 and f_apo == 1:
        fig_index = 2
        axs[2].set_title(r'$f_{apo}=1.0$')
    else:
        continue
    with open (data_dic['CG_SP_file'], 'rb') as _file:
        CG_SP_result = pickle.load(_file)
        chi2 = CG_SP_result['chi2_hist']
        axs[fig_index].plot(
            (chi2-chi2_min)/(chi2[0]-chi2_min),
            label='CG'
        )
    with open (data_dic['CG_eta_file'], 'rb') as _file:
        CG_eta_result = pickle.load(_file)
        chi2 = CG_eta_result['chi2_hist']
        axs[fig_index].plot(
            (chi2-chi2_min)/(chi2[0]-chi2_min),
            label='CG with $\eta$'
        )
    axs[fig_index].set_yscale('log')
    axs[fig_index].set_ylim(1e-12,1)
    axs[fig_index].grid()
axs[0].set_ylabel(r'$\frac{\chi^2-\chi^2_{min}} {\chi^2_{ini}- \chi^2_{min}}$') 
axs[1].set_xlabel('number of iteration')
axs[2].legend(bbox_to_anchor=(1,1), loc="upper left")
for ax in axs:
    ax.label_outer()
#plt.show()
plt.savefig(figs_dir/'apodized_noise_chi2.pdf', bbox_inches="tight")
plt.close()

# plot differnt eta results
fig = plt.figure(figsize=(9,3))
gs = fig.add_gridspec(1,3, wspace=0)
axs = gs.subplots(sharey='row')
for data_dic in data_list:
    f_scan = data_dic['f_scan']
    f_knee = data_dic['f_knee']
    f_apo = data_dic['f_apo']
    chi2_min = data_dic['chi2_min']
    if f_knee == 10 and f_apo == 0:
        fig_index = 0
        axs[0].set_title('$f_{knee}=10$\n$f_{apo}=0$')
    elif f_knee == 1 and f_apo == 0:
        fig_index = 1
        axs[1].set_title('$f_{knee}=1$\n$f_{apo}=0$')
    elif f_knee == 0.1 and f_apo == 0.001:
        fig_index = 2
        axs[2].set_title('$f_{knee}=0.1$\n$f_{apo}=0.001$')
    else:
        continue
    with open (data_dic['CG_SP_file'], 'rb') as _file:
        CG_SP_result = pickle.load(_file)
        chi2 = CG_SP_result['chi2_hist']
        axs[fig_index].plot(
            (chi2-chi2_min)/(chi2[0]-chi2_min),
            label='CG'
        )
    with open (data_dic['CG_eta_file'], 'rb') as _file:
        CG_eta_result = pickle.load(_file)
        chi2 = CG_eta_result['chi2_hist']
        axs[fig_index].plot(
            (chi2-chi2_min)/(chi2[0]-chi2_min),
            label='CG with $\eta$'
        )
    for num_eta in num_eta_arr:
        with open (data_dic['CG_manual_ln_{:d}_eta_file'.format(num_eta)],
                'rb') as _file:
            CG_eta_result = pickle.load(_file)
            chi2 = CG_eta_result['chi2_hist']
            axs[fig_index].plot(
                (chi2-chi2_min)/(chi2[0]-chi2_min),
                label='CG with $n_{{\eta}} = {:d}$'.format(num_eta)
            )
    axs[fig_index].set_yscale('log')
    axs[fig_index].set_ylim(1e-12,1)
    axs[fig_index].grid()
axs[0].set_ylabel(r'$\frac{\chi^2-\chi^2_{min}} {\chi^2_{ini}- \chi^2_{min}}$') 
axs[1].set_xlabel('number of iteration')
axs[2].legend(bbox_to_anchor=(1,1), loc="upper left")
for ax in axs:
    ax.label_outer()
#plt.show()
plt.savefig(figs_dir/'chi2_neta.pdf', bbox_inches="tight")
plt.close()

# plot exact eta results
fig = plt.figure(figsize=(9,3))
gs = fig.add_gridspec(1,3, wspace=0)
axs = gs.subplots(sharey='row')
for data_dic in data_list:
    f_scan = data_dic['f_scan']
    f_knee = data_dic['f_knee']
    f_apo = data_dic['f_apo']
    chi2_min = data_dic['chi2_min']
    if f_knee == 10 and f_apo == 0:
        fig_index = 0
        axs[0].set_title('$f_{knee}=10$\n$f_{apo}=0$')
    elif f_knee == 1 and f_apo == 0:
        fig_index = 1
        axs[1].set_title('$f_{knee}=1$\n$f_{apo}=0$')
    elif f_knee == 0.1 and f_apo == 0.001:
        fig_index = 2
        axs[2].set_title('$f_{knee}=0.1$\n$f_{apo}=0.001$')
    else:
        continue
    with open (data_dic['CG_SP_file'], 'rb') as _file:
        CG_SP_result = pickle.load(_file)
        chi2 = CG_SP_result['chi2_hist']
        axs[fig_index].plot(
            (chi2-chi2_min)/(chi2[0]-chi2_min),
            label='CG'
        )
    with open (data_dic['CG_eta_file'], 'rb') as _file:
        CG_eta_result = pickle.load(_file)
        chi2 = CG_eta_result['chi2_hist']
        axs[fig_index].plot(
            (chi2-chi2_min)/(chi2[0]-chi2_min),
            label='CG with $\eta$'
        )
    with open (data_dic['CG_exact_eta_file'], 'rb') as _file:
        CG_exact_eta_result = pickle.load(_file)
        chi2 = CG_exact_eta_result['chi2_hist']
        axs[fig_index].plot(
            (chi2-chi2_min)/(chi2[0]-chi2_min),
            label=r'CG with $\delta\eta = \frac{\chi^2}{-\frac{d}{d\eta}\chi^2}$'
        )
    #for num_eta in num_eta_arr:
    #    with open (data_dic['CG_manual_ln_{:d}_eta_file'.format(num_eta)],
    #            'rb') as _file:
    #        CG_eta_result = pickle.load(_file)
    #        chi2 = CG_eta_result['chi2_hist']
    #        axs[fig_index].plot(
    #            (chi2-chi2_min)/(chi2[0]-chi2_min),
    #            label='CG with $n_{{\eta}} = {:d}$'.format(num_eta)
    #        )
    axs[fig_index].set_yscale('log')
    axs[fig_index].set_ylim(1e-12,1)
    axs[fig_index].grid()
axs[0].set_ylabel(r'$\frac{\chi^2-\chi^2_{min}} {\chi^2_{ini}- \chi^2_{min}}$') 
axs[1].set_xlabel('number of iteration')
axs[2].legend(bbox_to_anchor=(1,1), loc="upper left")
for ax in axs:
    ax.label_outer()
#plt.show()
plt.savefig(figs_dir/'chi2_exact_eta.pdf', bbox_inches="tight")
plt.close()
