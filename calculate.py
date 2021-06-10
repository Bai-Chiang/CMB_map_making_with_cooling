import numpy as np
from mpi4py import MPI
import pickle
from pathlib import Path, PurePath

from Map import Map

force_recalculate = False

with open('parameters_dic', 'rb') as _file:
    parameters_dic = pickle.load(_file)

num_sample = parameters_dic['num_sample']
x_max = parameters_dic['x_max']
y_max = parameters_dic['y_max']
sig_amp = parameters_dic['sig_amp']
noise_sigma2 = parameters_dic['noise_sigma2']
noise_index = parameters_dic['noise_index']
num_pix_x = parameters_dic['num_pix_x']
num_pix_y = parameters_dic['num_pix_y']
crosslink = parameters_dic['crosslink']
#num_snapshots = parameters_dic['num_snapshots']
#max_iter = parameters_dic['max_iter']
num_iter = parameters_dic['num_iter']
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


data_list_rank = []

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

i_case = 0
for f_scan in f_scan_list:
    for f_sample, f_knee, f_apo in f_sample_knee_apo_arr:

        if i_case%size != rank:
            i_case += 1
            continue
        
        data_dic = {}
        data_dic['f_scan'] = f_scan
        data_dic['f_sample'] = f_sample
        data_dic['f_knee'] = f_knee
        data_dic['f_apo'] = f_apo


        ## generate TOD
        _map = Map(
            x_max, y_max, num_pix_x, num_pix_y,
            seed, cache_dir, force_recalculate
        )
        _map.generate_scan_info(f_scan, offsets[:,1], offsets[:,2],
            comps, num_sample, f_sample, crosslink)
        scan_info = ('f_scan={:.3g} f_knee={:.5g} f_apo={:.5g} '
            'f_sample={:g}').format(f_scan, f_knee, f_apo, f_sample)
        scan_info_latex = ('$f_{{scan}}={:.3g}$ $f_{{knee}}={:.5g}$ '
            '$f_{{apo}}={:.5g}$ $f_{{sample}}={:g}$').format(
            f_scan, f_knee, f_apo, f_sample)

        _map.generate_noiseless_signal(sig_amp=sig_amp)
        _map.generate_noise(f_knee, f_apo, 
            noise_index=noise_index, noise_sigma2=noise_sigma2)
        _map.get_tod()

        data_dic['scan_info'] = scan_info
        data_dic['scan_info_latex'] = scan_info_latex
        data_dic['noiseless_map'] = _map.noiseless_map
        data_dic['frequencies'] = _map.f

        # powerspctrum is monotonic decreasing function so condition
        # number is 0th element devided by last element
        condition_number = _map.N_f_diag[0] / _map.N_f_diag[-1]  
        condition_number_wo_f0 = _map.N_f_diag[1] / _map.N_f_diag[-1]
        #total_noise = noise_sigma2 * f_sample/2 \
        #    + noise_sigma2 / f_apo * (f_knee**2 + f_apo**2) \
        #    * np.arctan(f_sample/(f_apo*2))
        data_dic['condition_number'] = condition_number
        data_dic['condition_number_wo_f0'] = condition_number_wo_f0
        data_dic['noise_power_spectrum'] = _map.noise_power_spectrum
        data_dic['relative_dir'] \
            = PurePath(_map.map_dir).relative_to(cache_dir)

        # get etas that makes chi2 decrease
        data_dic['chi2_vs_eta'] = _map.get_chi2_vs_eta()
        data_dic['chi2_min'] = _map.chi2_min

        # CG with simple preconditioner
        print('CG simple preconditioner, max iter = {:d}'.format(num_iter))
        CG_SP_description = ('CG step={:d} '
            'preconditioner=simple preconditioner').format(num_iter)
        CG_SP_description_latex = ('CG step={:d} '
            'preconditioner=$P^T P$').format(num_iter)
        CG_SP_file,_ = _map.conjugate_gradient_solver(
            num_iter,
            preconditioner_inv=_map.PTP_preconditioner,
            preconditioner_description='PTP',
            #num_snapshots=num_snapshots,
            )
        data_dic['CG_SP_description'] = CG_SP_description
        data_dic['CG_SP_description_latex'] = CG_SP_description_latex
        data_dic['CG_SP_file'] = CG_SP_file


        # perturbative auto eta
        print('CG perturbative eta')
        CG_PT_description = ('CG perturbative eta preconditioner=PTP')
        CG_PT_description_latex =\
            ('CG perturbative $\eta$ preconditioner=$P^T P$')
        CG_PT_file, CG_PT_result =\
            _map.conjugate_gradient_solver_perturbative_eta(
                num_iter,
                preconditioner_inv=_map.PTP_preconditioner,
                preconditioner_description='PTP',
                #next_eta_ratio = 1e-2,
            )
        data_dic['CG_PT_auto_eta_description'] = CG_PT_description
        data_dic['CG_PT_auto_eta_description_latex'] = CG_PT_description_latex
        data_dic['CG_PT_auto_eta_file'] = CG_PT_file
        data_dic['etas'] = CG_PT_result['etas']


        # CG perturbation manual eta
        for num_eta in num_eta_arr:
            tau = np.min(_map.N_f_diag)
            Nbar_f = _map.N_f_diag - tau
            eta_min = tau/Nbar_f.max()
            etas=np.logspace(
                np.log(eta_min), 0, num=num_eta, base=np.e
            )
            print('CG manual ln {:d} eta'.format(num_eta))
            CG_PT_description = ('CG manual ln {:d} eta '
                'preconditioner=PTP').format(num_eta)
            CG_PT_description_latex = ('CG manual $\ln$ scale, '
                '$n_{{\eta}}={:d}$ preconditioner=$P^T P$').format(num_eta)
            CG_PT_file,_ =\
                _map.conjugate_gradient_solver_perturbative_eta(
                    num_iter,
                    preconditioner_inv=_map.PTP_preconditioner,
                    preconditioner_description='PTP',
                    #next_eta_ratio = 1e-2,
                    etas=etas,
                )
            data_dic['CG_PT_manual_ln_{:d}_eta_description'.format(num_eta)]\
                = CG_PT_description
            data_dic['CG_PT_manual_ln_{:d}_eta_description_latex'
                .format(num_eta)]\
                = CG_PT_description_latex
            data_dic['CG_PT_manual_ln_{:d}_eta_file'.format(
                num_eta)] = CG_PT_file

        ## MF iteration
        #for num_eta in num_eta_arr:
        #    tau = np.min(_map.N_f_diag)
        #    Nbar_f = _map.N_f_diag - tau
        #    eta_min = tau/Nbar_f.max()
        #    etas=np.logspace(
        #        np.log(eta_min), 0, num=num_eta, base=np.e
        #    )
        #    print('MF ln {:d}x1 eta'.format(num_eta))
        #    MF_description = ('MF ln {:d}x1 eta '.format(num_eta))
        #    MF_description_latex = (r'MF $\ln$ scale, '
        #        r'$\lambda$ ${:d}$ $\times$ $1$').format(num_eta)
        #    MF_file,_ =\
        #        _map.messenger_field_solver(
        #            lambs=1/etas,
        #            num_iter_per_lamb=1,
        #            num_iter=num_iter,
        #        )
        #    data_dic['MF_ln_{:d}_eta_description'.format(num_eta)]\
        #        = MF_description
        #    data_dic['MF_ln_{:d}_eta_description_latex'
        #        .format(num_eta)]\
        #        = MF_description_latex
        #    data_dic['MF_ln_{:d}_eta_file'.format(
        #        num_eta)] = MF_file


        data_list_rank.append(data_dic)

        i_case += 1


data_list_gathered = comm.gather(data_list_rank)
if rank == 0:
    data_list = []
    for _list in data_list_gathered:
        if len(_list) != 0:
            for i in _list:
                data_list.append(i)
    with open('data_list', 'wb') as _file:
        pickle.dump(data_list, _file)
            
