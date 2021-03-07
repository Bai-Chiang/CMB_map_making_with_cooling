import numpy as np
from mpi4py import MPI
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
num_snapshots = parameters_dic['num_snapshots']
#max_iter = parameters_dic['max_iter']
num_iter = parameters_dic['num_iter']
results_dir = parameters_dic['results_dir']
figs_dir = parameters_dic['figs_dir']
cache_dir = parameters_dic['cache_dir']
seed = parameters_dic['seed']
f_scan_list = parameters_dic['f_scan_list']
condition_number_arr = parameters_dic['condition_number_arr']
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
        _map = Map(x_max, y_max, num_pix_x, num_pix_y, seed, cache_dir)
        _map.generate_scan_info(f_scan, offsets[:,1], offsets[:,2],
            comps, num_sample, f_sample, crosslink)
        scan_info = ('f_scan={:.3g} f_knee={:.5g} f_apo={:.5g} '
            'f_sample={:g}').format(f_scan, f_knee, f_apo, f_sample)
        scan_info_latex = ('$f_{{scan}}={:.3g}$ $f_{{knee}}={:.5g}$ '
            '$f_{{apo}}={:.5g}$ $f_{{sample}}={:g}$').format(
            f_scan, f_knee, f_apo, f_sample)

        _map.generate_noiseless_signal(sig_amp=sig_amp)
        _map.generate_noise(f_knee, f_apo, noise_sigma2=noise_sigma2)
        _map.get_tod()

        data_dic['scan_info'] = scan_info
        data_dic['scan_info_latex'] = scan_info_latex
        data_dic['noiseless_map'] = _map.noiseless_map
        data_dic['frequencies'] = _map.f

        # powerspctrum is monotonic decreasing function so condition
        # number is 0th element devided by last element
        condition_number = _map.N_f_diag[0] / _map.N_f_diag[-1]  
        condition_number_wo_f0 = _map.N_f_diag[1] / _map.N_f_diag[-1]
        total_noise = noise_sigma2 * f_sample/2 \
            + noise_sigma2 / f_apo * (f_knee**2 + f_apo**2) \
            * np.arctan(f_sample/(f_apo*2))
        data_dic['condition_number'] = condition_number
        data_dic['condition_number_wo_f0'] = condition_number_wo_f0
        data_dic['noise_power_spectrum'] = _map.noise_power_spectrum
        data_dic['relative_dir'] \
            = PurePath(_map.map_dir).relative_to(cache_dir)

        # get etas that makes chi2 decrease
        data_dic['chi2_vs_eta'] = _map.get_chi2_vs_eta()
        data_dic['chi2_min'] = _map.chi2_min
        #log_eta_linear_chi2_interp = _map.get_log_eta_linear_chi2()
        #data_dic['log_eta_linear_chi2_interp'] = log_eta_linear_chi2_interp
        #log_eta_quadratic_chi2_interp = _map.get_log_eta_quadratic_chi2()
        #data_dic['log_eta_quadratic_chi2_interp'] = log_eta_quadratic_chi2_interp
        #log_eta_exp_chi2_interp = _map.get_log_eta_exp_chi2()
        #data_dic['log_eta_exp_chi2_interp'] = log_eta_exp_chi2_interp

        # CG with simple preconditioner
        print('CG simple preconditioner, max iter = {:d}'.format(num_iter))
        CG_SP_description = ('CG step={:d} '
            'preconditioner=simple preconditioner').format(num_iter)
        CG_SP_description_latex = ('CG step={:d} '
            'preconditioner=$P^T P$').format(num_iter)
        CG_SP_file,_ = _map.conjugate_gradient_solve_map(
            num_iter,
            preconditioner_inv=_map.PTP_preconditioner,
            preconditioner_description='PTP',
            num_snapshots=num_snapshots,
            )
        data_dic['CG_SP_description'] = CG_SP_description
        data_dic['CG_SP_description_latex'] = CG_SP_description_latex
        data_dic['CG_SP_file'] = CG_SP_file


        # Perturbative optimal precondionter 
        #eta1 = _map.N_f_diag.min()/(_map.N_f_diag.max()-_map.N_f_diag.min())
        #_i = np.arange(int(np.floor(np.log2(1/eta1 + 1)))) + 1
        #opt_lambs = eta1*( 2**(_i) - 1 )
        #opt_lambs = np.append(opt_lambs, 1)
        #data_dic['opt_lambs'] = opt_lambs
        #print('{:d}x{:d} CG opt perturbation '.format(
        #    len(opt_lambs), 1,))
        #CG_PT_description = ('CG with opt eta {:d}x{:d} '
        #    'preconditioner=PTP').format(len(opt_lambs), 1 )
        #CG_PT_description_latex = ('CG with opt $\eta$ '
        #    '{:d}x{:d} preconditioner=$P^T P$').format(
        #    len(opt_lambs), 1, )
        #CG_PT_file,_ = _map.conjugate_gradient_solver_perturbative_noise(
        #    opt_lambs,
        #    1,
        #    num_iter,
        #    preconditioner_inv=_map.PTP_preconditioner, 
        #    preconditioner_description='PTP',
        #    num_snapshots=num_snapshots,
        #    )
        #data_dic['CG_PT_opt_{:d}x{:d}_description'
        #    .format(len(opt_lambs), 1)]\
        #    = CG_PT_description
        #data_dic['CG_PT_opt_{:d}x{:d}_description_latex'
        #    .format(len(opt_lambs), 1)]\
        #    = CG_PT_description_latex
        #data_dic['CG_PT_opt_{:d}x{:d}_file'.format(
        #    len(opt_lambs), 1)] = CG_PT_file


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
            CG_PT_description_latex = ('CG manual $\ln$ {:d} $\eta$ '
                'preconditioner=$P^T P$').format(num_eta)
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

        #for num_eta, num_iter_eta in num_eta_iter_per_eta:

        #    # CG perturbation ln eta
        #    etas = np.logspace(
        #        np.log(
        #             _map.N_f_diag.min()
        #             /(_map.N_f_diag.max()-_map.N_f_diag.min())
        #        ),
        #        0, num=num_eta, base=np.e,)
        #    print('{:d}x{:d} CG ln perturbation '.format(
        #        num_eta, num_iter_eta, ))
        #    CG_PT_description = ('CG with ln eta {:d}x{:d} '
        #        'preconditioner=PTP').format(num_eta, num_iter_eta, )
        #    CG_PT_description_latex = ('CG with $\ln$ $\eta$ '
        #        '{:d}x{:d} preconditioner=$P^T P$').format(
        #        num_eta, num_iter_eta, )
        #    CG_PT_file,_ = _map.conjugate_gradient_solver_perturbative_manual_eta(
        #        etas,
        #        num_iter_eta,
        #        num_iter,
        #        preconditioner_inv=_map.PTP_preconditioner, 
        #        preconditioner_description='PTP',
        #        num_snapshots=num_snapshots,
        #        )
        #    data_dic['CG_PT_ln_{:d}x{:d}_description'
        #        .format(num_eta, num_iter_eta)]\
        #        = CG_PT_description
        #    data_dic['CG_PT_ln_{:d}x{:d}_description_latex'
        #        .format(num_eta, num_iter_eta)]\
        #        = CG_PT_description_latex
        #    data_dic['CG_PT_ln_{:d}x{:d}_file'.format(
        #        num_eta, num_iter_eta)] = CG_PT_file



            # CG perturbation log10 lambda
            #lambs = np.logspace(
            #    np.log10(
            #         _map.N_f_diag.min()
            #         /(_map.N_f_diag.max()-_map.N_f_diag.min())
            #    ),
            #    0, num=num_lamb, base=10)
            #print('{:d}x{:d} CG log10 perturbation '.format(
            #    num_lamb, num_iter_lamb, ))
            #CG_PT_description = ('CG with log10 eta {:d}x{:d} '
            #    'preconditioner=PTP').format(num_lamb, num_iter_lamb, )
            #CG_PT_description_latex = ('CG with $\log_{{10}}$ $\eta$ '
            #    '{:d}x{:d} preconditioner=$P^T P$').format(
            #    num_lamb, num_iter_lamb, )
            #CG_PT_file,_ = _map.conjugate_gradient_solver_perturbative_noise(
            #    lambs,
            #    num_iter_lamb,
            #    num_iter,
            #    preconditioner_inv=_map.PTP_preconditioner, 
            #    preconditioner_description='PTP',
            #    num_snapshots=num_snapshots,
            #    )
            #data_dic['CG_PT_log10_{:d}x{:d}_description'
            #    .format(num_lamb, num_iter_lamb)]\
            #    = CG_PT_description
            #data_dic['CG_PT_log10_{:d}x{:d}_description_latex'
            #    .format(num_lamb, num_iter_lamb)]\
            #    = CG_PT_description_latex
            #data_dic['CG_PT_log10_{:d}x{:d}_file'.format(
            #    num_lamb, num_iter_lamb)] = CG_PT_file


            ## CG with MF cooling
            #lambs = np.logspace(
            #    np.log(_map.N_f_diag.max()/_map.N_f_diag.min()), 0,
            #    num=num_lamb, base=np.e)
            #lambs = np.concatenate([
            #    _map.N_f_diag.max()/_map.N_f_diag.min()*np.ones(3),
            #    lambs
            #    ])
            #lambs = np.linspace(_map.N_f_diag.max()/_map.N_f_diag.min(), 1,
            #    num_lamb)
            #print('{:d}x{:d} CG messenger field cooling'
            #    .format(num_lamb, num_iter_lamb))
            #CG_MF_description = ('CG with MF cooling {:d}x{:d} '
            #    'preconditioner=PTP λ0={:.5g}').format(
            #    num_lamb, num_iter_lamb, lambs[0])
            #CG_MF_description_latex = ('CG with MF cooling {:d}x{:d} '
            #    'preconditioner=$P^T P$ $\lambda_0={:.5g}$').format(
            #    num_lamb, num_iter_lamb, lambs[0])
            #CG_MF_file,_ = _map.conjugate_gradient_solve_map_cooling(
            #    lambs,
            #    num_iter_lamb,
            #    num_iter,
            #    preconditioner_inv=_map.PTP_preconditioner,
            #    preconditioner_description='PTP',
            #    num_snapshots=num_snapshots, 
            #    )
            #data_dic['CG_MF_{:d}x{:d}_description'
            #    .format(num_lamb, num_iter_lamb)]\
            #    = CG_MF_description
            #data_dic['CG_MF_{:d}x{:d}_description_latex'
            #    .format(num_lamb, num_iter_lamb)]\
            #    = CG_MF_description_latex
            #data_dic['CG_MF_{:d}x{:d}_file'.format(num_lamb, num_iter_lamb)]\
            #    = CG_MF_file


            ## CG with chi2 decrease linearly
            #x = np.linspace(0,1,num=num_lamb+1)[1:]
            #eta_arr = np.exp(log_eta_linear_chi2_interp(x))
            #print('{:d}x{:d} CG linear chi2'.format(
            #    num_lamb, num_iter_lamb, ))
            #CG_PT_description = ('CG linear chi2 {:d}x{:d} '
            #    'preconditioner=PTP').format(num_lamb, num_iter_lamb, )
            #CG_PT_description_latex = ('CG linear $\chi^2$ $\eta$ '
            #    '{:d}x{:d} preconditioner=$P^T P$').format(
            #    num_lamb, num_iter_lamb, )
            #CG_PT_file,_ = _map.conjugate_gradient_solver_perturbative_noise(
            #    eta_arr,
            #    num_iter_lamb,
            #    num_iter,
            #    preconditioner_inv=_map.PTP_preconditioner, 
            #    preconditioner_description='PTP',
            #    num_snapshots=num_snapshots,
            #    )
            #data_dic['CG_PT_linear_chi2_eta_{:d}x{:d}_description'
            #    .format(num_lamb, num_iter_lamb)]\
            #    = CG_PT_description
            #data_dic['CG_PT_linear_chi2_eta_{:d}x{:d}_description_latex'
            #    .format(num_lamb, num_iter_lamb)]\
            #    = CG_PT_description_latex
            #data_dic['CG_PT_linear_chi2_eta_{:d}x{:d}_file'.format(
            #    num_lamb, num_iter_lamb)] = CG_PT_file


            ## CG with chi2 decrease quadratically
            #x = np.linspace(0,1,num=num_lamb+1)[1:]
            #eta_arr = np.exp(log_eta_quadratic_chi2_interp(x))
            #print('{:d}x{:d} CG quadratic chi2'.format(
            #    num_lamb, num_iter_lamb, ))
            #CG_PT_description = ('CG quadratic chi2 {:d}x{:d} '
            #    'preconditioner=PTP').format(num_lamb, num_iter_lamb, )
            #CG_PT_description_latex = ('CG quadratic $\chi^2$ $\eta$ '
            #    '{:d}x{:d} preconditioner=$P^T P$').format(
            #    num_lamb, num_iter_lamb, )
            #CG_PT_file,_ = _map.conjugate_gradient_solver_perturbative_noise(
            #    eta_arr,
            #    num_iter_lamb,
            #    num_iter,
            #    preconditioner_inv=_map.PTP_preconditioner, 
            #    preconditioner_description='PTP',
            #    num_snapshots=num_snapshots,
            #    )
            #data_dic['CG_PT_quadratic_chi2_eta_{:d}x{:d}_description'
            #    .format(num_lamb, num_iter_lamb)]\
            #    = CG_PT_description
            #data_dic['CG_PT_quadratic_chi2_eta_{:d}x{:d}_description_latex'
            #    .format(num_lamb, num_iter_lamb)]\
            #    = CG_PT_description_latex
            #data_dic['CG_PT_quadratic_chi2_eta_{:d}x{:d}_file'.format(
            #    num_lamb, num_iter_lamb)] = CG_PT_file

            ## CG with chi2 decrease exponentially
            #x = np.linspace(0,1,num=num_lamb+1)[1:]
            #eta_arr = np.exp(log_eta_exp_chi2_interp(x))
            #print('{:d}x{:d} CG exp chi2'.format(
            #    num_lamb, num_iter_lamb, ))
            #CG_PT_description = ('CG exp chi2 {:d}x{:d} '
            #    'preconditioner=PTP').format(num_lamb, num_iter_lamb, )
            #CG_PT_description_latex = ('CG exp $\chi^2$ $\eta$ '
            #    '{:d}x{:d} preconditioner=$P^T P$').format(
            #    num_lamb, num_iter_lamb, )
            #CG_PT_file,_ = _map.conjugate_gradient_solver_perturbative_noise(
            #    eta_arr,
            #    num_iter_lamb,
            #    num_iter,
            #    preconditioner_inv=_map.PTP_preconditioner, 
            #    preconditioner_description='PTP',
            #    num_snapshots=num_snapshots,
            #    )
            #data_dic['CG_PT_exp_chi2_eta_{:d}x{:d}_description'
            #    .format(num_lamb, num_iter_lamb)]\
            #    = CG_PT_description
            #data_dic['CG_PT_exp_chi2_eta_{:d}x{:d}_description_latex'
            #    .format(num_lamb, num_iter_lamb)]\
            #    = CG_PT_description_latex
            #data_dic['CG_PT_exp_chi2_eta_{:d}x{:d}_file'.format(
            #    num_lamb, num_iter_lamb)] = CG_PT_file

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
            
