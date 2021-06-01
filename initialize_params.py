import numpy as np
from pathlib import Path, PurePath
import pickle


num_iter = 100
cache_dir = Path('cache').expanduser()
seed = 0
f_scan_list = [
    0.1,
]
num_eta_arr = np.array([5,15,30],dtype=int)
f_sample_list = [100,]
f_knee_list = [
    0.001,
    0.1,
    1,
    10,
    100,
    1000,
]


num_sample = int(2**20)
#condition_number_arr = np.logspace(0,20,11)[1:]
condition_number_arr = np.array([
    1e2,
    1e6,
    1e12,
])


x_max = y_max = 1*np.pi/180
sig_amp = 100
noise_sigma2 = 10
noise_index = 2
num_pix_x = num_pix_y = 512
crosslink=True
num_snapshots = 5

figs_dir = Path('figs/').expanduser()
results_dir = Path('results/').expanduser()


# f_apo calculated from f_knee, f_sample, and condition number
if noise_index == 2:
    f_sample_knee_apo_list = []
    for k in condition_number_arr:
        for f_sample in f_sample_list:
            for f_knee in f_knee_list:
                #f_min = f_sample/num_sample    # condition number wo 0f
                f_min = 0.0   # condition number
                f_max = f_sample/2
                a = 2*(k-1)
                b = (k-1)*f_knee**2 + (k-2)*f_max**2 + (2*k-1)*f_min**2
                c = (k-1)*f_min**2*f_max**2 + (k*f_min**2-f_max**2)*f_knee**2
                Delta = b**2 - 4*a*c
                if Delta < 0:
                    continue
                f_apo2_1 = (-b + np.sqrt(Delta))/(2*a)
                f_apo2_2 = (-b - np.sqrt(Delta))/(2*a)
                if f_apo2_1 > 0:
                    f_apo1 = np.sqrt(f_apo2_1)
                    if f_apo1 < f_knee :
                        f_sample_knee_apo_list.append([
                            f_sample,
                            f_knee,
                            f_apo1
                        ])
                if f_apo2_2 > 0:
                    f_apo2 = np.sqrt(f_apo2_2)
                    if f_apo2 < f_knee:
                        f_sample_knee_apo_list.append([
                            f_sample,
                            f_knee,
                            f_apo2
                        ])
                # add 1/f noise, fapo = 0
                f_sample_knee_apo_list.append([f_sample, f_knee, 0.0])
    f_sample_knee_apo_arr = np.array(f_sample_knee_apo_list)


# detector info
offsets = np.array([0.0,   1.0091918717241837E-4,  6.770994003507864E-5,
                    0.0,   -1.0091918717241837E-4,  6.770994003507864E-5,
                    0.0,   1.0091918717241837E-4,  6.770994003507864E-5,
                    0.0,   -1.5349591551799984E-4,  -5.000278804091876E-4,
                    0.0,   1.5349591551799984E-4,  5.000278804091876E-4,
                    0.0,   -1.5349591551799984E-4,  5.000278804091876E-4,
                    0.0,   3.8136392789889527E-4,  -2.7541850136581645E-5,
                    0.0,   -3.8136392789889527E-4,  2.7541850136581645E-5,
                    0.0,   3.8136392789889527E-4,  2.7541850136581645E-5] ).reshape([9,3]) * 20
comps = np.array([ 1.0,    1.0,    0.0,    0.0,
                   1.0,    -0.5,   0.8660254037844387, 0.0,
                   1.0,    -0.5,   -0.8660254037844384,    0.0,
                   1.0,    1.0,    0.0,    0.0,
                   1.0,    -0.5,   0.8660254037844392, 0.0,
                   1.0,    -0.5,   -0.8660254037844387,    0.0,
                   1.0,    1.0,    0.0,    0.0,
                   1.0,    -0.5,   0.8660254037844393, 0.0,
                   1.0,    -0.5,   -0.8660254037844377,    0.0]).reshape([9,4])


parameters_dic = {
    'num_sample':num_sample,
    'x_max':x_max,
    'y_max':y_max,
    'sig_amp':sig_amp,
    'noise_sigma2':noise_sigma2,
    'noise_index':noise_index,
    'num_pix_x':num_pix_x,
    'num_pix_y':num_pix_y,
    'crosslink':crosslink,
    'num_snapshots':num_snapshots,
    #'max_iter':max_iter,
    'num_iter':num_iter,
    'results_dir':results_dir,
    'figs_dir':figs_dir,
    'cache_dir':cache_dir,
    'seed':seed,
    'f_scan_list':f_scan_list,
    'condition_number_arr':condition_number_arr,
    #'num_eta_iter_per_eta':num_eta_iter_per_eta,
    'num_eta_arr':num_eta_arr,
    'f_sample_knee_apo_arr':f_sample_knee_apo_arr,
    'offsets':offsets,
    'comps':comps,
    'f_sample_list':f_sample_list,
    'f_knee_list':f_knee_list,
    }
    

cache_dir.mkdir(parents=True, exist_ok=True)
figs_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

with open('parameters_dic', 'wb') as _file:
    pickle.dump(parameters_dic, _file)
