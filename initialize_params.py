import numpy as np
from pathlib import Path, PurePath
import pickle


num_iter = 150
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
    0.25,
    0.5,
    1,
    10,
    100,
]


num_sample = int(2**20)


x_max = y_max = 1*np.pi/180
sig_amp = 100
noise_sigma2 = 10
noise_index = 3
num_pix_x = num_pix_y = 512
crosslink=True
next_eta_ratio = 1e-1

figs_dir = Path('figs/').expanduser()
results_dir = Path('results/').expanduser()


# f_apo = 0.1 f_knee and f_apo = 0.01 f_knee
f_sample_knee_apo_list = []
for f_sample in f_sample_list:
    for f_knee in f_knee_list:
        f_sample_knee_apo_list.append([f_sample, f_knee, 0.0])
        f_sample_knee_apo_list.append([f_sample, f_knee, 0.1*f_knee])
        f_sample_knee_apo_list.append([f_sample, f_knee, 0.01*f_knee])
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
    'num_iter':num_iter,
    'results_dir':results_dir,
    'figs_dir':figs_dir,
    'cache_dir':cache_dir,
    'seed':seed,
    'f_scan_list':f_scan_list,
    'num_eta_arr':num_eta_arr,
    'f_sample_knee_apo_arr':f_sample_knee_apo_arr,
    'offsets':offsets,
    'comps':comps,
    'f_sample_list':f_sample_list,
    'f_knee_list':f_knee_list,
    'next_eta_ratio':next_eta_ratio,
    }
    

cache_dir.mkdir(parents=True, exist_ok=True)
figs_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

with open('parameters_dic', 'wb') as _file:
    pickle.dump(parameters_dic, _file)
