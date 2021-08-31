import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from numpy import random
from scipy import signal, interpolate
from numpy import fft
from numba import jit, njit, prange
from pathlib import Path
import hashlib
import pickle

@njit(parallel=True)
def bincount_2D(x, length, weights=None):
    """
    same args as np.bincount, except x is 2D array here, retrun will
    also be a 2D array.
    with dim0 the same as x. and dim1 is length 
    weight has the shape of x or None
    """
    num_cols, num_rows = x.shape
    bin_count = np.zeros(shape=(num_cols, length))
    if weights is None:
        for i in prange(num_cols):
            for j in prange(num_rows):
                bin_count[i,x[i,j]] += 1
    else:
        for i in prange(num_cols):
            for j in prange(num_rows):
                bin_count[i,x[i,j]] += weights[i,j]
    return bin_count



class Map:
    """
    """
### some initializations
    def __init__(self, x_max,
            y_max,
            num_pix_x,
            num_pix_y,
            seed=0,
            cache_dir='.cache',
            force_recalculate=False,
            ):
        """
        args:
            num_pix_x, num_pix_y
                --- int
                --- number of pixel
            x_range, y_range
                --- float/int
                --- maximum angle in x (-x_max, x_max) 
                    and y (-y_max, y_max) in radian
            seed
                --- int
                --- seed of random number
            cache_dir
                --- string
                --- cache file directory
            force_recalculate
                --- bool
                --- if True, recalculate everything
        """
        assert isinstance(num_pix_x, int)
        assert isinstance(num_pix_y, int)
        assert isinstance(seed, int)
        self.num_pix_x = num_pix_x
        self.num_pix_y = num_pix_y
        self.x_max = x_max
        self.y_max = y_max
        self.seed = seed
        self.force_recalculate = force_recalculate
        self.map_dir = Path(cache_dir)/('seed={:d} pix=({:d},{:d}) '
            'x_max={:.3g} y_max={:.3g}').format(
            seed, num_pix_x, num_pix_y, x_max, y_max)
        self.map_dir.mkdir(parents=True, exist_ok=True)


    def generate_scan_info(self,
            f_scan,
            offsets_x,
            offsets_y,
            comps,
            num_sample,
            f_sample,
            crosslink=True
            ):
        """
        generate scan info based on telescope info {offsets, comps}

        args:
            f_scan
                --- float/int
                --- 1 / time of scaning one back and forth
            offsets_x/y
                --- np.array(num_detector,)
                --- offset of each detector in x / y
            comps 
                --- np.array(N_detector, 3 or 4)
                --- components of each detector {I, Q, U, V}, only 
                    store component {I,Q,U} into self.comps
            num_sample
                --- int
                --- number of samples
            f_sample
                --- float/int
                --- sampling frequency in Hz
            crosslink
                --- bool
                --- if is true will generate scan signal sweep both x y
                    direction otherwise only x direction
        
        stored values:
            self.pointing_x/y --- pointing position (countinious)
            self.P_x/y --- pointing pixel (digitized)
            self.num_data --- number of toal data
        """
        assert isinstance(num_sample, int)
        assert offsets_x.shape[0] == offsets_y.shape[0] == comps.shape[0]
        assert comps.shape[1] == 3 or comps.shape[1] == 4
        assert isinstance(crosslink, bool) or crosslink == 1 or crosslink == 0
        num_detector = comps.shape[0]
        comps = comps[:,:3]    # store component for {I,Q,U}
        self.crosslink = crosslink = bool(crosslink)

        # directory info
        scan_info = [
            f_scan,
            crosslink,
            offsets_x,
            offsets_y,
            comps,
            num_sample,
            f_sample
            ]
        scan_hash = hashlib.md5(str(scan_info).encode()).hexdigest()
        scan_dir = ('f_scan={:.3g} crosslink={} num_sample={:d} '
            'f_sample={:.3g} {}').format(
            f_scan, crosslink, num_sample, f_sample, scan_hash)
        self.map_dir = self.map_dir/scan_dir
        self.map_dir.mkdir(parents=True, exist_ok=True)

        num_pix_x = self.num_pix_x
        num_pix_y = self.num_pix_y
        x_max = self.x_max
        y_max = self.y_max
        v_scan = 4 * x_max * f_scan

        # time domain
        self.num_t = num_t = num_sample
        self.dt = dt = 1/f_sample
        self.t = t = np.arange(num_sample) * dt
        # frequency domain
        self.df = df = f_sample/num_t   # = 1/(num_t*dt)
        self.num_f = num_f = num_t//2 + 1
        self.f = f = np.arange(num_f)*df

        # sweep in x direction
        # period of sweep -x_max -> x_max and x_max -> -x_max
        T_sweep = 4 * x_max / v_scan
        # back and forth in x
        boresight_x = x_max * signal.sawtooth(2*np.pi*t/T_sweep, width=0.5) 
        # slowly drift in y
        boresight_y = np.linspace(-y_max, y_max, num_sample)
        pointing_x = boresight_x[np.newaxis,:] + offsets_x[:,np.newaxis]
        pointing_y = boresight_y[np.newaxis,:] + offsets_y[:,np.newaxis]
        if crosslink:
            # sweep in y direction (change variable from x to y)
            T_sweep = 4 * y_max / v_scan
            boresight_x = np.linspace(-x_max, x_max, num_sample)
            boresight_y = y_max * signal.sawtooth(2*np.pi*t/T_sweep, width=0.5)
            pointing_x = np.concatenate((pointing_x,
                boresight_x[np.newaxis,:] + offsets_x[:,np.newaxis]), axis=0
                )
            pointing_y = np.concatenate((pointing_y,
                boresight_y[np.newaxis,:] + offsets_y[:,np.newaxis]), axis=0
                )
            comps = np.concatenate((comps, comps), axis=0)

        # digitized pointing matrix P
        bins_x = np.linspace(-x_max, x_max, num_pix_x-1)
        bins_y = np.linspace(-y_max, y_max, num_pix_y-1)
        P_x = np.digitize(pointing_x, bins_x)
        P_y = np.digitize(pointing_y, bins_y)
        self.pointing_x = pointing_x  # poining position (continious)
        self.pointing_y = pointing_y
        self.P_x = P_x   # pointing pixel (digitized)
        self.P_y = P_y
        self.num_data = pointing_x.shape[0]
        self.comps = comps


    def generate_noiseless_signal(self,
            num_blob=100,
            R_blob_ratio_range=(0.05, 0.2),
            sig_amp=100,
            ):
        """
        generate fake signal (w/o noise)

        args:
            num_blob
                --- int
                --- number of random blobs
            R_blob_ratio_range
                --- (min, max)
                --- range of blob radius relative to min(x_max, y_max)
            sig_amp
                --- maximum signal amplitude
        """
        assert isinstance(num_blob, int)
        assert isinstance(R_blob_ratio_range, tuple)
        x_max = self.x_max
        y_max = self.y_max
        R_blob_min, R_blob_max = R_blob_ratio_range

        signal_info = [num_blob, R_blob_ratio_range, sig_amp]
        signal_str = str(signal_info).encode()
        signal_hash = hashlib.md5(signal_str).hexdigest()
        signal_dir = ('sig_amp={:.2f} num_blob={:d} '
            'R_blob_ratio_range=({:.3f},{:.3f}) {}').format(
            sig_amp, num_blob, R_blob_min, R_blob_max, signal_hash)
        self.map_dir = self.map_dir/signal_dir
        self.map_dir.mkdir(parents=True, exist_ok=True)
        signal_file = self.map_dir/'signal_data'

        try:
            if self.force_recalculate:
                raise FileNotFoundError
            else:
                with open(signal_file, 'rb') as _file:
                    signal_data = pickle.load(_file)
                self.noiseless_signal = signal_data['noiseless_signal']
                self.noiseless_map = signal_data['noiseless_map']
        
        except FileNotFoundError:
            
            # generate fake signal based on these random blob
            random.seed(self.seed)
            R_blob = min(x_max,y_max) * random.uniform(R_blob_min, R_blob_max,
                size=num_blob)     # radius for each blob
            A_blob = random.uniform(-sig_amp, sig_amp,
                size=num_blob)    # amplitude for each blob
            x_blob = random.uniform(-x_max, x_max,
                size=num_blob)    # x position of each blob
            y_blob = random.uniform(-y_max, y_max,
                size=num_blob)    # y position of each blob
            phi_blob = random.uniform(0, np.pi,
                size=num_blob)    # polarization angle of each blob
            pol_frac_blob = random.uniform(0,1,
                size=num_blob)    # polarization fraction of each blob

            # 3 stocks parameters {I, Q, U}
            noiseless_signal = np.zeros(shape=(self.num_data, self.num_t, 3))

            for i_blob in range(num_blob):
                # the distance² to each blob
                r2 = (self.pointing_x - x_blob[i_blob])**2 \
                    + (self.pointing_y - y_blob[i_blob])**2
                # the decrease of signal from center of each blob
                g = np.exp(-1/2 * r2/(R_blob[i_blob]**2) )    
                noiseless_signal[:,:,0] += A_blob[i_blob] * g
                noiseless_signal[:,:,1] += pol_frac_blob[i_blob] \
                    * A_blob[i_blob] * np.cos(2*phi_blob[i_blob]) * g
                noiseless_signal[:,:,2] += pol_frac_blob[i_blob] \
                    * A_blob[i_blob] * np.sin(2*phi_blob[i_blob]) * g

            self.noiseless_signal \
                = np.einsum('ic,itc->it', self.comps, noiseless_signal)

            # signal map
            num_pix_x = self.num_pix_x
            num_pix_y = self.num_pix_y
            x_position = np.linspace(-x_max, x_max, num_pix_x, endpoint=True)
            y_position = np.linspace(-y_max, y_max, num_pix_y, endpoint=True)
            xx, yy = np.meshgrid(x_position, y_position)
            noiseless_map = np.zeros(shape=(num_pix_y, num_pix_x, 3))
            for  i_blob in range(num_blob):
                r2 = (xx- x_blob[i_blob])**2 + (yy - y_blob[i_blob])**2
                g = np.exp(-1/2 * r2/R_blob[i_blob]**2)
                noiseless_map[:,:,0] += A_blob[i_blob] * g
                noiseless_map[:,:,1] += pol_frac_blob[i_blob] \
                    * A_blob[i_blob] * np.cos(2*phi_blob[i_blob]) * g
                noiseless_map[:,:,2] += pol_frac_blob[i_blob] \
                    * A_blob[i_blob] * np.sin(2*phi_blob[i_blob]) * g
            self.noiseless_map = noiseless_map

            signal_data = {}
            signal_data['noiseless_signal'] = self.noiseless_signal
            signal_data['noiseless_map'] = self.noiseless_map
            with open(signal_file, 'wb') as _file:
                pickle.dump(signal_data, _file)



    def generate_noise(self,
            f_knee=0,
            f_apo=0,
            noise_index=2,
            noise_sigma2=1,
            ):
        """
        generate noise based on noise power spectrum:
        σ² * (1 + ((f_knee/f_apo)**noise_index + 1)) 
        / ((f/f_apo)**noise_index + 1)

        args:
            f_knee
                --- int/float
                --- knee frequency
            f_apo
                --- int/float
                --- apo frequency
            noise_sigma2
                --- int/float
                --- variance of noise, σ²
            noise_index
                --- int/float
        """
        noise_info = [f_knee, f_apo, noise_sigma2, noise_index]
        noise_str = str(noise_info).encode()
        noise_hash = hashlib.md5(noise_str).hexdigest()
        noise_dir = ('f_knee={:.5g}/f_apo={:.5g}/noise_index={:.2g}/'
            'noise_sigma2={:.2g} {}').format(
            f_knee, f_apo, noise_index, noise_sigma2, noise_hash)
        self.map_dir = self.map_dir/noise_dir
        self.map_dir.mkdir(parents=True, exist_ok=True)
        noise_file = self.map_dir/'noise_data'
        self.noise_sigma2 = noise_sigma2
        self.f_knee = f_knee
        self.f_apo = f_apo

        if f_apo == 0 and f_knee == 0:
            # white noise
            noise_power_spectrum = noise_sigma2 * np.ones(self.f.shape)    
            N_f_diag = noise_power_spectrum/self.df
        else:
            noise_power_spectrum  = \
                noise_sigma2 * (1 + 
                    (f_knee**noise_index + f_apo**noise_index)\
                    /(self.f**noise_index + f_apo**noise_index) 
                    )
            # for 1/f noise, noise_power_spectrum[0] would be np.inf,
            # make the zeroth element equal to the first one to get
            # finite Χ² value.
            if np.isinf(noise_power_spectrum[0]):
                noise_power_spectrum[0] = noise_power_spectrum[1]
            N_f_diag = noise_power_spectrum/self.df
        self.N_f_diag = N_f_diag
        self.noise_power_spectrum = noise_power_spectrum
        try:
            if self.force_recalculate:
                raise FileNotFoundError
            else:
                with open(noise_file, 'rb') as _file:
                    self.noise = pickle.load(_file)
        except FileNotFoundError:
            random.seed(self.seed)
            g_real = random.normal(size=(self.num_data,self.num_f))/np.sqrt(2)
            g_imag = random.normal(size=(self.num_data,self.num_f))/np.sqrt(2)
            # the first element should be real due to hermitian
            g_real[0] *= np.sqrt(2)
            g_imag[0] = 0.0
            # last element being real, because num_t is even
            # https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html
            g_real[-1] *= np.sqrt(2)
            g_imag[-1] = 0.0
            noise_f = np.sqrt(noise_power_spectrum/self.df)\
                *(g_real + 1j*g_imag)
            # https://numpy.org/doc/stable/reference/routines.fft.html
            # based on Numpy inverse DFT definition, need to multiply num_t and 
            # df to agree with FT and inverse FT definition from Wikipedia
            # https://en.wikipedia.org/wiki/Fourier_transform#Definition
            self.noise = fft.irfft(noise_f, axis=1) * self.num_t * self.df
            with open(noise_file, 'wb') as _file:
                pickle.dump(self.noise, _file)


    def get_tod(self):
        """
        combine signal and noise as time ordered data, 
        stored in self.tod --- np.array shape=(N_detector, N_sample)
        """
        # hits per pixel
        num_pix_x = self.num_pix_x
        num_pix_y = self.num_pix_y
        P_y = self.P_y
        P_x = self.P_x

        # position in 1D space, notice Nx is inner dimension [Ny_pix, Nx_pix]
        P_1D = P_y * num_pix_x + P_x    

        self.hits_per_pixel = hits = (
            bincount_2D(P_1D, length=num_pix_y*num_pix_x)
            .reshape(self.num_data, num_pix_y, num_pix_x)
        )

        self.tod = self.noiseless_signal + self.noise
        self.tod_f = fft.rfft(self.tod, axis=1)

        # TOD from pixelized noiseless map
        noiseless_pixelized_signal = \
            np.einsum(
                'ic,itc->it',
                self.comps,
                noiseless_map[P_y, P_x,:]
            )
        self.tod = self.noiseless_pixelized_signal + self.noise
        self.tod_f = fft.rfft(self.tod, axis=1)

        # some calculation in self._PTP_inv
        comps = self.comps
        CCT = comps[:,:,np.newaxis] @ comps[:,np.newaxis,:]
        PTP = np.einsum('iyx,ick->yxck', hits, CCT)
        self.PTP_inv = la.pinv(PTP)
        self.cond_num_pixel = la.cond(CCT)
        eigvals_PTP = la.eigvals(PTP).reshape(-1)
        self.cond_num_PTP \
            = np.abs(np.max(eigvals_PTP))/np.abs(np.min(eigvals_PTP))

        self.get_b()
        self.get_simple_binned_map()



#### some operations

    def _P(self, m):
        """
        pointing matrix P acting on map size object
        P{it;pc} and m{pc} 
        (index: i -- data, t -- time, p -- position, c -- components)
        get tod size object

        Pm{it}
        = P{it;pc} m{pc}
        = Σpc P'{it;p} C{ic} m{pc}
        = Σc Pm'{it;c} C{ic}
        where P{it;pc} = P'{it;p} C{ic} , C{ic} is compoent for data i.
        and Pm' = P' m

        args:
            m
                --- np.array shape=(num_pix_y, num_pix_x, 3)
                ---  map size object
        return:
            Pm
                --- np.array shape=(num_data, num_t)
        """
        assert m.shape == (self.num_pix_y, self.num_pix_x, 3)

        P_x = self.P_x
        P_y = self.P_y
        comps = self.comps
        Pm_comps = m[P_y, P_x, :]

        # or in other way Pm_comps @ comps
        return np.einsum('itc,ic -> it', Pm_comps, comps)


    def _PT(self, d):
        """
        P.T acting on TOD size object
        P{it;pc} d{it}
        = Σit P'{it;p} C{ic} d{it}
        = Σi P'Td{i;p} C{ic}
        = PTd{pc}

        args: 
            d
                --- np.array shape=(num_t,)
                --- 1D array of TOD
        return:
            PTd
                --- np.array shape=(num_pix_y,num_pix_x,3)
                --- dimension of map 
        """
        num_pix_x = self.num_pix_x
        num_pix_y = self.num_pix_y
        P_y = self.P_y
        P_x = self.P_x

        # position in 1D space, notice Nx is inner dimension [Ny_pix, Nx_pix]
        P_1D = P_y * num_pix_x + P_x    

        PTd_comps = bincount_2D(P_1D, length=num_pix_y*num_pix_x, weights=d)\
            .reshape(self.num_data, num_pix_y, num_pix_x)
        PTd = np.einsum('iyx,ic->yxc', PTd_comps, self.comps)
        return PTd


    def _PTP_inv(self, m):
        """
        P.T P acting on map size object
        P.T P {pc;p'c'}
        = P{it;pc} P{it;p'c'}
        = Σit P'{it;p} C{ic} P'{it;p'} C{ic'}
        = Σi (P'TP'){i;p;p'} C{ic} C{ic'}
        = Σi (# of hits of pixel p for data i) δ{p;p'} C{ic} C{ic'}
        = Σi (# of hits of pixel p for data i) δ{p;p'} CCT{i;c;c'}
        which is 3x3 block diagonaled matrix, the inverse is the 
        inverse of each block

        args:
            m
                --- np.array shape=(num_pix_y, num_pix_x, 3)
                ---  map size object
        return:
            PTP_inv_m
                --- np.array shape=(num_pix_y, num_pix_x, 3)
                --- map size object
        """
        num_pix_x = self.num_pix_x
        num_pix_y = self.num_pix_y
        #comps = self.comps
        P_y = self.P_y
        P_x = self.P_x

        # position in 1D space, notice Nx is inner dimension [Ny_pix, Nx_pix]
        P_1D = P_y * num_pix_x + P_x    
        
        #hits = self.hits_per_pixel
        #CCT = comps[:,:,np.newaxis] @ comps[:,np.newaxis,:]
        #PTP = np.einsum('iyx,ick->yxck', hits, CCT)
        #PTP_inv = la.pinv(PTP)
        PTP_inv = self.PTP_inv
        PTP_inv_m = np.einsum('yxck,yxk->yxc', PTP_inv, m)
        return PTP_inv_m
        

    def get_b(self):
        """ b vector in map making equation b = P.T N^(-1) d """
        b = self._PT( fft.irfft(
            self.tod_f/self.N_f_diag,
            axis=1))
        self.b = b


    def get_simple_binned_map(self):
        """ simple binned map (P.T P)^-1 P.T d """
        self.simple_binned_map = self._PTP_inv(self._PT(self.tod))


    def get_chi2(self, m, freq_mode=False):
        """ 
            calculate Χ²(m) = (d - Pm)† N^{-1} (d - Pm)
        args:
            m
                --- np.array shape=(num_pix_y, num_pix_x, 3)
                --- map size object
            freq_mode
                --- bool
                --- if True, it will return chi2 per frequency mode
        return:
            chi2
                --- Χ² value
            chi2_per_f
                --- np.array shape=(num_f,)
        """
        Pm = self._P(m)
        Pm_f = fft.rfft(Pm, axis=1)
        d_f = self.tod_f
        if not freq_mode:
            chi2 = np.sum(  
                ( np.conj(d_f - Pm_f) * (d_f - Pm_f)/self.N_f_diag ).real
                )
            return chi2
        else:
            chi2_vs_f = np.sum(
                (np.conj(d_f - Pm_f) * (d_f - Pm_f)/self.N_f_diag).real 
                , axis=0)
            return chi2_vs_f

    def get_chi2_eta(self, m, eta, tau, Nbar_f):
        """
            calculate Χ²(m,η) = (d - Pm)† (τ + η Nbar)^{-1} (d - Pm)
        args:
            m
                --- np.array shape=(num_pix_y, num_pix_x, 3)
                --- map size object
            eta
                --- float/int
                --- parameter η
            tau, Nbar_f
                --- tau = np.min(self.N_f_diag)
                --- Nbar_f = self.N_f_diag - tau
        return:
            chi2_eta
                --- Χ²* value
        """
        Pm = self._P(m)
        Pm_f = fft.rfft(Pm, axis=1)
        d_f = self.tod_f
        chi2_eta = np.sum(  
            ( np.conj(d_f - Pm_f) * (d_f - Pm_f)/(tau + eta*Nbar_f) ).real
        )
        return chi2_eta

    def get_dchi2_deta(self, m, eta, tau, Nbar_f):
        """
            #calculate per frequency mode for
            |dΧ²/dη| = (d - Pm)† N(η)^-1 Nbar N(η)^-1 (d - Pm)
            with N(η) = τ I + η Nbar
        args:
            m
                --- np.array shape=(num_pix_y, num_pix_x, 3)
                --- map size object
            eta
                --- float/int
                --- parameter η
            tau, Nbar_f
                --- tau = np.min(self.N_f_diag)
                --- Nbar_f = self.N_f_diag - tau
        """
        Pm = self._P(m)
        Pm_f = fft.rfft(Pm, axis=1)
        d_f = self.tod_f
        N_eta = tau + eta*Nbar_f
        dchi2_deta = np.sum(
            (np.conj(d_f - Pm_f) * (d_f - Pm_f)).real * Nbar_f/N_eta**2
        )
        return dchi2_deta
            
        
    def _A(self, m):
        """ map making A(m) = P.T N^(-1) P m
        args:
            m
                --- np.array shape=(num_pix_y, num_pix_x, 3)
                --- map size object
        return:
            Am
                --- np.array shape=(num_pix_y, num_pix_x, 3)
                --- map size object
        """
        Pm =  self._P(m)
        Pm_f = fft.rfft(Pm, axis=1)
        Am = self._PT( fft.irfft(
            Pm_f/self.N_f_diag,
            axis=1))
        return  Am

    def _get_2norm(self, m):
        """
        calculate Euclidean norm of map size object m
        """
        return la.norm(m.reshape(-1), ord=2)

    def PTP_preconditioner(self, m, eta=1):
        """
        M = P.T P
        return M^-1(m)
        """
        return self._PTP_inv(m)

    def get_chi2_vs_eta(self, 
            preconditioner_inv,
            preconditioner_description,
            num_eta=100,
            next_eta_ratio=1e-3,
            stop_ratio=1e-5,
            max_iter_per_eta=100,
            ):
        """
        get function of Χ²(hatm(η),η) as function of η
        args:
            preconditioner_inv
                --- function with input (m, η)
                    map size object m and cooling parameter η
                --- the inverse of preconditioner
            preconditioner_description
                --- description of preconditioner
            num_eta
                --- int
                --- n_η, the eta array is {η_1, η_2, ..., η_{n_η}},
                    with η_1 = 1e-5 * τ/max(Nbar)
                    and η_{n_η} = 1
            next_eta_ratio
                --- float
                --- if ||r||₂/||b(η_i)|| < next_eta_ratio, advance to η_{i+1}
            stop_ratio
                --- float
                --- when η=1 and the 2-norm of the residual
                    ||r||₂ / ||b||₂ < stop_ratio the iteration stops
            max_iter_per_eta
                --- int
                --- maximum number of iterations for each η value,
                    this number prevents endless loop if next_eta_ratio
                    or stop_ratio cannot be achieved.
        """
        assert isinstance(num_eta, int)
        assert isinstance(preconditioner_description, str)
        assert isinstance(next_eta_ratio, float)
        assert isinstance(stop_ratio, float)
        assert isinstance(max_iter_per_eta, int)
        tau = np.min(self.N_f_diag)
        Nbar_f = self.N_f_diag - tau
        eta_min = 1e-3 * tau/Nbar_f.max()
        etas_arr = np.logspace(
            np.log10(eta_min), 0, num=num_eta, base=10
        )
        random.seed(self.seed)
        num_pix_x = self.num_pix_x
        num_pix_y = self.num_pix_y

        chi2_vs_eta_info = str([
            num_eta,
            preconditioner_description,
            next_eta_ratio,
            stop_ratio,
            max_iter_per_eta,
            preconditioner_inv(
                random.rand(num_pix_y, num_pix_x, 3),1) ]
                ).encode()
        chi2_vs_eta_hash = hashlib.md5(chi2_vs_eta_info).hexdigest()
        chi2_vs_eta_file_name = (
            'chi2(hatm(eta), eta) num_eta={:d} preconditioner={} {}'
            ).format(
            num_eta,
            preconditioner_description,
            chi2_vs_eta_hash,
        )
        chi2_vs_eta_file = self.map_dir/chi2_vs_eta_file_name

        try:
            if self.force_recalculate:
                raise FileNotFoundError
            else:
                with open(chi2_vs_eta_file, 'rb') as _file:
                    chi2_vs_eta_results = pickle.load(_file)
                    assert np.all(chi2_vs_eta_results['etas_arr'] == etas_arr)
                    chi2_vs_eta = chi2_vs_eta_results['chi2_vs_eta']
        except FileNotFoundError:
            chi2_vs_eta = np.zeros(num_eta)

            print(('chi2(hatm(eta), eta) num_eta={:d} preconditioner={}'
                .format(num_eta, preconditioner_description)
            ))
            b = lambda eta: self._PT( fft.irfft(
                self.tod_f/(eta*Nbar_f + tau),
                axis=1))
            def A(m, eta):
                Pm = self._P(m)
                Pm_f = fft.rfft(Pm)
                return self._PT( fft.irfft(
                    Pm_f/(eta*Nbar_f + tau),
                    axis=1))

            # dot product in conjugate gradient algorithm
            dot = lambda x,y: np.sum(x*y)
                
            # preconditioned algorithm for conjugate gradient method 
            m = self.simple_binned_map.copy()

            print('i_η={:<5d} i={:<5d} η={:.5e}'.format(0,0,0))
            for i_eta,eta in enumerate(etas_arr):

                b_eta = b(eta)
                b_eta_2norm = self._get_2norm(b_eta)
                r_eta = b_eta - A(m, eta)
                r_eta_2norm = self._get_2norm(r_eta)
                z = preconditioner_inv(r_eta, eta)
                p = z.copy()

                for i_iter in range(max_iter_per_eta):
                    A_p_eta = A(p, eta)
                    alpha = dot(r_eta,z) / dot(p, A_p_eta)
                    m += alpha * p
                    r_eta_old = r_eta.copy()
                    r_eta -= alpha * A_p_eta
                    z_old = z.copy()
                    z = preconditioner_inv(r_eta, eta)
                    beta = dot(r_eta,z) / dot(r_eta_old,z_old)
                    p = z + beta * p

                    r_eta_2norm = self._get_2norm(r_eta)

                    print('i_η={:<5d} i={:<5d} η={:.5e}'
                        .format(i_eta, i_iter, eta)
                    )

                    if (eta != 1
                            and r_eta_2norm/b_eta_2norm < next_eta_ratio) :
                        chi2_vs_eta[i_eta] \
                            = self.get_chi2_eta(m, eta, tau, Nbar_f)
                        break
                    elif (eta == 1
                            and r_eta_2norm/b_eta_2norm < stop_ratio) :
                        chi2_vs_eta[i_eta] \
                            = self.get_chi2_eta(m, eta, tau, Nbar_f)
                        break
                    elif i_iter == max_iter_per_eta - 1:
                        chi2_vs_eta[i_eta] \
                            = self.get_chi2_eta(m, eta, tau, Nbar_f)
                        break
                        

            chi2_vs_eta_results = {}
            chi2_vs_eta_results['etas_arr'] = etas_arr
            chi2_vs_eta_results['chi2_vs_eta'] = chi2_vs_eta
            with open(chi2_vs_eta_file, 'wb') as _file:
                pickle.dump(chi2_vs_eta_results, _file)
        self.chi2_min = chi2_vs_eta[-1]
        return etas_arr, chi2_vs_eta
        

### solvers

    def conjugate_gradient_solver(self,
            num_iter,
            preconditioner_inv, 
            preconditioner_description,
            stop_ratio=1e-5,
            ):
        """ 
        solve map making equation with conjugate gradient method 
        P.T N^(-1) P m = P.T N^(-1) d  <==> A x = b
        where x = m, A = P.T N^(-1) P, b = P.T N^(-1) d

        the algorithm notation follows:
        https://en.wikipedia.org/wiki/Conjugate_gradient_method
        #The_preconditioned_conjugate_gradient_method

        args:
            num_iter
                --- int
                --- number of iteration
            preconditioner_inv
                --- function with map size object as input
                --- the inverse of preconditioner
            preconditioner_description
                --- string
                --- description of preconditioner
            stop_ratio
                --- float
                --- when the 2-norm of the residual
                    ||r||₂ / ||b||₂ < stop_ratio the iteration stops
        return:
            CG_file
                --- pathlib.Path
                --- file of calculation results
            CG_results
                --- dictionary {}, with keys:
                    [
                        'chi2_hist',
                        'chi2_eta_hist',
                        'm_hist',
                        'r_hist',
                        'r_2norm_hist',
                        'chi2_f_hist',
                        'etas_iter',
                    ]
        """
        assert isinstance(num_iter, int)
        assert isinstance(preconditioner_description, str)
        assert isinstance(stop_ratio, float)
        num_pix_x = self.num_pix_x
        num_pix_y = self.num_pix_y
        random.seed(self.seed)
        CG_info = str([
            num_iter,
            preconditioner_description,
            stop_ratio,
            preconditioner_inv(
                random.rand(num_pix_y, num_pix_x, 3)
                ),
            ]).encode()
        CG_hash = hashlib.md5(CG_info).hexdigest()
        CG_file_name = ('CG num_iter={:d} preconditioner={} {}')\
            .format(
                num_iter,
                preconditioner_description,
                CG_hash
            )
        CG_file = self.map_dir/CG_file_name

        try:
            if self.force_recalculate:
                raise FileNotFoundError
            else:
                with open(CG_file, 'rb') as _file:
                    CG_results = pickle.load(_file)
        except FileNotFoundError:

            m_hist = np.zeros(shape=(2, num_pix_y, num_pix_x, 3),
                dtype=np.float32)
            r_hist = np.zeros(shape=(2, num_pix_y, num_pix_x, 3),
                dtype=np.float32)
            chi2_hist = np.zeros(num_iter+1, dtype=np.float32)
            r_2norm_hist = np.zeros(num_iter+1, dtype=np.float32)
            chi2_f_hist = np.zeros(shape=(2, self.num_f), dtype=np.float32)

            b = self.b
            b_2norm = self._get_2norm(b)
            A = self._A

            # dot product in conjugate gradient algorithm
            dot = lambda x,y: np.sum(x*y)
                
            # preconditioned algorithm for conjugate gradient method
            print('CG preconditioner={} num_iter={:d}'.format(
                preconditioner_description, num_iter))
            m = self.simple_binned_map.copy()
            r = b - A(m)
            z = preconditioner_inv(r)
            p = z.copy()

            chi2 = self.get_chi2(m)
            chi2_hist[0] = chi2
            r_2norm_hist[0] = r0_2norm = self._get_2norm(r)
            m_hist[0,:,:,:] = m
            r_hist[0,:,:,:] = r
            chi2_f_hist[0,:] = self.get_chi2(m, freq_mode=True)
            print('iter={:<5d}  Χ²={:.10e}'.format(0, chi2))
            for i_iter in range(1, num_iter+1):
                alpha = dot(r,z) / dot(p, A(p))
                m += alpha * p
                r_old = r.copy()
                r -= alpha * A(p)
                z_old = z.copy()
                z = preconditioner_inv(r)
                beta = dot(r,z) / dot(r_old,z_old)
                p = z + beta * p

                chi2 = self.get_chi2(m)
                chi2_hist[i_iter] = chi2
                r_2norm_hist[i_iter] = r_2norm = self._get_2norm(r)
                print('iter={:<5d}  Χ²={:.10e} ||r||₂/||r0||₂={:.10e}'.format(
                    i_iter, chi2, r_2norm/r0_2norm))
                if (r_2norm/b_2norm < stop_ratio
                        and i_iter != num_iter):
                    stop_point = i_iter
                    chi2_hist[stop_point:] = chi2
                    r_2norm_hist[stop_point:] = r_2norm
                    break

            m_hist[1,:,:,:] = m
            r_hist[1,:,:,:] = r
            chi2_f_hist[1,:] = self.get_chi2(m, freq_mode=True)

            CG_results = {}
            CG_results['chi2_hist'] = chi2_hist
            CG_results['chi2_eta_hist'] = chi2_hist
            CG_results['m_hist'] = m_hist
            CG_results['r_hist'] = r_hist
            CG_results['r_2norm_hist'] = r_2norm_hist
            CG_results['chi2_f_hist'] = chi2_f_hist
            CG_results['etas_iter'] = np.ones(num_iter+1)
            with open(CG_file, 'wb') as _file:
                pickle.dump(CG_results, _file)
        return CG_file, CG_results


    def messenger_field_solver(self,
            lambs_arr,
            num_iter_per_lamb,
            num_iter,
            stop_ratio=1e-5,
            ):
        """
        messenger field solve map based on Huffenberger 2018: 
        https://arxiv.org/abs/1705.01893v2
        the iterative equation Eq(8) with T = τ I and Nbar = N - τ
        m_{i+1} = m_i 
            + τ (P.T P)^{-1} P.T (λ^{-1} Nbar + τ )^{-1} (d - P m_i)

        args:
            lambs_arr
                --- np.array, lambs_arr[-1] = 1 
                --- values of cooling parameter λ
            num_iter_per_lamb
                --- int
                --- number of iteration per lambda
            num_iter
                --- int
                --- total number of iteration
            stop_ratio
                --- float
                --- when the 2-norm of the residual
                    ||r||₂ / ||b||₂ < stop_ratio the iteration stops
        return:
            MF_file
                --- pathlib.Path
                --- file of calculation results
            MF_results
                --- dictionary {}, with keys:
                    [
                        'chi2_hist',   # Χ²(m)
                        'chi2_eta_hist',   # Χ²(m,η)
                        'm_hist',
                        'r_hist',
                        'r_2norm_hist',
                        'chi2_f_hist',
                        'etas_iter',
                        'etas_arr',
                    ]
        """
        assert isinstance(lambs_arr, np.ndarray)
        assert lambs_arr.ndim == 1
        assert lambs_arr[-1] == 1
        assert isinstance(num_iter_per_lamb, int)
        assert isinstance(num_iter, int)
        assert num_iter >= lambs_arr.size * num_iter_per_lamb
        MF_info = str([
            lambs_arr,
            num_iter_per_lamb,
            num_iter,
            stop_ratio,
            ]).encode()
        MF_hash = hashlib.md5(MF_info).hexdigest()
        MF_file_name = (
            'MF method {:d}x{:d} num_iter={:d} {}'
            ).format(
            len(lambs_arr),
            num_iter_per_lamb,
            num_iter,
            MF_hash
        )
        MF_file = self.map_dir/MF_file_name

        try:
            if self.force_recalculate:
                raise FileNotFoundError
            else:
                with open(MF_file, 'rb') as _file:
                    MF_results = pickle.load(_file)
        except FileNotFoundError:
            num_pix_x = self.num_pix_x
            num_pix_y = self.num_pix_y
            m_hist = np.zeros(shape=(2, num_pix_y, num_pix_x, 3), 
                dtype=np.float32)
            r_hist = np.zeros(shape=(2, num_pix_y, num_pix_x, 3), 
                dtype=np.float32)
            chi2_hist = np.zeros(num_iter+1, dtype=np.float32)
            chi2_eta_hist = np.zeros(num_iter+1, dtype=np.float32)
            r_2norm_hist = np.zeros(num_iter+1, dtype=np.float32)
            chi2_f_hist = np.zeros(shape=(2, self.num_f), 
                dtype=np.float32)
            etas_iter = np.zeros(num_iter+1, dtype=np.float32)

            print('MF num_iter={:d}'.format(num_iter))
            b = self.b
            b_2norm = self._get_2norm(b)
            A = self._A
            m = self.simple_binned_map.copy()
            r = b - A(m)
            tau = np.min(self.N_f_diag)
            Nbar_f = self.N_f_diag - tau
            chi2 = self.get_chi2(m)
            chi2_hist[0] = chi2
            chi2_eta_hist[0] = self.get_chi2_eta(m, 0, tau, Nbar_f)
            etas_iter[0] = 0
            r_2norm_hist[0] = r0_2norm = self._get_2norm(r)
            m_hist[0,:,:,:] = m
            r_hist[0,:,:,:] = r
            chi2_f_hist[0,:] = self.get_chi2(m, freq_mode=True)

            print('iter={:<5d}  Χ²={:.10e}'.format(0, chi2))
            for i_iter in range(1, num_iter+1):
                i_lamb = (i_iter-1)//num_iter_per_lamb
                if i_lamb < len(lambs_arr):
                    lamb = lambs_arr[i_lamb]
                else:
                    lamb = 1
                etas_iter[i_iter] = 1/lamb
                Pm = self._P(m)
                Pm_f = fft.rfft(Pm, axis=1)
                m += tau * self._PTP_inv( self._PT( fft.irfft(
                    1/(Nbar_f/lamb + tau) * (self.tod_f - Pm_f),
                    axis=1)))

                r = b - A(m)
                chi2 = self.get_chi2(m)
                chi2_hist[i_iter] = chi2
                chi2_eta = self.get_chi2_eta(m, 1/lamb, tau, Nbar_f)
                chi2_eta_hist[i_iter] = chi2_eta
                r_2norm_hist[i_iter] = r_2norm = self._get_2norm(r)
                print('iter={:<5d}  Χ²={:.10e} ||r||₂/||r0||₂={:.10e}'\
                    .format(i_iter, chi2, r_2norm/r0_2norm))
                if (r_2norm/b_2norm < stop_ratio
                        and lamb == 1
                        and i_iter != num_iter):
                    # stop calculation
                    stop_point = i_iter
                    chi2_hist[stop_point:] = chi2
                    chi2_eta_hist[stop_point:] = chi2_eta
                    r_2norm_hist[stop_point:] = r_2norm
                    etas_iter[stop_point:] = 1/lamb 
                    break

            m_hist[1,:,:,:] = m
            r_hist[1,:,:,:] = r
            chi2_f_hist[1,:] = self.get_chi2(m, freq_mode=True)

            MF_results = {}
            MF_results['chi2_hist'] = chi2_hist
            MF_results['chi2_eta_hist'] = chi2_eta_hist
            MF_results['m_hist'] = m_hist
            MF_results['r_hist'] = r_hist
            MF_results['r_2norm_hist'] = r_2norm_hist
            MF_results['chi2_f_hist'] = chi2_f_hist
            MF_results['etas_iter'] = etas_iter
            MF_results['etas_arr'] = 1/lambs_arr
            with open(MF_file, 'wb') as _file:
                pickle.dump(MF_results, _file)
        return MF_file, MF_results


    def conjugate_gradient_solver_eta(self,
            num_iter,
            preconditioner_inv,
            preconditioner_description,
            etas_arr=None,
            next_eta_ratio = 1e-1,
            stop_ratio=1e-5,
            single_step=False,
            ):
        """ 
        solve map making equation with conjugate gradient method 
        P.T N(η)^(-1) P m = P.T N(η)^(-1) d(η)  <==> A x = b 
        where x = m, A(η) = P.T N(η)^(-1) P, b(η) = P.T N(η)^(-1) d
        
        N = Nbar + τI
        with τ = min(diag(N)) in frequency space

        and perterbative parameter η from 0 to 1
        N(η) = η*Nbar + τI
        A(η) = P.T N(η)^{-1} P
        b(η) = P.T N(η)^{-1} d

        the algorithm notation follows:
        https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method

        args:
            num_iter
                --- int
                --- number of iteration
            preconditioner_inv
                --- function with input (m, η)
                    map size object m and cooling parameter η
                --- the inverse of preconditioner
            preconditioner_description
                --- description of preconditioner
            etas_arr
                --- np.array or None
                --- optional eta values array, etas_arr[-1] = 1
                    if None, etas series is given by (2^i - 1) * τ/max(Nbar_f)
            next_eta_ratio
                --- float
                --- if ||r||₂/||b(η_i)|| < next_eta_ratio, advance to η_{i+1}
            stop_ratio
                --- float
                --- when η=1 and the 2-norm of the residual
                    ||r||₂ / ||b||₂ < stop_ratio the iteration stops
            single_step
                --- bool
                --- if true, only do one single step for each conjugate gradient algorithm
        return:
            CG_file
                --- pathlib.Path
                --- file of calculation results
            CG_results
                --- dictionary {}, with keys:
                    [
                        'chi2_hist',   # Χ²(m)
                        'chi2_eta_hist',   # Χ²(m,η)
                        'm_hist',
                        'r_hist',
                        'r_2norm_hist',
                        'chi2_f_hist',
                        'etas_iter',
                        'etas_arr',
                    ]
        """
        assert isinstance(num_iter, int)
        assert isinstance(preconditioner_description, str)
        assert (etas_arr is None) or (isinstance(etas_arr, np.ndarray))
        assert isinstance(next_eta_ratio, float)
        assert isinstance(stop_ratio, float)
        tau = np.min(self.N_f_diag)
        Nbar_f = self.N_f_diag - tau
        if etas_arr is None:
            eta1 = tau/Nbar_f.max()
            _i = np.arange(int(np.floor(np.log2(1/eta1 + 1)))) + 1
            etas_arr = eta1*( 2**(_i) - 1 )
            etas_arr = np.append(etas_arr, 1)
        else:
            assert etas_arr[-1] == 1

        random.seed(self.seed)
        num_pix_x = self.num_pix_x
        num_pix_y = self.num_pix_y

        CG_info = str([
            etas_arr,
            num_iter,
            preconditioner_description,
            next_eta_ratio,
            stop_ratio,
            single_step,
            preconditioner_inv(
                random.rand(num_pix_y, num_pix_x, 3),1) ]
                ).encode()
        CG_hash = hashlib.md5(CG_info).hexdigest()
        CG_file_name = (
            'CG eta={:.2e},{:.2e},{:.2e},... preconditioner={} num_iter={:d} single_step={} {}'
            ).format(
            etas_arr[0], etas_arr[1], etas_arr[2],
            preconditioner_description,
            num_iter,
            str(single_step),
            CG_hash
        )
        CG_file = self.map_dir/CG_file_name

        try:
            if self.force_recalculate:
                raise FileNotFoundError
            else:
                with open(CG_file, 'rb') as _file:
                    CG_results = pickle.load(_file)
        except FileNotFoundError:
            # m_hist r_hist only store initial and final result
            m_hist = np.zeros(shape=(2, num_pix_y, num_pix_x, 3),
                dtype=np.float32)
            r_hist = np.zeros(shape=(2, num_pix_y, num_pix_x, 3),
                dtype=np.float32)
            chi2_hist = np.zeros(num_iter+1, dtype=np.float32)
            chi2_eta_hist = np.zeros(num_iter+1, dtype=np.float32)
            r_2norm_hist = np.zeros(num_iter+1, dtype=np.float32)
            chi2_f_hist = np.zeros(shape=(2, self.num_f), dtype=np.float32)
            etas_iter = np.zeros(num_iter+1, dtype=np.float32)

            print(('CG with eta solver preconditioner={} '
                'num_iter={:d}').format(
                preconditioner_description,
                num_iter))
            b = lambda eta: self._PT( fft.irfft(
                self.tod_f/(eta*Nbar_f + tau),
                axis=1))
            def A(m, eta):
                Pm = self._P(m)
                Pm_f = fft.rfft(Pm)
                return self._PT( fft.irfft(
                    Pm_f/(eta*Nbar_f + tau),
                    axis=1))

            # dot product in conjugate gradient algorithm
            dot = lambda x,y: np.sum(x*y)
                
            # preconditioned algorithm for conjugate gradient method 
            m = self.simple_binned_map.copy()
            r_true = self.b - self._A(m)
            b_2norm = self._get_2norm(self.b)
            chi2 = self.get_chi2(m)
            chi2_hist[0] = chi2
            chi2_eta = self.get_chi2_eta(m, 0, tau, Nbar_f)
            chi2_eta_hist[0] = chi2_eta
            etas_iter[0] = 0
            r_2norm_hist[0] = r0_2norm = self._get_2norm(r_true)
            m_hist[0,:,:,:] = m
            r_hist[0,:,:,:] = r_true
            chi2_f_hist[0,:] = self.get_chi2(m, freq_mode=True)

            print('iter={:<5d} η={:.5e}  Χ²(m)={:.10e}  Χ²(m,η)={:.10e}'
                .format(0, 0, chi2, chi2_eta)
            )
            stop_point = num_iter
            i_iter = 1
            i_eta = 0
            while i_eta < len(etas_arr) and i_iter <= num_iter:
                eta = etas_arr[i_eta]
                i_eta += 1
                b_eta = b(eta)
                b_eta_2norm = self._get_2norm(b_eta)
                r_eta = b_eta - A(m, eta)
                r_eta_2norm = self._get_2norm(r_eta)
                z = preconditioner_inv(r_eta, eta)
                p = z.copy()

                #chi2 = self.get_chi2(m)
                chi2_hist[i_iter] = chi2
                chi2_eta = self.get_chi2_eta(m, eta, tau, Nbar_f)
                chi2_eta_hist[i_iter] = chi2_eta
                etas_iter[i_iter] = eta
                r_true = self.b - self._A(m)
                r_2norm_hist[i_iter] = r_2norm = self._get_2norm(r_true)
                r_eta_2norm = self._get_2norm(r_eta)

                print('iter={:<5d} η={:.5e}  Χ²(m)={:.10e}  Χ²(m,η)={:.10e}'
                    .format( i_iter, eta, chi2, chi2_eta)
                )
                i_iter += 1

                if (eta < 1 and r_eta_2norm/b_eta_2norm < next_eta_ratio):
                    # to next eta value
                    continue
                elif (eta == 1 and r_eta_2norm/b_eta_2norm < stop_ratio):
                    # stop calculation
                    stop_point = i_iter - 1
                    i_iter = num_iter + 1
                    continue
                
                while i_iter <= num_iter:
                    A_p_eta = A(p, eta)
                    alpha = dot(r_eta,z) / dot(p, A_p_eta)
                    m += alpha * p
                    r_eta_old = r_eta.copy()
                    r_eta -= alpha * A_p_eta
                    z_old = z.copy()
                    z = preconditioner_inv(r_eta, eta)
                    beta = dot(r_eta,z) / dot(r_eta_old,z_old)
                    p = z + beta * p

                    chi2 = self.get_chi2(m)
                    chi2_hist[i_iter] = chi2
                    chi2_eta = self.get_chi2_eta(m, eta, tau, Nbar_f)
                    chi2_eta_hist[i_iter] = chi2_eta
                    etas_iter[i_iter] = eta
                    r_true = self.b - self._A(m)
                    r_2norm_hist[i_iter] = r_2norm = self._get_2norm(r_true)
                    r_eta_2norm = self._get_2norm(r_eta)

                    print('iter={:<5d} η={:.5e}  Χ²(m)={:.10e}  Χ²(m,η)={:.10e}'
                        .format( i_iter, eta, chi2, chi2_eta)
                    )
                    i_iter += 1

                    if single_step and eta<1:
                        break

                    if (eta < 1 and r_eta_2norm/b_eta_2norm < next_eta_ratio):
                        # to next eta value
                        break
                    elif (eta == 1 and r_eta_2norm/b_eta_2norm < stop_ratio):
                        # stop calculation
                        stop_point = i_iter - 1
                        i_iter = num_iter + 1
                        break

            chi2_hist[stop_point:] = chi2
            chi2_eta_hist[stop_point:] = chi2_eta
            etas_iter[stop_point:] = eta
            r_2norm_hist[stop_point:] = r_2norm
            m_hist[1,:,:,:] = m
            r_hist[1,:,:,:] = r_true
            chi2_f_hist[1,:] = self.get_chi2(m, freq_mode=True)

            CG_results = {}
            CG_results['chi2_hist'] = chi2_hist
            CG_results['chi2_eta_hist'] = chi2_eta_hist
            CG_results['m_hist'] = m_hist
            CG_results['r_hist'] = r_hist
            CG_results['r_2norm_hist'] = r_2norm_hist
            CG_results['chi2_f_hist'] = chi2_f_hist
            CG_results['etas_iter'] = etas_iter
            CG_results['etas_arr'] = etas_arr
            with open(CG_file, 'wb') as _file:
                pickle.dump(CG_results, _file)
        return CG_file, CG_results
        

    def conjugate_gradient_solver_exact_eta(self,
            num_iter,
            preconditioner_inv,
            preconditioner_description,
            next_eta_ratio=1e-1,
            stop_ratio=1e-5,
            single_step=False
            ):
        """
        The algorithm is the same as self.conjugate_gradient_solver_eta,
        except η update is different.
        η0 = 0 all other ηs update are based on 
        δη = - Χ²(hatm(η), η) / d/dη Χ²(hatm(η), η) 

        args:
            num_iter
                --- int
                --- number of iteration
            preconditioner_inv
                --- function with input (m, η)
                    map size object m and cooling parameter η
                --- the inverse of preconditioner
            preconditioner_description
                --- description of preconditioner
            next_eta_ratio
                --- float
                --- if ||r||₂/||b(η_i)|| < next_eta_ratio, advance to η_{i+1}
            stop_ratio
                --- float
                --- when η=1 and the 2-norm of the residual
                    ||r||₂ / ||b||₂ < stop_ratio the iteration stops
            single_step
                --- bool
                --- if true, only do one single step for each conjugate gradient algorithm
        return:
            CG_file
                --- pathlib.Path
                --- file of calculation results
            CG_results
                --- dictionary {}, with keys:
                    [
                        'chi2_hist',   # Χ²(m)
                        'chi2_eta_hist',   # Χ²(m,η)
                        'm_hist',
                        'r_hist',
                        'r_2norm_hist',
                        'chi2_f_hist',
                        'etas_iter',
                        'etas_arr',
                    ]
        """
        assert isinstance(num_iter, int)
        assert isinstance(preconditioner_description, str)
        assert isinstance(next_eta_ratio, float)
        assert isinstance(stop_ratio, float)
        random.seed(self.seed)
        num_pix_x = self.num_pix_x
        num_pix_y = self.num_pix_y

        tau = np.min(self.N_f_diag)
        Nbar_f = self.N_f_diag - tau
        etas_list = []

        CG_info = str([
            num_iter,
            preconditioner_description,
            next_eta_ratio,
            single_step,
            preconditioner_inv(
                random.rand(num_pix_y, num_pix_x, 3),1) ]
                ).encode()
        CG_hash = hashlib.md5(CG_info).hexdigest()
        CG_file_name = (
            'CG exact eta preconditioner={} single_step={} {}'
            ).format(
            preconditioner_description,
            str(single_step),
            CG_hash
        )
        CG_file = self.map_dir/CG_file_name

        try:
            if self.force_recalculate:
                raise FileNotFoundError
            else:
                with open(CG_file, 'rb') as _file:
                    CG_results = pickle.load(_file)
        except FileNotFoundError:
            m_hist = np.zeros(shape=(2, num_pix_y, num_pix_x, 3),
                dtype=np.float32)
            r_hist = np.zeros(shape=(2, num_pix_y, num_pix_x, 3),
                dtype=np.float32)
            chi2_hist = np.zeros(num_iter+1, dtype=np.float32)
            chi2_eta_hist = np.zeros(num_iter+1, dtype=np.float32)
            r_2norm_hist = np.zeros(num_iter+1, dtype=np.float32)
            chi2_f_hist = np.zeros(shape=(2, self.num_f),
                dtype=np.float32)
            etas_iter = np.zeros(num_iter+1, dtype=np.float32)

            print(('CG with exact eta solver preconditioner={} '
                'num_iter={:d}').format(
                preconditioner_description,
                num_iter))
            b = lambda eta: self._PT( fft.irfft(
                self.tod_f/(eta*Nbar_f + tau),
                axis=1))
            def A(m, eta):
                Pm = self._P(m)
                Pm_f = fft.rfft(Pm)
                return self._PT( fft.irfft(
                    Pm_f/(eta*Nbar_f + tau),
                    axis=1))

            # dot product in conjugate gradient algorithm
            dot = lambda x,y: np.sum(x*y)
                
            # preconditioned algorithm for conjugate gradient method 
            m = self.simple_binned_map.copy()
            r_true = self.b - self._A(m)
            b_2norm = self._get_2norm(self.b)
            chi2 = self.get_chi2(m)
            chi2_hist[0] = chi2
            chi2_eta = self.get_chi2_eta(m, 0, tau, Nbar_f)
            chi2_eta_hist[0] = chi2_eta
            etas_iter[0] = eta = 0
            r_2norm_hist[0] = r0_2norm = self._get_2norm(r_true)
            m_hist[0,:,:,:] = m
            r_hist[0,:,:,:] = r_true
            chi2_f_hist[0,:] = self.get_chi2(m, freq_mode=True)

            print('iter={:<5d} η={:.5e}  Χ²(m)={:.10e}  Χ²(m,η)={:.10e}'
                .format(0, 0, chi2, chi2_eta)
            )
            stop_point = num_iter
            i_iter = 1
            while eta < 1 and i_iter <= num_iter:
                dchi2_deta = self.get_dchi2_deta(m, eta, tau, Nbar_f)
                eta += chi2_eta/dchi2_deta
                eta = min(eta, 1.0)
                etas_list.append(eta)

                b_eta = b(eta)
                b_eta_2norm = self._get_2norm(b_eta)
                r_eta = b_eta - A(m, eta)
                r_eta_2norm = self._get_2norm(r_eta)
                z = preconditioner_inv(r_eta, eta)
                p = z.copy()

                #chi2 = self.get_chi2(m)
                chi2_hist[i_iter] = chi2
                chi2_eta = self.get_chi2_eta(m, eta, tau, Nbar_f)
                chi2_eta_hist[i_iter] = chi2_eta
                etas_iter[i_iter] = eta
                r_true = self.b - self._A(m)
                r_2norm_hist[i_iter] = r_2norm = self._get_2norm(r_true)
                r_eta_2norm = self._get_2norm(r_eta)

                print('iter={:<5d} η={:.5e}  Χ²(m)={:.10e}  Χ²(m,η)={:.10e}'
                    .format( i_iter, eta, chi2, chi2_eta)
                )
                i_iter += 1
                
                if (eta < 1 and r_eta_2norm/b_eta_2norm < next_eta_ratio):
                    # to next eta value
                    continue
                elif (eta == 1 and r_eta_2norm/b_eta_2norm < stop_ratio):
                    # stop calculation
                    stop_point = i_iter - 1
                    i_iter = num_iter + 1
                    continue

                while i_iter <= num_iter :

                    alpha = dot(r_eta,z) / dot(p, A(p, eta))
                    m += alpha * p
                    r_eta_old = r_eta.copy()
                    r_eta -= alpha * A(p, eta)
                    z_old = z.copy()
                    z = preconditioner_inv(r_eta, eta)
                    beta = dot(r_eta,z) / dot(r_eta_old,z_old)
                    p = z + beta * p

                    chi2 = self.get_chi2(m)
                    chi2_hist[i_iter] = chi2
                    chi2_eta = self.get_chi2_eta(m, eta, tau, Nbar_f)
                    chi2_eta_hist[i_iter] = chi2_eta
                    etas_iter[i_iter] = eta
                    r_true = self.b - self._A(m)
                    r_2norm_hist[i_iter] = r_2norm = self._get_2norm(r_true)
                    r_eta_2norm = self._get_2norm(r_eta)

                    print('iter={:<5d} η={:.5e}  Χ²(m)={:.10e}  Χ²(m,η)={:.10e}'
                        .format( i_iter, eta, chi2, chi2_eta)
                    )
                    i_iter += 1

                    if single_step and eta<1:
                        break

                    if (eta < 1 and r_eta_2norm/b_eta_2norm < next_eta_ratio):
                        # next eta value
                        break

                    elif (eta==1 and r_eta_2norm/b_eta_2norm < stop_ratio):
                        # stop calculation
                        stop_point = i_iter - 1
                        i_iter = num_iter + 1
                        break

            chi2_hist[stop_point:] = chi2
            chi2_eta_hist[stop_point:] = chi2_eta
            etas_iter[stop_point:] = eta
            r_2norm_hist[stop_point:] = r_2norm
            m_hist[1,:,:,:] = m
            r_hist[1,:,:,:] = r_true
            chi2_f_hist[1,:] = self.get_chi2(m, freq_mode=True)

            CG_results = {}
            CG_results['chi2_hist'] = chi2_hist
            CG_results['chi2_eta_hist'] = chi2_eta_hist
            CG_results['m_hist'] = m_hist
            CG_results['r_hist'] = r_hist
            CG_results['r_2norm_hist'] = r_2norm_hist
            CG_results['chi2_f_hist'] = chi2_f_hist
            CG_results['etas_iter'] = etas_iter
            CG_results['etas_arr'] = np.array(etas_list)
            with open(CG_file, 'wb') as _file:
                pickle.dump(CG_results, _file)
        return CG_file, CG_results
