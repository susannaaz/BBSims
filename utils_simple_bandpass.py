import healpy as hp
import numpy as np
import os
from astropy.table import QTable


def iter_cls(nfreq):
    map_combs = []
    for i in range(nfreq):
        map_combs.append([i, 0])
        map_combs.append([i, 1])
    nmaps = len(map_combs)

    ix = 0
    for im1, mn1 in enumerate(map_combs):
        inu1, ipol1 = mn1
        for im2, mn2 in enumerate(map_combs):
            if im2 < im1:
                continue
            inu2, ipol2 = mn2
            yield inu1, ipol1, im1, inu2, ipol2, im2, ix
            ix += 1


def get_vector_and_covar(ls, cls, fsky=1.):
    """ Vectorizes an array of C_ells and computes their
    associated covariance matrix.
    Args:
        ls: array of multipole values.
        cls: array of power spectra with shape [nfreq, npol, nfreq, npol, nell]
    Returns:
        translator: an array of shape [nfreq*npol, nfreq*npol] that contains
            the vectorized indices for a given pair of map indices.
        cl_vec: vectorized power spectra. Shape [n_pairs, nell]
        cov: vectorized covariance. Shape [n_pairs, n_ell, n_pair, n_ell]
    """
    nfreq, npol, _, _, nls = cls.shape
    nmaps = nfreq*npol
    nx = (nmaps * (nmaps+1)) // 2

    # 2D to 1D translator
    translator = np.zeros([nmaps, nmaps], dtype=int)
    for _, _, i1, _, _, i2, ix in iter_cls(nfreq):
        translator[i1, i2] = ix
        if i1 != i2:
            translator[i2, i1] = ix

    delta_ell = np.mean(np.diff(ls))
    fl = 1./((2*ls+1)*delta_ell*fsky)
    # covariance calculated with Knox formula
    cov = np.zeros([nx, nls, nx, nls])
    cl_vec = np.zeros([nx, nls])
    cl_maps = cls.reshape([nmaps, nmaps, nls])
    for _, _, i1, _, _, i2, ii in iter_cls(nfreq):
        cl_vec[ii, :] = cl_maps[i1, i2, :]
        for _, _, j1, _, _, j2, jj in iter_cls(nfreq):
            covar = (cl_maps[i1, j1, :] * cl_maps[i2, j2, :] +
                     cl_maps[i1, j2, :] * cl_maps[i2, j1, :]) * fl
            cov[ii, :, jj, :] = np.diag(covar)
    return translator, cl_vec, cov


def bin_cls(cls, delta_ell=10):
    """ Returns a binned-version of the power spectra.
    """
    nls = cls.shape[-1]
    ells = np.arange(nls)
    delta_ell = 10
    N_bins = (nls-2)//delta_ell
    w = 1./delta_ell
    W = np.zeros([N_bins, nls])
    for i in range(N_bins):
        W[i, 2+i*delta_ell:2+(i+1)*delta_ell] = w
    l_eff = np.dot(ells, W.T)
    cl_binned = np.dot(cls, W.T)
    return l_eff, W, cl_binned


def map2cl(maps, maps2=None, iter=0):
    """ Returns an array with all auto- and cross-correlations
    for a given set of Q/U frequency maps.
    Args:
        maps: set of frequency maps with shape [nfreq, 2, npix].
        maps2: set of frequency maps with shape [nfreq, 2, npix] to cross-correlate with.
        iter: iter parameter for anafast (default 0).
    Returns:
        Set of power spectra with shape [nfreq, 2, nfreq, 2, n_ell].
    """
    nfreq, npol, npix = maps.shape
    nside = hp.npix2nside(npix)
    nls = 3*nside
    ells = np.arange(nls)
    cl2dl = ells*(ells+1)/(2*np.pi)
    if maps2 is None:
        maps2 = maps

    cl_out = np.zeros([nfreq, npol, nfreq, npol, nls])
    for i in range(nfreq):
        m1 = np.zeros([3, npix])
        m1[1:,:]=maps[i, :, :]
        for j in range(i,nfreq):
            m2 = np.zeros([3, npix])
            m2[1:,:]=maps2[j, :, :]

            cl = hp.anafast(m1, m2, iter=0)
            cl_out[i, 0, j, 0] = cl[1] * cl2dl
            cl_out[i, 1, j, 1] = cl[2] * cl2dl
            if j!=i:
                cl_out[j, 0, i, 0] = cl[1] * cl2dl
                cl_out[j, 1, i, 1] = cl[2] * cl2dl
    return cl_out


def get_default_params():
    pars = {
        'r_tensor': 0,
        'A_d_BB': 28.0,
        'A_d_EE': 56.0,
        'alpha_d_EE': -0.32,
        'alpha_d_BB': -0.16,
        'nu0_d': 353.,
        'beta_d': 1.54,
        'temp_d': 20.0,
        'A_s_BB': 1.6,
        'A_s_EE': 9.0,
        'alpha_d_EE': -0.7,
        'alpha_d_BB': -0.93,
        'nu0_s': 23.,
        'beta_s': -3.0,
        'include_CMB': True,
        'include_dust': True,
        'include_sync': True,
        'include_E': True,
        'include_B': True,
        'dust_SED': 'mbb',
        'sync_SED': 'plaw',
        'use_bandpass': False,  # NEW
        'bandpass_dir': "/global/homes/s/susannaz/Software/bandpass_sampler/SAT/"    # NEW (optional)
    }
    return pars


def fcmb(nu):
    """ CMB SED (in antenna temperature units).
    """
    x=0.017608676067552197*nu
    ex=np.exp(x)
    return ex*(x/(ex-1))**2


def comp_sed(nu,nu0,beta,temp,typ):
    """ Component SEDs (in antenna temperature units).
    """
    if typ=='cmb':
        return fcmb(nu)
    elif typ=='dust':
        x_to=0.04799244662211351*nu/temp
        x_from=0.04799244662211351*nu0/temp
        return (nu/nu0)**(1+beta)*(np.exp(x_from)-1)/(np.exp(x_to)-1)
    elif typ=='sync':
        return (nu/nu0)**beta
    return None


def get_mean_spectra(lmax, params):
    """ Computes amplitude power spectra for all components
    """
    ells = np.arange(lmax+1)
    dl2cl = np.ones(len(ells))
    dl2cl[1:] = 2*np.pi/(ells[1:]*(ells[1:]+1.))
    cl2dl = (ells*(ells+1.))/(2*np.pi)

    # Dust
    A_dust_BB = params['A_d_BB'] * fcmb(params['nu0_d'])**2
    A_dust_EE = params['A_d_EE'] * fcmb(params['nu0_d'])**2
    dl_dust_bb = A_dust_BB * ((ells+1E-5) / 80.)**params['alpha_d_BB']
    dl_dust_ee = A_dust_EE * ((ells+1E-5) / 80.)**params['alpha_d_EE']
    cl_dust_bb = dl_dust_bb * dl2cl
    cl_dust_ee = dl_dust_ee * dl2cl
    if not params['include_E']:
        cl_dust_ee *= 0 
    if not params['include_B']:
        cl_dust_bb *= 0
    if not params['include_dust']:
        cl_dust_bb *= 0
        cl_dust_ee *= 0

    # Sync
    A_sync_BB = params['A_s_BB'] * fcmb(params['nu0_s'])**2
    A_sync_EE = params['A_s_EE'] * fcmb(params['nu0_s'])**2
    dl_sync_bb = A_sync_BB * ((ells+1E-5) / 80.)**params['alpha_s_BB']
    dl_sync_ee = A_sync_EE * ((ells+1E-5) / 80.)**params['alpha_s_EE']
    cl_sync_bb = dl_sync_bb * dl2cl
    cl_sync_ee = dl_sync_ee * dl2cl
    if not params['include_E']:
        cl_sync_ee *= 0 
    if not params['include_B']:
        cl_sync_bb *= 0
    if not params['include_sync']:
        cl_sync_bb *= 0
        cl_sync_ee *= 0

    # CMB amplitude
    # Lensing
    l, dtt, dee, dbb, dte=np.loadtxt("data/camb_lens_nobb.dat",unpack=True)
    l = l.astype(int)
    msk = l <= lmax
    l = l[msk]
    dltt = np.zeros(len(ells)); dltt[l]=dtt[msk]
    dlee = np.zeros(len(ells)); dlee[l]=dee[msk]
    dlbb = np.zeros(len(ells)); dlbb[l]=dbb[msk]
    dlte = np.zeros(len(ells)); dlte[l]=dte[msk]  
    cl_cmb_bb_lens = dlbb * dl2cl
    cl_cmb_ee_lens = dlee * dl2cl
    if not params['include_E']:
        cl_cmb_ee_lens *= 0 
    if not params['include_B']:
        cl_cmb_bb_lens *= 0
    if not params['include_CMB']:
        cl_cmb_bb_lens *= 0
        cl_cmb_ee_lens *= 0

    # Lensing + r=1
    l,dtt,dee,dbb,dte=np.loadtxt("data/camb_lens_r1.dat",unpack=True)
    l = l.astype(int)
    msk = l <= lmax
    l = l[msk]
    dltt = np.zeros(len(ells)); dltt[l]=dtt[msk]
    dlee = np.zeros(len(ells)); dlee[l]=dee[msk]
    dlbb = np.zeros(len(ells)); dlbb[l]=dbb[msk]
    dlte = np.zeros(len(ells)); dlte[l]=dte[msk]  
    cl_cmb_bb_r1 = dlbb * dl2cl
    cl_cmb_ee_r1 = dlee * dl2cl
    if not params['include_E']:
        cl_cmb_ee_r1 *= 0 
    if not params['include_B']:
        cl_cmb_bb_r1 *= 0
    if not params['include_CMB']:
        cl_cmb_bb_r1 *= 0
        cl_cmb_ee_r1 *= 0
    cl_cmb_ee = cl_cmb_ee_lens + params['r_tensor'] * (cl_cmb_ee_r1-cl_cmb_ee_lens)
    cl_cmb_bb = cl_cmb_bb_lens + params['r_tensor'] * (cl_cmb_bb_r1-cl_cmb_bb_lens)
    return(ells, dl2cl, cl2dl,
           cl_dust_bb, cl_dust_ee,
           cl_sync_bb, cl_sync_ee,
           cl_cmb_bb, cl_cmb_ee)

def get_sacc(leff, cls, l_unbinned, windows, params, cov=None):
    import sacc

    nus = params['freqs']
    nfreq = len(nus)

    nbands, nls = windows.shape
    s_wins = sacc.BandpowerWindow(l_unbinned, windows.T)

    s = sacc.Sacc()

    use_bandpass = params.get('use_bandpass', False)
    bandpass_dir = params.get('bandpass_dir', None)
    beam_fwhm_file = params.get('beam_fwhm_file', None)

    for inu, nu in enumerate(nus):
        # --- Bandpass ---
        if use_bandpass and bandpass_dir is not None:
            nu_bp, w_bp = load_bandpass(nu, bandpass_dir)
        else:
            nu_bp = np.array([nu - 1, nu, nu + 1])
            w_bp = np.array([0.0, 1.0, 0.0])

        # --- Beam ---
        beam_fwhm_rad = None
        if 'beam_fwhm_file' in params and params['beam_fwhm_file'] is not None:
            rad_factor = np.pi / (60. * 180.)
            beam_fwhm = np.loadtxt(params['beam_fwhm_file'])
            sigma_rad = (beam_fwhm[inu] * rad_factor) / np.sqrt(8 * np.log(2))  # in radians
            beam = np.exp(-0.5 * l_unbinned * (l_unbinned + 1) * sigma_rad**2)
        else:
            beam = np.ones_like(l_unbinned)

        s.add_tracer('NuMap', f'band{inu+1}',
                     quantity='cmb_polarization',
                     spin=2,
                     nu=nu_bp,
                     bandpass=w_bp,
                     ell=l_unbinned,
                     beam=beam,
                     nu_unit='GHz',
                     map_unit='uK_CMB')

    pdict = ['e', 'b']
    for inu1, ipol1, i1, inu2, ipol2, i2, ix in iter_cls(nfreq):
        n1 = f'band{inu1+1}'
        n2 = f'band{inu2+1}'
        p1 = pdict[ipol1]
        p2 = pdict[ipol2]
        cl_type = f'cl_{p1}{p2}'
        s.add_ell_cl(cl_type, n1, n2, leff, cls[ix], window=s_wins)

    if cov is not None:
        ncls = len(cls.flatten())
        cv = cov.reshape([ncls, ncls])
        s.add_covariance(cv)

    return s


def load_bandpass(freq_tag, bandpass_dir):
    """
    Loads bandpass for a given frequency tag from disk.
    Returns:
        freqs (GHz), weights (normalized)
    """
    bp_file_map = {
        27: "resampled_bpasses_LF1_w0.tbl",
        39: "resampled_bpasses_LF2_w0.tbl",
        93: "resampled_bpasses_MF1_w0.tbl",
        145: "resampled_bpasses_MF2_w0.tbl",
        225: "resampled_bpasses_HF1_w0.tbl",
        280: "resampled_bpasses_HF2_w0.tbl"
    }
    if freq_tag not in bp_file_map:
        raise ValueError(f"No bandpass file for {freq_tag} GHz")
    fname = os.path.join(bandpass_dir, bp_file_map[freq_tag])
    table = QTable.read(fname, format='ascii.ipac')
    nu = table['bandpass_frequency'].to('GHz').value
    w = table['bandpass_weight'].value
    w /= np.trapz(w, nu)  # normalize
    return nu, w


def get_sky_realization(nside, seed, params,
                        delta_ell=10):
    """
    """
    npix = hp.nside2npix(nside)
    if seed is not None:
        np.random.seed(seed)
    lmax = 3*nside-1
    ells, dl2cl, cl2dl, cl_dust_bb, cl_dust_ee, cl_sync_bb, cl_sync_ee, cl_cmb_bb, cl_cmb_ee = get_mean_spectra(lmax, params)
    cl0 = 0 * cl_dust_bb

    # Dust amplitudes
    Q_dust, U_dust = hp.synfast([cl0, cl_dust_ee, cl_dust_bb, cl0, cl0, cl0],
                                nside, new=True)[1:]
    # Sync amplitudes
    Q_sync, U_sync = hp.synfast([cl0, cl_sync_ee, cl_sync_bb, cl0, cl0, cl0],
                                nside, new=True)[1:]
    # CMB amplitude
    Q_cmb, U_cmb = hp.synfast([cl0, cl_cmb_ee, cl_cmb_bb, cl0, cl0, cl0],
                              nside, new=True)[1:]

    if not params['include_dust']:
        Q_dust *= 0
        U_dust *= 0
    if not params['include_sync']:
        Q_sync *= 0
        U_sync *= 0
    if not params['include_CMB']:
        Q_cmb *= 0
        U_cmb *= 0

    if params.get('use_bandpass', False):
        if params.get('bandpass_dir', None) is None:
            raise ValueError("You must set 'bandpass_dir' in params if 'use_bandpass' is True")
    
        seds_dust = []
        seds_sync = []
        seds_cmb = []
        for freq in params['freqs']:
            nu_bp, w_bp = load_bandpass(freq, params['bandpass_dir'])
    
            dust_sed = comp_sed(nu_bp, params['nu0_d'], params['beta_d'], params['temp_d'], typ='dust')
            sync_sed = comp_sed(nu_bp, params['nu0_s'], params['beta_s'], None, typ='sync')
            cmb_sed = fcmb(nu_bp)
    
            norm = np.trapz(fcmb(nu_bp) * w_bp, nu_bp)
    
            seds_dust.append(np.trapz(dust_sed * w_bp, nu_bp) / norm)
            seds_sync.append(np.trapz(sync_sed * w_bp, nu_bp) / norm)
            seds_cmb.append(np.trapz(cmb_sed * w_bp, nu_bp) / norm)
    
        seds = np.array([seds_dust, seds_sync, seds_cmb])  # shape (3, nfreq)

    else:
        seds = np.array([
            comp_sed(params['freqs'], params['nu0_d'], params['beta_d'], params['temp_d'], typ='dust'),
            comp_sed(params['freqs'], params['nu0_s'], params['beta_s'], None, typ='sync'),
            comp_sed(params['freqs'], None, None, None, 'cmb')
        ])
        seds /= fcmb(params['freqs'])[None, :]  # Normalize to uK_CMB


    # SEDs of theory spectra should be unaffected by beams/bandpasses
    seds_theory = np.array([
    comp_sed(params['freqs'], params['nu0_d'], params['beta_d'], params['temp_d'], typ='dust'),
    comp_sed(params['freqs'], params['nu0_s'], params['beta_s'], None, typ='sync'),
    comp_sed(params['freqs'], None, None, None, 'cmb')])
    seds_theory /= fcmb(params['freqs'])[None, :]  # Normalize to uK_CMB

    # Generate C_ells from theory
    nnu = len(params['freqs'])
    nell = lmax+1
    cl_sky = np.zeros([nnu, 2, nnu, 2, nell])
    cl_sky[:, 0, :, 0, :] = (cl_dust_ee[None, None, :]*np.outer(seds_theory[0], seds_theory[0])[:, :, None] +
                             cl_sync_ee[None, None, :]*np.outer(seds_theory[1], seds_theory[1])[:, :, None] +
                             cl_cmb_ee[None, None, :]*np.outer(seds_theory[2], seds_theory[2])[:, :, None])
    cl_sky[:, 1, :, 1, :] = (cl_dust_bb[None, None, :]*np.outer(seds_theory[0], seds_theory[0])[:, :, None] +
                             cl_sync_bb[None, None, :]*np.outer(seds_theory[1], seds_theory[1])[:, :, None] +
                             cl_cmb_bb[None, None, :]*np.outer(seds_theory[2], seds_theory[2])[:, :, None])
    cl_sky *= cl2dl[None, None, None, None, :]
    l_binned, windows, cl_sky_binned = bin_cls(cl_sky, delta_ell=delta_ell)
    _, cl_sky_binned, _ = get_vector_and_covar(l_binned, cl_sky_binned)

    # Create full and single component maps
    nnu = len(params['freqs'])
    maps_signal = np.zeros((nnu, 2, npix))
    maps_dust_all = np.zeros_like(maps_signal)
    maps_sync_all = np.zeros_like(maps_signal)
    maps_cmb_all = np.zeros_like(maps_signal)

    beam_fwhm_rad = None
    if 'beam_fwhm_file' in params and params['beam_fwhm_file'] is not None:
        rad_factor = np.pi / (60. * 180.)
        beam_fwhm = np.loadtxt(params['beam_fwhm_file'])
        #beam_fwhm_rad = np.radians(np.array(params['beam_fwhm_arcmin']) / 60.)
        beam_fwhm_rad = np.array(beam_fwhm*rad_factor)
        if len(beam_fwhm_rad) != nnu:
            raise ValueError("Length of beam_fwhm_arcmin must match number of frequency channels")
    
    # Loop over frequency bands
    for i in range(nnu):
        # SED scaling
        f_dust = seds[0, i]
        f_sync = seds[1, i]
        f_cmb = seds[2, i]
    
        # Scale
        q_dust, u_dust = f_dust * Q_dust, f_dust * U_dust
        q_sync, u_sync = f_sync * Q_sync, f_sync * U_sync
        q_cmb,  u_cmb  = f_cmb  * Q_cmb,  f_cmb  * U_cmb

        t_dummy = np.zeros_like(q_dust)
        
        # Beam smooth if applicable
        if beam_fwhm_rad is not None:
            _, q_dust, u_dust = hp.smoothing([t_dummy, q_dust, u_dust], 
                                             fwhm=beam_fwhm_rad[i], 
                                             pol=True)
            _, q_sync, u_sync = hp.smoothing([t_dummy, q_sync, u_sync], 
                                             fwhm=beam_fwhm_rad[i], 
                                             pol=True)
            _, q_cmb, u_cmb = hp.smoothing([t_dummy, q_cmb, u_cmb], 
                                           fwhm=beam_fwhm_rad[i], 
                                           pol=True)

        # Total map = sum of all components per freq
        maps_signal[i, 0] = q_dust + q_sync + q_cmb
        maps_signal[i, 1] = u_dust + u_sync + u_cmb
    
        maps_dust_all[i, 0] = q_dust
        maps_dust_all[i, 1] = u_dust
        maps_sync_all[i, 0] = q_sync
        maps_sync_all[i, 1] = u_sync
        maps_cmb_all[i, 0]  = q_cmb
        maps_cmb_all[i, 1]  = u_cmb

    dict_out = {
        'maps_dust': maps_dust_all,   # shape [nfreq, 2, npix]
        'maps_sync': maps_sync_all,
        'maps_cmb': maps_cmb_all,
        'freq_maps': maps_signal,
        'seds': seds,
        'amp_dust': np.array([Q_dust, U_dust]),
        'amp_sync': np.array([Q_sync, U_sync]),
        }

    # Generate C_ells from data
    cls_unbinned = map2cl(maps_signal)
    _, _, cls_binned = bin_cls(cls_unbinned,
                               delta_ell=delta_ell)
    indices, cls_binned, cov_binned = get_vector_and_covar(l_binned,
                                                           cls_binned)
    dict_out['ls_unbinned'] = ells
    dict_out['ls_binned'] = l_binned
    dict_out['cls_data'] = cls_binned
    dict_out['cls_theory'] = cl_sky_binned
    dict_out['cls_theory_unbinned'] = cl_sky
    dict_out['cov'] = cov_binned
    dict_out['ind_cl'] = indices
    dict_out['windows'] = windows

    return dict_out
