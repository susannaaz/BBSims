import healpy as hp
import numpy as np
import os


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
    pars = {'r_tensor': 0,
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
            'sync_SED': 'plaw'}
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

    for inu, nu in enumerate(nus):
        nu_s = np.array([nu-1, nu, nu+1])
        bnu_s = np.array([0.0, 1.0, 0.0])
        s.add_tracer('NuMap', 'band%d' % (inu+1),
                     quantity='cmb_polarization',
                     spin=2,
                     nu=nu_s,
                     bandpass=bnu_s,
                     ell=l_unbinned,
                     beam=np.ones_like(l_unbinned),
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

    seds = np.array([comp_sed(params['freqs'], params['nu0_d'], params['beta_d'],
                              params['temp_d'], typ='dust'),
                     comp_sed(params['freqs'], params['nu0_s'], params['beta_s'],
                              None, typ='sync'),
                     comp_sed(params['freqs'], None, None, None, 'cmb')])
    seds /= fcmb(params['freqs'])[None, :]

    # Generate C_ells from theory
    nnu = len(params['freqs'])
    nell = lmax+1
    cl_sky = np.zeros([nnu, 2, nnu, 2, nell])
    cl_sky[:, 0, :, 0, :] = (cl_dust_ee[None, None, :]*np.outer(seds[0], seds[0])[:, :, None] +
                             cl_sync_ee[None, None, :]*np.outer(seds[1], seds[1])[:, :, None] +
                             cl_cmb_ee[None, None, :]*np.outer(seds[2], seds[2])[:, :, None])
    cl_sky[:, 1, :, 1, :] = (cl_dust_bb[None, None, :]*np.outer(seds[0], seds[0])[:, :, None] +
                             cl_sync_bb[None, None, :]*np.outer(seds[1], seds[1])[:, :, None] +
                             cl_cmb_bb[None, None, :]*np.outer(seds[2], seds[2])[:, :, None])
    cl_sky *= cl2dl[None, None, None, None, :]
    l_binned, windows, cl_sky_binned = bin_cls(cl_sky, delta_ell=delta_ell)
    _, cl_sky_binned, _ = get_vector_and_covar(l_binned, cl_sky_binned)

    # Generate sky maps
    maps_signal = np.sum(np.array([[Q_dust, U_dust],
                                   [Q_sync, U_sync],
                                   [Q_cmb, U_cmb]])[:, None, :, :] *
                         seds[:, :, None, None], axis=0)

    dict_out = {'maps_dust': np.array([Q_dust, U_dust]),
                'maps_sync': np.array([Q_sync, U_sync]),
                'maps_cmb': np.array([Q_cmb, U_cmb]),
                'freq_maps': maps_signal,
                'seds': seds}

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
