import utils_simple_bandpass as ut
import healpy as hp
import numpy as np
import os
from optparse import OptionParser
import yaml

parser = OptionParser()
parser.add_option('--output-dir', dest='dirname', default='none',
                  type=str, help='Output directory')
parser.add_option('--seed', dest='seed',  default=1000, type=int,
                  help='Set to define seed, default=1000')
parser.add_option('--nside', dest='nside', default=256, type=int,
                  help='Set to define Nside parameter, default=256')

parser.add_option('--include-cmb', dest='include_cmb', default=True, action='store_false',
                  help='Set to remove CMB from simulation, default=True.')
parser.add_option('--include-sync', dest='include_sync', default=True, action='store_false',
                  help='Set to remove synchrotron from simulation, default=True.')
parser.add_option('--include-dust', dest='include_dust', default=True, action='store_false',
                  help='Set to remove dust from simulation, default=True.')
parser.add_option('--include-E', dest='include_E', default=True, action='store_false',
                  help='Set to remove E-modes from simulation, default=True.')
parser.add_option('--include-B', dest='include_B', default=True, action='store_false',
                  help='Set to remove B-modes from simulation, default=True.')
parser.add_option('--nu0-dust', dest='nu0_d', default=353., type=int,
                  help='Set to change dust pivot frequency, default=353 GHz.')
parser.add_option('--nu0-sync', dest='nu0_s', default=23., type=int,
                  help='Set to change synchrotron pivot frequency, default=23 GHz.')

parser.add_option('--A-dust-EE', dest='A_d_EE', default=56.0, type=float,
                  help='Set to modify the E-mode dust power spectrum amplitude, default=56.0')
parser.add_option('--alpha-dust-EE', dest='alpha_d_EE', default=-0.32, type=float,
                  help='Set to mofify tilt in D_l^EE for dust, default=-0.32')
parser.add_option('--A-dust-BB', dest='A_d_BB', default=28.0, type=float,
                  help='Set to modify the B-mode dust power spectrum amplitude, default=28.0')
parser.add_option('--alpha-dust-BB', dest='alpha_d_BB', default=-0.16, type=float,
                  help='Set to mofify tilt in D_l^BB for dust, default=-0.16')
parser.add_option('--beta-dust', dest='beta_d', default=1.54, type=float,
                  help='Set to mofify dust spectral index, default=1.54')
parser.add_option('--temp-dust', dest='temp_d', default=20.0, type=float,
                  help='Set to mofify dust temperature, default=20.0')

parser.add_option('--A-sync-EE', dest='A_s_EE', default=9.0, type=float,
                  help='Set to modify the E-mode dust power spectrum amplitude, default=9.0')
parser.add_option('--alpha-sync-EE', dest='alpha_s_EE', default=-0.7, type=float,
                  help='Set to mofify tilt in D_l^EE for synchrotron, default=-0.7')
parser.add_option('--A-sync-BB', dest='A_s_BB', default=1.6, type=float,
                  help='Set to modify the B-mode dust power spectrum amplitude, default=1.6')
parser.add_option('--alpha-sync-BB', dest='alpha_s_BB', default=-0.93, type=float,
                  help='Set to mofify tilt in D_l^BB for synchrotron, default=-0.93')
parser.add_option('--beta-sync', dest='beta_s', default=-3.0, type=float,
                  help='Set to mofify synchrotron spectral index, default=-3.0')

parser.add_option('--r-tensor', dest='r_tensor', default=0.0, type=float,
                  help='Set to mofify tensor-to-scalar ratio')
parser.add_option('--frequencies', dest='freq_file', default=None, type=str,
                  help='Path to file containing frequencies to use in GHz')

parser.add_option('--use-bandpass', dest='use_bandpass', default=False, action='store_true',
                  help='Use bandpass integration instead of mono-frequency SEDs')
parser.add_option('--bandpass-dir', dest='bandpass_dir',
                  default="/global/homes/s/susannaz/Software/bandpass_sampler/SAT/", type=str,
                  help='Directory containing bandpass files (required if --use-bandpass is set)')

parser.add_option('--beam-fwhm-file', dest='beam_file', default="beam_fwhm.txt", type=str,
                  help='Path to file with beam FWHM values in arcmin, one per line matching the frequency file.')


(o, args) = parser.parse_args()

if o.use_bandpass and o.bandpass_dir is None:
    raise ValueError("If --use-bandpass is set, you must also specify --bandpass-dir.")

nside = o.nside
seed = o.seed

if o.freq_file is None:
    raise ValueError("Pass a frequency file in --frequencies")

freqs = np.loadtxt(o.freq_file)

if o.beam_file is None:
    raise ValueError("Must provide --beam-fwhm-file to apply beam smoothing")

beam_fwhm_arcmin = np.loadtxt(o.beam_file)  # Shape: (nfreq,)
if len(beam_fwhm_arcmin) != len(freqs):
    raise ValueError("Mismatch: beam file and frequency file must have same number of entries")


os.system('mkdir -p ' + o.dirname)
prefix = o.dirname + f'/sim_seed{o.seed}'
pars = {k: getattr(o, k)
        for k in ['seed', 'nside', 'include_cmb', 'include_sync', 'include_E', 'include_B',
                  'nu0_d', 'A_d_EE', 'A_d_BB', 'alpha_d_EE', 'alpha_d_BB', 'beta_d', 'temp_d',
                  'nu0_s', 'A_s_EE', 'A_s_BB', 'alpha_s_EE', 'alpha_s_BB', 'beta_s', 'r_tensor',
                  'use_bandpass', 'bandpass_dir']}  # <- add new options here

pars['freqs'] = [float(f) for f in freqs]
with open(f'{prefix}_params.yaml', 'w') as file:
    yaml.safe_dump(pars, file)
pars['freqs'] = np.array(pars['freqs'])

# Decide whether spectral index is constant or varying
params = ut.get_default_params()
params.update(pars)

data = ut.get_sky_realization(o.nside, o.seed, params)

# Write data power spectra
s = ut.get_sacc(data['ls_binned'], data['cls_data'], data['ls_unbinned'],
                data['windows'], params, cov=data['cov'])
s.save_fits(prefix+'_cl_data.fits', overwrite=True)

# Write theory power spectra
s = ut.get_sacc(data['ls_binned'], data['cls_theory'], data['ls_unbinned'],
                data['windows'], params, cov=data['cov'])
s.save_fits(prefix+'_cl_theory.fits', overwrite=True)

# Write sky maps
for i in range(len(freqs)):
    hp.write_map(prefix+f'_sky_band{i+1}.fits', data['freq_maps'][i], overwrite=True)

if o.use_bandpass:
    # Write actual component maps per freq (m = amp x sed)
    for comp in ['dust', 'sync', 'cmb']:
        for i in range(len(freqs)):
            hp.write_map(prefix + f'_{comp}_band{i+1}.fits',
                         data[f'maps_{comp}'][i], overwrite=True)

        # single fits file:
        #    Q_band1, U_band1, Q_band2, U_band2, ..., Q_bandN, U_bandN
        maps = data[f'maps_{comp}']  # shape: [nband, 2, npix]
        maps_reshaped = maps.reshape(-1, maps.shape[-1])  # shape: [nband*2, npix]
        hp.write_map(prefix + f'_{comp}.fits', maps_reshaped, overwrite=True)
else:
    # Write component amplitude maps
    for comp in ['dust', 'sync', 'cmb']:
        hp.write_map(prefix+f'_{comp}.fits', data[f'maps_{comp}'], overwrite=True)

# Write component spectra
sdic = {f'sed_{comp}': data['seds'][i] for i, comp in enumerate(['dust', 'sync', 'cmb'])}
np.savez(prefix+f'_seds.npz', **sdic)


#python simulation_simple_bandpass.py \
#  --output-dir ./outputs/bandpass_sim \
#  --seed 1234 \
#  --nside 256 \
#  --frequencies frequencies.txt \
#  --beam-fwhm-file beams.txt \
#  --use-bandpass \
#  --bandpass-dir /global/homes/s/susannaz/Software/bandpass_sampler/SAT/

# python simulation_simple_bandpass.py --output-dir ./outputs/bandpass_sim --seed 1234 --nside 256 --frequencies frequencies.txt --beam-fwhm-file beams.txt --use-bandpass --bandpass-dir /global/homes/s/susannaz/Software/bandpass_sampler/SAT/


