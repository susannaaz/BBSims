#!/bin/bash

# data-path: where all the camb best-fit spectra and rest templates live 
# simple-sky assigns beta_sync = -3 and beta_dust=1.6 instead of beta maps -- could that be a problem?
# conv_space: map -- do smoothing of maps instead of harmonic space convolution

for seed in {1001..1100}
do
    echo ${seed}
    python simulation.py --data-path ../data \
    --output-dir /cfs/home/koda4949/simonsobs/beam_chromaticity/sims_dir/epsilon_${epsilon}_ --seed ${seed} --nside 256 \
    --band-names LF1 LF2 MF1 MF2 UHF1 UHF2 --simple-sky --conv_space map
   
done





