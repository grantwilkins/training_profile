python3 ../profiling/profile.py \
    --gpu 0 \
    --mode sweep \
    --duty_min 0.75 \
    --duty_max 0.95 \
    --duty_step 0.01 \
    --dwell_seconds 5.0 \
    --csv duty_sweep_gpu0_fine.csv
