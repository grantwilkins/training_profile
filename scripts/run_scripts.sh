python3 ../profiling/profile.py \
    --gpu 0 \
    --mode sweep \
    --duty_min 0.1 \
    --duty_max 1.0 \
    --duty_step 0.1 \
    --dwell_seconds 5.0 \
    --csv duty_sweep_gpu0.csv
