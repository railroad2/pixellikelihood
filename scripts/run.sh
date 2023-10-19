#!/bin/bash

wait_until_cpu_low(){
   tinit=`date +%s`
   awk -v target="$1" -v t0=$tinit '
    $13 ~ /^[0-9.]+$/ {
        current = 100 - $13;
        if (current <= target) { 
            printf "\n"
            exit(0); 
        }
        else { 
            t1 = systime() 
            dt = t1 - t0
            printf "\r Waiting for CPU usage to go below %s%, current: %s% ... %s s elapsed", target, current, dt; 
        }
    }' < <(mpstat 2)
}

ntest=9

configpath='./configs/'

for i in {0..9}; do
    # 1 nofg
    wait_until_cpu_low 50
    CMD="export OMP_NUM_THREADS=2; python3 ensemble_test.py ${configpath}/0_nofg.ini $i; "
    echo $CMD
    screen -S 1_nofg_seed$i -d -m bash -c "$CMD"

    # 2 2bands
    wait_until_cpu_low 50
    CMD="export OMP_NUM_THREADS=1; python3 ensemble_test.py ${configpath}/1_2bands.ini $i; "
    echo $CMD
    screen -S 2_2bands_seed$i -d -m bash -c "$CMD"

    # 2 2bands with residual
    wait_until_cpu_low 50
    CMD="export OMP_NUM_THREADS=1; python3 ensemble_test.py ${configpath}/2_2bands_res.ini $i; "
    echo $CMD
    screen -S 2_2bands_res_seed$i -d -m bash -c "$CMD"

    # 3 3bands
    wait_until_cpu_low 50
    CMD="export OMP_NUM_THREADS=1; python3 ensemble_test.py ${configpath}/3_3bands.ini $i; "
    echo $CMD
    screen -S 3_3bands_seed$i -d -m bash -c "$CMD"

    # 4 3bands_res
    wait_until_cpu_low 50
    CMD="export OMP_NUM_THREADS=1; python3 ensemble_test.py ${configpath}/4_3bands_res.ini $i; "
    echo $CMD
    screen -S 4_3bands_res_seed$i -d -m bash -c "$CMD"

    # 5 8 bands
    wait_until_cpu_low 50
    CMD="export OMP_NUM_THREADS=1; python3 ensemble_test.py ${configpath}/5_8bands.ini $i; "
    echo $CMD
    screen -S 5_8bands_seed$i -d -m bash -c "$CMD"

    # 5 8 bands res
    wait_until_cpu_low 50
    CMD="export OMP_NUM_THREADS=1; python3 ensemble_test.py ${configpath}/6_8bands_res.ini $i; "
    echo $CMD
    screen -S 5_8bands_res_seed$i -d -m bash -c "$CMD"

done

