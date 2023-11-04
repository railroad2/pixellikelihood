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

ntest=0

configpath='./configs'

for i in `seq -s ' ' 0 $ntest`; do
    # 1 nofg
    wait_until_cpu_low 50
    CMD="export OMP_NUM_THREADS=2; python3 ensemble_test.py ${configpath}/nofg_lownoise.ini $i; "
    echo $CMD
    #screen -S 1_nofg_seed$i -d -m bash -c "$CMD"
    bash -c "$CMD"
done

