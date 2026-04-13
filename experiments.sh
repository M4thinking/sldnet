
# python dimmad_cli.py --dataset ztf --scheme rid --num_runs 1 --epochs 1000
# python dimmad_cli.py --dataset ztf --scheme ood --num_runs 1 --epochs 1000
# python dimmad_cli.py --dataset elasticc --scheme rid --num_runs 1 --epochs 1000
# python dimmad_cli.py --dataset elasticc --scheme ood --num_runs 1 --epochs 1000

# python alerce_cli.py --epochs 10000 --fold all --scheme Transient  --all_outliers
# python alerce_cli.py --epochs 10000 --fold all --scheme Periodic   --all_outliers 
# python alerce_cli.py --epochs  5000 --fold all --scheme Stochastic --all_outliers --lr 2e-5

# python alerce_cli.py --epochs 10000 --fold all --scheme Transient  --all_outliers               > log_transient.log  2>&1 &
# python alerce_cli.py --epochs 10000 --fold all --scheme Periodic   --all_outliers               > log_periodic.log   2>&1 &
# python alerce_cli.py --epochs  5000 --fold all --scheme Stochastic --all_outliers --lr 2e-5     > log_stochastic.log 2>&1 &

# wait


# python alerce_cli.py --epochs 10000 --patience -1 --fold all --scheme Transient  --all_outliers --L 4 > log_transient.log  2>&1 &
python alerce_cli.py --epochs 10000 --patience -1 --fold all --scheme Periodic   --all_outliers --L 4 > log_periodic.log   2>&1 &
# python alerce_cli.py --epochs  5000 --patience -1 --fold all --scheme Stochastic --all_outliers --lr 2e-5 > log_stochastic.log 2>&1 &

# wait

# periodic classes in parallel
# python alerce_cli.py --epochs 10000 --patience -1 --fold all --scheme Periodic --outlier CEP  --L 4 > log_periodic_cep.log  2>&1 &
# python alerce_cli.py --epochs 10000 --patience -1 --fold all --scheme Periodic --outlier DSCT --L 4 > log_periodic_dsct.log 2>&1 &
# python alerce_cli.py --epochs 10000 --patience -1 --fold all --scheme Periodic --outlier E    --L 4 > log_periodic_e.log    2>&1 &
# python alerce_cli.py --epochs 10000 --patience -1 --fold all --scheme Periodic --outlier RRL  --L 4 > log_periodic_rrl.log  2>&1 &
# python alerce_cli.py --epochs 10000 --patience -1 --fold all --scheme Periodic --outlier LPV  --L 4 > log_periodic_lpv.log  2>&1 &

wait
