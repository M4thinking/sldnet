
python dimmad_cli.py --dataset ztf --scheme rid --num_runs 20 --epochs 1000
python dimmad_cli.py --dataset ztf --scheme ood --num_runs 20 --epochs 1000
python dimmad_cli.py --dataset elasticc --scheme rid --num_runs 20 --epochs 1000
python dimmad_cli.py --dataset elasticc --scheme ood --num_runs 20 --epochs 1000

python alerce_cli.py --epochs 10000 --fold all --scheme Transient  --all_outliers
python alerce_cli.py --epochs 10000 --fold all --scheme Periodic   --all_outliers 
python alerce_cli.py --epochs  5000 --fold all --scheme Stochastic --all_outliers --lr 2e-5