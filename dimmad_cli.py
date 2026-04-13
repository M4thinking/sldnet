import json
import os
import pickle
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from distclassipy.anomaly import DistanceAnomaly
from fastparquet import ParquetFile
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from tqdm.auto import tqdm

from dimmad_benchmark import alerceanomalies, run_experiments, utils
from dimmad_benchmark.sbgm import SBGMAnomalyDetector

warnings.filterwarnings("ignore", category=FutureWarning)

epsilon = np.finfo(np.float32).eps


with open("settings.txt") as f:
    settings_dict = json.load(f)
seed_val = settings_dict["seed_choice"]
np.random.seed(seed_val)
sns_dict = settings_dict["sns_dict"]
sns.set_theme(**sns_dict)

set3colors = [matplotlib.colors.rgb2hex(x) for x in plt.cm.Set3.colors]
knowncolors = set3colors[:4]
unknowncolors = ['#e6194B', '#3cb44b', '#ffe119',
                 '#4363d8', '#f58231', '#911eb4',
                 '#42d4f4', '#f032e6', '#bfef45',
                 '#fabed4', '#469990', '#dcbeff',
                 '#9A6324', '#fffac8', '#800000',
                 '#aaffc3', '#808000', '#ffd8b1',
                 '#000075', '#a9a9a9']
unknowncolors = set3colors[4:]+unknowncolors


def _print_dataset_stats(features_df, new_knowns, new_unknowns, features_to_use):
    filtered_df = features_df[features_df["class"].isin(new_knowns + new_unknowns)]

    tot_frac_withoutnans = len(filtered_df.loc[:, features_to_use].dropna()) / len(filtered_df)
    print(f"{tot_frac_withoutnans:.1%} of the dataset is clean and free of NaNs")

    normal_frac_withoutnans = []
    for col in filtered_df.columns:
        if col == "class":
            continue
        normal_frac_withoutnans.append(len(filtered_df[col].dropna()) / len(filtered_df))

    normal_frac_withoutnans = np.median(normal_frac_withoutnans)
    print(f"The median no. of objects across columns which are not NaNs is {normal_frac_withoutnans:.1%}")

ELASTICC_FEATURES = [
    'SPM_A_g', 'SPM_gamma_g', 'SPM_beta_g', 'SPM_tau_rise_g', 'SPM_tau_fall_g',
    'SPM_A_r', 'SPM_gamma_r', 'SPM_beta_r', 'SPM_tau_rise_r', 'SPM_tau_fall_r',
    'SPM_A_i', 'SPM_gamma_i', 'SPM_beta_i', 'SPM_tau_rise_i', 'SPM_tau_fall_i',
    'SPM_A_z', 'SPM_gamma_z', 'SPM_beta_z', 'SPM_tau_rise_z', 'SPM_tau_fall_z',
    'SPM_A_Y', 'SPM_gamma_Y', 'SPM_beta_Y', 'SPM_tau_rise_Y', 'SPM_tau_fall_Y',
]

ZTF_FEATURES = [
    "SPM_A_1", "SPM_t0_1", "SPM_gamma_1", "SPM_beta_1", "SPM_tau_rise_1", "SPM_tau_fall_1", "SPM_chi_1",
    "SPM_A_2", "SPM_t0_2", "SPM_gamma_2", "SPM_beta_2", "SPM_tau_rise_2", "SPM_tau_fall_2", "SPM_chi_2",
]

DATASET_SPECS = {
    ("elasticc", "ood"): {
        "path": "data/dimmad/elasticc_features.parquet",
        "index_col": "snid",
        "knowns": ['EB', 'DSCT', 'RRL', 'CEP'],
        "unknowns": [
            'PISN-STELLA_HYDROGENIC', 'PISN-MOSFIT', 'uLens-Single_PyLIMA',
            'TDE', 'SNIcBL+HostXT_V19', 'KN_B19', 'uLens-Binary', 'SL-SNII',
            'SNIc-Templates', 'SLSN-I+host', 'SNIa-SALT3', 'SNIb-Templates',
            'SNII+HostXT_V19', 'SNIa-91bg', 'SL-SNIb', 'Mdwarf-flare', 'ILOT',
            'KN_K17', 'CART', 'SNIIb+HostXT_V19', 'SNIb+HostXT_V19', 'SL-SN1a',
            'SNII-NMF', 'SNIIn+HostXT_V19', 'SNII-Templates',
            'SNIc+HostXT_V19', 'SNIax', 'SNIIn-MOSFIT', 'uLens-Single-GenLens',
            'PISN-STELLA_HECORE', 'AGN', 'SLSN-I_no_host', 'SL-SNIc',
            'dwarf-nova',
        ],
        "features": ELASTICC_FEATURES,
    },
    ("elasticc", "rid"): {
        "path": "data/dimmad/elasticc_features.parquet",
        "index_col": "snid",
        "knowns": [
            'SNIcBL+HostXT_V19', 'SNIc-Templates', 'SNIa-SALT3', 'SNIb-Templates',
            'SNII+HostXT_V19', 'SNIa-91bg', 'SNIIb+HostXT_V19', 'SNIb+HostXT_V19',
            'SNII-NMF', 'SNIIn+HostXT_V19', 'SNII-Templates', 'SNIc+HostXT_V19', 'SNIax',
            'SNIIn-MOSFIT', 'SLSN-I+host', 'SLSN-I_no_host',
        ],
        "unknowns": ['PISN-STELLA_HYDROGENIC', 'PISN-MOSFIT', 'PISN-STELLA_HECORE'],
        "features": ELASTICC_FEATURES,
    },
    ("ztf", "ood"): {
        "path": "data/dimmad/alerceztf_features.parquet",
        "index_col": "oid",
        "knowns": ['RRL', 'LPV', 'E', 'QSO', 'YSO', 'AGN', 'CEP', 'Periodic-Other', 'DSCT', 'Blazar', 'CV/Nova'],
        "unknowns": ['SNIa', "SLSN", 'SNIbc', 'SNII'],
        "features": ZTF_FEATURES,
    },
    ("ztf", "rid"): {
        "path": "data/dimmad/alerceztf_features.parquet",
        "index_col": "oid",
        "knowns": ['LPV', 'E', 'Periodic-Other', 'Blazar', 'CV/Nova'],
        "unknowns": ['RRL', 'CEP', 'DSCT'],
        "features": ZTF_FEATURES,
    },
}


def load_dataset_split(dataset, scheme):
    spec = DATASET_SPECS[(dataset, scheme)]
    features_df = ParquetFile(spec["path"]).to_pandas(index=False).set_index(spec["index_col"])
    new_knowns = spec["knowns"]
    new_unknowns = spec["unknowns"]
    features_to_use = spec["features"]

    features_df = features_df[features_df["class"].isin(new_knowns + new_unknowns)]

    # Could use features we know are better for the 4 variable star classification
    # but don't want to bias. additionally, the above features are simple and general
    # and have very little NaNs.
    _print_dataset_stats(features_df, new_knowns, new_unknowns, features_to_use)

    full_df = features_df.loc[:, features_to_use + ["class"]].dropna(subset=features_to_use)
    full_df = full_df[~full_df.index.duplicated(keep='first')]

    # full_df['status'] = np.where(full_df['class'].isin(new_knowns), 'normal', 'anomalous')
    # will use this for stratification
    full_inlier_df = full_df[full_df["class"].isin(new_knowns)]
    full_outlier_df = full_df[full_df["class"].isin(new_unknowns)]

    return full_inlier_df, full_outlier_df, new_knowns, features_to_use

def run(result_path, full_inlier_df, full_outlier_df, new_knowns, features_to_use, models_to_test, seed_val, num_runs=20):
    if not os.path.exists(result_path):
        all_run_results = run_experiments.run_experiment(
            N_RUNS = num_runs, 
            full_inlier_df=full_inlier_df, 
            models_to_test=models_to_test,
            full_outlier_df=full_outlier_df,
            new_knowns = new_knowns, 
            features_to_use = features_to_use,
            seed_val=seed_val
        )
        
        with open(result_path,"wb") as fp:
            pickle.dump(all_run_results, fp)

    else:
        with open(result_path,"rb") as fp:
            all_run_results=pickle.load(fp)
            
    return all_run_results

def plot_diversity_purity(models_to_test, all_run_results, results_path, name, budget=300):
    from pathlib import Path
    import matplotlib as mpl
    import matplotlib.font_manager as font_manager
    fpath = Path("font/cmunrm.ttf")
    font_manager.fontManager.addfont(fpath)
    prop = font_manager.FontProperties(fname=fpath)
    font_name = prop.get_name()
    plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False
    
    print(f"Budget {budget}")
    fig = run_experiments.analysis_with_errors(
        models_to_test, all_run_results, budget=budget, metric="purity", show_legend=True
    )
    purity_max = max((np.nanmax(line.get_ydata()) for line in plt.gca().lines), default=0)
    plt.ylim(0, max(0.1, np.ceil(purity_max * 10) / 10))
    # plt.gca().get_legend().remove()
    plt.savefig(f"{results_path}/{name}_a.pdf", bbox_inches="tight")
    plt.close()

    fig = run_experiments.analysis_with_errors(
        models_to_test, all_run_results, budget=budget, metric="diversity", show_legend=True
    )
    plt.savefig(f"{results_path}/{name}_b.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    from argparse import ArgumentParser
    
    p = ArgumentParser()
    p.add_argument('--result_path', type=str, default="results/dimmad")
    p.add_argument('--num_runs', type=int, default=20)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--dataset', type=str, default="elasticc", choices=["elasticc","ztf"])
    p.add_argument('--scheme', type=str, default="rid", choices=["rid","ood"])
    args = p.parse_args()
    
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    
    result_path = os.path.join(args.result_path, f"{args.dataset}_{args.scheme}_results.pkl")
    num_runs = args.num_runs
    dataset = args.dataset
    scheme = args.scheme
    epochs = args.epochs
    
    full_inlier_df, full_outlier_df, new_knowns, features_to_use = load_dataset_split(dataset, scheme)
    
    epochs = args.epochs
    
    # UNCOMMENT all the baselines you want to run.
    # Can be time consuming, especially the autoencoder and MCSVDD.
    models_to_test = {
        "SLDNet (L:4)": SBGMAnomalyDetector(
            epochs=epochs,
            L=4,
            betas=(0.9, 0.999),
        ),
        "SLDNet (L:0.001)": SBGMAnomalyDetector(
            epochs=epochs,
            L=0.001,
            betas=(0.9, 0.999),
        ),
        # "iForest": IsolationForest( #using alerce anomaly params
        #     n_estimators=100,
        #     max_samples=256,
        #     contamination=0.001,
        #     random_state=seed_val
        # ),
        # "LOF": LocalOutlierFactor(
        #     n_neighbors=20, 
        #     novelty=True,  # IMPORTANT: Allows use on new data
        #     contamination='auto'
        # ),
        
        # "OC-SVM": OneClassSVM(
        #     kernel='rbf',
        #     nu=0.001
        # ),

        # "Autoencoder": alerceanomalies.AutoencoderAnomalyDetector(
        #     encoding_dim=32, # Can be tuned
        #     epochs=500,
        #     patience=15,
        # ),

        # "MCSVDD": alerceanomalies.ClassSVDDAnomalyDetector(
        #     z_dim=64,
        #     epochs=500,
        #     patience=15,
        #     lr=1e-4,
        #     verbose=False
        # ),
            
        # "DiMMAD (med-med)": DistanceAnomaly(
        #     cluster_agg='median',
        #     metric_agg='median',
        #     normalize_scores=True
        # ),

        # "DiMMAD (min-med)": DistanceAnomaly(
        #     cluster_agg='min',
        #     metric_agg='median',
        #     normalize_scores=True
        # ),
    }
    all_runs_combined = run(
        result_path=result_path,
        full_inlier_df=full_inlier_df,
        full_outlier_df=full_outlier_df,
        new_knowns=new_knowns,
        features_to_use=features_to_use,
        models_to_test=models_to_test,
        seed_val=seed_val,
        num_runs=num_runs
    )
    newnames = {
        'score_DistClassiPy (min-median)': 'score_DiMMAD (min-med)',
        'score_DistClassiPy (median-median)': 'score_DiMMAD (med-med)',
        'score_IsolationForest': 'score_iForest',
        'score_LocalOutlierFactor': 'score_LOF',
        'score_OneClassSVM': 'score_OC-SVM',
        'score_Autoencoder': 'score_Autoencoder',
        'score_ClassSVDD': 'score_MCSVDD',
        'score_SLDNet (L:0.001)': 'score_SLDNet (L:0.001)',
        'score_SLDNet (L:4)': 'score_SLDNet (L:4)',
    }


    for i in range(len(all_runs_combined)):
        ar = all_runs_combined[i]
        all_runs_combined[i] = ar.rename(columns=newnames)
    
    plot_diversity_purity(
        models_to_test=models_to_test,
        all_run_results=all_runs_combined,
        results_path=args.result_path,
        name=f"{dataset}_{scheme}",
        budget=300
    )
    
    # Save a .txt with printed configs
    with open(os.path.join(args.result_path, f"{dataset}_{scheme}_config.txt"), "w") as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Scheme: {scheme}\n")
        f.write(f"Num runs: {num_runs}\n")
        f.write("\nModels tested:\n")
        for model_name, model in models_to_test.items():
            f.write(model_name + ": " + str(model) + "\n")


# Commands are
# python dimmad_cli.py --dataset elasticc --scheme rid
# python dimmad_cli.py --dataset elasticc --scheme ood
# python dimmad_cli.py --dataset ztf --scheme rid
# python dimmad_cli.py --dataset ztf --scheme ood
