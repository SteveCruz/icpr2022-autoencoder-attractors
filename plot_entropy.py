##############################################################################################################################################################
##############################################################################################################################################################
"""
Create the entropy plots used in the paper. Simply provide the experiment for which you want to create it.
Then select which classifier and recursion you want to use.
"""
##############################################################################################################################################################
##############################################################################################################################################################

import toml
import numpy as np

from scipy import stats

from pathlib import Path
from collections import defaultdict

import eval_ae

import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use(['seaborn-white', 'seaborn-paper'])
plt.rc('font', family='serif')
# force matplotlib to not plot images
mpl.use("Agg")
plt.style.use(['seaborn-white', 'seaborn-paper'])
plt.rc('font', family='serif')
plt.rc('axes', labelsize=8)
plt.rc('font', size=8)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('legend', fontsize=10)
plt.rc('figure', figsize=[9, 4.5])

##############################################################################################################################################################
##############################################################################################################################################################

def plot_entropy(experiment, which_classifier):

    # get the path to all the folder and files
    folders = eval_ae.get_folder(experiment)

    train_config = toml.load(folders["config"])

    # get all the performance files in the data folder
    files = list(folders["data-entropy"].glob("*.npy"))

    results_ood = dict()
    results_eval = dict()

    # for each performance file
    for file in sorted(files):

        # make sure to use only 2 recursions
        if "recursion" in file.name:
            if not which_classifier in file.name:
                continue

        what_is_it = file.name.split("_")[0]
        ood_dataset_name = file.name.split("eval_")[-1].split("_")[0].split(".npy")[0]
        eval_dataset_name = file.name.split("train_")[-1].split("_")[0].split(".npy")[0]

        # avoid using itself as ood
        if ood_dataset_name == train_config["dataset"]["name"]:
            continue

        if what_is_it.lower() != "ood":
            results_eval[ood_dataset_name] = np.load(file)
        else:
            results_ood[ood_dataset_name] = np.load(file)

    fig = plt.figure()

    plt.hist(list(results_eval.values())[0], bins=50, histtype='stepfilled', linewidth=1.0, alpha=0.75, density=False, label=f"$D_{{in}}$ - {eval_dataset_name.upper()}")

    for name, data in results_ood.items():
        plt.hist(data, bins=50, histtype='step', linewidth=1.0, alpha=0.99, density=False, label=f"$D_{{out}}$ - {name.upper()}")

    plt.xlim([0, 1.0])
    plt.ylim([0, 1600])

    plt.legend()
    plt.grid(color="0.9", linestyle='-', linewidth=1)
    plt.box(False)
    plt.xlabel("Entropy")

    filename = folders["images"].parent / "entropy.png"

    fig.savefig(filename, dpi=fig.dpi, bbox_inches='tight')
    plt.close(fig=fig)


    cifar_results = results_ood["cifar10"]
    test_results = list(results_eval.values())[0]
    cifar_all_distances = 0
    test_all_distances = 0
    for x,y in results_ood.items():
        if x == "test":
            continue
        cifar_all_distances += stats.wasserstein_distance(cifar_results, y) 
        test_all_distances += stats.wasserstein_distance(test_results, y) 

    print(f"Sum of distances between test and all D_out: {test_all_distances:.3f}")
    print(f"Sum of distances between Cifar10 and all other D_out: {cifar_all_distances:.3f}")

    return test_all_distances, cifar_all_distances

##############################################################################################################################################################
##############################################################################################################################################################

if __name__ == "__main__":

    # which classifier and how many recursions to use
    # nbr_recursions_classifier
    # which_classifier = "20_mlp"
    # which_classifier = "5_mlp"
    # which_classifier = "4_mlp"
    # which_classifier = "3_mlp"
    which_classifier = "2_mlp"
    # which_classifier = "1_mlp"
    # which_classifier = "0_mlp"

    all_test_distances = []
    all_cifar_distances = []

    experiments = [

    ]

    for experiment in experiments:
        print("-" * 37)
        print("Experiment: ", str(experiment))
        test_distances, cifar_distances = plot_entropy(experiment, which_classifier)
        all_test_distances.append(test_distances)
        all_cifar_distances.append(cifar_distances)

    print()
    print("*" * 77)
    print("*" * 77)
    print(f"Mean of distances between test and all D_out: {np.mean(np.array(all_test_distances)):.3f}  +- {np.std(np.array(all_test_distances)):.3f}")
    print(f"Mean of distances between Cifar10 and all other D_out: {np.mean(np.array(all_cifar_distances)):.3f} +- {np.std(np.array(all_cifar_distances)):.3f}")