##############################################################################################################################################################
##############################################################################################################################################################
"""
Helperfunctions used by other scripts.
"""
##############################################################################################################################################################
##############################################################################################################################################################

import math
import toml
import time
import torch
import pickle
import datetime
import dateutil

import numpy as np

from pathlib import Path
from collections import defaultdict

from PIL import Image
from sklearn import metrics
from torchvision.utils import save_image as save_grid
from torchvision.transforms import functional as TF

from matplotlib import pyplot as plt
plt.style.use(['seaborn-white', 'seaborn-paper'])
plt.rc('font', family='serif')

##############################################################################################################################################################
##############################################################################################################################################################

class Tee(object):
    """
    Class to make it possible to print text to the console and also write the 
    output to a file.
    """

    def __init__(self, original_stdout, file):

        # keep the original stdout
        self.original_stdout = original_stdout

        # the file to write to
        self.log_file_handler= open(file, 'w')

        # all the files the print statement should be saved to
        self.files = [self.original_stdout, self.log_file_handler]

    def write(self, obj):

        # for each file
        for f in self.files:

            # write to the file
            f.write(obj)

            # If you want the output to be visible immediately
            f.flush() 

    def flush(self):

        # for each file
        for f in self.files:

            # If you want the output to be visible immediately
            f.flush()

    def end(self):

        # close the file
        self.log_file_handler.close()

        # return the original stdout
        return self.original_stdout

##############################################################################################################################################################

def write_toml_to_file(cfg, file_path):
    """
    Write the parser to a file.
    """

    with open(file_path, 'w') as output_file:
        toml.dump(cfg, output_file)

    print('=' * 57)
    print("Config file saved: ", file_path)
    print('=' * 57)
    
##############################################################################################################################################################

def save_accuracy(save_path, score):

    with save_path.open('w') as file:
        file.write(str(score))

##############################################################################################################################################################

class TrainingTimer():
    """
    Keep track of training times.
    """

    def __init__(self):

        # get the current time
        self.start = datetime.datetime.fromtimestamp(time.time())
        self.eval_start = datetime.datetime.fromtimestamp(time.time())

        # print it human readable
        print("Training start: ", self.start)
        print('=' * 57)

    def print_end_time(self):

        # get datetimes for simplicity
        datetime_now = datetime.datetime.fromtimestamp(time.time())

        print("Training finish: ", datetime_now)

        # compute the time different between now and the input
        rd = dateutil.relativedelta.relativedelta(datetime_now, self.start)

        # print the duration in human readable
        print(f"Training duration: {rd.hours} hours, {rd.minutes} minutes, {rd.seconds} seconds")

    
    def print_time_delta(self):

        # get datetimes for simplicity
        datetime_now = datetime.datetime.fromtimestamp(time.time())

        # compute the time different between now and the input
        rd = dateutil.relativedelta.relativedelta(datetime_now, self.eval_start)

        # print the duration in human readable
        print(f"Duration since last evaluation: {rd.hours} hours, {rd.minutes} minutes, {rd.seconds} seconds")
        print('=' * 57)

        # update starting time
        self.eval_start = datetime.datetime.fromtimestamp(time.time())

##############################################################################################################################################################

def plot_progress(images, save_folder, nbr_epoch, text, max_nbr_samples=16, transpose=False, **kwargs):

    # check which number is smaller and use it to extract the correct amount of images per element in the list
    nbr_samples = min(images[0].shape[0], max_nbr_samples)

    # collect only the number of images we want to plot
    gather_sample_results = [x[:nbr_samples].detach() for x in images]
    
    # if we want to transpose the whole grid, e.g. plot from left to right instead of 
    # plotting from top to down
    if transpose:

        # collect the transposed version
        transposed_gather = []

        # for each sample, i.e. batch element
        for idx in range(nbr_samples):

            # collect from each list the idx'th sample
            # e.g. each reconstuction of batch element idx
            samples = [x[idx] for x in gather_sample_results]

            # make sure its a tensor and add the empty dimension for single channel image
            transposed_gather.append(torch.cat(samples).unsqueeze(1))
        
        # since we plot differently, we need to change the number of samples
        # to start the new row at the correct position
        nbr_samples = len(gather_sample_results)

        # change pointer for ease of use 
        gather_sample_results = transposed_gather

    # concat all the tensors in the list to a single tensor of the correct dimension
    gather_sample_results = torch.cat(gather_sample_results, dim=0)

    # create the filename
    save_name = save_folder["images"] / (text + "_" + str(nbr_epoch) + ".png")

    # save images as grid
    # each element in the batch has its own column
    save_grid(gather_sample_results, save_name, nrow=nbr_samples, **kwargs)
        
##############################################################################################################################################################

def compute_scores(accurate_certain, accurate_uncertain, inaccurate_certain, inaccurate_uncertain, threshold_to_consider, which_dataset, folders, idx_recursion, which_classifier, fix_dropout_mask):

    # true_positive = accurate_certain = true certainty
    # true_negative = inaccurate_uncertain = true uncertainty 
    # false_positive = inaccurate_certain = false certainty
    # false_negative = accurate_uncertain = false uncertainty

    # tu / (tu + fc)
    # tc / (tc + fu)
    # tu / (tu + fu)
    # (tu + tc) / (tu + tc + fu + fc)

    u_sen = [inaccurate_uncertain[key]/(inaccurate_uncertain[key]+inaccurate_certain[key]) for key in sorted(threshold_to_consider)]
    u_spec = [accurate_certain[key]/(accurate_certain[key]+accurate_uncertain[key]) for key in sorted(threshold_to_consider)]
    u_pre = [inaccurate_uncertain[key]/(inaccurate_uncertain[key]+accurate_uncertain[key]) for key in sorted(threshold_to_consider)]
    u_acc = [(accurate_certain[key]+inaccurate_uncertain[key])/(accurate_certain[key]+inaccurate_uncertain[key]+accurate_uncertain[key]+inaccurate_certain[key]) for key in sorted(threshold_to_consider)]

    with open(folders["data-all-proba"] / f'{which_dataset}_u_sen_recursion_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.pkl', 'wb') as f:
        pickle.dump(u_sen, f)
    with open(folders["data-all-proba"] / f'{which_dataset}_u_spec_recursion_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.pkl', 'wb') as f:
        pickle.dump(u_spec, f)
    with open(folders["data-all-proba"] / f'{which_dataset}_u_pre_uncertainty_recursion_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.pkl', 'wb') as f:
        pickle.dump(u_pre, f)
    with open(folders["data-all-proba"] / f'{which_dataset}_u_acc_recursion_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.pkl', 'wb') as f:
        pickle.dump(u_acc, f)

    accurate_certain_probability = [accurate_certain[key]/(accurate_certain[key]+inaccurate_certain[key]) for key in sorted(threshold_to_consider)]
    uncertain_inaccurate_probability = [inaccurate_uncertain[key]/(inaccurate_uncertain[key]+inaccurate_certain[key]) for key in sorted(threshold_to_consider)]
    patch_accuracy_v_patch_uncertainty = [(accurate_certain[key]+inaccurate_uncertain[key])/(accurate_certain[key]+inaccurate_uncertain[key]+accurate_uncertain[key]+inaccurate_certain[key]) for key in sorted(threshold_to_consider)]

    with open(folders["data-all-proba"] / f'{which_dataset}_accurate_certain_probability_recursion_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.pkl', 'wb') as f:
        pickle.dump(accurate_certain_probability, f)
    with open(folders["data-all-proba"] / f'{which_dataset}_uncertain_inaccurate_probability_recursion_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.pkl', 'wb') as f:
        pickle.dump(uncertain_inaccurate_probability, f)
    with open(folders["data-all-proba"] / f'{which_dataset}_patch_accuracy_v_patch_uncertainty_recursion_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.pkl', 'wb') as f:
        pickle.dump(patch_accuracy_v_patch_uncertainty, f)

    # tpr = tp / (tp + fn)
    tpr = [accurate_certain[key]/(accurate_certain[key]+accurate_uncertain[key]) for key in sorted(threshold_to_consider)]
    # fpr = fp / (fp + tn)
    fpr = [inaccurate_certain[key]/(inaccurate_certain[key]+inaccurate_uncertain[key]) for key in sorted(threshold_to_consider)]
    
    with open(folders["data-all-proba"] / f'{which_dataset}_tpr_recursion_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.pkl', 'wb') as f:
        pickle.dump(tpr, f)
    with open(folders["data-all-proba"] / f'{which_dataset}_fpr_recursion_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.pkl', 'wb') as f:
        pickle.dump(fpr, f)

    # precision = tp / (tp + fp)
    precision = [accurate_certain[key]/(accurate_certain[key]+inaccurate_certain[key]) for key in sorted(threshold_to_consider)]
    # recall = tp / (tp + fn)
    recall = [accurate_certain[key]/(accurate_certain[key]+accurate_uncertain[key]) for key in sorted(threshold_to_consider)]

    # replace nan values with 1, otherwise aupr is always 0
    precision = [1 if math.isnan(x) else x for x in precision]

    with open(folders["data-all-proba"] / f'{which_dataset}_precision_recursion_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.pkl', 'wb') as f:
        pickle.dump(precision, f)
    with open(folders["data-all-proba"] / f'{which_dataset}_recall_recursion_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.pkl', 'wb') as f:
        pickle.dump(recall, f)

    # the ratio of the number of certain but incorrect samples to all samples
    rer = [inaccurate_certain[key]/(accurate_certain[key]+inaccurate_uncertain[key]+accurate_uncertain[key]+inaccurate_certain[key]) for key in sorted(threshold_to_consider)]
    # the ratio of the number of certain and correct samples to all samples
    rar = [accurate_certain[key]/(accurate_certain[key]+inaccurate_uncertain[key]+accurate_uncertain[key]+inaccurate_certain[key]) for key in sorted(threshold_to_consider)]

    with open(folders["data-all-proba"] / f'{which_dataset}_rer_recursion_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.pkl', 'wb') as f:
        pickle.dump(rer, f)
    with open(folders["data-all-proba"] / f'{which_dataset}_rar_recursion_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.pkl', 'wb') as f:
        pickle.dump(rar, f)

    tpr_rate = 0.3
    save_accuracy(Path(folders["data-all-proba"]) / f'fdpr-{tpr_rate}_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance', np.nan_to_num(np.array(fpr).flat[np.abs(np.array(tpr) - tpr_rate).argmin()], nan=0.0))
    tpr_rate = 0.5
    save_accuracy(Path(folders["data-all-proba"]) / f'fdpr-{tpr_rate}_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance', np.nan_to_num(np.array(fpr).flat[np.abs(np.array(tpr) - tpr_rate).argmin()], nan=0.0))
    tpr_rate = 0.8
    save_accuracy(Path(folders["data-all-proba"]) / f'fdpr-{tpr_rate}_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance', np.nan_to_num(np.array(fpr).flat[np.abs(np.array(tpr) - tpr_rate).argmin()], nan=0.0))
    tpr_rate = 0.9
    save_accuracy(Path(folders["data-all-proba"]) / f'fdpr-{tpr_rate}_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance', np.nan_to_num(np.array(fpr).flat[np.abs(np.array(tpr) - tpr_rate).argmin()], nan=0.0))
    tpr_rate = 0.95
    save_accuracy(Path(folders["data-all-proba"]) / f'fdpr-{tpr_rate}_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance', np.nan_to_num(np.array(fpr).flat[np.abs(np.array(tpr) - tpr_rate).argmin()], nan=0.0))

    
    save_accuracy(Path(folders["data-all-proba"]) / f"auroc_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance", metrics.auc(fpr, tpr))
    save_accuracy(Path(folders["data-all-proba"]) / f"accurate_certain_probability_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance", np.nan_to_num(np.abs(np.trapz(threshold_to_consider, accurate_certain_probability)), nan=0.0))
    save_accuracy(Path(folders["data-all-proba"]) / f"uncertain_inaccurate_probability_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance", np.nan_to_num(np.abs(np.trapz(threshold_to_consider, uncertain_inaccurate_probability)), nan=0.0))
    save_accuracy(Path(folders["data-all-proba"]) / f"patch_accuracy_v_patch_uncertainty_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance", np.nan_to_num(np.abs(np.trapz(threshold_to_consider, patch_accuracy_v_patch_uncertainty)), nan=0.0))
    save_accuracy(Path(folders["data-all-proba"]) / f"aupr_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance", metrics.auc(recall, precision))
    save_accuracy(Path(folders["data-all-proba"]) / f"rervsrar_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance", np.nan_to_num(np.abs(np.trapz(rer, rar)), nan=0.0))

    save_accuracy(Path(folders["data-all-proba"]) / f"u_sen_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance", np.nan_to_num(np.abs(np.trapz(threshold_to_consider, u_sen)), nan=0.0))
    save_accuracy(Path(folders["data-all-proba"]) / f"u_spec_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance", np.nan_to_num(np.abs(np.trapz(threshold_to_consider, u_spec)), nan=0.0))
    save_accuracy(Path(folders["data-all-proba"]) / f"u_pre_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance", np.nan_to_num(np.abs(np.trapz(threshold_to_consider, u_pre)), nan=0.0))
    save_accuracy(Path(folders["data-all-proba"]) / f"u_accy_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance", np.nan_to_num(np.abs(np.trapz(threshold_to_consider, u_acc)), nan=0.0))

##############################################################################################################################################################


def compute_ood_scores(true_positive, true_negative, false_positive, false_negative, threshold_to_consider, which_dataset, folders, idx_recursion, which_classifier, fix_dropout_mask):

    # tpr = tp / (tp + fn)
    tpr = [true_positive[key]/(true_positive[key]+false_negative[key]) for key in sorted(threshold_to_consider)]
    # fpr = fp / (fp + tn)
    fpr = [false_positive[key]/(false_positive[key]+true_negative[key]) for key in sorted(threshold_to_consider)]
    
    with open(folders["data-ood"] / f'{which_dataset}_tpr_recursion_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.pkl', 'wb') as f:
        pickle.dump(tpr, f)
    with open(folders["data-ood"] / f'{which_dataset}_fpr_recursion_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.pkl', 'wb') as f:
        pickle.dump(fpr, f)

    # precision = tp / (tp + fp)
    precision = [true_positive[key]/(true_positive[key]+false_positive[key]) for key in sorted(threshold_to_consider)]
    # recall = tp / (tp + fn)
    recall = [true_positive[key]/(true_positive[key]+false_negative[key]) for key in sorted(threshold_to_consider)]

    # replace nan values with 1, otherwise aupr is always 0
    precision = [1 if math.isnan(x) else x for x in precision]

    with open(folders["data-ood"] / f'{which_dataset}_precision_recursion_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.pkl', 'wb') as f:
        pickle.dump(precision, f)
    with open(folders["data-ood"] / f'{which_dataset}_recall_recursion_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.pkl', 'wb') as f:
        pickle.dump(recall, f)

    save_accuracy(Path(folders["data-ood"]) / f"ood_auroc_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance", metrics.auc(fpr, tpr))
    save_accuracy(Path(folders["data-ood"]) / f"ood_aupr_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance", metrics.auc(recall, precision))

    tpr_rate = 0.3
    save_accuracy(Path(folders["data-ood"]) / f'ood_fdpr-{tpr_rate}_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance', np.nan_to_num(np.array(fpr).flat[np.abs(np.array(tpr) - tpr_rate).argmin()], nan=0.0))
    tpr_rate = 0.5
    save_accuracy(Path(folders["data-ood"]) / f'ood_fdpr-{tpr_rate}_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance', np.nan_to_num(np.array(fpr).flat[np.abs(np.array(tpr) - tpr_rate).argmin()], nan=0.0))
    tpr_rate = 0.8
    save_accuracy(Path(folders["data-ood"]) / f'ood_fdpr-{tpr_rate}_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance', np.nan_to_num(np.array(fpr).flat[np.abs(np.array(tpr) - tpr_rate).argmin()], nan=0.0))
    tpr_rate = 0.9
    save_accuracy(Path(folders["data-ood"]) / f'ood_fdpr-{tpr_rate}_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance', np.nan_to_num(np.array(fpr).flat[np.abs(np.array(tpr) - tpr_rate).argmin()], nan=0.0))
    tpr_rate = 0.95
    save_accuracy(Path(folders["data-ood"]) / f'ood_fdpr-{tpr_rate}_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance', np.nan_to_num(np.array(fpr).flat[np.abs(np.array(tpr) - tpr_rate).argmin()], nan=0.0))


##############################################################################################################################################################

