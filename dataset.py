####################################################################################################################################################
####################################################################################################################################################
"""
Dataloader definitions for all the datasets used in our paper.
The datasets need to be downloaded manually and placed inside a same folder.
Specify your folder location in the following line

# directory containing all the datasets
ROOT_DATA_DIR = Path("")
"""
####################################################################################################################################################
####################################################################################################################################################

import os
import random
import numpy as np
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.datasets import FashionMNIST as TFashionMNIST
from torchvision.datasets import CIFAR10 as TCIFAR10
from torchvision.datasets import SVHN as TSVHN
from torchvision.datasets import Omniglot as TOmniglot
from torchvision.datasets import Places365 as TPlaces365
from torchvision.datasets import LSUN as TLSUN
from torchvision.datasets import MNIST as TMNIST

import albumentations as album

from collections import defaultdict

####################################################################################################################################################
####################################################################################################################################################

# directory containing all the datasets
ROOT_DATA_DIR = Path("")

####################################################################################################################################################

class BaseDatasetCar(Dataset):
    """
    Base class for all dataset classes for the vehicle interior.
    """

    def __init__(self, root_dir, car, split, make_scene_impossible, make_instance_impossible, augment=False, nbr_of_samples_per_class=-1):

        # path to the main folder
        self.root_dir =  Path(root_dir)

        # which car are we using?
        self.car = car

        # train or test split
        self.split = split

        # are we using training data
        self.is_train = True if "train" in self.split else False

        # normal or impossible reconstruction loss?
        self.make_scene_impossible = make_scene_impossible
        self.make_instance_impossible = make_instance_impossible

        # pre-process the data if necessary
        self._pre_process_dataset()

        # load the data into the memory
        self._get_data()

        # number of samples to keep for each class
        self.nbr_of_samples_per_class = nbr_of_samples_per_class

        # only get a subset of the data
        self._get_subset_of_data()

        # augmentations if needed
        if augment:
            self.augment = album.Compose(
                [
                    album.Blur(always_apply=False, p=0.4, blur_limit=7),
                    album.RandomBrightnessContrast(always_apply=False, p=0.4, brightness_limit=(0.0, 0.33), contrast_limit=(0.0, 0.33), brightness_by_max=False),
                    album.MultiplicativeNoise(always_apply=False, p=0.4, elementwise=True, multiplier=(0.75, 1.25)),
                ]
            )
        else:
            self.augment = False

        # dict to match the concatenations of the three seat position classes into a single integer
        self.label_str_to_int = {'0_0_0': 0, '0_0_3': 1, '0_3_0': 2, '3_0_0': 3, '0_3_3': 4, '3_0_3': 5, '3_3_0': 6, '3_3_3': 7}

        # the revers of the above, to transform int labels back into strings
        self.int_to_label_str = {v:k for k,v in self.label_str_to_int.items()}

    def _get_subset_of_data(self):

        # if we are training
        if self.nbr_of_samples_per_class > 0:

            # keep track of samples per class
            counter = defaultdict(int)
            # and the corresponding indices
            keep_indices = []

            # for each label
            for idx, label in enumerate(self.labels):
                
                # make sure its a string
                label =  self._get_classif_str(label)

                # increase the counter for this label
                counter[label] += 1
                
                # if we are above the theshold for this label
                if counter[label] >= (self.nbr_of_samples_per_class+1):
                    # then skip it
                    continue
                else:
                    # otherwise keep track of the label
                    keep_indices.append(idx)
            
            # only take the subset of indices based on how many samples per class to keep
            self.images = [x for idx, x in enumerate(self.images) if idx in keep_indices]
            self.labels = [x for idx, x in enumerate(self.labels) if idx in keep_indices]

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        # number of images to use
        return len(self.images)

    def _get_data(self):

        # get all folders with the sceneries
        if self.car.lower() == "all":
            self.folders = sorted(list(self.root_dir.glob(f"*/pp_{self.split}_64/*")))
        else:
            self.folders = sorted(list(self.root_dir.glob(f"{self.car}/pp_{self.split}_64/*")))

        # placeholder for all images and labels
        self.images = []
        self.labels = []

        # for each folder
        for idx, folder in enumerate(self.folders):

            # get classification labels for each seat from folder name
            classif_labels = self._get_classif_label(folder)

            # each scene will be an array of images
            self.images.append([])

            # get all the images for this scene
            files = sorted(list(folder.glob("*.png")))

            # for each file
            for file in files:
        
                # open the image specified by the path
                # make sure it is a grayscale image
                img = np.array(Image.open(file).convert("L"))

                # append the image to the placeholder
                self.images[idx].append(img)
            
            # append label to placeholder
            self.labels.append(classif_labels)

    def _get_classif_label(self, file_path):

        # get the filename only of the path
        name = file_path.stem

        # split at GT 
        gts = name.split("GT")[-1]
        
        # split the elements at _
        # first element is empty string, remove it
        clean_gts = gts.split("_")[1:]

        # convert the strings to ints
        clean_gts = [int(x) for x in clean_gts]

        # convert sviro labels to compare with other datasets
        for index, value in enumerate(clean_gts):
            # everyday objects and child seats to background
            if value in [1,2,4,5,6]:
                clean_gts[index] = 0

        return clean_gts

    def _get_classif_str(self, label):
        return str(label[0]) + "_" + str(label[1]) + "_" + str(label[2])

    def _pre_process_dataset(self):

        # get all the subfolders inside the dataset folder
        data_folder_variations = self.root_dir.glob("*")

        # for each variation
        for folder in data_folder_variations:

            # for each split
            for pre_processed_split in ["pp_train_64", "pp_test_64"]:

                # create the path
                path_to_preprocessed_split = folder / pre_processed_split
                path_to_vanilla_split = folder / pre_processed_split.split("_")[1]

                # if no pre-processing for these settings exists, then create them
                if not path_to_preprocessed_split.exists():

                    print("-" * 37)
                    print(f"Pre-process and save data for folder: {folder} and split: {pre_processed_split} and downscale size: 64 ...")
                    
                    self.pre_process_and_save_data(path_to_preprocessed_split, path_to_vanilla_split)
                    
                    print("Pre-processing and saving finished.")
                    print("-" * 37)

    def pre_process_and_save_data(self, path_to_preprocessed_split, path_to_vanilla_split):
        """
        To speed up training, it is beneficial to do the rudementary pre-processing once and save the data.
        """

        # create the folders to save the pre-processed data
        path_to_preprocessed_split.mkdir()

        # get all the files in all the subfolders 
        files = list(path_to_vanilla_split.glob(f"**/*.png"))

        # for each image 
        for curr_file in files:

            # open the image specified by the path
            img = Image.open(curr_file).convert("L")

            # center crop the image using the smaller size (i.e. width or height)
            # to define the new size of the image (basically we remove only either the width or height)
            img = TF.center_crop(img, np.min(img.size))

            # then resize the image to the one we want to use for training
            img = TF.resize(img, 64)

            # create the folder for the experiment
            save_folder = path_to_preprocessed_split / curr_file.parent.stem
            save_folder.mkdir(exist_ok=True)

            # save the processed image
            img.save(save_folder / curr_file.name)

    def _get_positive(self, rand_indices, positive_label, positive_images):

        # get all the potential candidates which have the same label 
        masked = [idx for idx, x in enumerate(self.labels) if x==positive_label] 

        # if there is no other image with the same label
        if not masked:
            
            new_rand_indices = random.sample(range(0,len(positive_images)), 2)
            positive_input_image = positive_images[new_rand_indices[0]]
            positive_output_image = positive_images[new_rand_indices[1]] if self.make_scene_impossible else positive_images[new_rand_indices[0]]
            positive_input_image = TF.to_tensor(positive_input_image)
            positive_output_image = TF.to_tensor(positive_output_image)

        else:
            # choose one index randomly from the masked subset
            index = np.random.choice(masked)

            positive_input_image = self.images[index][rand_indices[0]]
            positive_output_image = self.images[index][rand_indices[1]] if self.make_scene_impossible else self.images[index][rand_indices[0]]
            positive_input_image = TF.to_tensor(positive_input_image)
            positive_output_image = TF.to_tensor(positive_output_image)

        return positive_input_image, positive_output_image

    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        images = self.images[index]
        label = self.labels[index]
        str_label = self._get_classif_str(label)

        # randomly selected
        # .) the input images 
        # .) the output images 
        rand_indices = random.sample(range(0,len(images)), 2)

        # get the image to be used as input
        input_image = images[rand_indices[0]]

        # get the image to be used for the reconstruction error
        output_image = images[rand_indices[1]] if self.make_scene_impossible else images[rand_indices[0]]

        # make sure its a tensor
        input_image = TF.to_tensor(input_image)
        output_image = TF.to_tensor(output_image)

        if self.make_instance_impossible:
            _, output_image = self._get_positive(rand_indices, label, images)

        # augment image if necessary (we need 0-channel input, not 1-channel input)
        if self.augment: input_image = torch.from_numpy(self.augment(image=np.array(input_image)[0])["image"][None,:])

        return {"image":input_image, "target":output_image, "gt": self.label_str_to_int[str_label]}
            
####################################################################################################################################################

class SVIRO(BaseDatasetCar):
    """
    https://sviro.kl.dfki.de

    You only need the grayscale images for the whole scene.
    Make sure to have a folder structure as follows:
    
    SVIRO 
    ├── aclass
    │   ├── train
    │   │   └──── grayscale_wholeImage
    │   └── test
    │       └──── grayscale_wholeImage
    ⋮
    ⋮
    ⋮
    └── zoe
        ├── train
        │   └──── grayscale_wholeImage
        └── test
            └──── grayscale_wholeImage
    """
    def __init__(self, car, which_split, make_instance_impossible, augment):

        # path to the main folder
        root_dir = ROOT_DATA_DIR / "SVIRO"  

        # call the init function of the parent class
        super().__init__(root_dir=root_dir, car=car, split=which_split, make_scene_impossible=False, make_instance_impossible=make_instance_impossible, augment=augment)
        
    def _get_data(self):

        # get all the png files, i.e. experiments
        if self.car.lower() == "all":
            self.files = sorted(list(self.root_dir.glob(f"*/{self.split}/grayscale_wholeImage_pp_640_64/*.png")))
        else:
            self.files = sorted(list(self.root_dir.glob(f"{self.car}/{self.split}/grayscale_wholeImage_pp_640_64/*.png")))

        # placeholder for all images and labels
        self.images = []
        self.labels = []

        # for each file
        for file in self.files:
        
            # get classification labels for each seat from folder name
            classif_labels = self._get_classif_label(file)

            # do not get child seats or everyday objects
            if 1 in classif_labels or 2 in classif_labels or 4 in classif_labels or 5 in classif_labels or 6 in classif_labels:
                continue

            # open the image specified by the path
            # make sure it is a grayscale image
            img = np.array(Image.open(file).convert("L"))

            # each scene will be an array of images
            # append the image to the placeholder
            self.images.append([img])
        
            # append label to placeholder
            self.labels.append(classif_labels)

    def _get_classif_label(self, file_path):

        # get the filename only of the path
        name = file_path.stem

        # split at GT 
        gts = name.split("GT")[-1]
        
        # split the elements at _
        # first element is empty string, remove it
        clean_gts = gts.split("_")[1:]

        # convert the strings to ints
        clean_gts = [int(x) for x in clean_gts]

        return clean_gts

    def _pre_process_dataset(self):

        # get all the subfolders inside the dataset folder
        data_folder_variations = self.root_dir.glob("*/*")

        # for each variation
        for folder in data_folder_variations:

            # create the path
            path_to_preprocessed_split = folder / "grayscale_wholeImage_pp_640_64"
            path_to_vanilla_split = folder / "grayscale_wholeImage"

            # if no pre-processing for these settings exists, then create them
            if not path_to_preprocessed_split.exists():

                print("-" * 37)
                print(f"Pre-process and save data for folder: {folder} and downscale size: 64 ...")
                
                self.pre_process_and_save_data(path_to_preprocessed_split, path_to_vanilla_split)
                
                print("Pre-processing and saving finished.")
                print("-" * 37)

    def pre_process_and_save_data(self, path_to_preprocessed_split, path_to_vanilla_split):
        """
        To speed up training, it is beneficial to do the rudementary pre-processing once and save the data.
        """

        # create the folders to save the pre-processed data
        path_to_preprocessed_split.mkdir()

        # get all the files in all the subfolders 
        files = list(path_to_vanilla_split.glob("*.png"))

        # for each image 
        for curr_file in files:

            # open the image specified by the path
            img = Image.open(curr_file).convert("L")

            # center crop the image using the smaller size (i.e. width or height)
            # to define the new size of the image (basically we remove only either the width or height)
            img = TF.center_crop(img, np.min(img.size))

            # then resize the image to the one we want to use for training
            img = TF.resize(img, 64)

            # create the path to the file
            save_path = path_to_preprocessed_split / curr_file.name

            # save the processed image
            img.save(save_path)

    def _get_positive(self, positive_label):

        # get all the potential candidates from the real images which have the same label as the synthetic one
        masked = [idx for idx, x in enumerate(self.labels) if x==positive_label] 

        # choose one index randomly from the masked subset
        index = np.random.choice(masked)

        input_image = self.images[index][0]
        input_image = TF.to_tensor(input_image)

        return input_image


    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        image = self.images[index][0]
        label = self.labels[index]
        str_label = self._get_classif_str(label)

        # transform it for pytorch (normalized and transposed)
        image = TF.to_tensor(image)
    
        if self.make_instance_impossible:
            output_image = self._get_positive(label)
        else:
            output_image = image.clone()

        # augment image if necessary (we need 0-channel input, not 1-channel input)
        if self.augment: image = torch.from_numpy(self.augment(image=np.array(image)[0])["image"][None,:])

        return {"image":image, "target":output_image, "gt": self.label_str_to_int[str_label]}

####################################################################################################################################################

class SVIROUncertainty(BaseDatasetCar):
    """
    https://sviro.kl.dfki.de
    
    Make sure to have a folder structure as follows:

    SVIRO-Illumination 
    └── sharan
        ├── train
        ├── test-adults
        ├── test-objects
        └── test-adults-and-objects

    """
    def __init__(self, car, which_split, make_instance_impossible, nbr_of_samples_per_class, augment):

        # path to the main folder
        root_dir = ROOT_DATA_DIR / "SVIRO-Uncertainty"

        # call the init function of the parent class
        super().__init__(root_dir=root_dir, car=car, split=which_split, make_scene_impossible=False, make_instance_impossible=make_instance_impossible, nbr_of_samples_per_class=nbr_of_samples_per_class, augment=augment)

    def _get_data(self):

        # get all the png files, i.e. experiments
        self.files = sorted(list(self.root_dir.glob(f"{self.car}/pp_{self.split}_64/*/ir.png")))

        # placeholder for all images and labels
        self.images = []
        self.labels = []

        # for each file
        for file in self.files:
        
            # get classification labels for each seat from folder name
            classif_labels = self._get_classif_label(file.parent)

            # open the image specified by the path
            # make sure it is a grayscale image
            img = np.array(Image.open(file).convert("L"))

            # each scene will be an array of images
            # append the image to the placeholder
            self.images.append([img])
        
            # append label to placeholder
            self.labels.append(classif_labels)

    def _pre_process_dataset(self):

        # get all the subfolders inside the dataset folder
        data_folder_variations = self.root_dir.glob("*")

        # for each variation
        for folder in data_folder_variations:

            # for each split
            for pre_processed_split in ["pp_train-adults_64", "pp_train-adults-and-seats_64", "pp_test-adults_64", "pp_test-objects_64", "pp_test-seats_64", "pp_test-adults-and-objects_64", "pp_test-adults-and-seats_64", "pp_test-adults-and-seats-and-objects_64"]:

                # create the path
                path_to_preprocessed_split = folder / pre_processed_split
                path_to_vanilla_split = folder / pre_processed_split.split("_")[1]

                # if no pre-processing for these settings exists, then create them
                if not path_to_preprocessed_split.exists():

                    print("-" * 37)
                    print(f"Pre-process and save data for folder: {folder} and split: {pre_processed_split} and downscale size: 64 ...")
                    
                    self.pre_process_and_save_data(path_to_preprocessed_split, path_to_vanilla_split)
                    
                    print("Pre-processing and saving finished.")
                    print("-" * 37)

    def pre_process_and_save_data(self, path_to_preprocessed_split, path_to_vanilla_split):
        """
        To speed up training, it is beneficial to do the rudementary pre-processing once and save the data.
        """

        # create the folders to save the pre-processed data
        path_to_preprocessed_split.mkdir()

        # get all the files in all the subfolders 
        files = list(path_to_vanilla_split.glob(f"**/ir.png")) + list(path_to_vanilla_split.glob(f"**/rgb.png"))

        # for each image 
        for curr_file in files:

            # open the image specified by the path
            img = Image.open(curr_file).convert("L") if "ir" in curr_file.name else Image.open(curr_file).convert("RGB")

            # center crop the image using the smaller size (i.e. width or height)
            # to define the new size of the image (basically we remove only either the width or height)
            img = TF.center_crop(img, np.min(img.size))

            # then resize the image to the one we want to use for training
            img = TF.resize(img, 64)

            # create the folder for the experiment
            save_folder = path_to_preprocessed_split / curr_file.parent.stem
            save_folder.mkdir(exist_ok=True)

            # save the processed image
            img.save(save_folder / curr_file.name)

    def _get_positive(self, positive_label):

        # get all the potential candidates from the real images which have the same label as the synthetic one
        masked = [idx for idx, x in enumerate(self.labels) if x==positive_label] 

        # choose one index randomly from the masked subset
        index = np.random.choice(masked)

        input_image = self.images[index][0]
        input_image = TF.to_tensor(input_image)

        return input_image

    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        image = self.images[index][0]
        label = self.labels[index]
        str_label = self._get_classif_str(label)

        # transform it for pytorch (normalized and transposed)
        image = TF.to_tensor(image)
    
        if self.make_instance_impossible:
            output_image = self._get_positive(label)
        else:
            output_image = image.clone()

        # augment image if necessary (we need 0-channel input, not 1-channel input)
        if self.augment: image = torch.from_numpy(self.augment(image=np.array(image)[0])["image"][None,:])

        return {"image":image, "target":output_image, "gt": self.label_str_to_int[str_label]}


####################################################################################################################################################

class Fashion(TFashionMNIST):

    # dict to transform integers to string labels 
    int_to_label_str = {x:str(x) for x in range(10)}

    def __init__(self, which_split, make_instance_impossible, nbr_of_samples_per_class, augment):

        # path to the main folder
        root_dir = Path(f"/data/local_data/workingdir_g02/sds/data/")

        # train or test split
        self.split = which_split
        self.is_train = True if self.split.lower() == "train" else False

        # normal or impossible reconstruction loss?
        self.make_instance_impossible = make_instance_impossible

        # number of samples to keep for each class
        self.nbr_of_samples_per_class = nbr_of_samples_per_class

        # call the init function of the parent class
        super().__init__(root=root_dir, train=self.is_train, download=False)

        # only get a subset of the data
        self._get_subset_of_data()

        # augmentations if needed
        if augment:
            self.augment = album.Compose(
                [
                    album.Blur(always_apply=False, p=0.4, blur_limit=7),
                    album.MultiplicativeNoise(always_apply=False, p=0.4, elementwise=True, multiplier=(0.75, 1.25)),
                ]
            )
        else:
            self.augment = False

    def _get_classif_str(self, label):
        return int(label)

    def _get_subset_of_data(self):

        self.images = self.data
        self.labels = self.targets

        # if we are training
        if self.nbr_of_samples_per_class > 0:

            # keep track of samples per class
            counter = defaultdict(int)
            # and the corresponding indices
            keep_indices = []

            # for each label
            for idx, label in enumerate(self.labels):
                
                # make sure its a string
                label =  self._get_classif_str(label)

                # increase the counter for this label
                counter[label] += 1
                
                # if we are above the theshold for this label
                if counter[label] >= (self.nbr_of_samples_per_class+1):
                    # then skip it
                    continue
                else:
                    # otherwise keep track of the label
                    keep_indices.append(idx)
            
            # only take the subset of indices based on how many samples per class to keep
            self.data = [x for idx, x in enumerate(self.images) if idx in keep_indices]
            self.targets = [x for idx, x in enumerate(self.labels) if idx in keep_indices]

    def _get_positive(self, positive_label):

        while True:
            index = random.randint(0, len(self.targets)-1)
            if int(self.targets[index]) == positive_label:
                image = self.data[index]
                image = Image.fromarray(image.numpy(), mode='L')
                image = TF.resize(image, [64, 64])
                image = TF.to_tensor(image)

                return image

    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        image = self.data[index]
        label = int(self.targets[index])

        # doing this so that it is consistent with all other datasets to return a PIL Image
        image = Image.fromarray(image.numpy(), mode='L')

        # transform it for pytorch (normalized and transposed)
        image = TF.resize(image, [64, 64])
        image = TF.to_tensor(image)

        if self.make_instance_impossible:
            output_image = self._get_positive(label)
        else:
            output_image = image.clone()

        # augment image if necessary (we need 0-channel input, not 1-channel input)
        if self.augment: image = torch.from_numpy(self.augment(image=np.array(image)[0])["image"][None,:])

        return {"image":image, "target":output_image, "gt":label}

####################################################################################################################################################

class MNIST(TMNIST):

    def __init__(self, which_split, make_instance_impossible, nbr_of_samples_per_class, augment):

        # path to the main folder
        root_dir = Path(f"/data/local_data/workingdir_g02/sds/data/MNIST")

        # train or test split, digits or letters
        self.split = which_split
        self.is_train = True if self.split.lower() == "train" else False

        # normal or impossible reconstruction loss?
        self.make_instance_impossible = make_instance_impossible

        # number of samples to keep for each class
        self.nbr_of_samples_per_class = nbr_of_samples_per_class

        # call the init function of the parent class
        super().__init__(root=root_dir, train=self.is_train, download=False)

        # only get a subset of the data
        self._get_subset_of_data()

        # dict to transform integers to string labels 
        self.int_to_label_str = {x:str(x) for x in range(10)}

        # augmentations if needed
        if augment:
            self.augment = album.Compose(
                [
                    album.Blur(always_apply=False, p=0.4, blur_limit=7),
                    album.MultiplicativeNoise(always_apply=False, p=0.4, elementwise=True, multiplier=(0.75, 1.25)),
                ]
            )
        else:
            self.augment = False

    def _get_classif_str(self, label):
        return int(label)

    def __len__(self):
        return len(self.images)

    def _get_subset_of_data(self):

        self.images = []
        self.labels = []

        # if we are training
        if self.nbr_of_samples_per_class > 0:

            # keep track of samples per class
            counter = defaultdict(int)
            # and the corresponding indices
            keep_indices = []

            # for each label
            for idx, label in enumerate(self.targets):
                
                # make sure its a string
                label =  self._get_classif_str(label)

                # increase the counter for this label
                counter[label] += 1
                
                # if we are above the theshold for this label
                if counter[label] >= (self.nbr_of_samples_per_class+1):
                    # then skip it
                    continue
                else:
                    # otherwise keep track of the label
                    keep_indices.append(idx)
        # testing
        else:
            keep_indices = [idx for idx, _ in enumerate(self.targets[0:10_000])]

        # only take the subset of indices based on how many samples per class to keep
        for idx in keep_indices:
                
            # get the image
            current_image = Image.fromarray(self.data[idx].numpy(), mode="L")

            # transform it for pytorch (normalized and transposed)
            current_image = TF.resize(current_image, [64, 64])
            current_image = TF.to_tensor(current_image)

            # get label
            current_label = self.targets[idx]

            # keep it
            self.images.append(current_image)
            self.labels.append(current_label)

        del self.targets
        del self.data
            
    def _get_positive(self, positive_label):

        while True:
            index = random.randint(0, len(self.labels)-1)
            if int(self.labels[index]) == positive_label:
                image = self.images[index]
                return image

    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        image = self.images[index]
        label = int(self.labels[index])

        if self.make_instance_impossible:
            output_image = self._get_positive(label)
        else:
            output_image = image.clone()

        # augment image if necessary (we need 0-channel input, not 1-channel input)
        if self.augment: image = torch.from_numpy(self.augment(image=np.array(image)[0])["image"][None,:])

        return {"image":image, "target":output_image, "gt":label}

####################################################################################################################################################

class GTSRB(Dataset):

    string_labels_to_integer_dict = dict()

    def __init__(self, which_split, make_instance_impossible, nbr_of_samples_per_class, augment):

        # train or test
        self.is_train = True if which_split.lower() == "train" else False

        if which_split.lower() == "train":
            self.folder = "train_png"
        elif which_split.lower() == "test":
            self.folder = "test_png"
        elif which_split.lower() == "ood":
            self.folder = "ood_png"
        else:
            raise ValueError

        # path to the main folder
        self.root_dir = Path(f"/data/local_data/workingdir_g02/sds/data/GTSRB") / self.folder

        # normal or impossible reconstruction loss?
        self.make_instance_impossible = make_instance_impossible

        # number of samples to keep for each class
        self.nbr_of_samples_per_class = nbr_of_samples_per_class

        # only get a subset of the data
        self._get_subset_of_data()

        # augmentations if needed
        if augment:
            self.augment = album.Compose(
                [
                    album.RandomBrightnessContrast(always_apply=False, p=0.4, brightness_limit=(0.0, 0.33), contrast_limit=(0.0, 0.33), brightness_by_max=False),
                    album.Blur(always_apply=False, p=0.4, blur_limit=7),
                    album.MultiplicativeNoise(always_apply=False, p=0.4, elementwise=True, multiplier=(0.75, 1.25)),
                ]
            )
        else:
            self.augment = False

    def _get_classif_str(self, label):
        return int(label)

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        # number of images to use
        return len(self.images)

    def _get_subset_of_data(self):

        self.all_images = list(self.root_dir.glob("*/*.png"))
        self.images = []
        self.labels = []

        # if we are training
        if self.nbr_of_samples_per_class > 0:

            # keep track of samples per class
            counter = defaultdict(int)
            # and the corresponding indices
            keep_indices = []

            # for each label
            for idx, img in enumerate(self.all_images):
                
                # get the label 
                label = self._get_label_from_path(img)

                # increase the counter for this label
                counter[label] += 1
                
                # if we are above the theshold for this label
                if counter[label] >= (self.nbr_of_samples_per_class+1):
                    # then skip it
                    continue
                else:
                    # otherwise keep track of the label
                    keep_indices.append(idx)
        # testing
        else:
            keep_indices = [idx for idx, _ in enumerate(self.all_images[0:10_000])]

        # only take the subset of indices based on how many samples per class to keep
        for idx in keep_indices:
                
                # get the image
                current_image = Image.open(self.all_images[idx]).convert("L")

                # transform it for pytorch (normalized and transposed)
                current_image = TF.resize(current_image, [64, 64])
                current_image = TF.to_tensor(current_image)

                # get label
                current_label = self._get_label_from_path(self.all_images[idx])

                # keep it
                self.images.append(current_image)
                self.labels.append(current_label)

    def _get_label_from_path(self, path):

        # get the name from the parent folder
        if self.folder == "ood":
            if int(path.parent.name) < 10:
                return int(path.parent.name)
            else:
                return int(path.parent.name)-10
        else:
            return int(path.parent.name)-10

    def _get_positive(self, positive_label):

        # get all the potential candidates from the real images which have the same label as the synthetic one
        masked = [idx for idx, x in enumerate(self.labels) if x==positive_label] 

        # choose one index randomly from the masked subset
        index = np.random.choice(masked)

        input_image = self.images[index]

        return input_image
        
    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        image = self.images[index]
        label = self.labels[index]

        if self.make_instance_impossible:
            output_image = self._get_positive(label)
        else:
            output_image = image.clone()

        # augment image if necessary (we need 0-channel input, not 1-channel input)
        if self.augment: image = torch.from_numpy(self.augment(image=np.array(image)[0])["image"][None,:])

        return {"image":image, "target":output_image, "gt":label}

####################################################################################################################################################

class CIFAR10(TCIFAR10):

    # dict to transform integers to string labels 
    int_to_label_str = {x:str(x) for x in range(10)}

    def __init__(self, which_split, make_instance_impossible, nbr_of_samples_per_class, augment):

        # path to the main folder
        root_dir = Path(f"/data/local_data/workingdir_g02/sds/data/CIFAR10")

        # train or test split
        self.split = which_split
        self.is_train = True if self.split.lower() == "train" else False

        # normal or impossible reconstruction loss?
        self.make_instance_impossible = make_instance_impossible

        # number of samples to keep for each class
        self.nbr_of_samples_per_class = nbr_of_samples_per_class

        # call the init function of the parent class
        super().__init__(root=root_dir, train=self.is_train, download=False)

        # only get a subset of the data
        self._get_subset_of_data()

        # augmentations if needed
        if augment:
            self.augment = album.Compose(
                [
                    album.RandomBrightnessContrast(always_apply=False, p=0.4, brightness_limit=(0.0, 0.33), contrast_limit=(0.0, 0.33), brightness_by_max=False),
                    album.Blur(always_apply=False, p=0.4, blur_limit=7),
                    album.MultiplicativeNoise(always_apply=False, p=0.4, elementwise=True, multiplier=(0.75, 1.25)),
                ]
            )
        else:
            self.augment = False

    def _get_classif_str(self, label):
        return int(label)

    def __len__(self):
        return len(self.images)

    def _get_subset_of_data(self):

        self.images = []
        self.labels = []

        # if we are training
        if self.nbr_of_samples_per_class > 0:

            # keep track of samples per class
            counter = defaultdict(int)
            # and the corresponding indices
            keep_indices = []

            # for each label
            for idx, label in enumerate(self.targets):
                
                # make sure its a string
                label =  self._get_classif_str(label)

                # increase the counter for this label
                counter[label] += 1
                
                # if we are above the theshold for this label
                if counter[label] >= (self.nbr_of_samples_per_class+1):
                    # then skip it
                    continue
                else:
                    # otherwise keep track of the label
                    keep_indices.append(idx)
        # testing
        else:
            keep_indices = [idx for idx, _ in enumerate(self.data[0:10_000])]

        # only take the subset of indices based on how many samples per class to keep
        for idx in keep_indices:
                
            # get the image
            current_image = Image.fromarray(self.data[idx]).convert("L")

            # transform it for pytorch (normalized and transposed)
            current_image = TF.resize(current_image, [64, 64])
            current_image = TF.to_tensor(current_image)

            # get label
            current_label = self.targets[idx]

            # keep it
            self.images.append(current_image)
            self.labels.append(current_label)

        del self.targets
        del self.data
            
    def _get_positive(self, positive_label):

        while True:
            index = random.randint(0, len(self.labels)-1)
            if int(self.labels[index]) == positive_label:
                image = self.images[index]
                return image

    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        image = self.images[index]
        label = int(self.labels[index])

        if self.make_instance_impossible:
            output_image = self._get_positive(label)
        else:
            output_image = image.clone()

        # augment image if necessary (we need 0-channel input, not 1-channel input)
        if self.augment: image = torch.from_numpy(self.augment(image=np.array(image)[0])["image"][None,:])

        return {"image":image, "target":output_image, "gt":label}

####################################################################################################################################################


class SVHN(TSVHN):

    # dict to transform integers to string labels 
    int_to_label_str = {x:str(x) for x in range(10)}

    def __init__(self, which_split, make_instance_impossible, nbr_of_samples_per_class, augment):

        # path to the main folder
        root_dir = Path(f"/data/local_data/workingdir_g02/sds/data/SVHN")

        # train or test split
        self.split = which_split
        self.is_train = True if self.split.lower() == "train" else False

        # normal or impossible reconstruction loss?
        self.make_instance_impossible = make_instance_impossible

        # number of samples to keep for each class
        self.nbr_of_samples_per_class = nbr_of_samples_per_class

        # call the init function of the parent class
        super().__init__(root=root_dir, split="train" if self.is_train else "test", download=False)

        # only get a subset of the data
        self._get_subset_of_data()

        # augmentations if needed
        if augment:
            self.augment = album.Compose(
                [
                    album.RandomBrightnessContrast(always_apply=False, p=0.4, brightness_limit=(0.0, 0.33), contrast_limit=(0.0, 0.33), brightness_by_max=False),
                    album.Blur(always_apply=False, p=0.4, blur_limit=7),
                    album.MultiplicativeNoise(always_apply=False, p=0.4, elementwise=True, multiplier=(0.75, 1.25)),
                ]
            )
        else:
            self.augment = False

    def _get_classif_str(self, label):
        return int(label)

    def __len__(self):
        return len(self.images)

    def _get_subset_of_data(self):

        self.targets = self.labels
        self.images = []
        self.labels = []

        # if we are training
        if self.nbr_of_samples_per_class > 0:

            # keep track of samples per class
            counter = defaultdict(int)
            # and the corresponding indices
            keep_indices = []

            # for each label
            for idx, label in enumerate(self.targets):
                
                # make sure its a string
                label =  self._get_classif_str(label)

                # increase the counter for this label
                counter[label] += 1
                
                # if we are above the theshold for this label
                if counter[label] >= (self.nbr_of_samples_per_class+1):
                    # then skip it
                    continue
                else:
                    # otherwise keep track of the label
                    keep_indices.append(idx)
        # testing
        else:
            keep_indices = [idx for idx, _ in enumerate(self.data[0:10_000])]

        # only take the subset of indices based on how many samples per class to keep
        for idx in keep_indices:
                
            # get the image
            current_image = Image.fromarray(np.transpose(self.data[idx], (1, 2, 0))).convert("L")

            # transform it for pytorch (normalized and transposed)
            current_image = TF.resize(current_image, [64, 64])
            current_image = TF.to_tensor(current_image)

            # get label
            current_label = self.targets[idx]

            # keep it
            self.images.append(current_image)
            self.labels.append(current_label)

        del self.targets
        del self.data
            
    def _get_positive(self, positive_label):

        while True:
            index = random.randint(0, len(self.labels)-1)
            if int(self.labels[index]) == positive_label:
                image = self.images[index]
                return image

    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        image = self.images[index]
        label = int(self.labels[index])

        if self.make_instance_impossible:
            output_image = self._get_positive(label)
        else:
            output_image = image.clone()

        # augment image if necessary (we need 0-channel input, not 1-channel input)
        if self.augment: image = torch.from_numpy(self.augment(image=np.array(image)[0])["image"][None,:])

        return {"image":image, "target":output_image, "gt":label}

####################################################################################################################################################

class Omniglot(TOmniglot):

    # dict to transform integers to string labels 
    int_to_label_str = {x:str(x) for x in range(10)}

    def __init__(self, which_split, make_instance_impossible, nbr_of_samples_per_class, augment):

        # path to the main folder
        root_dir = Path(f"/data/local_data/workingdir_g02/sds/data/Omniglot")

        # train or test split
        self.split = which_split
        self.is_train = True if self.split.lower() == "train" else False

        # normal or impossible reconstruction loss?
        self.make_instance_impossible = make_instance_impossible

        # number of samples to keep for each class
        self.nbr_of_samples_per_class = nbr_of_samples_per_class

        # call the init function of the parent class
        super().__init__(root=root_dir, background=self.is_train, download=False)

        # only get a subset of the data
        self._get_subset_of_data()

        # augmentations if needed
        if augment:
            self.augment = album.Compose(
                [
                    album.Blur(always_apply=False, p=0.4, blur_limit=7),
                    album.MultiplicativeNoise(always_apply=False, p=0.4, elementwise=True, multiplier=(0.75, 1.25)),
                ]
            )
        else:
            self.augment = False

    def _get_classif_str(self, label):
        return int(label)

    def __len__(self):
        return len(self.images)

    def _get_subset_of_data(self):

        self.images = []
        self.labels = []

        # if we are training
        if self.nbr_of_samples_per_class > 0:

            # keep track of samples per class
            counter = defaultdict(int)
            # and the corresponding indices
            keep_indices = []

            # for each label
            for idx, (_, character_class) in enumerate(self._flat_character_images):
                
                # increase the counter for this label
                counter[character_class] += 1
                
                # if we are above the theshold for this label
                if counter[character_class] >= (self.nbr_of_samples_per_class+1):
                    # then skip it
                    continue
                else:
                    # otherwise keep track of the label
                    keep_indices.append(idx)
        # testing
        else:
            keep_indices = [idx for idx, _ in enumerate(self._flat_character_images[0:10_000])]

        # only take the subset of indices based on how many samples per class to keep
        for idx in keep_indices:
                
            # get the image
            image_name, character_class = self._flat_character_images[idx]
            image_path = os.path.join(self.target_folder, self._characters[character_class], image_name)
            current_image = Image.open(image_path, mode='r').convert('L')

            # transform it for pytorch (normalized and transposed)
            current_image = TF.resize(current_image, [64, 64])
            current_image = TF.to_tensor(current_image)

            # keep it
            self.images.append(current_image)
            self.labels.append(character_class)

    def _get_positive(self, positive_label):

        while True:
            index = random.randint(0, len(self.labels)-1)
            if int(self.labels[index]) == positive_label:
                image = self.images[index]
                return image

    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        image = self.images[index]
        label = int(self.labels[index])

        if self.make_instance_impossible:
            output_image = self._get_positive(label)
        else:
            output_image = image.clone()

        # augment image if necessary (we need 0-channel input, not 1-channel input)
        if self.augment: image = torch.from_numpy(self.augment(image=np.array(image)[0])["image"][None,:])

        return {"image":image, "target":output_image, "gt":label}

####################################################################################################################################################

class Places365(TPlaces365):

    # dict to transform integers to string labels 
    int_to_label_str = {x:str(x) for x in range(10)}

    def __init__(self, which_split, make_instance_impossible, nbr_of_samples_per_class, augment):

        # path to the main folder
        root_dir = Path(f"/data/local_data/workingdir_g02/sds/data/Places365")

        # train or test split
        self.split = which_split
        self.is_train = True if self.split.lower() == "train" else False

        # normal or impossible reconstruction loss?
        self.make_instance_impossible = make_instance_impossible

        # number of samples to keep for each class
        self.nbr_of_samples_per_class = nbr_of_samples_per_class

        # call the init function of the parent class
        super().__init__(root=root_dir, split="train-standard" if self.is_train else "val", small=True, download=False)

        # only get a subset of the data
        self._get_subset_of_data()

        # augmentations if needed
        if augment:
            self.augment = album.Compose(
                [
                    album.Blur(always_apply=False, p=0.4, blur_limit=7),
                    album.MultiplicativeNoise(always_apply=False, p=0.4, elementwise=True, multiplier=(0.75, 1.25)),
                ]
            )
        else:
            self.augment = False

    def _get_classif_str(self, label):
        return int(label)

    def __len__(self):
        return len(self.images)

    def _get_subset_of_data(self):

        self.images = []
        self.labels = []

        # if we are training
        if self.nbr_of_samples_per_class > 0:

            # keep track of samples per class
            counter = defaultdict(int)
            # and the corresponding indices
            keep_indices = []

            # for each label
            for idx, (_, target) in enumerate(self.imgs):
                
                # increase the counter for this label
                counter[target] += 1
                
                # if we are above the theshold for this label
                if counter[target] >= (self.nbr_of_samples_per_class+1):
                    # then skip it
                    continue
                else:
                    # otherwise keep track of the label
                    keep_indices.append(idx)
        # testing
        else:
            keep_indices = [idx for idx, _ in enumerate(self.imgs[0:10_000])]

        # only take the subset of indices based on how many samples per class to keep
        for idx in keep_indices:
                
            # get the image
            file, target = self.imgs[idx]
            current_image = self.loader(file)

            # transform it for pytorch (normalized and transposed)
            current_image = TF.rgb_to_grayscale(current_image, num_output_channels=1)
            current_image = TF.resize(current_image, [64, 64])
            current_image = TF.to_tensor(current_image)

            # keep it
            self.images.append(current_image)
            self.labels.append(target)

    def _get_positive(self, positive_label):

        while True:
            index = random.randint(0, len(self.labels)-1)
            if int(self.labels[index]) == positive_label:
                image = self.images[index]
                return image

    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        image = self.images[index]
        label = int(self.labels[index])

        if self.make_instance_impossible:
            output_image = self._get_positive(label)
        else:
            output_image = image.clone()

        # augment image if necessary (we need 0-channel input, not 1-channel input)
        if self.augment: image = torch.from_numpy(self.augment(image=np.array(image)[0])["image"][None,:])

        return {"image":image, "target":output_image, "gt":label}

####################################################################################################################################################

class LSUN(TLSUN):

    # dict to transform integers to string labels 
    int_to_label_str = {x:str(x) for x in range(10)}

    def __init__(self, which_split, make_instance_impossible, nbr_of_samples_per_class, augment):

        # path to the main folder
        root_dir = Path(f"/data/local_data/workingdir_g02/sds/data/LSUN")

        # train or test split
        self.split = which_split
        self.is_train = True if self.split.lower() == "train" else False

        # normal or impossible reconstruction loss?
        self.make_instance_impossible = make_instance_impossible

        # number of samples to keep for each class
        self.nbr_of_samples_per_class = nbr_of_samples_per_class

        # call the init function of the parent class
        super().__init__(root=root_dir, classes="train" if self.is_train else "test")

        # only get a subset of the data
        self._get_subset_of_data()

        # augmentations if needed
        if augment:
            self.augment = album.Compose(
                [
                    album.Blur(always_apply=False, p=0.4, blur_limit=7),
                    album.MultiplicativeNoise(always_apply=False, p=0.4, elementwise=True, multiplier=(0.75, 1.25)),
                ]
            )
        else:
            self.augment = False

    def _get_classif_str(self, label):
        return int(label)

    def __len__(self):
        return len(self.images)

    def _get_subset_of_data(self):

        self.images = []
        self.labels = []

        # if we are training
        if self.nbr_of_samples_per_class > 0:

            # keep track of samples per class
            counter = defaultdict(int)
            # and the corresponding indices
            keep_indices = []

            # for each label
            for idx in range(self.length):

                target = 0
                for ind in self.indices:
                    if idx < ind:
                        break
                    target += 1
                
                # increase the counter for this label
                counter[target] += 1
                
                # if we are above the theshold for this label
                if counter[target] >= (self.nbr_of_samples_per_class+1):
                    # then skip it
                    continue
                else:
                    # otherwise keep track of the label
                    keep_indices.append(idx)
        # testing
        else:
            keep_indices = [idx for idx in range(10_000)]

        # only take the subset of indices based on how many samples per class to keep
        for idx in keep_indices:

            target = 0
            sub = 0
            for ind in self.indices:
                if idx < ind:
                    break
                target += 1
                sub = ind
                
            db = self.dbs[target]
            idx = idx - sub

            current_image, _ = db[idx]

            # transform it for pytorch (normalized and transposed)
            current_image = TF.rgb_to_grayscale(current_image, num_output_channels=1)
            current_image = TF.resize(current_image, [64, 64])
            current_image = TF.to_tensor(current_image)

            # keep it
            self.images.append(current_image)
            self.labels.append(target)

    def _get_positive(self, positive_label):

        while True:
            index = random.randint(0, len(self.labels)-1)
            if int(self.labels[index]) == positive_label:
                image = self.images[index]
                return image

    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        image = self.images[index]
        label = int(self.labels[index])

        if self.make_instance_impossible:
            output_image = self._get_positive(label)
        else:
            output_image = image.clone()

        # augment image if necessary (we need 0-channel input, not 1-channel input)
        if self.augment: image = torch.from_numpy(self.augment(image=np.array(image)[0])["image"][None,:])

        return {"image":image, "target":output_image, "gt":label}

####################################################################################################################################################

def print_dataset_statistics(dataset, which_dataset, which_split):

    # if a vehicle dataset
    if which_dataset.lower() in ["sviro", "sviro_uncertainty"]:

        # get the int label for all labels
        labels = np.array([dataset.label_str_to_int["_".join([str(y) for y in x])] for x in dataset.labels])
        int_to_label_str = dataset.int_to_label_str
    
    elif hasattr(dataset, "labels"):
        labels = np.array(dataset.labels)
        int_to_label_str = None

    elif hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
        int_to_label_str = None
    
    else:
        print("No targets or labels attribute.")
        return 

    unique_labels, labels_counts = np.unique(labels, return_counts=True)

    if int_to_label_str is None:
        int_to_label_str = {x:str(x) for x in unique_labels}

    print("=" * 37)
    print("Dataset used: \t", dataset)
    print("Split: \t\t", which_split)
    print("Samples: \t", len(dataset))
    print("-" * 37)

    # print the label and its number of occurences
    for label, count in zip(unique_labels, labels_counts):
        print(f"Label {int_to_label_str[label]}: {count}")

    print("=" * 37)

####################################################################################################################################################

def create_dataset(which_dataset, which_factor, which_split, make_scene_impossible=False, make_instance_impossible=False, augment=False, batch_size=64, shuffle=True, nbr_of_samples_per_class=-1, print_dataset=True):

    # create the dataset
    if which_dataset.lower() == "sviro":
        dataset = SVIRO(car=which_factor, which_split=which_split, make_instance_impossible=make_instance_impossible, augment=augment)
    elif which_dataset.lower() == "sviro_uncertainty":
        dataset = SVIROUncertainty(car=which_factor, which_split=which_split, make_instance_impossible=make_instance_impossible, nbr_of_samples_per_class=nbr_of_samples_per_class, augment=augment)
    elif which_dataset.lower() == "fashion":
        dataset = Fashion(which_split=which_split, make_instance_impossible=make_instance_impossible, nbr_of_samples_per_class=nbr_of_samples_per_class, augment=augment)
    elif which_dataset.lower() == "mnist":
        dataset = MNIST(which_split=which_split, make_instance_impossible=make_instance_impossible, nbr_of_samples_per_class=nbr_of_samples_per_class, augment=augment)
    elif which_dataset.lower() == "gtsrb":
        dataset = GTSRB(which_split=which_split, make_instance_impossible=make_instance_impossible, nbr_of_samples_per_class=nbr_of_samples_per_class, augment=augment)
    elif which_dataset.lower() == "cifar10":
        dataset = CIFAR10(which_split=which_split, make_instance_impossible=make_instance_impossible, nbr_of_samples_per_class=nbr_of_samples_per_class, augment=augment)
    elif which_dataset.lower() == "svhn":
        dataset = SVHN(which_split=which_split, make_instance_impossible=make_instance_impossible, nbr_of_samples_per_class=nbr_of_samples_per_class, augment=augment)
    elif which_dataset.lower() == "omniglot":
        dataset = Omniglot(which_split=which_split, make_instance_impossible=make_instance_impossible, nbr_of_samples_per_class=nbr_of_samples_per_class, augment=augment)
    elif which_dataset.lower() == "places365":
        dataset = Places365(which_split=which_split, make_instance_impossible=make_instance_impossible, nbr_of_samples_per_class=nbr_of_samples_per_class, augment=augment)
    elif which_dataset.lower() == "lsun":
        dataset = LSUN(which_split=which_split, make_instance_impossible=make_instance_impossible, nbr_of_samples_per_class=nbr_of_samples_per_class, augment=augment)
    else:
        raise ValueError

    if len(dataset) == 0:
        raise ValueError("The length of the dataset is zero. There is probably a problem with the folder structure for the dataset you want to consider. Have you downloaded the dataset and used the correct folder name and folder tree structure?")

    # for reproducibility
    # https://pytorch.org/docs/1.9.0/notes/randomness.html?highlight=reproducibility
    g = torch.Generator()
    g.manual_seed(0)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # create loader for the defined dataset
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=4, 
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    if print_dataset:
        print_dataset_statistics(dataset, which_dataset, which_split)

    return train_loader

####################################################################################################################################################