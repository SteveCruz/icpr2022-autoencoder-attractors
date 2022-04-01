##############################################################################################################################################################
##############################################################################################################################################################
"""
Autoencoder model definitions are located inside this script.
"""
##############################################################################################################################################################
##############################################################################################################################################################

import math
import torch
from torch import nn
from pytorch_msssim import SSIM, MS_SSIM
from pytorch_metric_learning import losses
import lpips
import torchvision

##############################################################################################################################################################
##############################################################################################################################################################

def get_activation_function(model_config):

    # from the torch.nn module, get the activation function specified in the config. 
    # make sure the naming is correctly spelled according to the torch name
    if hasattr(nn, model_config["activation"]):
        activation = getattr(nn, model_config["activation"])
        return activation()
    else:
        raise ValueError("Activation function does not exist.")

##############################################################################################################################################################

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        # As an example, we print the shape here.
        print(x.shape)
        return x

##############################################################################################################################################################

def create_ae_model(config, print_model=True):

    # define the autoencoder model
    if config["model"]["type"] == "dropout-cae":
        return DropoutCAE(config, print_model=print_model)
    elif config["model"]["type"] == "conv":
        return Conv(config, print_model=print_model)

##############################################################################################################################################################

class BaseAE(nn.Module):
    def __init__(self, config, print_model=True):
        super().__init__()

        # for ease of use, get the different configs
        self.model_config, self.data_config, self.training_config = config["model"], config["dataset"], config["training"]

        # some definitions
        self.type =  self.model_config["type"]
        self.nbr_input_channels = 1 
        self.pixel_height = 64
        self.pixel_width = 64

        # latent space dimension
        self.latent_dim = self.model_config["dimension"]

        # width of all layers
        # scale with dropout rate for a fair comparison
        if "width" in self.model_config:
            if "dropout" in self.model_config:
                self.dropout_rate = self.model_config["dropout"] 
                self.layer_width = int(self.model_config["width"] * (1+self.dropout_rate))
            else:
                self.layer_width = self.model_config["width"]

        # define the activation function to use
        self.activation = get_activation_function(self.model_config)      

        # define the loss function to use
        self.which_recon_loss = self.training_config["loss"]
        self.which_metric_loss = self.model_config["metric"]
        self._init_loss(print_model)

    def _define_decoder(self):

        decoder = nn.Sequential()
        decoder.add_module("fc_1", nn.Linear(in_features=self.latent_dim, out_features=self.layer_width, bias=True))
        decoder.add_module("activation_1", self.activation)
        decoder.add_module("fc_2", nn.Linear(in_features=self.layer_width, out_features=self.layer_width, bias=True))
        decoder.add_module("activation_2", self.activation)
        decoder.add_module("fc_3", nn.Linear(in_features=self.layer_width, out_features=self.layer_width, bias=True))
        decoder.add_module("activation_3", self.activation)
        decoder.add_module("fc_4", nn.Linear(in_features=self.layer_width, out_features=self.pixel_height*self.pixel_width, bias=True))

        return decoder

    def print_model(self):

        print("=" * 57)
        print("The autoencoder is defined as: ")
        print("=" * 57)
        print(self)
        print("=" * 57)
        print("Parameters of the model to learn:")
        print("=" * 57)
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                print(name)
        print('=' * 57)

    def _init_recon_loss(self):

        # define the loss function to use
        if self.which_recon_loss == "MSE":
            self.criterion = nn.MSELoss(reduction="sum")
        elif self.which_recon_loss == "L1":     
            self.criterion = nn.L1Loss(reduction="sum")
        elif self.which_recon_loss == "BCE":     
            self.criterion = nn.BCEWithLogitsLoss(reduction="sum")
        elif self.which_recon_loss == "Huber":     
            self.criterion = nn.HuberLoss(reduction="sum", delta=1.0)
        elif self.which_recon_loss == "SSIM":     
            self.criterion = SSIM(data_range=1.0, size_average=True, channel=self.nbr_input_channels)
        elif self.which_recon_loss == "MSSSIM":     
            self.criterion = MS_SSIM(data_range=1.0, size_average=True, channel=self.nbr_input_channels, win_size=7)
        elif self.which_recon_loss == "Perceptual":     
            self.criterion = lpips.LPIPS(net='vgg', lpips=False)
            self.criterion_preprocess = torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        else:
            raise ValueError("Loss definition does not exist.")

    def _init_metric_loss(self):

        # init metric  loss
        if self.which_metric_loss == "":
            self._metric_loss_fn = None
        elif self.which_metric_loss == "triplet":
            self._metric_loss_fn = losses.TripletMarginLoss(margin=0.2, swap=True)
        elif self.which_metric_loss == "angular":
            self._metric_loss_fn = losses.AngularLoss()
        elif self.which_metric_loss == "npair":
            self._metric_loss_fn = losses.NPairsLoss()
        elif self.which_metric_loss == "ntxent":
            self._metric_loss_fn = losses.NTXentLoss()
        elif self.which_metric_loss == "contrastive":
            self._metric_loss_fn = losses.ContrastiveLoss()
        else:
            raise ValueError("Metric function does not exist.")

    def _init_loss(self, print_model):

        # define both losses
        self._init_recon_loss()
        self._init_metric_loss()

        if print_model:
            print("Loss function used: ", self.criterion)
            print("Metric function used: ", self._metric_loss_fn)
            print('=' * 57)

    def metric_loss(self, embedding, labels) : 
        return self._metric_loss_fn(embedding, labels)

    def loss(self, prediction, target):

        if self.which_recon_loss in ["SSIM", "MSSSIM"]:
            return 1-self.criterion(prediction, target)
        elif self.which_recon_loss == "BCE":   
            # for BCE its better not to mean over batch dimension
            return self.criterion(prediction, target) 
        elif self.which_recon_loss == "Perceptual":   
            if prediction.shape[1] != 3:
                prediction = prediction.repeat(1, 3, 1, 1)
                target = target.repeat(1, 3, 1, 1)
            prediction = self.criterion_preprocess(prediction)
            target = self.criterion_preprocess(target)
            return self.criterion.forward(prediction, target).mean()
        else:
            # sum over the pixel dimension and divide over batch dimension
            # this is better than the mean over all pixels
            return self.criterion(prediction, target) / target.shape[0]

    def encode(self, x):
        return self.encoder(x)

    def decode(self, mu):
        return self.decoder(mu)

    def forward(self, x):

        # do the reshaping outside the model definition
        # otherwise the jacobian for the reshape layers does not work
        x = x.reshape(-1, self.pixel_height*self.pixel_width*self.nbr_input_channels)
        mu = self.encode(x)
        z = self.decode(mu)
        z = z.reshape(-1, self.nbr_input_channels, self.pixel_height, self.pixel_width)

        return {"xhat":z, "mu": mu}

##############################################################################################################################################################

class DropoutCAE(BaseAE):
    def __init__(self, config, print_model=True):
        # call the init function of the parent class
        super().__init__(config, print_model)

        # get the dropout rata
        self.dropout_rate = self.model_config["dropout"] 

        # increase the number of channels and fc neurons based on the dropout rate
        # we need to round up because pytorch seems to round up as well
        self.base_number_chnanels = math.ceil(32 * (1+self.dropout_rate))
        self.channels_before_fc = math.ceil(64 * (1+self.dropout_rate))
        self.fc_features = math.ceil(256 * (1+self.dropout_rate))
        
        self.reshape_channels = 4
        self.dimension_before_fc = 2 * self.base_number_chnanels * self.reshape_channels * self.reshape_channels

        # encoder
        self.encoder_first = nn.Sequential()
        self.encoder_first.add_module("conv_1", nn.Conv2d(in_channels=self.nbr_input_channels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.encoder_first.add_module("activation_1", self.activation)

        self.encoder_second = nn.Sequential()
        self.encoder_second.add_module("conv_2", nn.Conv2d(in_channels=self.base_number_chnanels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.encoder_second.add_module("activation_2", self.activation)

        self.encoder_third = nn.Sequential()
        self.encoder_third.add_module("conv_3", nn.Conv2d(in_channels=self.base_number_chnanels, out_channels=2*self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.encoder_third.add_module("activation_3", self.activation)

        self.encoder_fourth = nn.Sequential()
        self.encoder_fourth.add_module("conv_4", nn.Conv2d(in_channels=2*self.base_number_chnanels, out_channels=2*self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.encoder_fourth.add_module("activation_4", self.activation)

        self.encoder_fifth = nn.Sequential()
        self.encoder_fifth.add_module("fc_1", nn.Linear(in_features=self.dimension_before_fc, out_features=self.fc_features, bias=True))
        self.encoder_fifth.add_module("activation_5", self.activation)

        self.encoder_final = nn.Linear(in_features=self.fc_features, out_features=self.latent_dim, bias=True)

        self.flatten = nn.Flatten()

        # decoder
        self.decoder_first = nn.Sequential()
        self.decoder_first.add_module("fc_1", nn.Linear(in_features=self.latent_dim, out_features=self.fc_features, bias=True))
        self.decoder_first.add_module("activation_1", self.activation)

        self.decoder_second = nn.Sequential()
        self.decoder_second.add_module("fc_2", nn.Linear(in_features=self.fc_features, out_features=self.dimension_before_fc, bias=True))
        self.decoder_second.add_module("activation_2", self.activation)

        self.decoder_third = nn.Sequential()
        self.decoder_third.add_module("conv_trans_1", nn.ConvTranspose2d(in_channels=2*self.base_number_chnanels, out_channels=2*self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.decoder_third.add_module("activation_3", self.activation)

        self.decoder_fourth = nn.Sequential()
        self.decoder_fourth.add_module("conv_trans_2", nn.ConvTranspose2d(in_channels=2*self.base_number_chnanels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.decoder_fourth.add_module("activation_4", self.activation)

        self.decoder_fifth = nn.Sequential()
        self.decoder_fifth.add_module("conv_trans_3", nn.ConvTranspose2d(in_channels=self.base_number_chnanels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.decoder_fifth.add_module("activation_5", self.activation)

        self.decoder_final = nn.ConvTranspose2d(in_channels=self.base_number_chnanels, out_channels=self.nbr_input_channels, kernel_size=4, padding=1, stride=2)

        self.unflatten = nn.Unflatten(1, (2 * self.base_number_chnanels, self.reshape_channels, self.reshape_channels))


    def _get_binary_mask(self, shape, device):
        # https://discuss.pytorch.org/t/how-to-fix-the-dropout-mask-for-different-batch/7119
        # Please note that the Bernoulli distribution samples 0 with the probability (1-p), 
        # contrary to dropout implementations, which sample 0 with probability p.
        if len(shape) == 4:
            return torch.bernoulli(torch.full([shape[0], shape[1], shape[2], shape[3]], 1-self.dropout_rate, device=device, requires_grad=False))/(1-self.dropout_rate)
        else:
            return torch.bernoulli(torch.full([shape[0], shape[1]], 1-self.dropout_rate, device=device, requires_grad=False))/(1-self.dropout_rate)

    def _define_and_fix_new_dropout_mask(self, x):

        self.encoder_first_mask = self._get_binary_mask([x.shape[0], self.base_number_chnanels, 8*self.reshape_channels, 8*self.reshape_channels], device=x.device)
        self.encoder_second_mask = self._get_binary_mask([x.shape[0], self.base_number_chnanels, 4*self.reshape_channels, 4*self.reshape_channels], device=x.device)
        self.encoder_third_mask = self._get_binary_mask([x.shape[0], 2*self.base_number_chnanels, 2*self.reshape_channels, 2*self.reshape_channels], device=x.device)
        self.encoder_fourth_mask = self._get_binary_mask([x.shape[0], 2*self.base_number_chnanels, self.reshape_channels, self.reshape_channels], device=x.device)
        self.encoder_fifth_mask = self._get_binary_mask([x.shape[0], self.fc_features], device=x.device)

        self.decoder_first_mask = self._get_binary_mask([x.shape[0], self.fc_features], device=x.device)
        self.decoder_second_mask = self._get_binary_mask([x.shape[0], self.dimension_before_fc], device=x.device)
        self.decoder_third_mask = self._get_binary_mask([x.shape[0], 2*self.base_number_chnanels, 2*self.reshape_channels, 2*self.reshape_channels], device=x.device)
        self.decoder_fourth_mask = self._get_binary_mask([x.shape[0], self.base_number_chnanels, 4*self.reshape_channels, 4*self.reshape_channels], device=x.device)
        self.decoder_fifth_mask = self._get_binary_mask([x.shape[0], self.base_number_chnanels, 8*self.reshape_channels, 8*self.reshape_channels], device=x.device)

    def encoding(self, x, random):

        x = self.encoder_first(x)
        if random:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=True)
        else:
            x = x * self.encoder_first_mask 

        x = self.encoder_second(x)
        if random:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=True)
        else:
            x = x * self.encoder_second_mask 
            
        x = self.encoder_third(x)
        if random:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=True)
        else:
            x = x * self.encoder_third_mask 

        x = self.encoder_fourth(x)
        if random:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=True)
        else:
            x = x * self.encoder_fourth_mask 

        x = self.flatten(x)

        x = self.encoder_fifth(x)
        if random:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=True)
        else:
            x = x * self.encoder_fifth_mask 
            
        x = self.encoder_final(x)

        return x

    def decoding(self, x, random):

        x = self.decoder_first(x)
        if random:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=True)
        else:
            x = x * self.decoder_first_mask 

        x = self.decoder_second(x)
        if random:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=True)
        else:
            x = x * self.decoder_second_mask 

        x = self.unflatten(x)

        x = self.decoder_third(x)
        if random:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=True)
        else:
            x = x * self.decoder_third_mask 

        x = self.decoder_fourth(x)
        if random:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=True)
        else:
            x = x * self.decoder_fourth_mask 

        x = self.decoder_fifth(x)
        if random:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=True)
        else:
            x = x * self.decoder_fifth_mask 
        
        x = self.decoder_final(x)

        return x

    def forward(self, x, random=True):

        mu = self.encoding(x, random)
        xhat = self.decoding(mu, random)

        return {"xhat":xhat, "mu": mu}

##############################################################################################################################################################

class Conv(nn.Module):
    def __init__(self, config, print_model=True):
        super().__init__()

        # for ease of use, get the different configs
        self.model_config, self.data_config, self.training_config = config["model"], config["dataset"], config["training"]

        # define the number of classes
        if "sviro" in config["dataset"]["name"].lower():
            self.nbr_classes = 8
        elif config["dataset"]["name"].lower() == "gtsrb":
            self.nbr_classes = 10
        elif config["dataset"]["name"].lower() == "mnist":
            self.nbr_classes = 10
        elif config["dataset"]["name"].lower() == "fashion":
            self.nbr_classes = 10
        elif config["dataset"]["name"].lower() == "svhn":
            self.nbr_classes = 10

        # some definitions
        self.type =  self.model_config["type"]
        self.dropout_rate = self.model_config["dropout"]
        self.nbr_input_channels = 1 
        self.base_number_chnanels = math.ceil(32 * (1+self.dropout_rate))
        self.fc_features = math.ceil(256 * (1+self.dropout_rate))
        self.reshape_channels = 4
        self.dimension_before_fc = 2 * self.base_number_chnanels * self.reshape_channels * self.reshape_channels

        # define the activation function to use
        self.activation = get_activation_function(self.model_config)      

        # loss function
        self.criterion = nn.CrossEntropyLoss()

        # define the model
        self.classifier = nn.Sequential()
        self.classifier.add_module("conv_1", nn.Conv2d(in_channels=self.nbr_input_channels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.classifier.add_module("activation_1", self.activation)
        self.classifier.add_module("dropout_1", nn.Dropout(p=self.dropout_rate))
        self.classifier.add_module("conv_2", nn.Conv2d(in_channels=self.base_number_chnanels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.classifier.add_module("activation_2", self.activation)
        self.classifier.add_module("dropout_2", nn.Dropout(p=self.dropout_rate))
        self.classifier.add_module("conv_3", nn.Conv2d(in_channels=self.base_number_chnanels, out_channels=2*self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.classifier.add_module("activation_3", self.activation)
        self.classifier.add_module("dropout_3", nn.Dropout(p=self.dropout_rate))
        self.classifier.add_module("conv_4", nn.Conv2d(in_channels=2*self.base_number_chnanels, out_channels=2*self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.classifier.add_module("activation_4", self.activation)
        self.classifier.add_module("dropout_4", nn.Dropout(p=self.dropout_rate))
        self.classifier.add_module("flatten", nn.Flatten())
        self.classifier.add_module("fc_1", nn.Linear(in_features=self.dimension_before_fc, out_features=self.fc_features, bias=True))
        self.classifier.add_module("activation_5", self.activation)
        self.classifier.add_module("fc_2", nn.Linear(in_features=self.fc_features, out_features=self.nbr_classes, bias=True))

    def _enable_dropout(self):
        if self.dropout_rate == 0:
            raise ValueError("Enabling dropout does not make any sense.")
        for each_module in self.modules():
            if each_module.__class__.__name__.startswith('Dropout'):
                each_module.train()

    def print_model(self):

        print("=" * 57)
        print("The autoencoder is defined as: ")
        print("=" * 57)
        print(self)
        print("=" * 57)
        print("Parameters of the model to learn:")
        print("=" * 57)
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                print(name)
        print('=' * 57)


    def loss(self, prediction, target):
        return self.criterion(prediction, target)

    def forward(self, x):

        # do the reshaping outside the model definition
        # otherwise the jacobian for the reshape layers does not work
        x = self.classifier(x)

        return {"prediction":x}

##############################################################################################################################################################
