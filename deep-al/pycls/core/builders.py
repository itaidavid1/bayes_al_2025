# This file is modified from official pycls repository

"""Model and loss construction functions."""

from pycls.core.net import SoftCrossEntropyLoss
from pycls.models.resnet import *
from pycls.models.vgg import *
from pycls.models.alexnet import *
from sklearn.neural_network import MLPClassifier

import torch
from torch import nn
from torch.nn import functional as F
import sys
import os

# Import the sklearn-compatible PyTorch MLP if available
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../tools'))
    from comapre_archs import MLPClassifier as MLPClassifierSklearnCompatible
except ImportError:
    MLPClassifierSklearnCompatible = None

# Supported models
_models = {
    # VGG style architectures
    'vgg11': vgg11,
    'vgg11_bn': vgg11_bn,
    'vgg13': vgg13,
    'vgg13_bn': vgg13_bn,
    'vgg16': vgg16,
    'vgg16_bn': vgg16_bn,
    'vgg19': vgg19,
    'vgg19_bn': vgg19_bn,

    # ResNet style archiectures
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'wide_resnet50_2': wide_resnet50_2,
    'wide_resnet101_2': wide_resnet101_2,

    # AlexNet architecture
    'alexnet': alexnet
}

# Supported loss functions
_loss_funs = {"cross_entropy": SoftCrossEntropyLoss}


class FeaturesNet(nn.Module):
    def __init__(self, in_layers, out_layers, use_mlp=False, penultimate_active=False):
        super().__init__()
        self.use_mlp = use_mlp
        self.penultimate_active = penultimate_active
        self.lin1 = nn.Linear(in_layers, in_layers)
        self.lin2 = nn.Linear(in_layers, in_layers)
        self.final = nn.Linear(in_layers, out_layers)

    def forward(self, x):
        feats = x
        if self.use_mlp:
            x = F.relu(self.lin1(x))
            x = F.relu((self.lin2(x)))
        out = self.final(x)
        if self.penultimate_active:
            return feats, out
        return out

class NNNet(nn.Module):
    def __init__(self, num_classes, device="cuda"):
        super().__init__()
        self.device = device
        self.num_classes = num_classes

    def compute_norm(self, x1, x2, batch_size=512):
        x1, x2 = x1.unsqueeze(0).to(self.device), x2.unsqueeze(0).to(self.device) # 1 x n x d, 1 x n' x d
        dist_matrix = []
        batch_round = x2.shape[1] // batch_size + int(x2.shape[1] % batch_size > 0)
        for i in range(batch_round):
            # distance comparisons are done in batches to reduce memory consumption
            x2_subset = x2[:, i * batch_size: (i + 1) * batch_size]
            dist = torch.cdist(x1, x2_subset, p=2.0)
            dist_matrix.append(dist.cpu())

        dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
        return dist_matrix

    def forward(self, x, y, x_test, return_logits=False):
        x, x_test = F.normalize(x, dim=1), F.normalize(x_test, dim=1)

        dist_matrix = self.compute_norm(x_test, x)
        if return_logits:
            try:
                topk = torch.topk(-dist_matrix, k=self.num_classes, largest=True, dim=1) # N_t x N
            except:
                print('RuntimeError: selected index k out of range')
            preds = topk.values
        else:
            nn_indices = torch.argmin(dist_matrix, dim=1)
            nn_labels = y[nn_indices]
            preds = F.one_hot(nn_labels, num_classes=self.num_classes)
            # preds = nn_labels

        output_dict = {'preds': preds}
        return output_dict


class MLPClassifierTorch(nn.Module):
    """
    PyTorch MLP for classification matching sklearn's MLPClassifier behavior.
    
    This is a simplified version that can be used as an nn.Module for training
    with custom loss functions and distillation.
    
    For full sklearn API compatibility, use the MLPClassifier from tools.comapre_archs.
    """
    def __init__(self, input_dim, hidden_dim, num_classes, activation='relu'):
        super(MLPClassifierTorch, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Build network matching sklearn's architecture
        layers = []
        # Support for tuple of hidden layer sizes (like sklearn)
        if isinstance(hidden_dim, (tuple, list)):
            in_size = input_dim
            for h in hidden_dim:
                layers.append(nn.Linear(in_size, h))
                layers.append(self._get_activation(activation))
                in_size = h
            layers.append(nn.Linear(in_size, num_classes))
        else:
            # Single hidden layer (default)
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Linear(hidden_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, activation):
        """Get activation function matching sklearn's options."""
        mapping = {
            'identity': nn.Identity(),
            'logistic': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
        }
        if activation not in mapping:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(mapping.keys())}.")
        return mapping[activation]

    def forward(self, x):
        return self.network(x)


class MLPClassifierDropout(nn.Module):
    """
    PyTorch MLP with Dropout for Monte Carlo uncertainty estimation.
    
    Architecture:
        Linear(embedding_dim, 256) -> ReLU -> Dropout(0.3) -> Linear(256, num_classes)
    
    During inference, dropout is kept active to enable MC dropout uncertainty estimation.
    """
    def __init__(self, input_dim, num_classes, dropout_p=0.3):
        super(MLPClassifierDropout, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        
        # Build MLP with dropout
        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.head(x)
    
    def mc_uncertainty(self, features, T=20):
        """
        Compute Monte Carlo uncertainty by running T forward passes with dropout active.
        
        Args:
            features: Input features (batch_size, input_dim)
            T: Number of forward passes (default: 20)
            
        Returns:
            uncertainty: Variance-based uncertainty scores (batch_size,)
        """
        self.train()  # Keep dropout active
        preds = torch.stack([
            F.softmax(self.head(features), dim=-1) 
            for _ in range(T)
        ])  # Shape: (T, batch_size, num_classes)
        
        # Compute variance across predictions as uncertainty
        uncertainty = preds.var(dim=0).mean(dim=-1)  # Shape: (batch_size,)
        return uncertainty

def get_model(cfg):
    """Gets the model class specified in the config."""
    err_str = "Model type '{}' not supported"
    assert cfg.MODEL.TYPE in _models.keys(), err_str.format(cfg.MODEL.TYPE)
    return _models[cfg.MODEL.TYPE]


def get_loss_fun(cfg):
    """Gets the loss function class specified in the config."""
    err_str = "Loss function type '{}' not supported"
    assert cfg.MODEL.LOSS_FUN in _loss_funs.keys(), err_str.format(cfg.TRAIN.LOSS)
    return _loss_funs[cfg.MODEL.LOSS_FUN]


def build_model(cfg):
    """Builds the model."""
    if cfg.MODEL.USE_1NN:
        # model = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
        # return model
        return NNNet(cfg.MODEL.NUM_CLASSES)

    elif cfg.MODEL.LINEAR_FROM_FEATURES:
        if cfg.EVAL_MODEL_TYPE == 'from_features':
            num_features = 384 if cfg.DATASET.NAME in ['IMAGENET50', 'IMAGENET100', 'IMAGENET200'] else 512
            num_features = 2 if cfg.DATASET.NAME in ['SCENARIO_A', 'HALF_MOON'] else num_features
            return FeaturesNet(num_features, cfg.MODEL.NUM_CLASSES).cuda()
        elif cfg.EVAL_MODEL_TYPE == 'from_mlp':
            # PyTorch MLP matching sklearn MLP configuration
            # Input dimension: feature dimension from the dataset
            input_dim = 384 if cfg.DATASET.NAME in ['IMAGENET50', 'IMAGENET100', 'IMAGENET200'] else 512
            input_dim = 2 if cfg.DATASET.NAME in ['SCENARIO_A', 'HALF_MOON'] else input_dim
            input_dim = 768 if "dino" in cfg.DATASET.NAME else input_dim
            input_dim = 2048 if cfg.DATASET.NAME == "TINYIMAGENET" else input_dim

            # Hidden layer size: matches sklearn's hidden_layer_sizes
            hidden_dim = 384 if cfg.DATASET.NAME in ['IMAGENET50', 'IMAGENET100', 'IMAGENET200'] else 256
            hidden_dim = 2 if cfg.DATASET.NAME in ['SCENARIO_A', 'HALF_MOON'] else hidden_dim
            hidden_dim = 384 if "dino" in cfg.DATASET.NAME else hidden_dim
            hidden_dim = 512 if cfg.DATASET.NAME == "TINYIMAGENET" else hidden_dim


            # Create PyTorch MLP with same architecture as sklearn
            # Using Kaiming initialization (PyTorch default) which is better for ReLU
            return MLPClassifierTorch(input_dim, hidden_dim, cfg.MODEL.NUM_CLASSES).cuda()
        elif cfg.EVAL_MODEL_TYPE == 'from_mlp_sklearn':
            num_features = 384 if cfg.DATASET.NAME in ['IMAGENET50', 'IMAGENET100', 'IMAGENET200'] else 256
            num_features = 2 if cfg.DATASET.NAME in ['SCENARIO_A', 'HALF_MOON'] else num_features

            # Fallback to sklearn's MLPClassifier
            print("Using sklearn's MLPClassifier (PyTorch version not available)")
            return MLPClassifier(
                hidden_layer_sizes=(num_features,),
                activation='relu',
                solver='adam',
                alpha=3e-2,
                max_iter=300,
                random_state=cfg.RNG_SEED,
                verbose=True
            )
        elif cfg.EVAL_MODEL_TYPE == 'mlp_dropout':
            # MLP with Dropout for Monte Carlo uncertainty estimation
            input_dim = 384 if cfg.DATASET.NAME in ['IMAGENET50', 'IMAGENET100', 'IMAGENET200'] else 512
            input_dim = 2 if cfg.DATASET.NAME in ['SCENARIO_A', 'HALF_MOON'] else input_dim
            
            dropout_p = getattr(cfg, 'MLP_DROPOUT_P', 0.3)
            print(f"Using MLP with Dropout (p={dropout_p}) for MC uncertainty estimation")
            return MLPClassifierDropout(input_dim, cfg.MODEL.NUM_CLASSES, dropout_p=dropout_p).cuda()


    model = get_model(cfg)(num_classes=cfg.MODEL.NUM_CLASSES, use_dropout=True)
    if cfg.DATASET.NAME == 'MNIST':
        model.conv1 =  torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    return model.cuda()


def build_loss_fun(cfg):
    """Build the loss function."""
    return get_loss_fun(cfg)()


def register_model(name, ctor):
    """Registers a model dynamically."""
    _models[name] = ctor


def register_loss_fun(name, ctor):
    """Registers a loss function dynamically."""
    _loss_funs[name] = ctor
