import torch
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics.functional import r2_score
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms
from EfficientNets import EfficientNetV2S
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import optuna


def determine_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps" # Apple M1

    print(f"We are using device: {device}")
    return device


DEVICE = determine_device()
RANDOM_SEED = np.random.randint(0, 1000)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def get_data_item(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

train_data = get_data_item('../data/data_meta_train_norm.csv')

ascentions_per_route = train_data['ascensionist_count'].values
del train_data

# sample_weights = (ascentions_per_route / sum(ascentions_per_route)) * len(ascentions_per_route)
sample_weights = ascentions_per_route * len(ascentions_per_route) # Because we already normalized the data
sample_weights_tensor = torch.Tensor(sample_weights)

sample_weights_tensor.shape, sample_weights_tensor[0]

default_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

class BoulderingDataset(Dataset):
    def __init__(self, *, train_path='../data/data_meta_train_norm.csv', img_transforms=default_transforms, img_path='../data/imgs'):
        super().__init__()
        train_data = get_data_item(train_path)
        self.Y = torch.Tensor(train_data['difficulty_average'].values).unsqueeze(1)
        self.ids = train_data['uuid'].values
        self.angles = torch.Tensor(train_data['angle'].values).unsqueeze(1)
        self.img_transforms = img_transforms
        self.img_path = img_path

    def __getitem__(self, index) -> tuple: # (img, angle, y)
        img = Image.open(os.path.join(self.img_path, self.ids[index] + '.png')).convert(mode='L') # L = grayscale

        if self.img_transforms:
            img = self.img_transforms(img)

        return img, self.angles[index], self.Y[index]

    def __len__(self):
        return len(self.Y)


def compute_regression_metrics(y_pred, y_true, *, metrics=['MSE', 'MAE', 'R2'], mse=None):
    metrics_dict = {}
    
    if 'MSE' in metrics or 'RMSE' in metrics:
        if mse:
            metrics_dict['MSE'] = mse
        else:
            mse = F.mse_loss(y_pred, y_true, reduction='sum') / y_pred.size(0)
            metrics_dict['MSE'] = mse.item()
    
    if 'RMSE' in metrics:
        metrics_dict['RMSE'] = torch.sqrt(metrics_dict['MSE']).item()
    
    if 'MAE' in metrics:
        mae = F.l1_loss(y_pred, y_true, reduction='sum') / y_pred.size(0)
        metrics_dict['MAE'] = mae.item()
    
    if 'R2' in metrics:
        metrics_dict['R2'] = r2_score(y_pred, y_true).item()
        
    return metrics_dict


def train_one_epoch(model, train_loader, optimizer, loss_fn, device=DEVICE):
    model.train()
    
    total_loss = 0.0
    for img, angle, y in train_loader:
        img, angle, y = img.to(device), angle.to(device), y.to(device)
        
        y_pred = model(img, angle)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, valid_loader, *, device=DEVICE, mse=None):
    model.eval()
    
    with torch.no_grad():
        ys_pred, ys_true = [], []
        for img, angle, y in valid_loader:
            img, angle = img.to(device), angle.to(device)
            y_pred = model(img, angle)
            ys_pred.append(y_pred)
            ys_true.append(y.to(device))
    
    ys_pred = torch.cat(ys_pred, dim=0)
    ys_true = torch.cat(ys_true, dim=0)
    
    metrics = compute_regression_metrics(ys_pred, ys_true, mse=mse)
    return metrics


def append_to_csv(file_path, data_frame):
    try:
        df_existing = pd.read_csv(file_path)
        df_combined = pd.concat([df_existing, data_frame]).drop_duplicates()
        df_combined.to_csv(file_path, index=False)
    except FileNotFoundError:
        data_frame.to_csv(file_path, index=False)


img_dim = 224

custom_transforms = transforms.Compose([
    transforms.Resize(img_dim),
    transforms.ToTensor(),
])

dataset = BoulderingDataset(img_transforms=custom_transforms)


# Source https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets

TEST_SIZE = 0.2
train_indices, test_indices, _, _ = train_test_split(
    range(len(dataset.Y)),
    dataset.Y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    shuffle=True
)

PARTIAL_DATASET = True
PARTIAL_PCT = 0.1
if PARTIAL_DATASET:
    train_indices = train_indices[:int(len(train_indices) * PARTIAL_PCT)]
    test_indices = test_indices[:int(len(test_indices) * PARTIAL_PCT)]

train_dataset = Subset(dataset, train_indices)
valid_dataset = Subset(dataset, test_indices)
del dataset
if DEVICE == "cuda":
    torch.cuda.empty_cache()

train_sample_weights_tensor = sample_weights_tensor[train_indices]
sampler = WeightedRandomSampler(train_sample_weights_tensor, len(train_sample_weights_tensor))

DEBUG = True
if PARTIAL_DATASET:
    print(f'ATTENTION: Using only {PARTIAL_PCT * 100}% of the dataset!')

print(f'Random seed: {RANDOM_SEED}, train size: {len(train_dataset)}, valid size: {len(valid_dataset)}. Device: {DEVICE}')

MAX_EPOCHS = 300

# Early stopping if MSE of the last n epochs is higher than the mean of the k previous epochs
n = 7
k = 14

PRETRAINED = True

def objective(trial):
    gradient_exploded = False
    metrics = {
        'MSE': { 'train': [], 'valid': [] },
        'MAE': { 'train': [], 'valid': [] },
        'R2': { 'train': [], 'valid': [] }
    }
    
    BATCH_NORM = trial.suggest_categorical('BATCH_NORM', [True, False])
    ACTIVATION = trial.suggest_categorical('ACTIVATION', ['ELU', 'ReLU', 'LeakyReLU', 'Tanh'])
    DROPOUT = trial.suggest_categorical('DROPOUT', [0.0, 0.1, 0.2, 0.3])
    LAYER_DIMS = trial.suggest_categorical('LAYER_DIMS', [
        '1024, 512',
        '512, 256',
        '256, 128',
    ])
    LAYER_DIMS = tuple(map(int, LAYER_DIMS.split(', ')))
    BATCH_SIZE = trial.suggest_categorical('BATCH_SIZE', [16, 32, 64, 128])
    LEARNING_RATE = trial.suggest_categorical('LEARNING_RATE', [0.0005, 0.001, 0.005, 0.01])
    SCHEDULER_FACTOR = trial.suggest_categorical('SCHEDULER_FACTOR', [0.1, 0.25])
    SCHEDULER_PATIENCE = trial.suggest_categorical('SCHEDULER_PATIENCE', [4, 5])
    WEIGHT_DECAY = trial.suggest_categorical('WEIGHT_DECAY', [0, 1e-5, 1e-4])

    model = EfficientNetV2S(img_dim, 1, pretrained=PRETRAINED, batch_norm=BATCH_NORM, activation=ACTIVATION, dropout=DROPOUT, layer_dims=LAYER_DIMS).to(DEVICE)

    DEBUG and print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
    DEBUG and print(f'Trial {trial.number} params: {trial.params}')

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR, verbose=True if DEBUG else False)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, pin_memory=True)

    for epoch in range(MAX_EPOCHS):
        gradient_exploded = False
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)

        train_metrics = evaluate(model, train_loader, mse=train_loss)
        valid_metrics = evaluate(model, valid_loader)

        scheduler.step(valid_metrics['MSE'])
        
        DEBUG and print(f'Epoch {epoch + 1}/{MAX_EPOCHS}, valid MAE {valid_metrics["MAE"]:.2f}, R2 {valid_metrics["R2"]:.2f}, MSE {valid_metrics["MSE"]:.2f}')
        for metric in train_metrics.keys():
            metrics[metric]['train'].append(train_metrics[metric])
            metrics[metric]['valid'].append(valid_metrics[metric])

        if epoch > 0 and train_loss > 100 * metrics['MSE']['train'][-1]:
            print(f'Gradient explosion at epoch {epoch + 1}. Stopping trial. train_loss: {train_loss}')
            gradient_exploded = True
            break

        # Early stopping if MSE of the last n epochs is higher than the mean of the k previous epochs, and the scheduler has reduced the learning rate, or we are at epoch > 100
        has_scheduler_reduced_lr = optimizer.param_groups[0]['lr'] < LEARNING_RATE
        if epoch > (n + k - 1) and np.mean(metrics['MSE']['valid'][-n:]) >= np.mean(metrics['MSE']['valid'][-k:-n]) and (has_scheduler_reduced_lr or epoch > 99):
            print(f'Early stopping at epoch {epoch + 1} because MSE appears to be increasing.')
            break

        if epoch == MAX_EPOCHS - 1:
            print(f'Wow! Max epochs reached ({MAX_EPOCHS})! Stopping...')

    best_valid_mse = min(metrics['MSE']['valid'])

    trial_results = pd.DataFrame([{
        'gradient_exploded': gradient_exploded,
        'n_epochs': epoch + 1,
        'best_valid_mse': best_valid_mse,
        'best_valid_mae': min(metrics['MAE']['valid']),
        'best_valid_r2': max(metrics['R2']['valid']),
        'BATCH_SIZE': BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'SCHEDULER_FACTOR': SCHEDULER_FACTOR,
        'SCHEDULER_PATIENCE': SCHEDULER_PATIENCE,
        'PRETRAINED': PRETRAINED,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'random_seed': RANDOM_SEED,
    }])
    append_to_csv('results-train-py.csv', trial_results)

    print(f"Trial {trial.number} finished. Best valid MSE: {best_valid_mse:.4f}")

    return best_valid_mse

study_name = "br-study-bayes"
storage_name = f"sqlite:///{study_name}.db"

study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction='minimize')

study.optimize(objective, n_trials=20)

print(f'Finished study with {len(study.trials)} trials. Lowest mean MSE valid: {study.best_value:.2f}, best params: {study.best_params}')
