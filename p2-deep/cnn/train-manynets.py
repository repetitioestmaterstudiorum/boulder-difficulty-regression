import torch
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics.functional import r2_score
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from smallnets import shufflenetx1, shufflenetx2, mobilenet, mnasnet
from bignets import cvt, convnextl, regnety32gf
from EfficientNets import EfficientNetV2S, EfficientNetB0
from tnet import tnet, tnextnet, tnet2, tnet3, tnet4, tnet5, tnet6
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import argparse
import wandb
from tqdm import tqdm
from utils import get_mean_std


parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, default='shufflenet', help='Model to use: shufflenet, mobilenet, mnasnet')
parser.add_argument('-pct', type=float, default=1, help='Percentage of the dataset to use')
parser.add_argument('-lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('-patience', type=int, default=0, help='Patience for ReduceLROnPlateau scheduler')
parser.add_argument('-img-dim', type=int, default=256, help='Image dimension')
parser.add_argument('-wd', type=float, default=0, help='Weight decay')
parser.add_argument('-factor', type=float, default=0.25, help='Factor for ReduceLROnPlateau scheduler')
parser.add_argument('-device', type=str, default='auto', help='Device to use: cpu, cuda, mps, auto')
parser.add_argument('-es-n', type=int, default=10, help='Early stopping n: number of epochs to look back')
parser.add_argument('-es-k', type=int, default=20, help='Early stopping k: number of epochs to look back')
parser.add_argument('-debug', type=bool, default=True, help='Debug mode')
parser.add_argument('-bs', type=int, default=16, help='Batch size')
parser.add_argument('-pin-memory', type=bool, default=False, help='Pin memory')
parser.add_argument('-mgpu', type=bool, default=False, help='Multi-process GPU')
parser.add_argument('-get-stats', type=bool, default=False, help='Get mean and std of dataset')
parser.add_argument('-img-standardize', type=bool, default=False, help='Standardize images')
parser.add_argument('-img-normalize', type=bool, default=False, help='Normalize images')
parser.add_argument('-img-mem', type=bool, default=False, help='Load img files into memory')
parser.add_argument('-sw', type=bool, default=False, help='Sample weights')
args = parser.parse_args()


def determine_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps" # Apple M1

    print(f"We are using device: {device}")
    return device


def get_data_item(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


class BoulderingDataset(Dataset):
    def __init__(self, *, train_path='../data/data_meta_train_norm.csv', img_transforms, root_dir='../data', res=256, img_mem=True):
        super().__init__()
        train_data = get_data_item(train_path)
        self.Y = torch.Tensor(train_data['difficulty_average'].values).unsqueeze(1)
        self.ids = train_data['uuid'].values
        self.angles = torch.Tensor(train_data['angle'].values).unsqueeze(1)
        self.img_transforms = img_transforms
        self.root_dir = root_dir
        self.res = res
        self.img_mem = img_mem
        if self.img_mem:
            self.preload_imgs()

    def load_img(self, uuid):
        imgfile = os.path.join(self.root_dir, f"imgs/{uuid}.png")
        image = Image.open(imgfile).convert(mode='L')
        if self.img_transforms:
            image = self.img_transforms(image)
        return image

    def preload_imgs(self):
        print(f'Preloading images with resolution {self.res}')
        index_path = f'{self.root_dir}/index{self.res}.npy'
        images_path = f'{self.root_dir}/images{self.res}.npy'
        if os.path.exists(index_path) and os.path.exists(images_path):
            self.index = np.load(index_path, allow_pickle=True)
            self.images = np.load(images_path, allow_pickle=True)
        else:
            print(f'First time loading images. Saving index and images to {self.root_dir} with resolution {self.res}')
            unique_ids = np.unique(self.ids)
            self.images = np.array([self.load_img(uuid).numpy() for uuid in unique_ids])
            self.index = unique_ids
            np.save(index_path, self.index)
            np.save(images_path, self.images)
        print('Done preloading images')

    def __getitem__(self, index) -> tuple:
        if self.img_mem:
            img_idx = np.where(self.index == self.ids[index])[0][0]
            img = self.images[img_idx]
            img = torch.from_numpy(img).float()
            return img, self.angles[index], self.Y[index]
        else:
            img = self.load_img(self.ids[index])
            if not args.img_standardize:
                img = torch.from_numpy(img.numpy()).float()
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


def train_one_epoch(model, train_loader, optimizer, loss_fn, device, *, epoch, clip_grad=False):
    model.train()
    
    total_loss = 0.0
    for img, angle, y in tqdm(train_loader, total=len(train_loader), desc=f'Training epoch {epoch}'):
        img, angle, y = img.to(device), angle.to(device), y.to(device)
        
        y_pred = model(img, angle)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, data_loader, device, *, mse=None, dataset_name='valid'):
    model.eval()
    
    with torch.no_grad():
        ys_pred, ys_true = [], []
        for img, angle, y in tqdm(data_loader, total=len(data_loader), desc=f'Evaluating on {dataset_name} data'):
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


nets = {
    'shufflenetx1': shufflenetx1,
    'shufflenetx2': shufflenetx2,
    'mobilenet': mobilenet,
    'mnasnet': mnasnet,
    'tnet': tnet,
    'tnet2': tnet2,
    'tnet3': tnet3,
    'tnet4': tnet4,
    'tnet5': tnet5,
    'tnet6': tnet6,
    'tnextnet': tnextnet,
    'effnetv2s': EfficientNetV2S,
    'effnetb0': EfficientNetB0,
    'cvt': cvt,
    'convnextl': convnextl,
    'regnet': regnety32gf,
}


if __name__=='__main__':
    DEVICE = determine_device() if args.device == 'auto' else args.device
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    RANDOM_SEED = np.random.randint(0, 1000)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    tf = [transforms.Resize(args.img_dim)]
    if args.img_standardize:
        print('Standardizing images')
        tf.append(transforms.ToTensor())
    else:
        tf.append(transforms.PILToTensor())
    if args.img_normalize:
        print('Normalizing images')
        tf.append(transforms.Normalize(mean=[0.0044], std=[0.0446]))
    custom_transforms = transforms.Compose(tf)
    dataset = BoulderingDataset(img_transforms=custom_transforms, res=args.img_dim, img_mem=args.img_mem)

    # Source https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
    TEST_SIZE = 0.2
    train_indices, test_indices, _, _ = train_test_split(
        range(len(dataset.Y)), dataset.Y, test_size=TEST_SIZE, random_state=RANDOM_SEED, shuffle=True
    )

    if (args.pct > 0 and args.pct < 1):
        print(f'ATTENTION: Using only {args.pct * 100}% of the dataset!')
        train_indices = train_indices[:int(len(train_indices) * args.pct)]
        test_indices = test_indices[:int(len(test_indices) * args.pct)]

    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, test_indices)
    del dataset

    metrics = {
        'MSE': { 'train': [], 'valid': [] },
        'MAE': { 'train': [], 'valid': [] },
        'R2': { 'train': [], 'valid': [] }
    }

    MAX_EPOCHS = 500

    BATCH_SIZE = args.bs
    LEARNING_RATE = args.lr
    WEIGHT_DECAY = args.wd

    USE_SCHEDULER = True if args.patience > 0 else False
    SCHEDULER_FACTOR = args.factor
    SCHEDULER_PATIENCE = args.patience

    model = nets[args.model](img_dim=args.img_dim, output_dim=1).to(DEVICE)
    torch_gpu_count = torch.cuda.device_count() if DEVICE == 'cuda' else 0
    if torch_gpu_count > 1 and args.mgpu:
        print(f"Using {torch_gpu_count} GPUs!")
        model = nn.DataParallel(model)

    DEBUG = args.debug
    PIN_MEMORY = args.pin_memory
    print(f'The {model.__class__.__name__} model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR, verbose=True if DEBUG else False)
    loss_fn = nn.MSELoss()

    if args.sw:
        train_data = get_data_item('../data/data_meta_train_norm.csv')
        ascentions_per_route = train_data['ascensionist_count'].values
        del train_data
        sample_weights = torch.Tensor(ascentions_per_route * len(ascentions_per_route)) # Data is already normalized
        train_sample_weights = sample_weights[train_indices]
        sampler = WeightedRandomSampler(train_sample_weights, len(train_sample_weights))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=PIN_MEMORY, num_workers=torch_gpu_count if torch_gpu_count > 1 else 0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY, num_workers=torch_gpu_count if torch_gpu_count > 1 else 0, shuffle=True)
    
    if args.get_stats:
        mean, std = get_mean_std(train_loader)
        print(f'Mean: {mean}, std: {std}')
        exit()

    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY, num_workers=torch_gpu_count if torch_gpu_count > 1 else 0)

    print(f'Random seed: {RANDOM_SEED}, train size: {len(train_dataset)}, valid size: {len(valid_dataset)}. Device: {DEVICE}, img_dim: {args.img_dim}, lr: {args.lr}, wd: {args.wd}, factor: {args.factor}, patience: {args.patience}, batch_size: {args.bs}, pin_memory: {args.pin_memory}')
    wandb.init(project="kilterb-scnn", entity="embereagle", config={
        'model': model.__class__.__name__,
        'img_dim': args.img_dim,
        'lr': args.lr,
        'wd': args.wd,
        'factor': args.factor,
        'patience': args.patience,
        'batch_size': args.bs,
        'pin_memory': args.pin_memory,
        'pct': args.pct,
        'seed': RANDOM_SEED,
        'model': args.model,
        'es_n': args.es_n,
        'es_k': args.es_k,
        'torch_gpu_count': torch_gpu_count,
        'mgpu': args.mgpu,
        'device': DEVICE,
    })

    for epoch in range(MAX_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device=DEVICE, epoch=epoch)

        train_metrics = evaluate(model, train_loader, device=DEVICE, mse=train_loss, dataset_name='train')
        valid_metrics = evaluate(model, valid_loader, device=DEVICE, dataset_name='valid')
        wandb.log({'valid_MSE': valid_metrics['MSE'], 'train_MSE': train_metrics['MSE'], 'valid_MAE': valid_metrics['MAE'], 'train_MAE': train_metrics['MAE'], 'valid_R2': valid_metrics['R2'], 'train_R2': train_metrics['R2'], 'epoch': epoch, 'lr': optimizer.param_groups[0]['lr']})

        if USE_SCHEDULER:
            scheduler.step(valid_metrics['MSE'])
        
        DEBUG and print(f'Epoch {epoch}/{MAX_EPOCHS}. MSE: v:{valid_metrics["MSE"]:.4f} t:{train_metrics["MSE"]:.4f} | MAE: v:{valid_metrics["MAE"]:.4f} t:{train_metrics["MAE"]:.4f} | R2: v:{valid_metrics["R2"]:.4f} t:{train_metrics["R2"]:.4f}')
        for metric in train_metrics.keys():
            metrics[metric]['train'].append(train_metrics[metric])
            metrics[metric]['valid'].append(valid_metrics[metric])

        if epoch > 0 and train_loss > 100 * metrics['MSE']['train'][-1]:
            print(f'Gradient explosion at epoch {epoch}. Stopping training. train_loss: {train_loss}')
            break

        # Early stopping if MSE of the last n epochs is higher than the mean of the k previous epochs, and the scheduler has reduced the learning rate, or we are at epoch > 100
        has_scheduler_reduced_lr = optimizer.param_groups[0]['lr'] < LEARNING_RATE
        if epoch > (args.es_n + args.es_k - 1) and np.mean(metrics['MSE']['valid'][-args.es_n:]) >= np.mean(metrics['MSE']['valid'][-args.es_k:]) and (has_scheduler_reduced_lr or epoch > 99):
            print(f'Early stopping at epoch {epoch} because MSE appears to be increasing.')
            break

        if epoch == MAX_EPOCHS - 1:
            print(f'Wow! Max epochs reached ({MAX_EPOCHS})! Stopping...')

    wandb.finish()
