import torch

def get_mean_std(dataloader):
    """
    This function computes the mean and the std from a dataloader object. 

    Inputs:
    - dataloader: A torchvision dataloader

    Outputs:
    - mean: a float32 tensor containing the mean values for each image channel
    - std: a float32 tensor containing the std values for each image channel
    """

    # Check the number of channels in the first batch
    first_batch, _, _ = next(iter(dataloader))
    num_channels = first_batch.shape[1]

    # Initializing the mean and std
    mean = torch.zeros(num_channels)
    std = torch.zeros(num_channels)

    for batch, _, _ in dataloader: # iterating over every batch
        for channel in range(num_channels):
            # Extracting the current channel
            current_channel = batch[:, channel, :, :]

            # Computing the mean and std for the current channel
            mean[channel] += torch.mean(current_channel)
            std[channel] += torch.std(current_channel)

    # Dividing by the number of batches
    mean /= len(dataloader)
    std /= len(dataloader)

    return mean, std
