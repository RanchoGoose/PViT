import torch
from tqdm import tqdm
import torch.nn.functional as F

def extract_features(model, dataloader, device):
    model.to(device)
    model.eval()

    feas = [[]] * len(dataloader)
    logits = [[]] * len(dataloader)
    labels = [[]] * len(dataloader)
    
    for i, labeled_data in tqdm(enumerate(dataloader), desc="Extracting features"):
        _x = labeled_data[0].to(device)
        _y = labeled_data[2]

        with torch.no_grad():
            _rawfeas, _logits = model(_x)
        _feas = F.normalize(_rawfeas, dim=1)

        feas[i] = _feas.cpu()
        logits[i] = _logits.cpu()
        labels[i] = _y.cpu()
    
    feas = torch.cat(feas, dim=0)
    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)

    print(f"Successfully extracted features")

    return {"feas": feas, "logits": logits, "labels": labels}

def extract_features_pvit(prior_model, model, dataloader, device):
    model.to(device)
    model.eval()

    feas = [[]] * len(dataloader)
    logits = [[]] * len(dataloader)
    labels = [[]] * len(dataloader)
    priors = [[]] * len(dataloader)
    
    for i, labeled_data in tqdm(enumerate(dataloader), desc="Extracting features"):
        _x = labeled_data[0].to(device)
        _y = labeled_data[2]

        with torch.no_grad():
            try:
                # Attempt to unpack two outputs from the prior model
                _prior_feas, _priors = prior_model(_x)
            except ValueError:
                # If a ValueError occurs, it means the model returned only one value
                _priors = prior_model(_x)
            if hasattr(_priors, 'logits'):
                _priors = _priors.logits
                
            _rawfeas, _logits = model(_x, _priors)
        _feas = F.normalize(_rawfeas, dim=1)

        feas[i] = _feas.cpu()
        logits[i] = _logits.cpu()
        labels[i] = _y.cpu()
        priors[i] = _priors.cpu()
    
    feas = torch.cat(feas, dim=0)
    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)
    priors = torch.cat(priors, dim=0)

    print(f"Successfully extracted features")

    return {"feas": feas, "logits": logits, "labels": labels, "priors": priors}

def extract_features_pvit_ablation(prior_model, model, dataloader, device):
    model.to(device)
    model.eval()

    feas = [[]] * len(dataloader)
    logits = [[]] * len(dataloader)
    labels = [[]] * len(dataloader)
    priors = [[]] * len(dataloader)
    
    for i, labeled_data in tqdm(enumerate(dataloader), desc="Extracting features"):
        _x = labeled_data[0].to(device)
        _y = labeled_data[2]

        with torch.no_grad():
            try:
                # Attempt to unpack two outputs from the prior model
                _prior_feas, _priors = prior_model(_x)
            except ValueError:
                # If a ValueError occurs, it means the model returned only one value
                _priors = prior_model(_x)
            if hasattr(_priors, 'logits'):
                _priors = _priors.logits
                
            _rawfeas, _logits = model(_x)
        _feas = F.normalize(_rawfeas, dim=1)

        feas[i] = _feas.cpu()
        logits[i] = _logits.cpu()
        labels[i] = _y.cpu()
        priors[i] = _priors.cpu()
    
    feas = torch.cat(feas, dim=0)
    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)
    priors = torch.cat(priors, dim=0)

    print(f"Successfully extracted features")

    return {"feas": feas, "logits": logits, "labels": labels, "priors": priors}

def extract_features_resnet18(model, dataloader, device):
    model.to(device)
    model.eval()

    feas = []
    logits = []
    labels = []
    
    for i, labeled_data in tqdm(enumerate(dataloader), desc="Extracting features"):
        _x = labeled_data[0].to(device)
        _y = labeled_data[2]

        with torch.no_grad():        
            _logits, _rawfeas = model(_x, return_feature=True)

        _feas = F.normalize(_rawfeas, dim=1)
        
        feas.append(_feas.cpu())
        logits.append(_logits.cpu())
        labels.append(_y.cpu())
    
    feas = torch.cat(feas, dim=0)
    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)
    
    print(f"Successfully extracted features")

    return {"feas": feas, "logits": logits, "labels": labels}

def extract_features_vit_pross(model, dataloader, device):
    model.to(device)
    model.eval()

    feas = [[]] * len(dataloader)
    logits = [[]] * len(dataloader)
    labels = [[]] * len(dataloader)
    
    for i, labeled_data in tqdm(enumerate(dataloader), desc="Extracting features"):
        _x = labeled_data[0].to(device)
        _y = labeled_data[2]

        with torch.no_grad():
            _logits, _, _rawfeas = model(_x)
        _feas = F.normalize(_rawfeas, dim=1)

        feas[i] = _feas.cpu()
        logits[i] = _logits.cpu()
        labels[i] = _y.cpu()
    
    feas = torch.cat(feas, dim=0)
    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)

    print(f"Successfully extracted features")

    return {"feas": feas, "logits": logits, "labels": labels}

def compute_energy_reward(energy_scores_list, ood_labels):
    """
    Computes energy-based rewards.

    Args:
        energy_scores_list (List[torch.Tensor]): List of energy scores from different layers.
        ood_labels (torch.Tensor): OOD labels (0 for ID, 1 for OOD).

    Returns:
        List[torch.Tensor]: Energy rewards for each layer.
    """
    rewards = []
    for energy_scores in energy_scores_list:
        # For ID samples (ood_label=0), reward negative energy (lower energy)
        # For OOD samples (ood_label=1), reward positive energy (higher energy)
        reward = energy_scores * (2 * ood_labels.float() - 1)  # Maps 0 -> -1, 1 -> 1
        rewards.append(reward)
    return rewards

def compute_loss(main_output, labels, energy_rewards, classification_criterion):
    """
    Computes the total loss combining classification loss and energy-based rewards.

    Args:
        main_output (torch.Tensor): Main classification logits.
        labels (torch.Tensor): Ground truth labels.
        energy_rewards (List[torch.Tensor]): Energy rewards from each layer.
        classification_criterion (nn.Module): Classification loss function.

    Returns:
        torch.Tensor: Total loss.
    """
    # Main classification loss
    classification_loss = classification_criterion(main_output, labels)

    # Energy-based reward loss
    energy_loss = 0
    for reward in energy_rewards:
        # We want to maximize the reward, so we minimize the negative reward
        energy_loss += -torch.mean(reward)
    energy_loss = energy_loss / len(energy_rewards)

    # Total loss
    total_loss = classification_loss + energy_loss
    return total_loss

def save_checkpoint(epoch, loss, filename=None):
    """
    Saves the current state of the model, optimizer, and scheduler.

    Args:
        epoch (int): Current epoch number.
        loss (float): Loss value at the current epoch.
        filename (str, optional): File path to save the checkpoint. 
                                    Defaults to a timestamped filename.
    """
    if filename is None:
        filename = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
    
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }

    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(checkpoint, filename)
    logging.info(f"Checkpoint saved at {filename}")

def load_checkpoint(args, filename):
    """
    Loads the model, optimizer, and scheduler states from a checkpoint.

    Args:
        filename (str): Path to the checkpoint file.
    """
    if not os.path.exists(filename):
        logging.error(f"Checkpoint file {filename} does not exist.")
        raise FileNotFoundError(f"Checkpoint file {filename} not found.")

    checkpoint = torch.load(filename, map_location=self._device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    logging.info(f"Loaded checkpoint from {filename}, Epoch {epoch}, Loss {loss:.4f}")
