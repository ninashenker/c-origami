import torch
import torchvision.utils as vision_utils

def record_csv_entry(file_dir, entry):
    with open(file_dir, 'a') as f:
        f.write(entry)

def save_image(image, name):
    image = torch.clip(image.unsqueeze(1) / 5, 0, 1)
    img_grid = vision_utils.make_grid(image)
    vision_utils.save_image(img_grid, name)

def show_device(device):
    if device.type == 'cuda':
        message = ' Memory:\n Allocated:{} GB\n Cached:{} GB\n'.format(
        round(torch.cuda.memory_allocated(0)/1024**3,1),
        round(torch.cuda.memory_cached(0)/1024**3,1))
        print(message)

def show_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    print(total_norm)

def show_grad_mean(model):
    total_norm = torch.tensor(0)
    for p in model.parameters():
        param_norm = p.grad.data.max()
        #param_norm = p.grad.data.mean()
        total_norm += param_norm.item() ** 2
    total_norm = torch.cat([total_norm, param_norm])
    print(total_norm.max())

