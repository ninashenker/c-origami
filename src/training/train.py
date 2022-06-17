import os
import datetime

import torch
from tqdm import tqdm

from dataloader import GenomeDataset
from model import CNN, CNN_dual_encoder
from log import record_csv_entry, save_image, show_device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using: ', device)

# Data 
batch_size = 8
# TODO changed
#data_root = '/gpfs/data/tsirigoslab/home/jt3545/hic_prediction/ingenious-evaluation/data/imr90/processed_data'
data_root = '/gpfs/data/tsirigoslab/home/jt3545/hic_prediction/ingenious-evaluation/data/imr90/processed_data_bigwig'
trainset = GenomeDataset(data_root, mode = 'train')
valset = GenomeDataset(data_root, mode = 'val')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4)

# Model
# TODO changed
in_channel = 5 + 1 + 1 # DNA + CTCF + ATAC
#in_channel = 5 + 1  # DNA + CTCF/ATAC
#model = CNN(in_channel)
model = CNN_dual_encoder(in_channel)
if torch.cuda.device_count() > 1:
  print("Using", torch.cuda.device_count(), "GPUs")
  model = torch.nn.DataParallel(model)
model.to(device)

# Loss and optimizer
lr = 0.002
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr, eps = 1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 200)

# Training loop
epochs = 200
save_interval = 1
# TODO changed
dna_only_epoches = 0
run_name = 'IMR-90_dense_dual_encoder_one_direction'
save_dir = f'runs/{run_name}_{datetime.datetime.now().isoformat()}'
os.mkdir(save_dir)
record_csv_entry(f'{save_dir}/loss.csv', 'Epoch, Train loss, Validation loss\n')

for epoch in range(epochs):
    epoch_loss = {'train' : 0, 'val' : 0}

    # Staged training 
    start_epi = model.module.encoder.conv_start_epi
    res_epi = model.module.encoder.res_blocks_epi

    if epoch < dna_only_epoches:
        for start_p in start_epi:
            start_p.requires_grad_(False)
        for res_p in res_epi:
            res_p.requires_grad_(False)
    else:
        for start_p in start_epi:
            start_p.requires_grad_(True)
        for res_p in res_epi:
            res_p.requires_grad_(True)
            
    print('Epi start requires grad:', model.module.encoder.conv_start_epi[0].weight.requires_grad)
    print('Epi start grad:', model.module.encoder.conv_start_epi[0].weight.grad)
    print('Epi blocks requires grad:', model.module.encoder.res_blocks_epi[0].res[0].weight.requires_grad)
    print('Epi blocks grad:', model.module.encoder.res_blocks_epi[0].res[0].weight.grad)

    # Training loop
    model.train()
    for train_i, data_batch in enumerate(tqdm(trainloader)): 
        seq, ctcf, atac, mat, start, end, chr_name, chr_idx = data_batch
        seq = seq.to(device)
        ctcf = ctcf.to(device)
        atac = atac.to(device)
        mat = mat.to(device)

        # TODO: step wise training
        if epoch < dna_only_epoches:
            ctcf = abs(torch.normal(0, 0.5, ctcf.shape).to(device))
            atac = abs(torch.normal(0, 0.5, atac.shape).to(device))

        # Default input
        inputs = torch.cat([seq, ctcf.unsqueeze(2), atac.unsqueeze(2)], dim = 2)
        # TODO: CTCF only
        #inputs = torch.cat([seq, ctcf.unsqueeze(2)], dim = 2)
        # TODO: ATAC only
        #inputs = torch.cat([seq, atac.unsqueeze(2)], dim = 2)

        outputs = model(inputs)
        loss = criterion(outputs, mat.float())
        
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 100) # grad clip

        optimizer.step()

        epoch_loss['train'] += loss.item()
        print(loss.item())
        show_device(device)

        # Save output 
        if train_i % 100 == 0 and epoch == 0:
            save_image(outputs, f'{save_dir}/train_{epoch}_{train_i}_pred.png')
            save_image(mat, f'{save_dir}/train_{epoch}_{train_i}_label.png')
    save_image(outputs, f'{save_dir}/train_{epoch}_{train_i}_pred.png')
    save_image(mat, f'{save_dir}/train_{epoch}_{train_i}_label.png')
    scheduler.step()

    # Validation loop
    model.eval()
    with torch.no_grad():
        for val_i, data_batch in enumerate(tqdm(valloader)):
            seq, ctcf, atac, mat, start, end, chr_name, chr_idx = data_batch
            seq = seq.to(device)
            ctcf = ctcf.to(device)
            atac = atac.to(device)
            mat = mat.to(device)

            # TODO: step wise training
            if epoch < dna_only_epoches:
                ctcf = abs(torch.normal(0, 0.5, ctcf.shape, device = device))
                atac = abs(torch.normal(0, 0.5, atac.shape, device = device))

            # Default input
            inputs = torch.cat([seq, ctcf.unsqueeze(2), atac.unsqueeze(2)], dim = 2)
            # TODO: CTCF only
            #inputs = torch.cat([seq, ctcf.unsqueeze(2)], dim = 2)
            # TODO: ATAC only
            #inputs = torch.cat([seq, atac.unsqueeze(2)], dim = 2)

            outputs = model(inputs)
            loss = criterion(outputs, mat.float())
            epoch_loss['val'] += loss.item()

            # Save output 
            if val_i == int(0.5 * len(valloader)):
                save_image(outputs, f'{save_dir}/val_{epoch}_pred.png')
                save_image(mat, f'{save_dir}/val_{epoch}_label.png')
        # Save model
        if (epoch + 1) % save_interval == 0 or epoch == 0:
            torch.save(model.state_dict(), f'{save_dir}/state_dict_{epoch}.pt')
    
    # Log values
    train_loss = epoch_loss["train"] / len(trainloader)
    val_loss = epoch_loss["val"] / len(valloader)
    print(f'Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}')
    record_csv_entry(f'{save_dir}/loss.csv', f'{epoch}, {train_loss}, {val_loss}\n')
