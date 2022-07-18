import sys
import torch
import hydra
import torchmetrics
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks

from omegaconf import DictConfig, OmegaConf

import model.corigami_models as corigami_models
from data import genome_dataset

torch.autograd.set_detect_anomaly(True)

@hydra.main(config_path="config", config_name="default")
def main(args : DictConfig) -> None:
    init_training(args)

def init_training(args):

    # Offline training

    if args.logger.wandb.mode == 'offline':
        print('*** wandb in offline mode ***')
        import os
        os.environ['WANDB_MODE'] = 'offline'

    # Early_stopping
    early_stop_callback = callbacks.EarlyStopping(monitor='val_loss', 
                                        min_delta=0.00, 
                                        patience=args.trainer.patience,
                                        verbose=False,
                                        mode="min")
    # Checkpoints
    checkpoint_callback = callbacks.ModelCheckpoint(dirpath=f'{args.run.save_path}/models',
                                        save_top_k=args.trainer.save_top_n, 
                                        monitor='val_loss')

    # LR monitor
    lr_monitor = callbacks.LearningRateMonitor(logging_interval='epoch')

    # Logger
    csv_logger = pl.loggers.CSVLogger(save_dir = f'{args.run.save_path}/csv')
    wandb_logger = pl.loggers.WandbLogger(name = args.run.name, project=args.logger.wandb.project, save_dir = args.run.save_path)
    if args.logger.wandb.mode == 'off':
        all_loggers = csv_logger
    else:
        all_loggers = [csv_logger, wandb_logger]
    
    # Assign seed
    pl.seed_everything(args.run.seed, workers=True)
    pl_module = TrainModule(args)
    pl_trainer = pl.Trainer(strategy='ddp',
                            accelerator="gpu", devices=args.trainer.num_gpu,
                            gradient_clip_val=1,
                            logger = all_loggers,
                            callbacks = [early_stop_callback,
                                         checkpoint_callback,
                                         lr_monitor],
                            max_epochs = args.trainer.max_epochs
                            )
    trainloader = pl_module.get_dataloader(args, 'train')
    valloader = pl_module.get_dataloader(args, 'val')
    testloader = pl_module.get_dataloader(args, 'test')
    pl_trainer.fit(pl_module, trainloader, valloader)

class TrainModule(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        self.model = self.get_model(args)
        self.args = args
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        seq, features, mat, start, end, chr_name, chr_idx = batch
        features = torch.cat([feat.unsqueeze(2) for feat in features], dim = 2)
        inputs = torch.cat([seq, features], dim = 2)
        mat = mat.float()
        outputs = self(inputs)
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, mat)

        metrics = {'train_step_loss': loss}
        self.log_dict(metrics, batch_size = inputs.shape[0], prog_bar=True)

        if batch_idx % 100 == 0 and self.current_epoch == 0:
            out_img = self.proc_image(outputs)
            mat_img = self.proc_image(mat)
            if self.args.logger.wandb.mode != 'off':
                self.logger[1].log_image(key = f'train_pred_vs_target_{batch_idx}', 
                                        images = [out_img, mat_img], 
                                        caption=['Prediction', 'Target'])
        return loss

    def validation_step(self, batch, batch_idx):
        ret_metrics = self._shared_eval_step(batch, batch_idx)
        return ret_metrics

    def test_step(self, batch, batch_idx):
        ret_metrics = self._shared_eval_step(batch, batch_idx)
        return ret_metrics

    def _shared_eval_step(self, batch, batch_idx):
        seq, features, mat, start, end, chr_name, chr_idx = batch
        features = torch.cat([feat.unsqueeze(2) for feat in features], dim = 2)
        inputs = torch.cat([seq, features], dim = 2)
        mat = mat.float()
        outputs = self(inputs)
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, mat)

        if batch_idx in (np.array([0.3, 0.5, 0.7]) * self.val_length).astype(int):
            out_img = self.proc_image(outputs)
            mat_img = self.proc_image(mat)
            if self.args.logger.wandb.mode != 'off':
                self.logger[1].log_image(key = f'val_pred_vs_target_{batch_idx}',
                                        images = [out_img, mat_img], 
                                        caption=['Prediction', 'Target'])
        return loss

    def proc_image(self, image):
        image = torch.clip(image / 5, 0, 1)[0].detach().cpu().numpy()
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        color_map = LinearSegmentedColormap.from_list("bright_red",
                                                    [(1,1,1),(1,0,0)])
        fig, ax = plt.subplots()
        ax.imshow(image, cmap = color_map, vmin = 0, vmax = 1)
        ax.set_axis_off()
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return data
        
    # Collect epoch statistics
    def training_epoch_end(self, step_outputs):
        step_outputs = [out['loss'] for out in step_outputs]
        ret_metrics = self._shared_epoch_end(step_outputs)
        metrics = {'train_loss' : ret_metrics['loss']
                  }
        self.log_dict(metrics, prog_bar=True)

    def validation_epoch_end(self, step_outputs):
        ret_metrics = self._shared_epoch_end(step_outputs)
        metrics = {'val_loss' : ret_metrics['loss']
                  }
        self.log_dict(metrics, prog_bar=True)

    def _shared_epoch_end(self, step_outputs):
        loss = torch.tensor(step_outputs).mean()
        return {'loss' : loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr = self.args.optim.lr,
                                     weight_decay = self.args.optim.weight_decay)

        import pl_bolts
        scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.args.optim.warmup_epoches, max_epochs=200)
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'val_loss',
            'strict': True,
            'name': 'WarmupCosineAnnealing',
        }
        return {'optimizer' : optimizer, 'lr_scheduler' : scheduler_config}

    def get_dataset(self, args, mode):

        if args.run.debug: # Small test set
            if mode == 'train':
                mode = 'test'


        celltype_root = f'{args.dataset.data_root}/{args.dataset.assembly}/{args.dataset.celltype}'
        dataset = genome_dataset.GenomeDataset(celltype_root, 
                                args.dataset.genomic_features, 
                                mode = mode)

        # Record length for printing validation image
        if mode == 'val':
            self.val_length = len(dataset) / args.dataloader.batch_size
            print('Validation loader length:', self.val_length)

        return dataset

    def get_dataloader(self, args, mode):
        dataset = self.get_dataset(args, mode)

        if args.run.debug:
            size = 150
            dataset = torch.utils.data.Subset(dataset, range(size))

        if mode == 'train':
            shuffle = True
        else: # validation and test settings
            shuffle = False
        
        if args.dataloader.ddp_enabled:
            gpus = args.dataloader.num_gpu
            batch_size = int(args.dataloader.batch_size / gpus)
            num_workers = int(args.dataloader.num_workers / gpus) 
        else:
            batch_size = args.dataloader.batch_size
            num_workers = args.dataloader.num_workers 
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle = shuffle,
            batch_size= batch_size,
            num_workers=num_workers,
            pin_memory=args.dataloader.pin_memory,
            prefetch_factor=args.dataloader.prefetch_factor,
            persistent_workers=True
        )
        return dataloader

    def get_model(self, args):
        model_name =  args.model.model_type
        num_genomic_features = len(args.dataset.genomic_features)
        ModelClass = getattr(corigami_models, model_name)
        model = ModelClass(num_genomic_features, mid_hidden = args.model.mid_hidden)
        return model

if __name__ == '__main__':
    main()
