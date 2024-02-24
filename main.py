from argparse import ArgumentParser
import random
import numpy as np
import torch
import torch.nn.functional as F
from utils.dataload import *
from torch.utils.data import Dataset, DataLoader
import os
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from configs import *
from d3 import *
from sklearn import metrics
import torchvision
from attacks.Attack import attack_loader


def seed_randomness(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    seed_randomness(0)

    # Config
    config = eval(args.config).config

    # Logging
    ckpt_callback = ModelCheckpoint(save_top_k=-1, monitor='train_loss',
                                    filename='{epoch}-{train_loss:.4f}-{val_loss:.4f}--{val_nat_acc:.4f}--{val_adv_acc:.4f}')
    logger = TensorBoardLogger(save_dir=os.path.join("./models_sym", args.config), name="logs")
    trainer = pl.Trainer(accelerator='gpu', strategy='ddp_find_unused_parameters_false',
                         devices=config['train_args']['num_gpus'],
                         max_epochs=config['train_args']['epochs'],
                         default_root_dir=os.path.join("training_logs", args.config),
                         logger=logger, callbacks=[ckpt_callback],
                         precision=16)

    # Datasets
    train_loader, val_loader, test_loader = None, None, None
    if config["task"] in ["train", "saliency"]:
        train_dataset = DiffusionDataset(split="train", type="LDM")
        train_loader = DataLoader(train_dataset, batch_size=config['train_args']['batch_size'], shuffle=True,
                                  num_workers=4,
                                  pin_memory=False)
        val_dataset = DiffusionDataset(split="val", type="LDM")
        val_loader = DataLoader(val_dataset, batch_size=config['train_args']['batch_size'], shuffle=False,
                                num_workers=4,
                                pin_memory=False)
    elif config["task"] == "adv_eval":
        test_dataset = DiffusionDataset(split="test", fake_only=True, type="LDM")
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                 num_workers=4,
                                 pin_memory=False)
    else:
        test_dataset = DiffusionDataset(split="test", fake_only=False, type="LDM")
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                                 num_workers=4,
                                 pin_memory=False)

    ########################################################
    #                        TRAIN                         #
    ########################################################
    if config["task"] == "train":
        model = D3(config['model_args'], config['train_args'])
        trainer.fit(model, train_loader, val_loader)

    ########################################################
    #                        SALIENCY                      #
    ########################################################
    elif config["task"] == "saliency":
        model = D3.load_from_checkpoint(config['eval_args']['ckpt'][0])
        model.eval()
        model.cuda()
        pbar = tqdm(train_loader, total=len(train_loader), colour='green', leave=True)
        saliencies = []
        count = 0
        for idx, (x, y) in enumerate(pbar):
            x = x.cuda()
            y = y.cuda()
            if y.sum() == 0:
                continue
            else:
                x = x[y == 1]
                y = y[y == 1]
                count += x.shape[0]
            x_adv, grad = carlini_wagner_l2(model, x, y, c=config['saliency_args']['c'],
                                            iterations=config['saliency_args']['steps'])
            saliency = torch.abs(grad) * torch.abs(x_adv - x)
            saliencies.append(saliency.detach().cpu())
            pbar.set_description("Count: %d" % count)
            if count > config['saliency_args']['num_samples']:
                break

        saliencies = torch.cat(saliencies, dim=0).mean(dim=0)
        torch.save(saliencies, os.path.join(config['eval_args']['ckpt'][0].split('.ckpt')[0] + '_saliency.pt'))
        for ensemble_size in tqdm(range(2, config['saliency_args']['max_ensemble_size'] + 1)):
            masks = [torch.zeros(3, 224, 224) for _ in range(ensemble_size)]
            list_indices = []
            for i in range(3):
                for j in range(224):
                    for k in range(224):
                        list_indices.append((saliencies[i, j, k], i, j, k))
            list_indices.sort()
            for l, (v, i, j, k) in enumerate(tqdm(list_indices)):
                masks[l % ensemble_size][i, j, k] = 1
            for i in range(ensemble_size):
                save_dir = os.path.join('/', *config['eval_args']['ckpt'][0].split('/')[:-2])
                save_dir = os.path.join(save_dir, 'masks', 'd3s{}'.format(ensemble_size))
                os.makedirs(save_dir, exist_ok=True)
                torch.save(masks[i], os.path.join(save_dir, 'd3s{}_mask_{}.pt'.format(ensemble_size, i)))
                torchvision.utils.save_image(masks[i],
                                             os.path.join(save_dir, 'd3s{}_mask_{}.png'.format(ensemble_size, i)))

    ########################################################
    #                        EVAL                          #
    ########################################################
    elif config["task"] == "nat_eval":
        models = [D3.load_from_checkpoint(ck) for ck in config['eval_args']['ckpt']]
        for m in models: m.eval()
        for m in models: m.cuda()
        thresholds = [th for th in config['eval_args']['threshold']]
        model = D3_HL(models, thresholds)
        y_true = []
        y_pred_prob = []
        pbar = tqdm(test_loader, total=len(test_loader), colour='green', leave=True)
        with torch.no_grad():
            for idx, (x, y) in enumerate(pbar):
                x = x.cuda()
                y = y.cuda()
                seed_randomness(0)
                ensemble_prob = model(x, mode='soft_avg')
                y_pred_prob.extend(F.softmax(ensemble_prob, dim=1)[:, 1].cpu().numpy().tolist())
                y_true.extend(y.cpu().numpy().tolist())
                pbar.set_description(f"AP: {metrics.average_precision_score(y_true, y_pred_prob):.4f}")

    elif config["task"] == "adv_eval":
        models = [D3.load_from_checkpoint(ck) for ck in config['eval_args']['ckpt']]
        for m in models: m.eval()
        for m in models: m.cuda()
        thresholds = [th for th in config['eval_args']['threshold']]
        model = D3_HL(models, thresholds)
        os.makedirs(config['eval_args']['log_dir'], exist_ok=True)
        attack_loader(model, test_loader,
                      attack=config['eval_args']['attack'],
                      eps=config['eval_args']['eps'],
                      budget=config['eval_args']['budget'],
                      start_idx=args.start_idx,
                      end_idx=args.end_idx,
                      log_file=os.path.join(config['eval_args']['log_dir'],
                                            '{}_{}.txt'.format(args.start_idx, args.end_idx)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config')
    parser.add_argument('--start_idx', type=int)
    parser.add_argument('--end_idx', type=int)
    args = parser.parse_args()
    print(args)
    main(args)
