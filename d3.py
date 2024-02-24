import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from models.image_models import RensetNoDownClassifier
import pytorch_lightning as pl
from torch.nn import ModuleList
from trades import trades_loss
import torchvision
from sklearn.metrics import average_precision_score
import random


class D3(pl.LightningModule):
    def __init__(self, model_args, train_args):
        super().__init__()
        self.save_hyperparameters()
        self.models = []
        for model_idx in range(len(self.hparams.model_args['masks'])):
            if self.hparams.model_args["arch"] == "resnet":
                model = RensetNoDownClassifier(self.hparams.model_args['mean'],
                                               self.hparams.model_args['var'],
                                               self.hparams.model_args['masks'][model_idx])
            self.models.append(model)
        self.models = ModuleList(self.models)

    def _avg_logits(self, x):
        if len(self.models) == 1:
            avg_logits = self.models[0](x)
        else:
            logits_list = []
            for model in self.models:
                logits = model(x)
                logits_list.append(logits)
            avg_logits = torch.log(torch.mean(torch.stack([F.softmax(l, dim=-1) for l in logits_list]), dim=0))
        return avg_logits

    def forward(self, x):
        return self._avg_logits(x)

    def configure_optimizers(self):
        all_params = []
        for model in self.models:
            all_params += list(model.parameters())
        optimizer = torch.optim.Adam(all_params, lr=self.hparams.train_args['lr'])
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = []
        for model in self.models:
            model.train()
            l = trades_loss(model=model,
                            x_natural=x,
                            y=y,
                            step_size=self.hparams.train_args['adv_train']['step_size'],
                            epsilon=self.hparams.train_args['adv_train']['eps'],
                            perturb_steps=self.hparams.train_args['adv_train']['steps'],
                            beta=1,
                            distance='l_2')
            loss.append(l)
        loss = torch.sum(torch.stack(loss))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = []
        with torch.enable_grad():
            for model in self.models:
                loss.append(F.cross_entropy(model(x), y))
        loss = torch.sum(torch.stack(loss))
        logits = self(x)

        try:
            nat_acc = average_precision_score(y.cpu(), F.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
        except:
            nat_acc = 0

        try:
            with torch.enable_grad():
                x_adv = self.l2_pgd(self.models[0], x, y)
                pred = torch.argmax(self(x_adv), dim=1)
                adv_acc = torch.sum(pred == y).float() / len(y)
        except:
            adv_acc = 0
        self.log('val_nat_acc', nat_acc)
        self.log('val_adv_acc', adv_acc)
        self.log('val_loss', loss)
        return loss

    def l2_pgd(self, model, x, y):
        batch_size = len(x)
        criterion_kl = nn.KLDivLoss(size_average=False)
        delta = 0.001 * torch.randn(x.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=self.hparams.train_args['adv_train']['eps'] /
                                                self.hparams.train_args['adv_train']['steps'] * 2)

        for _ in range(self.hparams.train_args['adv_train']['steps']):
            adv = x + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x)
            delta.data.clamp_(0, 1).sub_(x)
            delta.data.renorm_(p=2, dim=0, maxnorm=self.hparams.train_args['adv_train']['eps'])
        x_adv = Variable(x + delta, requires_grad=False)
        return x_adv


class D3_HL(pl.LightningModule):
    def __init__(self, model_list, threshold_list):
        super().__init__()
        self.save_hyperparameters()
        self.models = model_list
        self.thresholds = threshold_list

    def forward(self, x, mode='hard_avg'):
        logits_list = []
        for model in self.models:
            logits = model(x)
            logits_list.append(logits)

        probs = torch.stack([F.softmax(l, dim=-1) for l in logits_list])

        if mode == 'soft_avg':
            probs = torch.mean(probs, dim=0)
            return probs
        elif mode == 'hard_avg':
            probs = torch.mean(probs, dim=0)
            decisions = probs[:, 1:] > self.thresholds[0]
            decisions = decisions.unsqueeze(0)
            at_least_one = torch.cat([torch.sum(decisions, dim=0) <= 0, torch.sum(decisions, dim=0) > 0], dim=-1)
            return at_least_one.long()
