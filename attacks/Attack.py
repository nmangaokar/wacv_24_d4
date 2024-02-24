from copy import deepcopy
import torch
import torchvision
import numpy as np
import attacks
from attacks.attacks import *
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import random
import logging


def seed_randomness(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class BudgetExhaustionError(Exception):
    pass


class Model(torch.nn.Module):
    def __init__(self, model, budget):
        super().__init__()
        self._model = model
        self._queries = 0
        self._budget = budget

    def forward(self, x):
        return self.batch_forward(x)

    def batch_forward(self, x):
        if len(x) > 20:
            out = []
            for i in range(0, len(x), 20):
                out.append(self._model(x[i:i + 20]))
            out = torch.cat(out, dim=0)
        else:
            out = self._model(x)
        self._queries += len(x)
        self._check_budget(self._budget)
        return out

    def get_total_queries(self):
        return self._queries

    def reset(self):
        self._queries = 0

    def get_queries(self):
        return self._queries

    def _check_budget(self, budget):
        if self._queries > budget:
            raise BudgetExhaustionError(
                f'Attack budget exhausted: {self.get_total_queries()} > {budget}')


@torch.no_grad()
def attack_loader(model, loader, attack, eps, budget, start_idx, end_idx, log_file):
    # create logger
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        filename=log_file,
                        level=logging.INFO)

    # Load model
    model = Model(model, budget).eval()

    # Load attack
    try:
        attack = globals()[attack]
    except KeyError:
        raise NotImplementedError(f'Attack {attack_config["attack"]} not implemented.')

    # Run attack and compute adversarial accuracy
    y_true, y_pred = [], []
    pbar = tqdm(loader, colour="yellow")
    for i, (x, y) in enumerate(pbar):
        if i < start_idx:
            continue
        if i >= end_idx:
            break

        x = x.cuda()
        y = y.cuda()

        seed_randomness()
        try:
            if model(x).argmax(dim=1) != y:
                x_adv = x
            else:
                x_adv = attack(model, x, y, eps=eps)
        except BudgetExhaustionError:
            x_adv = x

        x_adv = x_adv.cuda()
        logits = model._model(x_adv)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        true = y.detach().cpu().numpy().tolist()

        y_true.extend(true)
        y_pred.extend(preds)
        pbar.set_description(
            "Running Accuracy: {} ({} / {}) ".format(accuracy_score(y_true, y_pred),
                                                     sum(np.array(y_true) == np.array(y_pred)),
                                                     len(y_true)))
        logging.info(
            f"True Label : {true[0]} | Predicted Label : {preds[0]} | Total Queries : {model.get_total_queries()}")
        model.reset()
    logging.info(f"Final Accuracy : {accuracy_score(y_true, y_pred)}")
    logging.info("Finished Attack")
