import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch_dct as dct
from torch_dct import idct_2d, dct_2d
from tqdm import tqdm


def carlini_wagner_l2(model, x, y, c=10, kappa=0, iterations=1000, learning_rate=0.01):
    w = torch.zeros_like(x, requires_grad=True)
    perturbation = (1 / 2) * (torch.tanh(w) + 1) - x
    optimizer = torch.optim.Adam([w], lr=learning_rate)
    best_adv = None
    for t in range(iterations):
        optimizer.zero_grad()
        perturbation = (1 / 2) * (torch.tanh(w) + 1) - x
        x_adv = x + perturbation
        assert torch.max(x_adv) <= 1 and torch.min(x_adv) >= 0
        logits = model(x_adv)
        real = logits[torch.arange(logits.size(0)), y]
        other = torch.topk(logits, 2, dim=1)[0][:, 1]
        assert other.shape == real.shape
        # optimize for making this class least likely. There are alot of ways to do this depending on the target class.
        classification_loss = torch.clamp(real - other + kappa, min=0.)
        norm_loss = torch.linalg.norm(torch.flatten(perturbation, start_dim=1), dim=1)
        loss = norm_loss + c * classification_loss
        loss.sum().backward()
        optimizer.step()

        # update best result to current result only if current result is misclassified
        if best_adv is None:
            best_adv = torch.clamp(x + perturbation, 0, 1).detach()
        else:
            candidate_adv = torch.clamp(x + perturbation, 0, 1).detach()
            candidate_logits = model(candidate_adv)
            best_adv[candidate_logits.argmax(dim=1) != y] = candidate_adv[candidate_logits.argmax(dim=1) != y]
        if t % 1 == 0:
            print("Iteration: {} | best norm: {}".format(t, torch.linalg.norm(torch.flatten(best_adv - x, start_dim=1),
                                                                              dim=1)))

    best_adv = best_adv.detach().requires_grad_(True)
    logits = model(best_adv)
    real = logits[torch.arange(logits.size(0)), y]
    other = torch.topk(logits, 2, dim=1)[0][:, 1]
    assert other.shape == real.shape
    classification_loss = torch.clamp(real - other + kappa, min=0.)
    norm_loss = torch.linalg.norm(torch.flatten(best_adv - x, start_dim=1), dim=1)
    loss = norm_loss + c * classification_loss
    grad = torch.autograd.grad(loss.sum(), best_adv)[0]

    return best_adv, grad


def surfree(model, x, y, eps=0.05, steps=np.iinfo(np.int32).max, theta_max=30, bs_gamma=0.01, bs_max_iter=10,
            freq_range=(0, 0.5),
            n_ortho=10, rho=0.98, eval_per_direction=1):
    dim = torch.prod(torch.tensor(x.shape[1:]))

    def is_adversarial(x):
        logits = model(x)
        return (logits.argmax(dim=1) != y).float()

    def binary_search_to_boundary(x, x_adv):
        high = 1
        low = 0
        threshold = bs_gamma / (dim * torch.sqrt(dim))
        boost_start = torch.clamp(0.2 * x + 0.8 * x_adv, 0, 1)
        if is_adversarial(boost_start) == 1:
            x_adv = boost_start
        iters = 0
        while high - low > threshold and iters < bs_max_iter:
            middle = (high + low) / 2
            interpolated = (1 - middle) * x + middle * x_adv
            if is_adversarial(interpolated) == 1:
                high = middle
            else:
                low = middle
            iters += 1
        interpolated = (1 - high) * x + high * x_adv
        return interpolated

    def dct2_8_8(image, mask):
        assert mask.shape[-2:] == (8, 8)
        imsize = image.shape
        reshaped = image.reshape(-1, *imsize[1:2], 8, 8)
        dct_reshaped = dct_2d(reshaped)
        masked = dct_reshaped * mask
        dct = masked.reshape(imsize)

        return dct

    def idct2_8_8(image):
        imsize = image.shape
        reshaped = image.reshape(-1, *imsize[1:2], 8, 8)
        dct_reshaped = idct_2d(reshaped)
        masked = dct_reshaped
        dct = masked.reshape(imsize)
        return dct

    def get_zig_zag_mask(mask_size):
        total_components = mask_size[0] * mask_size[1]
        n_coeff_kept = int(total_components * min(1, freq_range[1]))
        n_coeff_to_start = int(total_components * max(0, freq_range[0]))
        mask_size = (x.shape[0], x.shape[1], mask_size[0], mask_size[1])
        zig_zag_mask = torch.zeros(mask_size)
        s = 0
        while n_coeff_kept > 0:
            for i in range(min(s + 1, mask_size[2])):
                for j in range(min(s + 1, mask_size[3])):
                    if i + j == s:
                        if n_coeff_to_start > 0:
                            n_coeff_to_start -= 1
                            continue
                        if s % 2:
                            zig_zag_mask[:, :, i, j] = 1
                        else:
                            zig_zag_mask[:, :, j, i] = 1
                        n_coeff_kept -= 1
                        if n_coeff_kept == 0:
                            return zig_zag_mask
            s += 1
        return zig_zag_mask

    def step_in_circular_direction(dir1, dir2, r, degree):
        degree = degree.reshape(degree.shape + (1,) * (len(dir1.shape) - len(degree.shape)))
        r = r.reshape(r.shape + (1,) * (len(dir1.shape) - len(r.shape)))
        result = dir1 * torch.cos(degree * np.pi / 180) + dir2 * torch.sin(degree * np.pi / 180)
        result = result * r * torch.cos(degree * np.pi / 180)
        return result

    def gram_schmidt(v, orthogonal_with):
        v_repeated = torch.cat([v] * len(orthogonal_with), axis=0)
        gs_coeff = (orthogonal_with * v_repeated).flatten(1).sum(1)
        proj = gs_coeff.reshape(
            gs_coeff.shape + (1,) * (len(orthogonal_with.shape) - len(gs_coeff.shape))) * orthogonal_with
        v = v - proj.sum(0)
        return v

    # Initialize
    x_adv = x.clone()
    while model(x_adv).argmax(dim=1) == y:
        x_adv = torch.clamp(x_adv + torch.randn_like(x_adv) * 0.5, 0, 1)
    x_adv = binary_search_to_boundary(x, x_adv)
    explored_orthogonal_directions = ((x_adv - x) / torch.linalg.norm(x_adv - x))

    norm_dist = torch.linalg.norm(x_adv - x) / (x.shape[-1] * x.shape[-2] * x.shape[-3]) ** 0.5
    history = []

    # Attack
    pbar = tqdm(range(steps))
    import time
    start_time = time.time()
    zig_mask = get_zig_zag_mask((8, 8)).to(x.device)
    for t in pbar:
        iteration_start_queries_expended = model.get_queries()
        # get candidates
        epsilon = 0
        stuck_counter = 0
        while epsilon == 0:
            # get new direction
            probs = torch.FloatTensor(size=x.shape).uniform_(0, 3).long().to(x.device) - 1
            dcts = torch.tanh(dct2_8_8(x, zig_mask))
            new_direction = idct2_8_8(dcts * probs) + torch.FloatTensor(size=x.shape).normal_(std=0).to(x.device)
            new_direction = gram_schmidt(new_direction, explored_orthogonal_directions)
            new_direction = new_direction / torch.linalg.norm(new_direction)

            explored_orthogonal_directions = torch.cat((
                explored_orthogonal_directions[:1],
                explored_orthogonal_directions[1 + len(explored_orthogonal_directions) - n_ortho:],
                new_direction), dim=0)
            # get best angle
            direction = ((x_adv - x) / torch.linalg.norm(x_adv - x))
            evolution_function = lambda degree: torch.clamp(
                x + step_in_circular_direction(direction, new_direction, torch.linalg.norm(x_adv - x), degree), 0, 1)
            coefficients = torch.zeros(2 * eval_per_direction).to(x.device)

            for i in range(0, eval_per_direction):
                coefficients[2 * i] = 1 - (i / eval_per_direction)
                coefficients[2 * i + 1] = - coefficients[2 * i]
            best_epsilon = 0
            for coeff in coefficients:
                possible_best_epsilon = coeff * theta_max
                x_evolved = evolution_function(possible_best_epsilon)
                if best_epsilon == 0 and is_adversarial(x_evolved) == 1:
                    best_epsilon = possible_best_epsilon
                if best_epsilon != 0:
                    break

            if best_epsilon == 0:
                theta_max = theta_max * rho
            if best_epsilon != 0 and epsilon == 0:
                theta_max = theta_max / rho
                epsilon = best_epsilon
            norm_dist = torch.linalg.norm(x_adv - x) / (x.shape[-1] * x.shape[-2] * x.shape[-3]) ** 0.5
            pbar.set_description(
                f"Step {t}: Norm L2: {norm_dist} | Queries: {model.get_queries()} | Stuck counter: {stuck_counter} / 200")
            stuck_counter += 1

        evolution_function = lambda degree: torch.clamp(
            x + step_in_circular_direction(direction, new_direction, torch.linalg.norm(x_adv - x), degree), 0, 1)

        # alpha binary search
        check_opposite = epsilon > 0
        lower = epsilon
        if abs(lower) != theta_max:
            upper = lower + torch.sign(lower) * theta_max / eval_per_direction
        else:
            upper = 0
        max_angle = 180
        keep_going = upper == 0
        while keep_going:
            new_upper = lower + torch.sign(lower) * theta_max / eval_per_direction
            new_upper = min(new_upper, max_angle)
            x_evolved_new_upper = evolution_function(new_upper)
            if is_adversarial(x_evolved_new_upper) == 1:
                lower = new_upper
            else:
                upper = new_upper
                keep_going = False
        step = 0
        over_gamma = abs(torch.cos(lower * np.pi / 180) - torch.cos(upper * np.pi / 180)) > bs_gamma
        while step < bs_max_iter and over_gamma:
            mid_bound = (upper + lower) / 2
            if mid_bound != 0:
                mid = evolution_function(mid_bound)
            else:
                mid = torch.zeros_like(x)
            is_mid_adversarial = is_adversarial(mid)
            if check_opposite:
                mid_opp = evolution_function(-mid_bound)
            else:
                mid_opp = torch.zeros_like(x)
            is_mid_opp_adversarial = is_adversarial(mid_opp)
            if is_mid_adversarial:
                lower = mid_bound
            elif not is_mid_adversarial and check_opposite and is_mid_opp_adversarial:
                lower = -mid_bound
            if not is_mid_adversarial and check_opposite and is_mid_opp_adversarial:
                upper = -upper
            if abs(lower) != abs(mid_bound):
                upper = mid_bound
            check_opposite = check_opposite and is_mid_opp_adversarial and lower > 0
            over_gamma = abs(torch.cos(lower * np.pi / 180) - torch.cos(upper * np.pi / 180)) > bs_gamma
            step += 1
        epsilon = lower
        # end alpha binary search

        candidate = evolution_function(epsilon)
        prev_norm_dist = norm_dist
        if torch.linalg.norm(candidate - x) < torch.linalg.norm(x_adv - x):
            x_adv = candidate

        # Logging current progress with normalized L2 distance
        norm_dist = torch.linalg.norm(x_adv - x) / (x.shape[-1] * x.shape[-2] * x.shape[-3]) ** 0.5
        pbar.set_description(f"Step {t}: Norm L2: {norm_dist} | Queries: {model.get_queries()}")
        if norm_dist < eps:
            return x_adv

        iteration_end_queries_expended = model.get_queries()
        history.append(
            (prev_norm_dist - norm_dist) / (iteration_end_queries_expended - iteration_start_queries_expended))
        history = history[-10:]

    # final binary search
    x_adv = binary_search_to_boundary(x, x_adv)
    norm_dist = torch.linalg.norm(x_adv - x) / (x.shape[-1] * x.shape[-2] * x.shape[-3]) ** 0.5
    if norm_dist < eps:
        return x_adv
    else:
        return x
