import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math

import copy
import types

def GraSP_fetch_data(train_dataloader, iter_times):
    inputs_list = []
    labels_list = []
    for batch in train_dataloader:
        inputs, labels = mask_tokens(batch, tokenizer, args)



def GraSP(net, ratio, train_dataloader, device, num_classes=10, samples_per_class=25, num_iters=1, T=200, reinit=True):
    eps = 1e-10
    keep_ratio = 1-ratio
    old_net = net

    net = copy.deepcopy(net)  # .eval()
    net.zero_grad()

    weights = []

    # total_parameters = count_total_parameters(net)
    # fc_parameters = count_fc_parameters(net)

    # # rescale_weights(net)
    # for layer in net.modules():
    #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    #         if isinstance(layer, nn.Linear) and reinit:
    #             nn.init.xavier_normal(layer.weight)
    #         weights.append(layer.weight)

    for ii in range(12):
        weights.append(model.bert.encoder.layer[ii].attention.self.query.weight)
        weights.append(model.bert.encoder.layer[ii].attention.self.key.weight)
        weights.append(model.bert.encoder.layer[ii].attention.self.value.weight)
        weights.append(model.bert.encoder.layer[ii].attention.output.dense.weight)
        weights.append(model.bert.encoder.layer[ii].intermediate.dense.weight)
        weights.append(model.bert.encoder.layer[ii].output.dense.weight)
    weights.append(model.bert.pooler.dense.weight)


    inputs_one = []
    targets_one = []

    grad_w = None
    for w in weights:
        w.requires_grad_(True)

    print_once = False
    for it in range(num_iters):
        print("(1): Iterations %d/%d." % (it, num_iters))
        inputs, targets = GraSP_fetch_data(train_dataloader, num_classes, samples_per_class)
        N = inputs.shape[0]
        din = copy.deepcopy(inputs)
        dtarget = copy.deepcopy(targets)
        inputs_one.append(din[:N//2])
        targets_one.append(dtarget[:N//2])
        inputs_one.append(din[N // 2:])
        targets_one.append(dtarget[N // 2:])
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = net.forward(inputs[:N//2])/T
        if print_once:
            # import pdb; pdb.set_trace()
            x = F.softmax(outputs)
            print(x)
            print(x.max(), x.min())
            print_once = False
        loss = F.cross_entropy(outputs, targets[:N//2])
        # ===== debug ================
        grad_w_p = autograd.grad(loss, weights)
        if grad_w is None:
            grad_w = list(grad_w_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx]

        outputs = net.forward(inputs[N // 2:])/T
        loss = F.cross_entropy(outputs, targets[N // 2:])
        grad_w_p = autograd.grad(loss, weights, create_graph=False)
        if grad_w is None:
            grad_w = list(grad_w_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx]

    ret_inputs = []
    ret_targets = []

    for it in range(len(inputs_one)):
        print("(2): Iterations %d/%d." % (it, num_iters))
        inputs = inputs_one.pop(0).to(device)
        targets = targets_one.pop(0).to(device)
        ret_inputs.append(inputs)
        ret_targets.append(targets)
        outputs = net.forward(inputs)/T
        loss = F.cross_entropy(outputs, targets)
        # ===== debug ==============

        grad_f = autograd.grad(loss, weights, create_graph=True)
        z = 0
        count = 0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                z += (grad_w[count].data * grad_f[count]).sum()
                count += 1
        z.backward()

    grads = dict()
    old_modules = list(old_net.modules())
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grads[old_modules[idx]] = -layer.weight.data * layer.weight.grad  # -theta_q Hg

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads.values()])
    norm_factor = torch.abs(torch.sum(all_scores)) + eps
    print("** norm factor:", norm_factor)
    all_scores.div_(norm_factor)

    num_params_to_rm = int(len(all_scores) * (1-keep_ratio))
    threshold, _ = torch.topk(all_scores, num_params_to_rm, sorted=True)
    # import pdb; pdb.set_trace()
    acceptable_score = threshold[-1]
    print('** accept: ', acceptable_score)
    keep_masks = dict()
    for m, g in grads.items():
        keep_masks[m] = ((g / norm_factor) <= acceptable_score).float()

    print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))

    return keep_masks




