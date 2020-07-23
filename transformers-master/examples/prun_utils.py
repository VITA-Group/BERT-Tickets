import torch.nn.utils.prune as prune
import numpy as np  
import torch  

def see_weight_varience(pre_weight):
    recover_dict = []
    for ii in range(12):
        recover_dict.append(pre_weight['bert.encoder.layer.'+str(ii)+'.attention.self.query.weight'].view(-1))
        recover_dict.append(pre_weight['bert.encoder.layer.'+str(ii)+'.attention.self.key.weight'].view(-1))
        recover_dict.append(pre_weight['bert.encoder.layer.'+str(ii)+'.attention.self.value.weight'].view(-1))
        recover_dict.append(pre_weight['bert.encoder.layer.'+str(ii)+'.attention.output.dense.weight'].view(-1))
        recover_dict.append(pre_weight['bert.encoder.layer.'+str(ii)+'.intermediate.dense.weight'].view(-1))
        recover_dict.append(pre_weight['bert.encoder.layer.'+str(ii)+'.output.dense.weight'].view(-1))
    recover_dict.append(pre_weight['bert.pooler.dense.weight'].view(-1))

    weight = torch.cat(recover_dict, dim=0)
    print(weight.size())
    print(torch.sqrt(torch.var(weight)))

def adding_noise(pre_weight, noise):
    recover_dict = {}
    for ii in range(12):
        recover_dict['bert.encoder.layer.'+str(ii)+'.attention.self.query.weight'] = pre_weight['bert.encoder.layer.'+str(ii)+'.attention.self.query.weight']
        recover_dict['bert.encoder.layer.'+str(ii)+'.attention.self.key.weight'] = pre_weight['bert.encoder.layer.'+str(ii)+'.attention.self.key.weight']
        recover_dict['bert.encoder.layer.'+str(ii)+'.attention.self.value.weight'] = pre_weight['bert.encoder.layer.'+str(ii)+'.attention.self.value.weight']
        recover_dict['bert.encoder.layer.'+str(ii)+'.attention.output.dense.weight'] = pre_weight['bert.encoder.layer.'+str(ii)+'.attention.output.dense.weight']
        recover_dict['bert.encoder.layer.'+str(ii)+'.intermediate.dense.weight'] = pre_weight['bert.encoder.layer.'+str(ii)+'.intermediate.dense.weight']
        recover_dict['bert.encoder.layer.'+str(ii)+'.output.dense.weight'] = pre_weight['bert.encoder.layer.'+str(ii)+'.output.dense.weight']
    recover_dict['bert.pooler.dense.weight'] = pre_weight['bert.pooler.dense.weight']

    for key in recover_dict.keys():
        print(key)
        weight_key = recover_dict[key]
        weight_key = weight_key + torch.randn_like(weight_key)*noise
        recover_dict[key] = weight_key

    return recover_dict

def rewind(pre_weight):
    recover_dict = {}
    for ii in range(12):
        recover_dict['bert.encoder.layer.'+str(ii)+'.attention.self.query.weight_orig'] = pre_weight['bert.encoder.layer.'+str(ii)+'.attention.self.query.weight']
        recover_dict['bert.encoder.layer.'+str(ii)+'.attention.self.key.weight_orig'] = pre_weight['bert.encoder.layer.'+str(ii)+'.attention.self.key.weight']
        recover_dict['bert.encoder.layer.'+str(ii)+'.attention.self.value.weight_orig'] = pre_weight['bert.encoder.layer.'+str(ii)+'.attention.self.value.weight']
        recover_dict['bert.encoder.layer.'+str(ii)+'.attention.output.dense.weight_orig'] = pre_weight['bert.encoder.layer.'+str(ii)+'.attention.output.dense.weight']
        recover_dict['bert.encoder.layer.'+str(ii)+'.intermediate.dense.weight_orig'] = pre_weight['bert.encoder.layer.'+str(ii)+'.intermediate.dense.weight']
        recover_dict['bert.encoder.layer.'+str(ii)+'.output.dense.weight_orig'] = pre_weight['bert.encoder.layer.'+str(ii)+'.output.dense.weight']
    # recover_dict['bert.pooler.dense.weight_orig'] = pre_weight['bert.pooler.dense.weight']

    return recover_dict

def rewind_first(pre_weight):
    recover_dict = {}
    for ii in range(12):
        recover_dict['bert.encoder.layer.'+str(ii)+'.attention.self.query.weight'] = pre_weight['bert.encoder.layer.'+str(ii)+'.attention.self.query.weight']
        recover_dict['bert.encoder.layer.'+str(ii)+'.attention.self.key.weight'] = pre_weight['bert.encoder.layer.'+str(ii)+'.attention.self.key.weight']
        recover_dict['bert.encoder.layer.'+str(ii)+'.attention.self.value.weight'] = pre_weight['bert.encoder.layer.'+str(ii)+'.attention.self.value.weight']
        recover_dict['bert.encoder.layer.'+str(ii)+'.attention.output.dense.weight'] = pre_weight['bert.encoder.layer.'+str(ii)+'.attention.output.dense.weight']
        recover_dict['bert.encoder.layer.'+str(ii)+'.intermediate.dense.weight'] = pre_weight['bert.encoder.layer.'+str(ii)+'.intermediate.dense.weight']
        recover_dict['bert.encoder.layer.'+str(ii)+'.output.dense.weight'] = pre_weight['bert.encoder.layer.'+str(ii)+'.output.dense.weight']
    recover_dict['bert.pooler.dense.weight'] = pre_weight['bert.pooler.dense.weight']

    return recover_dict

def rewind_orig(pre_weight):
    recover_dict = {}
    for ii in range(12):
        recover_dict['bert.encoder.layer.'+str(ii)+'.attention.self.query.weight_orig'] = pre_weight['bert.encoder.layer.'+str(ii)+'.attention.self.query.weight_orig']
        recover_dict['bert.encoder.layer.'+str(ii)+'.attention.self.key.weight_orig'] = pre_weight['bert.encoder.layer.'+str(ii)+'.attention.self.key.weight_orig']
        recover_dict['bert.encoder.layer.'+str(ii)+'.attention.self.value.weight_orig'] = pre_weight['bert.encoder.layer.'+str(ii)+'.attention.self.value.weight_orig']
        recover_dict['bert.encoder.layer.'+str(ii)+'.attention.output.dense.weight_orig'] = pre_weight['bert.encoder.layer.'+str(ii)+'.attention.output.dense.weight_orig']
        recover_dict['bert.encoder.layer.'+str(ii)+'.intermediate.dense.weight_orig'] = pre_weight['bert.encoder.layer.'+str(ii)+'.intermediate.dense.weight_orig']
        recover_dict['bert.encoder.layer.'+str(ii)+'.output.dense.weight_orig'] = pre_weight['bert.encoder.layer.'+str(ii)+'.output.dense.weight_orig']
    recover_dict['bert.pooler.dense.weight_orig'] = pre_weight['bert.pooler.dense.weight_orig']

    return recover_dict

def rewind_distribution(pre_weight):
    recover_dict = {}
    for ii in range(12):
        recover_dict['module.bert.encoder.layer.'+str(ii)+'.attention.self.query.weight_orig'] = pre_weight['bert.encoder.layer.'+str(ii)+'.attention.self.query.weight']
        recover_dict['module.bert.encoder.layer.'+str(ii)+'.attention.self.key.weight_orig'] = pre_weight['bert.encoder.layer.'+str(ii)+'.attention.self.key.weight']
        recover_dict['module.bert.encoder.layer.'+str(ii)+'.attention.self.value.weight_orig'] = pre_weight['bert.encoder.layer.'+str(ii)+'.attention.self.value.weight']
        recover_dict['module.bert.encoder.layer.'+str(ii)+'.attention.output.dense.weight_orig'] = pre_weight['bert.encoder.layer.'+str(ii)+'.attention.output.dense.weight']
        recover_dict['module.bert.encoder.layer.'+str(ii)+'.intermediate.dense.weight_orig'] = pre_weight['bert.encoder.layer.'+str(ii)+'.intermediate.dense.weight']
        recover_dict['module.bert.encoder.layer.'+str(ii)+'.output.dense.weight_orig'] = pre_weight['bert.encoder.layer.'+str(ii)+'.output.dense.weight']
    recover_dict['module.bert.pooler.dense.weight_orig'] = pre_weight['bert.pooler.dense.weight']

    return recover_dict

def pruning_model(model,px):

    parameters_to_prune =[]
    for ii in range(12):
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.query, 'weight'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.key, 'weight'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.value, 'weight'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.output.dense, 'weight'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].intermediate.dense, 'weight'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].output.dense, 'weight'))

    # parameters_to_prune.append((model.bert.pooler.dense, 'weight'))
    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def pruning_model_random(model,px):

    parameters_to_prune =[]
    for ii in range(12):
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.query, 'weight'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.key, 'weight'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.value, 'weight'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.output.dense, 'weight'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].intermediate.dense, 'weight'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].output.dense, 'weight'))

    # parameters_to_prune.append((model.bert.pooler.dense, 'weight'))
    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.RandomUnstructured,
        amount=px,
    )

def pruning_model_custom(model, mask):

    parameters_to_prune =[]
    for ii in range(12):
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.query, mask['bert.encoder.layer.'+str(ii)+'.attention.self.query']))
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.key, mask['bert.encoder.layer.'+str(ii)+'.attention.self.key']))
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.value, mask['bert.encoder.layer.'+str(ii)+'.attention.self.value']))
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.output.dense, mask['bert.encoder.layer.'+str(ii)+'.attention.output.dense']))
        parameters_to_prune.append((model.bert.encoder.layer[ii].intermediate.dense, mask['bert.encoder.layer.'+str(ii)+'.intermediate.dense']))
        parameters_to_prune.append((model.bert.encoder.layer[ii].output.dense, mask['bert.encoder.layer.'+str(ii)+'.output.dense']))
    parameters_to_prune.append((model.bert.pooler.dense, mask['bert.pooler.dense']))

    for idx in range(len(parameters_to_prune)):
        prune.CustomFromMask.apply(parameters_to_prune[idx][0], 'weight', mask=parameters_to_prune[idx][1])

def remove_prune_model_custom(model):

    parameters_to_prune =[]
    for ii in range(12):
        parameters_to_prune.append(model.bert.encoder.layer[ii].attention.self.query)
        parameters_to_prune.append(model.bert.encoder.layer[ii].attention.self.key)
        parameters_to_prune.append(model.bert.encoder.layer[ii].attention.self.value)
        parameters_to_prune.append(model.bert.encoder.layer[ii].attention.output.dense)
        parameters_to_prune.append(model.bert.encoder.layer[ii].intermediate.dense)
        parameters_to_prune.append(model.bert.encoder.layer[ii].output.dense)
    parameters_to_prune.append(model.bert.pooler.dense)

    for idx in range(len(parameters_to_prune)):
        prune.remove(parameters_to_prune[idx], 'weight')

def pruning_model_distribution(model,px):

    parameters_to_prune =[]
    for ii in range(12):
        parameters_to_prune.append((model.module.bert.encoder.layer[ii].attention.self.query, 'weight'))
        parameters_to_prune.append((model.module.bert.encoder.layer[ii].attention.self.key, 'weight'))
        parameters_to_prune.append((model.module.bert.encoder.layer[ii].attention.self.value, 'weight'))
        parameters_to_prune.append((model.module.bert.encoder.layer[ii].attention.output.dense, 'weight'))
        parameters_to_prune.append((model.module.bert.encoder.layer[ii].intermediate.dense, 'weight'))
        parameters_to_prune.append((model.module.bert.encoder.layer[ii].output.dense, 'weight'))

    parameters_to_prune.append((model.module.bert.pooler.dense, 'weight'))
    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def see_weight_rate(model):

    sum_list = 0
    zero_sum = 0
    for ii in range(12):
        sum_list = sum_list+float(model.bert.encoder.layer[ii].attention.self.query.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].attention.self.query.weight == 0))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].attention.self.key.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].attention.self.key.weight == 0))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].attention.self.value.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].attention.self.value.weight == 0))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].attention.output.dense.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].attention.output.dense.weight == 0))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].intermediate.dense.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].intermediate.dense.weight == 0))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].output.dense.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].output.dense.weight == 0))


    # sum_list = sum_list+float(model.bert.pooler.dense.weight.nelement())
    # zero_sum = zero_sum+float(torch.sum(model.bert.pooler.dense.weight == 0))
 

    return 100*zero_sum/sum_list

def see_weight_rate_distribution(model):

    sum_list = 0
    zero_sum = 0
    for ii in range(12):
        sum_list = sum_list+float(model.module.bert.encoder.layer[ii].attention.self.query.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.module.bert.encoder.layer[ii].attention.self.query.weight == 0))

        sum_list = sum_list+float(model.module.bert.encoder.layer[ii].attention.self.key.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.module.bert.encoder.layer[ii].attention.self.key.weight == 0))

        sum_list = sum_list+float(model.module.bert.encoder.layer[ii].attention.self.value.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.module.bert.encoder.layer[ii].attention.self.value.weight == 0))

        sum_list = sum_list+float(model.module.bert.encoder.layer[ii].attention.output.dense.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.module.bert.encoder.layer[ii].attention.output.dense.weight == 0))

        sum_list = sum_list+float(model.module.bert.encoder.layer[ii].intermediate.dense.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.module.bert.encoder.layer[ii].intermediate.dense.weight == 0))

        sum_list = sum_list+float(model.module.bert.encoder.layer[ii].output.dense.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.module.bert.encoder.layer[ii].output.dense.weight == 0))


    sum_list = sum_list+float(model.module.bert.pooler.dense.weight.nelement())
    zero_sum = zero_sum+float(torch.sum(model.module.bert.pooler.dense.weight == 0))
 

    return 100*zero_sum/sum_list



