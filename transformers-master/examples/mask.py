from transformers import BertForMaskedLM, BertForSequenceClassification, BertForQuestionAnswering
from transformers import BertConfig
import torch.nn.utils.prune as prune
import numpy as np  
import torch  



def pruning_model(model,px):

    parameters_to_prune =[]
    for ii in range(12):
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.query, 'weight'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.key, 'weight'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.value, 'weight'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.output.dense, 'weight'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].intermediate.dense, 'weight'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].output.dense, 'weight'))

    parameters_to_prune.append((model.bert.pooler.dense, 'weight'))
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


    sum_list = sum_list+float(model.bert.pooler.dense.weight.nelement())
    zero_sum = zero_sum+float(torch.sum(model.bert.pooler.dense.weight == 0))
 

    return 100*zero_sum/sum_list


config = BertConfig.from_pretrained(
    'bert-base-uncased'
)

list_all = [['qqp',0.7],['qnli',0.5],['mrpc',0.4],['sst2',0.1],['cola',0.4]]



# rand = True
# for kkk in list_all:

#     if rand:
#         print('random')
#         model = BertForSequenceClassification(config=config)
#         output = 'random_mask2/'

#     else:
#         model = BertForSequenceClassification.from_pretrained(
#             'bert-base-uncased',
#             from_tf=bool(".ckpt" in 'bert-base-uncased'),
#             config=config
#         )
#         output = 'pretrain_mask2/'

#     pruning_model(model, kkk[1])
#     zero = see_weight_rate(model)
#     print(kkk, zero)

#     mask_dict = {}
#     model_dict = model.state_dict()
#     for key in model_dict.keys():
#         if 'mask' in key:
#             mask_dict[key] = model_dict[key]
#             # print(key)

#     torch.save(mask_dict, output+kkk[0]+'.pt')


# if rand:
#     print('random')
#     model = BertForQuestionAnswering(config = config)
#     output = 'random_mask/'

# else:

#     model = BertForQuestionAnswering.from_pretrained(
#         'bert-base-uncased',
#         from_tf=bool(".ckpt" in 'bert-base-uncased'),
#         config=config
#     )
#     output = 'pretrain_mask/'

# pruning_model(model, 0.7)
# zero = see_weight_rate(model)
# print('squad', zero)

# mask_dict = {}
# model_dict = model.state_dict()
# for key in model_dict.keys():
#     if 'mask' in key:
#         mask_dict[key] = model_dict[key]
#         # print(key)

# torch.save(mask_dict, output+'squad.pt')


# if rand:
#     print('random')
#     model = BertForMaskedLM(config = config)
#     output = 'random_mask2/'

# else:

#     model = BertForMaskedLM.from_pretrained(
#         'bert-base-uncased',
#         from_tf=bool(".ckpt" in 'bert-base-uncased'),
#         config=config
#     )
#     output = 'pretrain_mask2/'


# pruning_model(model, 0.5)
# zero = see_weight_rate(model)
# print('pretrain', zero)

# mask_dict = {}
# model_dict = model.state_dict()
# for key in model_dict.keys():
#     if 'mask' in key:
#         mask_dict[key] = model_dict[key]
#         # print(key)

# torch.save(mask_dict, output+'pretrain.pt')



rand = False
# for kkk in list_all:

#     if rand:
#         print('random')
#         model = BertForSequenceClassification(config=config)
#         output = 'random_mask2/'

#     else:
#         model = BertForSequenceClassification.from_pretrained(
#             'bert-base-uncased',
#             from_tf=bool(".ckpt" in 'bert-base-uncased'),
#             config=config
#         )
#         output = 'pretrain_mask2/'

#     pruning_model(model, kkk[1])
#     zero = see_weight_rate(model)
#     print(kkk, zero)

#     mask_dict = {}
#     model_dict = model.state_dict()
#     for key in model_dict.keys():
#         if 'mask' in key:
#             mask_dict[key] = model_dict[key]
#             # print(key)

#     torch.save(mask_dict, output+kkk[0]+'.pt')


# if rand:
#     print('random')
#     model = BertForQuestionAnswering(config = config)
#     output = 'random_mask/'

# else:

#     model = BertForQuestionAnswering.from_pretrained(
#         'bert-base-uncased',
#         from_tf=bool(".ckpt" in 'bert-base-uncased'),
#         config=config
#     )
#     output = 'pretrain_mask/'

# pruning_model(model, 0.7)
# zero = see_weight_rate(model)
# print('squad', zero)

# mask_dict = {}
# model_dict = model.state_dict()
# for key in model_dict.keys():
#     if 'mask' in key:
#         mask_dict[key] = model_dict[key]
#         print(key)

# torch.save(mask_dict, output+'squad.pt')


if rand:
    print('random')
    model = BertForMaskedLM(config = config)
    output = 'random_mask2/'

else:

    model = BertForMaskedLM.from_pretrained(
        'bert-base-uncased',
        from_tf=bool(".ckpt" in 'bert-base-uncased'),
        config=config
    )
    output = 'pretrain_mask2/'


pruning_model(model, 0.5)
zero = see_weight_rate(model)
print('pretrain', zero)

mask_dict = {}
model_dict = model.state_dict()
for key in model_dict.keys():
    if 'mask' in key:
        mask_dict[key] = model_dict[key]
        # print(key)

torch.save(mask_dict, output+'pretrain.pt')


