import torch 
from transformers import BertForMaskedLM, BertForSequenceClassification, BertForQuestionAnswering
from transformers import BertConfig
import numpy as np  

config = BertConfig.from_pretrained(
    'bert-base-uncased'
)

model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        from_tf=bool(".ckpt" in 'bert-base-uncased'),
        config=config
    )

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


# orig_dict = model.state_dict()
# for k in [0.01,0.02,0.05,0.1,0.2,0.5,1]:
#     noise_dict = adding_noise(orig_dict, k*0.0387)
#     print(k)
#     torch.save(noise_dict, str(k)+'pre_weight.pt')

# for key in model.state_dict().keys():
#     print(key, model.state_dict()[key].size())


def shuffle_weight(pre_weight):
    seq = np.random.permutation(12)
    recover_dict = {}
    for ii in range(12):
        recover_dict['bert.encoder.layer.'+str(ii)+'.attention.self.query.weight'] = pre_weight['bert.encoder.layer.'+str(seq[ii])+'.attention.self.query.weight']
        recover_dict['bert.encoder.layer.'+str(ii)+'.attention.self.key.weight'] = pre_weight['bert.encoder.layer.'+str(seq[ii])+'.attention.self.key.weight']
        recover_dict['bert.encoder.layer.'+str(ii)+'.attention.self.value.weight'] = pre_weight['bert.encoder.layer.'+str(seq[ii])+'.attention.self.value.weight']
        recover_dict['bert.encoder.layer.'+str(ii)+'.attention.output.dense.weight'] = pre_weight['bert.encoder.layer.'+str(seq[ii])+'.attention.output.dense.weight']
        recover_dict['bert.encoder.layer.'+str(ii)+'.intermediate.dense.weight'] = pre_weight['bert.encoder.layer.'+str(seq[ii])+'.intermediate.dense.weight']
        recover_dict['bert.encoder.layer.'+str(ii)+'.output.dense.weight'] = pre_weight['bert.encoder.layer.'+str(seq[ii])+'.output.dense.weight']
    recover_dict['bert.pooler.dense.weight'] = pre_weight['bert.pooler.dense.weight']

    # for key in recover_dict.keys():
    #     print(key)
    #     weight_key = recover_dict[key]
    #     weight_key = weight_key + torch.randn_like(weight_key)*noise
    #     recover_dict[key] = weight_key

    return recover_dict

new_dict = shuffle_weight(model.state_dict())

torch.save(new_dict, 'shuffle_weight.pt')

dict_new = torch.load('tmp/shuffle_weight.pt')

model_dict = model.state_dict()
model_dict.update(dict_new)
model.load_state_dict(model_dict)
print('load')
