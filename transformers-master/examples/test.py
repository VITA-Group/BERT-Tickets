import torch  
import argparse
import torch.nn.utils.prune as prune

def tensor_equal_nonzero(tensor1,tensor2):
    return tensor1*tensor2 != 0

def tensor_or_nonzero(tensor1,tensor2):
    return (tensor1+tensor2) != 0


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

def meature_difference(model, pretrain_model):

    sum_list = 0
    zero_sum = 0
    for ii in range(12):
        sum_list = sum_list+float(model.bert.encoder.layer[ii].attention.self.query.weight.nelement())
        temp = tensor_equal_zero(model.bert.encoder.layer[ii].attention.self.query.weight, pretrain_model.bert.encoder.layer[ii].attention.self.query.weight)
        zero_sum = zero_sum+float(torch.sum(temp))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].attention.self.key.weight.nelement())
        temp = tensor_equal_zero(model.bert.encoder.layer[ii].attention.self.key.weight, pretrain_model.bert.encoder.layer[ii].attention.self.key.weight)
        zero_sum = zero_sum+float(torch.sum(temp))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].attention.self.value.weight.nelement())
        temp = tensor_equal_zero(model.bert.encoder.layer[ii].attention.self.value.weight, pretrain_model.bert.encoder.layer[ii].attention.self.value.weight)
        zero_sum = zero_sum+float(torch.sum(temp))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].attention.output.dense.weight.nelement())
        temp = tensor_equal_zero(model.bert.encoder.layer[ii].attention.output.dense.weight, pretrain_model.bert.encoder.layer[ii].attention.output.dense.weight)
        zero_sum = zero_sum+float(torch.sum(temp))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].intermediate.dense.weight.nelement())
        temp = tensor_equal_zero(model.bert.encoder.layer[ii].intermediate.dense.weight, pretrain_model.bert.encoder.layer[ii].intermediate.dense.weight)
        zero_sum = zero_sum+float(torch.sum(temp))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].output.dense.weight.nelement())
        temp = tensor_equal_zero(model.bert.encoder.layer[ii].output.dense.weight, pretrain_model.bert.encoder.layer[ii].output.dense.weight)
        zero_sum = zero_sum+float(torch.sum(temp))

    sum_list = sum_list+float(model.bert.pooler.dense.weight.nelement())
    temp = tensor_equal_zero(model.bert.pooler.dense.weight, pretrain_model.bert.pooler.dense.weight)
    zero_sum = zero_sum+float(torch.sum(temp))
 
    return 100*(1-zero_sum/sum_list)

def meature_relative_difference(model, pretrain_model):

    sum_list = 0
    zero_sum = 0
    for ii in range(12):
        sum_temp = tensor_or_nonzero(model.bert.encoder.layer[ii].attention.self.query.weight, pretrain_model.bert.encoder.layer[ii].attention.self.query.weight)
        temp = tensor_equal_nonzero(model.bert.encoder.layer[ii].attention.self.query.weight, pretrain_model.bert.encoder.layer[ii].attention.self.query.weight)
        zero_sum = zero_sum+float(torch.sum(temp))
        sum_list = sum_list+float(torch.sum(sum_temp))

        sum_temp = tensor_or_nonzero(model.bert.encoder.layer[ii].attention.self.key.weight, pretrain_model.bert.encoder.layer[ii].attention.self.key.weight)
        temp = tensor_equal_nonzero(model.bert.encoder.layer[ii].attention.self.key.weight, pretrain_model.bert.encoder.layer[ii].attention.self.key.weight)
        zero_sum = zero_sum+float(torch.sum(temp))
        sum_list = sum_list+float(torch.sum(sum_temp))

        sum_temp = tensor_or_nonzero(model.bert.encoder.layer[ii].attention.self.value.weight, pretrain_model.bert.encoder.layer[ii].attention.self.value.weight)
        temp = tensor_equal_nonzero(model.bert.encoder.layer[ii].attention.self.value.weight, pretrain_model.bert.encoder.layer[ii].attention.self.value.weight)
        zero_sum = zero_sum+float(torch.sum(temp))
        sum_list = sum_list+float(torch.sum(sum_temp))

        sum_temp = tensor_or_nonzero(model.bert.encoder.layer[ii].attention.output.dense.weight, pretrain_model.bert.encoder.layer[ii].attention.output.dense.weight)
        temp = tensor_equal_nonzero(model.bert.encoder.layer[ii].attention.output.dense.weight, pretrain_model.bert.encoder.layer[ii].attention.output.dense.weight)
        zero_sum = zero_sum+float(torch.sum(temp))
        sum_list = sum_list+float(torch.sum(sum_temp))

        sum_temp = tensor_or_nonzero(model.bert.encoder.layer[ii].intermediate.dense.weight, pretrain_model.bert.encoder.layer[ii].intermediate.dense.weight)
        temp = tensor_equal_nonzero(model.bert.encoder.layer[ii].intermediate.dense.weight, pretrain_model.bert.encoder.layer[ii].intermediate.dense.weight)
        zero_sum = zero_sum+float(torch.sum(temp))
        sum_list = sum_list+float(torch.sum(sum_temp))

        sum_temp = tensor_or_nonzero(model.bert.encoder.layer[ii].output.dense.weight, pretrain_model.bert.encoder.layer[ii].output.dense.weight)
        temp = tensor_equal_nonzero(model.bert.encoder.layer[ii].output.dense.weight, pretrain_model.bert.encoder.layer[ii].output.dense.weight)
        zero_sum = zero_sum+float(torch.sum(temp))
        sum_list = sum_list+float(torch.sum(sum_temp))

    sum_temp = tensor_or_nonzero(model.bert.pooler.dense.weight, pretrain_model.bert.pooler.dense.weight)
    temp = tensor_equal_nonzero(model.bert.pooler.dense.weight, pretrain_model.bert.pooler.dense.weight)
    zero_sum = zero_sum+float(torch.sum(temp))
    sum_list = sum_list+float(torch.sum(sum_temp))
 
    return 100*zero_sum/sum_list

def see_weight_rate_query(model):

    sum_list = 0
    zero_sum = 0
    for ii in range(12):
        sum_list = sum_list+float(model.bert.encoder.layer[ii].attention.self.query.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].attention.self.query.weight == 0))

    return 100*zero_sum/sum_list

def see_weight_rate_key(model):

    sum_list = 0
    zero_sum = 0
    for ii in range(12):

        sum_list = sum_list+float(model.bert.encoder.layer[ii].attention.self.key.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].attention.self.key.weight == 0))

    
    return 100*zero_sum/sum_list

def see_weight_rate_value(model):

    sum_list = 0
    zero_sum = 0
    for ii in range(12):

        sum_list = sum_list+float(model.bert.encoder.layer[ii].attention.self.value.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].attention.self.value.weight == 0))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].attention.output.dense.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].attention.output.dense.weight == 0))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].intermediate.dense.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].intermediate.dense.weight == 0))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].output.dense.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].output.dense.weight == 0))

    return 100*zero_sum/sum_list

def see_weight_rate_atdense(model):

    sum_list = 0
    zero_sum = 0
    for ii in range(12):

        sum_list = sum_list+float(model.bert.encoder.layer[ii].attention.output.dense.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].attention.output.dense.weight == 0))

    return 100*zero_sum/sum_list

def see_weight_rate_intedense(model):

    sum_list = 0
    zero_sum = 0
    for ii in range(12):

        sum_list = sum_list+float(model.bert.encoder.layer[ii].intermediate.dense.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].intermediate.dense.weight == 0))

    return 100*zero_sum/sum_list

def see_weight_rate_dense(model):

    sum_list = 0
    zero_sum = 0
    for ii in range(12):

        sum_list = sum_list+float(model.bert.encoder.layer[ii].output.dense.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].output.dense.weight == 0))

    return 100*zero_sum/sum_list

def see_weight_rate_pooler(model):

    sum_list = 0
    zero_sum = 0

    sum_list = sum_list+float(model.bert.pooler.dense.weight.nelement())
    zero_sum = zero_sum+float(torch.sum(model.bert.pooler.dense.weight == 0))
 
    return 100*zero_sum/sum_list

def see_weight_layer(model, ii):

    sum_list = 0
    zero_sum = 0

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

    return 100*zero_sum/sum_list


for path in ['tmp/SST2_new_model/checkpoint-36000/model.pt','tmp/squad_new_model/checkpoint-60000/model.pt','tmp/Co_new_model/checkpoint-4800/model.pt','output_bert/checkpoint-50000/model.pt']:
    print(path)
    model = torch.load(path, map_location='cpu')
    r1 = see_weight_rate(model)
    print('zeros_rate:', r1)
    print('********************************************************************')
    print('layer type')
    a1 = see_weight_rate_query(model)
    a2 = see_weight_rate_key(model)
    a3 = see_weight_rate_value(model)
    a4 = see_weight_rate_atdense(model)
    a5 = see_weight_rate_intedense(model)
    a6 = see_weight_rate_dense(model)
    a7 = see_weight_rate_pooler(model)

    print('query',a1)
    print('key',a2)
    print('value',a3)
    print('attention dense',a4)
    print('intermedia dense',a5)
    print('dense',a6)
    print('pooler',a7)

    for ii in range(12):
        tt = see_weight_layer(model,ii)
        print('layer',ii,tt)






# parser = argparse.ArgumentParser()
# parser.add_argument("--a", default=None, type=str, required=True, help="The input training data file (a text file).")
# parser.add_argument("--b", default=None, type=str, required=True, help="The input training data file (a text file).")
# args = parser.parse_args()

# for i in range(9):

#     m2_path = 'tmp/random_prun_CoLA/checkpoint-'+str((i+2)*800)+'/model.pt'
#     m3_path = 'tmp/SST2_new_model/checkpoint-'+str((i+2)*6000)+'/model.pt'
#     m4_path = 'tmp/squad_new_model/checkpoint-'+str((i+2)*10000)+'/model.pt'

#     print('********************************************************************')
#     print(m2_path)
#     print(m3_path)

#     model = torch.load(m2_path, map_location='cpu')
#     pretrain_model = torch.load(m3_path, map_location='cpu')

#     r1 = see_weight_rate(model)
#     r2 = see_weight_rate(pretrain_model)
#     print('model1', r1)
#     print('pre_model', r2)

#     r3 = meature_relative_difference(model, pretrain_model)
#     print('difference', r3)
#     print('********************************************************************')


# for i in range(9):

#     m2_path = 'tmp/random_prun_CoLA/checkpoint-'+str((i+2)*800)+'/model.pt'
#     m3_path = 'tmp/SST2_new_model/checkpoint-'+str((i+2)*6000)+'/model.pt'
#     m4_path = 'tmp/squad_new_model/checkpoint-'+str((i+2)*10000)+'/model.pt'

#     print('********************************************************************')
#     print(m2_path)
#     print(m4_path)

#     model = torch.load(m2_path, map_location='cpu')
#     pretrain_model = torch.load(m4_path, map_location='cpu')

#     r1 = see_weight_rate(model)
#     r2 = see_weight_rate(pretrain_model)
#     print('model1', r1)
#     print('pre_model', r2)

#     r3 = meature_relative_difference(model, pretrain_model)
#     print('difference', r3)
#     print('********************************************************************')


# for i in range(9):
#     m1_path = 'output_bert/checkpoint-'+str((i+1)*10000)+'/model.pt'

#     m2_path = 'tmp/Co_new_model/checkpoint-'+str((i+2)*800)+'/model.pt'
#     m3_path = 'tmp/SST2_new_model/checkpoint-'+str((i+2)*6000)+'/model.pt'
#     m4_path = 'tmp/squad_new_model/checkpoint-'+str((i+2)*10000)+'/model.pt'

#     print('********************************************************************')
#     print(m1_path)
#     print(m2_path)

#     model = torch.load(m1_path, map_location='cpu')
#     pretrain_model = torch.load(m2_path, map_location='cpu')

#     r1 = see_weight_rate(model)
#     r2 = see_weight_rate(pretrain_model)
#     print('model1', r1)
#     print('pre_model', r2)

#     r3 = meature_relative_difference(model, pretrain_model)
#     print('difference', r3)
#     print('********************************************************************')

# for i in range(9):
#     m1_path = 'output_bert/checkpoint-'+str((i+1)*10000)+'/model.pt'

#     m2_path = 'tmp/Co_new_model/checkpoint-'+str((i+2)*800)+'/model.pt'
#     m3_path = 'tmp/SST2_new_model/checkpoint-'+str((i+2)*6000)+'/model.pt'
#     m4_path = 'tmp/squad_new_model/checkpoint-'+str((i+2)*10000)+'/model.pt'

#     print('********************************************************************')
#     print(m1_path)
#     print(m3_path)

#     model = torch.load(m1_path, map_location='cpu')
#     pretrain_model = torch.load(m3_path, map_location='cpu')

#     r1 = see_weight_rate(model)
#     r2 = see_weight_rate(pretrain_model)
#     print('model1', r1)
#     print('pre_model', r2)

#     r3 = meature_relative_difference(model, pretrain_model)
#     print('difference', r3)
#     print('********************************************************************')

# for i in range(9):
#     m1_path = 'output_bert/checkpoint-'+str((i+1)*10000)+'/model.pt'

#     m2_path = 'tmp/Co_new_model/checkpoint-'+str((i+2)*800)+'/model.pt'
#     m3_path = 'tmp/SST2_new_model/checkpoint-'+str((i+2)*6000)+'/model.pt'
#     m4_path = 'tmp/squad_new_model/checkpoint-'+str((i+2)*10000)+'/model.pt'

#     print('********************************************************************')
#     print(m1_path)
#     print(m4_path)

#     model = torch.load(m1_path, map_location='cpu')
#     pretrain_model = torch.load(m4_path, map_location='cpu')

#     r1 = see_weight_rate(model)
#     r2 = see_weight_rate(pretrain_model)
#     print('model1', r1)
#     print('pre_model', r2)

#     r3 = meature_relative_difference(model, pretrain_model)
#     print('difference', r3)
#     print('********************************************************************')

# for i in range(9):
#     m1_path = 'output_bert/checkpoint-'+str((i+1)*10000)+'/model.pt'

#     m2_path = 'tmp/Co_new_model/checkpoint-'+str((i+2)*800)+'/model.pt'
#     m3_path = 'tmp/SST2_new_model/checkpoint-'+str((i+2)*6000)+'/model.pt'
#     m4_path = 'tmp/squad_new_model/checkpoint-'+str((i+2)*10000)+'/model.pt'

#     print('********************************************************************')
#     print(m2_path)
#     print(m3_path)

#     model = torch.load(m2_path, map_location='cpu')
#     pretrain_model = torch.load(m3_path, map_location='cpu')

#     r1 = see_weight_rate(model)
#     r2 = see_weight_rate(pretrain_model)
#     print('model1', r1)
#     print('pre_model', r2)

#     r3 = meature_relative_difference(model, pretrain_model)
#     print('difference', r3)
#     print('********************************************************************')

# for i in range(9):
#     m1_path = 'output_bert/checkpoint-'+str((i+1)*10000)+'/model.pt'

#     m2_path = 'tmp/Co_new_model/checkpoint-'+str((i+2)*800)+'/model.pt'
#     m3_path = 'tmp/SST2_new_model/checkpoint-'+str((i+2)*6000)+'/model.pt'
#     m4_path = 'tmp/squad_new_model/checkpoint-'+str((i+2)*10000)+'/model.pt'

#     print('********************************************************************')
#     print(m2_path)
#     print(m4_path)

#     model = torch.load(m2_path, map_location='cpu')
#     pretrain_model = torch.load(m4_path, map_location='cpu')

#     r1 = see_weight_rate(model)
#     r2 = see_weight_rate(pretrain_model)
#     print('model1', r1)
#     print('pre_model', r2)

#     r3 = meature_relative_difference(model, pretrain_model)
#     print('difference', r3)
#     print('********************************************************************')

# for i in range(9):
#     m1_path = 'output_bert/checkpoint-'+str((i+1)*10000)+'/model.pt'

#     m2_path = 'tmp/Co_new_model/checkpoint-'+str((i+2)*800)+'/model.pt'
#     m3_path = 'tmp/SST2_new_model/checkpoint-'+str((i+2)*6000)+'/model.pt'
#     m4_path = 'tmp/squad_new_model/checkpoint-'+str((i+2)*10000)+'/model.pt'

#     print('********************************************************************')
#     print(m3_path)
#     print(m4_path)

#     model = torch.load(m3_path, map_location='cpu')
#     pretrain_model = torch.load(m4_path, map_location='cpu')

#     r1 = see_weight_rate(model)
#     r2 = see_weight_rate(pretrain_model)
#     print('model1', r1)
#     print('pre_model', r2)

#     r3 = meature_relative_difference(model, pretrain_model)
#     print('difference', r3)
#     print('********************************************************************')
    
    


