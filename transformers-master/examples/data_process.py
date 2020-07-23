import torch 
import pdb 
from prun_utils import see_weight_rate

def Union(mask1,mask2):
    return torch.max(mask1,mask2)

def Intersection(mask1,mask2):
    return mask1*mask2

def mask_out(pre_weight):
    recover_dict = {}
    for ii in range(12):
        recover_dict['bert.encoder.layer.'+str(ii)+'.attention.self.query.weight_mask'] = pre_weight['bert.encoder.layer.'+str(ii)+'.attention.self.query.weight_mask']
        recover_dict['bert.encoder.layer.'+str(ii)+'.attention.self.key.weight_mask'] = pre_weight['bert.encoder.layer.'+str(ii)+'.attention.self.key.weight_mask']
        recover_dict['bert.encoder.layer.'+str(ii)+'.attention.self.value.weight_mask'] = pre_weight['bert.encoder.layer.'+str(ii)+'.attention.self.value.weight_mask']
        recover_dict['bert.encoder.layer.'+str(ii)+'.attention.output.dense.weight_mask'] = pre_weight['bert.encoder.layer.'+str(ii)+'.attention.output.dense.weight_mask']
        recover_dict['bert.encoder.layer.'+str(ii)+'.intermediate.dense.weight_mask'] = pre_weight['bert.encoder.layer.'+str(ii)+'.intermediate.dense.weight_mask']
        recover_dict['bert.encoder.layer.'+str(ii)+'.output.dense.weight_mask'] = pre_weight['bert.encoder.layer.'+str(ii)+'.output.dense.weight_mask']
    recover_dict['bert.pooler.dense.weight_mask'] = pre_weight['bert.pooler.dense.weight_mask']

    return recover_dict

def new_mask_process(model1,model2,type1=0):
    all_mask = {}
    for key in model1.keys():
        tensor1 = model1[key]
        tensor2 = model2[key]
        if type1 == 0:
            new_mask = Union(tensor1,tensor2)
        else:
            new_mask = Intersection(tensor1,tensor2)
        all_mask[key] = new_mask
    return all_mask

for ii in range(2,11):
    print(ii-1,'cola_sst')
    path1 = 'Co_new_model/checkpoint-'+str(ii*800)+'/model.pt'
    path2 = 'SST2_new_model/checkpoint-'+str(ii*6000)+'/model.pt'
    outpath1 = str(ii-1)+'mask_union_cola_sst.pt'
    outpath2 = str(ii-1)+'mask_intersection_cola_sst.pt'

    model1 = torch.load(path1, map_location = 'cpu')
    model2 = torch.load(path2, map_location = 'cpu')

    a1 = see_weight_rate(model1)
    a2 = see_weight_rate(model2)
    print(a1,a2)
 
    mask1 = mask_out(model1.state_dict())
    mask2 = mask_out(model2.state_dict())

    umask = new_mask_process(mask1,mask2,0)
    imask = new_mask_process(mask1,mask2,1)

    torch.save(umask,outpath1)
    torch.save(imask,outpath2)

for ii in range(2,11):
    print(ii-1,'cola_squad')
    path1 = 'Co_new_model/checkpoint-'+str(ii*800)+'/model.pt'
    path2 = 'squad_new_model/checkpoint-'+str(ii*10000)+'/model.pt'
    outpath1 = str(ii-1)+'mask_union_cola_squad.pt'
    outpath2 = str(ii-1)+'mask_intersection_cola_squad.pt'

    model1 = torch.load(path1, map_location = 'cpu')
    model2 = torch.load(path2, map_location = 'cpu')

    a1 = see_weight_rate(model1)
    a2 = see_weight_rate(model2)
    print(a1,a2)
 
    mask1 = mask_out(model1.state_dict())
    mask2 = mask_out(model2.state_dict())

    umask = new_mask_process(mask1,mask2,0)
    imask = new_mask_process(mask1,mask2,1)

    torch.save(umask,outpath1)
    torch.save(imask,outpath2)

for ii in range(2,11):
    print(ii-1,'all')
    path1 = 'Co_new_model/checkpoint-'+str(ii*800)+'/model.pt'
    path2 = 'squad_new_model/checkpoint-'+str(ii*10000)+'/model.pt'
    path3 = 'SST2_new_model/checkpoint-'+str(ii*6000)+'/model.pt'
    outpath1 = str(ii-1)+'mask_union_cola_sst_squad.pt'
    outpath2 = str(ii-1)+'mask_intersection_cola_sst_squad.pt'

    model1 = torch.load(path1, map_location = 'cpu')
    model2 = torch.load(path2, map_location = 'cpu')
    model3 = torch.load(path3, map_location = 'cpu')

    a1 = see_weight_rate(model1)
    a2 = see_weight_rate(model2)
    a3 = see_weight_rate(model3)
    print(a1,a2,a3)
 
    mask1 = mask_out(model1.state_dict())
    mask2 = mask_out(model2.state_dict())
    mask3 = mask_out(model3.state_dict())

    umask = new_mask_process(mask1,mask2,0)
    all_umask = new_mask_process(umask,mask3,0)
    imask = new_mask_process(mask1,mask2,1)
    all_imask = new_mask_process(imask,mask3,1)

    torch.save(all_umask,outpath1)
    torch.save(all_imask,outpath2)


