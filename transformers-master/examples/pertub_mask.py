import torch 



def pertuba_mask(mask_dict, px):
    new_mask_dict={}

    for key in mask_dict.keys():
        mask_orig = mask_dict[key]
        reverse_mask = 1-mask_orig

        sam_tensor = sample_mask(mask_orig, px)
        del_tensor = sample_mask(reverse_mask, px)


        new_mask = mask_orig+sam_tensor-del_tensor
        new_mask_dict[key] = new_mask

    return new_mask_dict

def sample_mask(tensor, px):
    para_num = tensor.nelement()
    keep_num = int((1-px)*para_num)

    rand_tensor = torch.rand_like(tensor)
    new_mask = torch.ones_like(tensor)
    op_tensor = torch.max(tensor, rand_tensor)
    topk = torch.topk(op_tensor.view(-1), k=keep_num)

    new_mask.view(-1)[topk.indices]=0

    return new_mask

def check_rate(mask_dict):
    zero = 0
    sum_all = 0
    for key in mask_dict.keys():
        zero += float(torch.sum(mask_dict[key] == 0))
        sum_all += float(mask_dict[key].nelement())

    print('zero rate = ', zero/sum_all)

def check_different(mask_dict1, mask_dict2):
    zero = 0
    sum_all = 0
    for key in mask_dict1.keys():
        zero += float(torch.sum(mask_dict1[key] == mask_dict2[key]))
        sum_all += float(mask_dict1[key].nelement())

    print('same rate = ', zero/sum_all)

orig_mask_dict = torch.load('pre60.pt', map_location='cpu')
check_rate(orig_mask_dict)


for idx in [0.1,1,2,3,5,10]:
    rate = idx/100
    new_dict = pertuba_mask(orig_mask_dict, rate)
    check_rate(new_dict)
    check_different(orig_mask_dict, new_dict)
    torch.save(new_dict, 'pertub_mask'+str(idx)+'.pt')

