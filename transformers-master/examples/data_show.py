import torch
import numpy as np 
import argparse
import os  


from transformers import BertForMaskedLM, BertForSequenceClassification, BertForQuestionAnswering, BertPreTrainedModel

class MultitaskBert_p1(BertPreTrainedModel):
    def __init__(self, config, original_model, num_label_task1):
        super().__init__(config)

        self.num_labels = [num_label_task1]
        self.bert = original_model.bert 
        self.cls = original_model.cls

        self.dropout_task1 = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_task1 = nn.Linear(config.hidden_size, num_label_task1)

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
        labels=None,
        task_ids=4,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForSequenceClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, logits = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        if task_ids == 4:

            sequence_output = outputs[0]
            prediction_scores = self.cls(sequence_output)

            outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
            # Although this may seem awkward, BertForMaskedLM supports two scenarios:
            # 1. If a tensor that contains the indices of masked labels is provided,
            #    the cross-entropy is the MLM cross-entropy that measures the likelihood
            #    of predictions for masked words.
            # 2. If `lm_labels` is provided we are in a causal scenario where we
            #    try to predict the next token for each input in the decoder.
            if masked_lm_labels is not None:
                loss_fct = CrossEntropyLoss()  # -100 index = padding token
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
                outputs = (masked_lm_loss,) + outputs

            if lm_labels is not None:
                # we are doing next-token prediction; shift prediction scores and input ids by one
                prediction_scores = prediction_scores[:, :-1, :].contiguous()
                lm_labels = lm_labels[:, 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
                outputs = (ltr_lm_loss,) + outputs

        else:
            pooled_output = outputs[1]

            if task_ids == 0:
                pooled_output = self.dropout_task1(pooled_output)
                logits = self.classifier_task1(pooled_output)

            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

            if labels is not None:
                if self.num_labels[task_ids] == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.float().view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels[task_ids]), labels.view(-1))
                outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class MultitaskBert_p2(BertPreTrainedModel):
    def __init__(self, config, original_model, num_label_task1, num_label_task2):
        super().__init__(config)

        self.num_labels = [num_label_task1,num_label_task2]
        self.bert = original_model.bert 
        self.cls = original_model.cls

        self.dropout_task1 = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_task1 = nn.Linear(config.hidden_size, num_label_task1)

        self.dropout_task2 = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_task2 = nn.Linear(config.hidden_size, num_label_task2)

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
        labels=None,
        task_ids=4,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForSequenceClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, logits = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        if task_ids == 4:

            sequence_output = outputs[0]
            prediction_scores = self.cls(sequence_output)

            outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
            # Although this may seem awkward, BertForMaskedLM supports two scenarios:
            # 1. If a tensor that contains the indices of masked labels is provided,
            #    the cross-entropy is the MLM cross-entropy that measures the likelihood
            #    of predictions for masked words.
            # 2. If `lm_labels` is provided we are in a causal scenario where we
            #    try to predict the next token for each input in the decoder.
            if masked_lm_labels is not None:
                loss_fct = CrossEntropyLoss()  # -100 index = padding token
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
                outputs = (masked_lm_loss,) + outputs

            if lm_labels is not None:
                # we are doing next-token prediction; shift prediction scores and input ids by one
                prediction_scores = prediction_scores[:, :-1, :].contiguous()
                lm_labels = lm_labels[:, 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
                outputs = (ltr_lm_loss,) + outputs

        else:
            pooled_output = outputs[1]

            if task_ids == 0:
                pooled_output = self.dropout_task1(pooled_output)
                logits = self.classifier_task1(pooled_output)

            elif task_ids == 1:
                pooled_output = self.dropout_task2(pooled_output)
                logits = self.classifier_task2(pooled_output)

            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

            if labels is not None:
                if self.num_labels[task_ids] == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.float().view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels[task_ids]), labels.view(-1))
                outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


# for path,j,k in os.walk('./'):
#     # print('*******************************************************************')
    
#     # if not path == './':

#     #     if not 'pnoise' in path:

#     #         if not 'pr_before_to_QQP' in path:

#     if 'pnoise' in path:
#         print('*******************************************************************')
#         print(path)

#         file_name = os.path.join(path,'result.pt')
#         if not os.path.isfile(file_name):
#             continue

#         data = torch.load(file_name)

#         for ii in range(len(data)):
#             sub_data = data[ii]
#             print(sub_data)

#     print('*******************************************************************')


# for type1 in ['CoLA','SST2']:
#     for type2 in ['_rand','']:
#         for type3 in ['1','001']:
#             for i in range(1,10):
#                 path = str(i)+'pnoise'+type3+type2+'_to_'+type1
#                 print(path)
#                 file_name = os.path.join(path,'result.pt')
#                 if not os.path.isfile(file_name):
#                     continue
#                 data = torch.load(file_name)
#                 for ii in range(len(data)):
#                     sub_data = data[ii]
#                     print(sub_data)
#                 print('*******************************************************************')


# for i in range(10):
#     for type2 in ['m1','m2']:
#         path = str(i)+type2+'_to_MNLI'
#         print(path)
#         file_name = os.path.join(path,'result.pt')
#         if not os.path.isfile(file_name):
#             continue
#         data = torch.load(file_name)
#         for ii in range(len(data)):
#             sub_data = data[ii]
#             print(sub_data)
#         print('*******************************************************************')

# for i in range(1,10):
#     for type2 in ['CoLA', 'SST2']:
#         path = 'pnoise/'+str(i)+'pnoise1_rand_to_'+type2
#         print(path)
#         file_name = os.path.join(path,'result.pt')
#         if not os.path.isfile(file_name):
#             continue
#         data = torch.load(file_name)
#         for ii in range(len(data)):
#             sub_data = data[ii]
#             print(sub_data)
#         print('*******************************************************************')



# for i in range(1,10):
#     for type2 in ['CoLA', 'SST2']:
#         path = 'converge_problem/'+str(i)+'pnoise1_10_rand_to_'+type2
#         print(path)
#         file_name = os.path.join(path,'result.pt')
#         if not os.path.isfile(file_name):
#             continue
#         data = torch.load(file_name)
#         for ii in range(len(data)):
#             sub_data = data[ii]
#             print(sub_data)
#         print('*******************************************************************')



# for i in range(1,10):
#     for type2 in ['CoLA', 'SST2']:
#         path = 'converge_problem/'+str(i)+'pnoise1_10_to_'+type2
#         print(path)
#         file_name = os.path.join(path,'result.pt')
#         if not os.path.isfile(file_name):
#             continue
#         data = torch.load(file_name)
#         for ii in range(len(data)):
#             sub_data = data[ii]
#             print(sub_data)
#         print('*******************************************************************')


# for type2 in ['CoLA', 'SST2']:
#     for i in range(1,10):
#         path = 'tmp/'+str(i)+'diff_noise_to_'+type2
#         print(path)
#         file_name = os.path.join(path,'result.pt')
#         if not os.path.isfile(file_name):
#             continue
#         data = torch.load(file_name)
#         for ii in range(len(data)):
#             sub_data = data[ii]
#             print(sub_data)
#         print('*******************************************************************')


# for type2 in ['CoLA', 'SST2']:
#     for i in range(1,10):
#         path = 'tmp/'+str(i)+'prand_to_'+type2
#         print(path)
#         file_name = os.path.join(path,'result.pt')
#         if not os.path.isfile(file_name):
#             continue
#         data = torch.load(file_name)
#         for ii in range(len(data)):
#             sub_data = data[ii]
#             print(sub_data)
#         print('*******************************************************************')


# for type2 in ['CoLA', 'SST2', 'SQ']:
#     for i in range(1,10):
#         path = 'tmp/'+str(i)+'noritergasp_to_'+type2
#         print(path)
#         file_name = os.path.join(path,'result.pt')
#         if not os.path.isfile(file_name):
#             continue
#         data = torch.load(file_name)
#         for ii in range(len(data)):
#             sub_data = data[ii]
#             print(sub_data)
#         print('*******************************************************************')

# for type2 in ['CoLA', 'SST2', 'SQ']:
#     for i in range(1,10):
#         path = 'tmp/'+str(i)+'gasp_rr_to_'+type2
#         print(path)
#         file_name = os.path.join(path,'result.pt')
#         if not os.path.isfile(file_name):
#             continue
#         data = torch.load(file_name)
#         for ii in range(len(data)):
#             sub_data = data[ii]
#             print(sub_data)
#         print('*******************************************************************')


# for type2 in ['CoLA', 'SST2', 'SQ']:
#     for i in range(1,10):
#         path = 'tmp/'+str(i)+'gasp_rp_to_'+type2
#         print(path)
#         file_name = os.path.join(path,'result.pt')
#         if not os.path.isfile(file_name):
#             continue
#         data = torch.load(file_name)
#         for ii in range(len(data)):
#             sub_data = data[ii]
#             print(sub_data)
#         print('*******************************************************************')


# for type2 in ['CoLA', 'SST2', 'SQ']:
#     for i in range(1,10):
#         path = 'tmp/'+str(i)+'gasp_pr_to_'+type2
#         print(path)
#         file_name = os.path.join(path,'result.pt')
#         if not os.path.isfile(file_name):
#             continue
#         data = torch.load(file_name)
#         for ii in range(len(data)):
#             sub_data = data[ii]
#             print(sub_data)
#         print('*******************************************************************')


# for type2 in ['CoLA', 'SST2', 'SQ']:
#     for i in range(1,10):
#         path = 'tmp/'+str(i)+'gasp_pp_to_'+type2
#         print(path)
#         file_name = os.path.join(path,'result.pt')
#         if not os.path.isfile(file_name):
#             continue
#         data = torch.load(file_name)
#         for ii in range(len(data)):
#             sub_data = data[ii]
#             print(sub_data)
#         print('*******************************************************************')

# for type2 in ['CoLA', 'SST2']:
#     for i in range(1,10):
#         path = 'tmp/'+str(i)+'nor_r_to_'+type2
#         print(path)
#         file_name = os.path.join(path,'result.pt')
#         if not os.path.isfile(file_name):
#             continue
#         data = torch.load(file_name)
#         for ii in range(len(data)):
#             sub_data = data[ii]
#             print(sub_data)
#         print('*******************************************************************')

# for type2 in ['CoLA', 'SST2']:
#     for i in range(1,10):
#         path = 'tmp/'+str(i)+'tr_r_to_'+type2
#         print(path)
#         file_name = os.path.join(path,'result.pt')
#         if not os.path.isfile(file_name):
#             continue
#         data = torch.load(file_name)
#         for ii in range(len(data)):
#             sub_data = data[ii]
#             print(sub_data)
#         print('*******************************************************************')

# for type2 in ['CoLA', 'SST2']:
#     for i in range(1,10):
#         path = 'tmp/'+str(i)+'tr_newr_to_'+type2
#         print(path)
#         file_name = os.path.join(path,'result.pt')
#         if not os.path.isfile(file_name):
#             continue
#         data = torch.load(file_name)
#         for ii in range(len(data)):
#             sub_data = data[ii]
#             print(sub_data)
#         print('*******************************************************************')

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('-t', default='211_It_s45_', type=str, help='file_dir')
parser.add_argument('-b', default='211_It_s45_', type=str, help='file_dir')
args = parser.parse_args()

kk = args.t

# path = os.listdir(kk)

# for ax in path:
#     file_name = os.path.join(kk,ax)
#     if not os.path.isfile(file_name):
#         continue
#     mask = torch.load(file_name)
#     rate = 0
#     sum1 = 0
#     for key in mask.keys():
#         rate += float(torch.sum(mask[key] == 0))
#         sum1 += float(mask[key].nelement())

#     print(file_name, rate/sum1)

# file_list = ['LTPRE90000','LTMNLI220890','LTQQP204660','LTSQUAD55400','LTSST37884','LTWNLI360','LTQNLI58914','LTRTE1398','LTSQUAD66480','LTSTSB3234']
# save_dir = ['pretrain90.pt','mnli50.pt','qqp50.pt','squad50.pt','sst250.pt','wnli50.pt','qnli50.pt','rte50.pt','squad60.pt','stsb50.pt']

# file_list = ['cola40','mrpc40','sst210']

# file_list = ['50000','60000','70000','80000','90000']


# for kk in file_list:
# # for ii in range(1,11):

    # path = 'LT_pretrain/checkpoint-'+kk
# path = kk
# model = torch.load(os.path.join(path, 'model.pt'), map_location='cpu')
# model_dict = model.state_dict()
# mask_dict = {}

# for key in model_dict.keys():
#     if 'mask' in key:
#         mask_dict[key] = model_dict[key]

# rate = 0
# sum1 = 0
# for key in mask_dict.keys():
#     rate += float(torch.sum(mask_dict[key] == 0))
#     sum1 += float(mask_dict[key].nelement())

# print(path)
# print('zero rate = ',rate/sum1)

# torch.save(mask_dict, args.b)






# file_name = os.path.join(kk,'result.pt')
# data = torch.load(file_name)
# for ii in range(len(data)):
#     print(ii, data[ii])
# print('*******************************************************************')


files = os.listdir(kk)
files.sort()

for path in files:

    print('*******************************************************************')
    print(path)

    file_name = os.path.join(kk, path, 'result.pt')
    if not os.path.isfile(file_name):
        continue

    data = torch.load(file_name)
    # all_result = []


    for ii in range(len(data)):
        sub_data = data[ii]
        # all_result.append(sub_data)
        print(sub_data)
    # print('best = ', np.max(np.array(all_result)))
    print('*******************************************************************')



# files = os.listdir(kk)
# files.sort()

# for path in files:

#     print('*******************************************************************')
#     print(path)

#     file_name = os.path.join(kk, path, 'result.pt')
#     if not os.path.isfile(file_name):
#         continue

#     data = torch.load(file_name)
#     all_result = []

#     if 'CoLA' in file_name:
#         key = 'mcc'
#     elif 'SST2' in file_name:
#         key = 'acc' 
#     elif 'SQ' in file_name:
#         key = 'f1'
#     else:
#         continue

#     for ii in range(len(data)):
#         sub_data = data[ii][key]
#         all_result.append(sub_data)
#         print(sub_data)
#     print('best = ', np.max(np.array(all_result)))
#     print('*******************************************************************')






# for i in range(1,10):
#     all_result = []
#     path = 'tmp/'+kk+'CoLA_'+str(i)
#     print(path)
#     file_name = os.path.join(path,'result.pt')
#     if not os.path.isfile(file_name):
#         continue
#     data = torch.load(file_name)
#     for ii in range(len(data)):
#         sub_data = data[ii]['mcc']
#         all_result.append(sub_data)
#         print(sub_data)
#     print('best = ', np.max(np.array(all_result)))
#     print('*******************************************************************')

# for i in range(1,10):
#     all_result = []
#     path = 'tmp/'+kk+'SST2_'+str(i)
#     print(path)
#     file_name = os.path.join(path,'result.pt')
#     if not os.path.isfile(file_name):
#         continue
#     data = torch.load(file_name)
#     for ii in range(len(data)):
#         sub_data = data[ii]['acc']
#         all_result.append(sub_data)
#         print(sub_data)
#     print('best = ', np.max(np.array(all_result)))
#     print('*******************************************************************')

# for i in range(1,10):
#     all_result = []
#     path = 'tmp/'+kk+'SQ_'+str(i)
#     print(path)
#     file_name = os.path.join(path,'result.pt')
#     if not os.path.isfile(file_name):
#         continue
#     data = torch.load(file_name)
#     for ii in range(len(data)):
#         sub_data = data[ii]['f1']
#         all_result.append(sub_data)
#         print(sub_data)
#     print('best = ', np.max(np.array(all_result)))
#     print('*******************************************************************')



# def Union(mask1,mask2):
#     return torch.max(mask1,mask2)

# def Intersection(mask1,mask2):
#     return mask1*mask2

# def see_zero(mask):
#     zero_list = 0
#     sum_num = 0

#     for key in mask.keys():
#         zero_list += float(torch.sum(mask[key] == 1))
#         sum_num += float(mask[key].nelement())

#     print('zero rate = ', zero_list/sum_num)

# def new_mask_process(model1,model2):

#     union_num = 0
#     inter_num = 0

#     for key in model1.keys():
#         tensor1 = model1[key]
#         tensor2 = model2[key]

#         union_mask = Union(tensor1,tensor2)
#         inter_mask = Intersection(tensor1,tensor2)

#         union_num += float(torch.sum(union_mask == 1))
#         inter_num += float(torch.sum(inter_mask == 1))


#     result = inter_num/union_num
#     print('different_result = ', result)

#     return result


# name_list = ['pre','qnli','sst2','cola','squad','wnli','mrpc','stsb','rte','mnli','qqp']
# rate_list = ['60.pt','70.pt']

# for rate in rate_list:
#     for a1 in name_list:
#         mask1_k = torch.load(a1+rate)
#         for b1 in name_list:
#             print('******************************************************************')
#             print('differenct of ', a1,b1)
#             mask2_k = torch.load(b1+rate)
#             see_zero(mask1_k)
#             see_zero(mask2_k)
#             new_mask_process(mask1_k, mask2_k)
#             print('******************************************************************')




















