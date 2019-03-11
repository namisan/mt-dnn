# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import os
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from module.dropout_wrapper import DropoutWrapper
from pytorch_pretrained_bert.modeling import BertConfig, BertEncoder, BertLayerNorm, BertModel
from module.san import SANClassifier, Classifier

class SANBertNetwork(nn.Module):
    def __init__(self, opt, bert_config=None):
        super(SANBertNetwork, self).__init__()
        self.dropout_list = []
        self.bert_config = BertConfig.from_dict(opt)
        self.bert = BertModel(self.bert_config)
        if opt['update_bert_opt'] > 0:
            for p in self.bert.parameters():
                p.requires_grad = False
        mem_size = self.bert_config.hidden_size
        self.decoder_opt = opt['answer_opt']
        self.scoring_list = nn.ModuleList()
        labels = [int(ls) for ls in opt['label_size'].split(',')]
        task_dropout_p = opt['tasks_dropout_p']
        self.bert_pooler = None

        for task, lab in enumerate(labels):
            decoder_opt = self.decoder_opt[task]
            dropout = DropoutWrapper(task_dropout_p[task], opt['vb_dropout'])
            self.dropout_list.append(dropout)
            if decoder_opt == 1:
                out_proj = SANClassifier(mem_size, mem_size, lab, opt, prefix='answer', dropout=dropout)
                self.scoring_list.append(out_proj)
            else:
                out_proj = nn.Linear(self.bert_config.hidden_size, lab)
                self.scoring_list.append(out_proj)

        self.opt = opt
        self._my_init()
        self.set_embed(opt)

    def _my_init(self):
        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.bert_config.initializer_range * self.opt['init_ratio'])
            elif isinstance(module, BertLayerNorm):
                # Slightly different from the BERT pytorch version, which should be a bug.
                # Note that it only affects on training from scratch. For detailed discussions, please contact xiaodl@.
                # Layer normalization (https://arxiv.org/abs/1607.06450)
                # support both old/latest version
                if 'beta' in dir(module) and 'gamma' in dir(module):
                    module.beta.data.zero_()
                    module.gamma.data.fill_(1.0)
                else:
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def nbert_layer(self):
        return len(self.bert.encoder.layer)

    def freeze_layers(self, max_n):
        assert max_n < self.nbert_layer()
        for i in range(0, max_n):
            self.freeze_layer(i)

    def freeze_layer(self, n):
        assert n < self.nbert_layer()
        layer = self.bert.encoder.layer[n]
        for p in layer.parameters():
            p.requires_grad = False

    def set_embed(self, opt):
        bert_embeddings = self.bert.embeddings
        emb_opt = opt['embedding_opt']
        if emb_opt == 1:
            for p in bert_embeddings.word_embeddings.parameters():
                p.requires_grad = False
        elif emb_opt == 2:
            for p in bert_embeddings.position_embeddings.parameters():
                p.requires_grad = False
        elif emb_opt == 3:
            for p in bert_embeddings.token_type_embeddings.parameters():
                p.requires_grad = False
        elif emb_opt == 4:
            for p in bert_embeddings.token_type_embeddings.parameters():
                p.requires_grad = False
            for p in bert_embeddings.position_embeddings.parameters():
                p.requires_grad = False

    def forward(self, input_ids, token_type_ids, attention_mask, premise_mask=None, hyp_mask=None, task_id=0):
        all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = all_encoder_layers[-1]
        if self.bert_pooler is not None:
            pooled_output = self.bert_pooler(sequence_output)
        decoder_opt = self.decoder_opt[task_id]
        if decoder_opt == 1:
            max_query = hyp_mask.size(1)
            assert max_query > 0
            assert premise_mask is not None
            assert hyp_mask is not None
            hyp_mem = sequence_output[:,:max_query,:]
            logits = self.scoring_list[task_id](sequence_output, hyp_mem, premise_mask, hyp_mask)
        else:
            pooled_output = self.dropout_list[task_id](pooled_output)
            logits = self.scoring_list[task_id](pooled_output)
        return logits
