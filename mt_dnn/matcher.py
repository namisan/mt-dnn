# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import os
import torch
import torch.nn as nn
from pretrained_models import MODEL_CLASSES
from module.dropout_wrapper import DropoutWrapper
from module.san import SANClassifier, MaskLmHeader
from module.san_model import SanModel
from module.pooler import Pooler
from torch.nn.modules.normalization import LayerNorm
from data_utils.task_def import EncoderModelType, TaskType
import tasks
from experiments.exp_def import TaskDef


def generate_decoder_opt(enable_san, max_opt):
    opt_v = 0
    if enable_san and max_opt < 2:
        opt_v = max_opt
    return opt_v

class SANBertNetwork(nn.Module):
    def __init__(self, opt, bert_config=None, initial_from_local=False):
        super(SANBertNetwork, self).__init__()
        self.dropout_list = nn.ModuleList()

        if opt['encoder_type'] not in EncoderModelType._value2member_map_:
            raise ValueError("encoder_type is out of pre-defined types")
        self.encoder_type = opt['encoder_type']
        self.preloaded_config = None

        literal_encoder_type = EncoderModelType(self.encoder_type).name.lower()
        config_class, model_class, _ = MODEL_CLASSES[literal_encoder_type]
        if not initial_from_local:
            # self.bert = model_class.from_pretrained(opt['init_checkpoint'], config=self.preloaded_config)
            self.bert = model_class.from_pretrained(opt['init_checkpoint'])
        else:
            self.preloaded_config = config_class.from_dict(opt)  # load config from opt
            self.preloaded_config.output_hidden_states = True # return all hidden states
            self.bert = model_class(self.preloaded_config)

        hidden_size = self.bert.config.hidden_size

        if opt.get('dump_feature', False):
            self.opt = opt
            return
        if opt['update_bert_opt'] > 0:
            for p in self.bert.parameters():
                p.requires_grad = False

        task_def_list = opt['task_def_list']
        self.task_def_list = task_def_list
        self.decoder_opt = []
        self.task_types = []
        for task_id, task_def in enumerate(task_def_list):
            self.decoder_opt.append(generate_decoder_opt(task_def.enable_san, opt['answer_opt']))
            self.task_types.append(task_def.task_type)

        # create output header
        self.scoring_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        for task_id in range(len(task_def_list)):
            task_def: TaskDef = task_def_list[task_id]
            lab = task_def.n_class
            decoder_opt = self.decoder_opt[task_id]
            task_type = self.task_types[task_id]
            task_dropout_p = opt['dropout_p'] if task_def.dropout_p is None else task_def.dropout_p
            dropout = DropoutWrapper(task_dropout_p, opt['vb_dropout'])
            self.dropout_list.append(dropout)
            task_obj = tasks.get_task_obj(task_def)
            if task_obj is not None:
                # quick hack
                self.pooler = Pooler(hidden_size, dropout_p= opt['dropout_p'], actf=opt['pooler_actf'])
                out_proj = task_obj.train_build_task_layer(decoder_opt, hidden_size, lab, opt, prefix='answer', dropout=dropout)
            elif task_type == TaskType.Span:
                assert decoder_opt != 1
                out_proj = nn.Linear(hidden_size, 2)
            elif task_type == TaskType.SeqenceLabeling:
                out_proj = nn.Linear(hidden_size, lab)
            elif task_type == TaskType.MaskLM:
                if opt['encoder_type'] == EncoderModelType.ROBERTA:
                    # TODO: xiaodl
                    out_proj = MaskLmHeader(self.bert.embeddings.word_embeddings.weight)
                else:
                    out_proj = MaskLmHeader(self.bert.embeddings.word_embeddings.weight)
            else:
                if decoder_opt == 1:
                    out_proj = SANClassifier(hidden_size, hidden_size, lab, opt, prefix='answer', dropout=dropout)
                else:
                    out_proj = nn.Linear(hidden_size, lab)
            self.scoring_list.append(out_proj)
        self.opt = opt


    def embed_encode(self, input_ids, token_type_ids=None, attention_mask=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        embedding_output = self.bert.embeddings(input_ids, token_type_ids)
        return embedding_output


    def encode(self, input_ids, token_type_ids, attention_mask, inputs_embeds=None):
        if self.encoder_type == EncoderModelType.T5:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)            
        else:
            outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                                                            attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        last_hidden_state = outputs.last_hidden_state
        all_hidden_states = outputs.hidden_states # num_layers + 1 (embeddings)
        return last_hidden_state, all_hidden_states

    def forward(self, input_ids, token_type_ids, attention_mask, premise_mask=None, hyp_mask=None, task_id=0, fwd_type=0, embed=None):
        if fwd_type == 2:
            assert embed is not None
            last_hidden_state, all_hidden_states = self.encode(None, token_type_ids, attention_mask, embed) 
        elif fwd_type == 1:
            return self.embed_encode(input_ids, token_type_ids, attention_mask)
        else:
            last_hidden_state, all_hidden_states = self.encode(input_ids, token_type_ids, attention_mask)
        decoder_opt = self.decoder_opt[task_id]
        task_type = self.task_types[task_id]
        task_obj = tasks.get_task_obj(self.task_def_list[task_id])
        if task_obj is not None:
            pooled_output = self.pooler(last_hidden_state)
            logits = task_obj.train_forward(last_hidden_state, pooled_output, premise_mask, hyp_mask, decoder_opt, self.dropout_list[task_id], self.scoring_list[task_id])
            return logits
        elif task_type == TaskType.Span:
            assert decoder_opt != 1
            last_hidden_state = self.dropout_list[task_id](last_hidden_state)
            logits = self.scoring_list[task_id](last_hidden_state)
            start_scores, end_scores = logits.split(1, dim=-1)
            start_scores = start_scores.squeeze(-1)
            end_scores = end_scores.squeeze(-1)
            return start_scores, end_scores
        elif task_type == TaskType.SeqenceLabeling:
            pooled_output = last_hidden_state
            pooled_output = self.dropout_list[task_id](pooled_output)
            pooled_output = pooled_output.contiguous().view(-1, pooled_output.size(2))
            logits = self.scoring_list[task_id](pooled_output)
            return logits
        elif task_type == TaskType.MaskLM:
            last_hidden_state = self.dropout_list[task_id](last_hidden_state)
            logits = self.scoring_list[task_id](last_hidden_state)
            return logits
        else:
            if decoder_opt == 1:
                max_query = hyp_mask.size(1)
                assert max_query > 0
                assert premise_mask is not None
                assert hyp_mask is not None
                hyp_mem = last_hidden_state[:, :max_query, :]
                logits = self.scoring_list[task_id](last_hidden_state, hyp_mem, premise_mask, hyp_mask)
            else:
                pooled_output = self.dropout_list[task_id](pooled_output)
                logits = self.scoring_list[task_id](pooled_output)
            return logits
