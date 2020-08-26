# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import os
import torch
import torch.nn as nn
from pretrained_models import MODEL_CLASSES
from transformers import BertConfig

from module.dropout_wrapper import DropoutWrapper
from module.san import SANClassifier, MaskLmHeader
from module.san_model import SanModel
from torch.nn.modules.normalization import LayerNorm
from data_utils.task_def import EncoderModelType, TaskType
import tasks
from experiments.exp_def import TaskDef

class LinearPooler(nn.Module):
    def __init__(self, hidden_size):
        super(LinearPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

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
        config_class, model_class, tokenizer_class = MODEL_CLASSES[literal_encoder_type]
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
        self._my_init()
        # if not loading from local, loading model weights from pre-trained model, after initialization
        if not initial_from_local:
            config_class, model_class, tokenizer_class = MODEL_CLASSES[literal_encoder_type]
            self.bert = model_class.from_pretrained(opt['init_checkpoint'], config=self.preloaded_config)

    def _my_init(self):
        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=0.02 * self.opt['init_ratio'])
            if isinstance(module, LayerNorm):
                # Layer normalization (https://arxiv.org/abs/1607.06450)
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    module.bias.data.zero_()
        self.apply(init_weights)


    def embed_encode(self, input_ids, token_type_ids=None, attention_mask=None):
        # support BERT now
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        embedding_output = self.bert.embeddings(input_ids, token_type_ids)
        return embedding_output


    def encode(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                                                          attention_mask=attention_mask)
        # all hidden states: outputs[1]
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        all_hidden_states = outputs[2]
        return sequence_output, pooled_output, all_hidden_states

    def embed_forward(self, embed, attention_mask=None, output_all_encoded_layers=True):
        device = embed.device
        input_shape = embed.size()[:-1]
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.bert.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = [None] * self.bert.config.num_hidden_layers

        #head_mask = self.bert.get_head_mask(head_mask, self.bert.config.num_hidden_layers)
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        #extended_attention_mask = self.bert.get_extended_attention_mask(
        #    attention_mask, input_shape, device
        #)
        encoder_outputs = self.bert.encoder(
            embed,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.bert.pooler(sequence_output)
        outputs = sequence_output, pooled_output
        return outputs

    def forward(self, input_ids, token_type_ids, attention_mask, premise_mask=None, hyp_mask=None, task_id=0, fwd_type=0, embed=None):
        if fwd_type == 2:
            assert embed is not None
            sequence_output, pooled_output = self.embed_forward(embed, attention_mask) 
        elif fwd_type == 1:
            return self.embed_encode(input_ids, token_type_ids, attention_mask)
        else:
            sequence_output, pooled_output, _ = self.encode(input_ids, token_type_ids, attention_mask)
        decoder_opt = self.decoder_opt[task_id]
        task_type = self.task_types[task_id]
        task_obj = tasks.get_task_obj(self.task_def_list[task_id])
        if task_obj is not None:
            logits = task_obj.train_forward(sequence_output, pooled_output, premise_mask, hyp_mask, decoder_opt, self.dropout_list[task_id], self.scoring_list[task_id])
            return logits
        elif task_type == TaskType.Span:
            assert decoder_opt != 1
            sequence_output = self.dropout_list[task_id](sequence_output)
            logits = self.scoring_list[task_id](sequence_output)
            start_scores, end_scores = logits.split(1, dim=-1)
            start_scores = start_scores.squeeze(-1)
            end_scores = end_scores.squeeze(-1)
            return start_scores, end_scores
        elif task_type == TaskType.SeqenceLabeling:
            pooled_output = sequence_output
            pooled_output = self.dropout_list[task_id](pooled_output)
            pooled_output = pooled_output.contiguous().view(-1, pooled_output.size(2))
            logits = self.scoring_list[task_id](pooled_output)
            return logits
        elif task_type == TaskType.MaskLM:
            sequence_output = self.dropout_list[task_id](sequence_output)
            logits = self.scoring_list[task_id](sequence_output)
            return logits
        else:
            if decoder_opt == 1:
                max_query = hyp_mask.size(1)
                assert max_query > 0
                assert premise_mask is not None
                assert hyp_mask is not None
                hyp_mem = sequence_output[:, :max_query, :]
                logits = self.scoring_list[task_id](sequence_output, hyp_mem, premise_mask, hyp_mask)
            else:
                pooled_output = self.dropout_list[task_id](pooled_output)
                logits = self.scoring_list[task_id](pooled_output)
            return logits
