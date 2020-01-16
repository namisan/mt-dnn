# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertConfig, BertLayerNorm, BertModel, BertEmbeddings

from module.dropout_wrapper import DropoutWrapper
from module.san import SANClassifier, MaskLmHeader
from module.sub_layers import RnnEncoder
from module.similarity import SelfAttnWrapper
from data_utils.task_def import EncoderModelType, TaskType


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

class SANNetwork(nn.Module):
    def __init__(self, config: BertConfig, opt):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.rnn = RnnEncoder(config.hidden_size, config.hidden_size, config.num_hidden_layers, True, 
                              config.hidden_dropout_prob)
        my_dropout = DropoutWrapper(config.hidden_dropout_prob, opt['vb_dropout'])
        self.self_att = SelfAttnWrapper(config.hidden_size, dropout=my_dropout)
        self.config = config

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        embedding_output = self.embeddings(input_ids, token_type_ids)
        rnn_output = self.rnn(embedding_output)[0]
        size = rnn_output.shape
        max_output = rnn_output.view(size[0], size[1], self.config.hidden_size, 2).max(-1)[0]
        pooled_output = self.self_att(max_output, attention_mask == 0)
        sequence_output = max_output
        return sequence_output, pooled_output

class SANBertNetwork(nn.Module):
    def __init__(self, opt, bert_config=None):
        super(SANBertNetwork, self).__init__()
        self.encoder_type = opt['encoder_type']
        if opt['encoder_type'] == EncoderModelType.ROBERTA:
            from fairseq.models.roberta import RobertaModel
            self.bert = RobertaModel.from_pretrained(opt['init_checkpoint'])
            hidden_size = self.bert.args.encoder_embed_dim
            self.pooler = LinearPooler(hidden_size)
        elif opt['encoder_type'] == EncoderModelType.BERT:
            self.bert_config = BertConfig.from_dict(opt)
            self.bert = BertModel(self.bert_config)
            hidden_size = self.bert_config.hidden_size
        else:
            self.bert_config = BertConfig.from_dict(opt)
            self.bert = SANNetwork(self.bert_config, opt)
            hidden_size = self.bert_config.hidden_size

        if opt.get('dump_feature', False):
            self.opt = opt
            return
        if opt['update_bert_opt'] > 0:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.decoder_opt = opt['answer_opt']
        self.task_types = opt["task_types"]

        # create output header
        self.scoring_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        labels = [int(ls) for ls in opt['label_size'].split(',')]
        task_dropout_p = opt['tasks_dropout_p']
        for task, lab in enumerate(labels):
            decoder_opt = self.decoder_opt[task]
            task_type = self.task_types[task]
            dropout = DropoutWrapper(task_dropout_p[task], opt['vb_dropout'])
            self.dropout_list.append(dropout)
            if task_type == TaskType.Span:
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

    def _my_init(self):
        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=0.02 * self.opt['init_ratio'])
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
                if module.bias is not None:
                    module.bias.data.zero_()

        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, premise_mask=None, hyp_mask=None, task_id=0):
        if self.encoder_type == EncoderModelType.ROBERTA:
            sequence_output = self.bert.extract_features(input_ids)
            pooled_output = self.pooler(sequence_output)
        elif self.encoder_type == EncoderModelType.BERT:
            all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]
        elif self.encoder_type == EncoderModelType.SAN:
            sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        else:
            raise NotImplemented("Unsupported encoder type %s" % self.encoder_type)

        decoder_opt = self.decoder_opt[task_id]
        task_type = self.task_types[task_id]
        if task_type == TaskType.Span:
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
