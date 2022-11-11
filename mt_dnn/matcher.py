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


class SANBertNetwork(nn.Module):
    def __init__(self, opt, bert_config=None, initial_from_local=False):
        super(SANBertNetwork, self).__init__()
        self.dropout_list = nn.ModuleList()

        if opt["encoder_type"] not in EncoderModelType._value2member_map_:
            raise ValueError("encoder_type is out of pre-defined types")
        self.encoder_type = opt["encoder_type"]
        self.preloaded_config = None

        literal_encoder_type = EncoderModelType(self.encoder_type).name.lower()
        config_class, model_class, _ = MODEL_CLASSES[literal_encoder_type]
        if not initial_from_local:
            # self.bert = model_class.from_pretrained(opt['init_checkpoint'], config=self.preloaded_config)
            self.bert = model_class.from_pretrained(
                opt["init_checkpoint"], cache_dir=opt["transformer_cache"]
            )
        else:
            self.preloaded_config = config_class.from_dict(opt)  # load config from opt
            self.preloaded_config.output_hidden_states = (
                True  # return all hidden states
            )
            self.bert = model_class(self.preloaded_config)

        hidden_size = self.bert.config.hidden_size

        if opt.get("dump_feature", False):
            self.config = opt
            return
        if opt["update_bert_opt"] > 0:
            for p in self.bert.parameters():
                p.requires_grad = False

        task_def_list = opt["task_def_list"]
        self.task_def_list = task_def_list

        # create output header
        self.scoring_list = nn.ModuleList()

        for task_id in range(len(task_def_list)):
            task_def: TaskDef = task_def_list[task_id]
            lab = task_def.n_class
            task_type = task_def.task_type
            task_obj = tasks.get_task_obj(task_def)
            if task_obj is not None:
                out_proj = task_obj.train_build_task_layer(
                    hidden_size, task_def, opt
                )

            elif task_type == TaskType.ClozeChoice:
                self.pooler = Pooler(
                    hidden_size, dropout_p=opt["dropout_p"], actf=opt["pooler_actf"]
                )
                out_proj = nn.Linear(hidden_size, lab)
            else:
                raise NotImplementedError()
            self.scoring_list.append(out_proj)
        self.config = opt

    def embed_encode(self, input_ids, token_type_ids=None, attention_mask=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        embedding_output = self.bert.embeddings(input_ids, token_type_ids)
        return embedding_output

    def encode(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        inputs_embeds=None,
        y_input_ids=None,
    ):
        if (self.encoder_type == EncoderModelType.T5 or
        self.encoder_type == EncoderModelType.OPT or
        self.encoder_type == EncoderModelType.OPTG
        ):
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            )
            last_hidden_state = outputs.last_hidden_state
            all_hidden_states = outputs.hidden_states  # num_layers + 1 (embeddings)
        elif self.encoder_type == EncoderModelType.T5G:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=y_input_ids,
            )
            # return logits from LM header
            last_hidden_state = outputs.logits
            all_hidden_states = (
                outputs.encoder_last_hidden_state
            )  # num_layers + 1 (embeddings)
        else:
            outputs = self.bert(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            )
            last_hidden_state = outputs.last_hidden_state
            all_hidden_states = outputs.hidden_states  # num_layers + 1 (embeddings)
        return last_hidden_state, all_hidden_states

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        premise_mask=None,
        hyp_mask=None,
        task_id=0,
        y_input_ids=None,
        fwd_type=0,
        embed=None,
    ):
        if fwd_type == 3:
            generated = self.bert.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.config["max_answer_len"],
                num_beams=self.config["num_beams"],
                repetition_penalty=self.config["repetition_penalty"],
                length_penalty=self.config["length_penalty"],
                early_stopping=True,
            )
            return generated
        elif fwd_type == 2:
            assert embed is not None
            last_hidden_state, all_hidden_states = self.encode(
                None, token_type_ids, attention_mask, embed, y_input_ids
            )
        elif fwd_type == 1:
            return self.embed_encode(input_ids, token_type_ids, attention_mask)
        else:
            last_hidden_state, all_hidden_states = self.encode(
                input_ids, token_type_ids, attention_mask, y_input_ids=y_input_ids
            )
        task_obj = tasks.get_task_obj(self.task_def_list[task_id])
        task_type = task_obj._task_def.task_type
        if task_obj is not None:
            logits = task_obj.train_forward(
                last_hidden_state,
                premise_mask,
                hyp_mask,
                self.scoring_list[task_id],
                enable_san=task_obj._task_def.enable_san
            )
            return logits
        elif task_type == TaskType.ClozeChoice:
            pooled_output = self.pooler(last_hidden_state)
            pooled_output = self.dropout_list[task_id](pooled_output)
            logits = self.scoring_list[task_id](pooled_output)
            return logits
        else:
            raise NotImplementedError()
