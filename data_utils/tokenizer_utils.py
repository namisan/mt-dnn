# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
# by xiaodong liu
from transformers import AutoTokenizer

def create_tokenizer(model, transformer_cache, use_fast=True):
    if 'opt' in model:
        use_fast = False
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=transformer_cache, use_fast=use_fast)
    return tokenizer