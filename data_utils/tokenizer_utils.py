# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
# by xiaodong liu
from transformers import AutoTokenizer

def create_tokenizer(model, transformer_cache):
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=transformer_cache)
    return tokenizer