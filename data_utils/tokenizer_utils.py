# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
# by xiaodong liu

import re
import argparse
import unicodedata
from tqdm import tqdm
import multiprocessing
from functools import partial
from transformers import AutoTokenizer

def create_tokenizer(model, transformer_cache, do_lower_case=False):
    if "msr" in model: 
        from msrt5.tokenization_t5 import MSRT5Tokenizer
        tokenizer = MSRT5Tokenizer.from_pretrained(model, cache_dir=transformer_cache)
        tokenizer.do_lower_case = do_lower_case
        tokenizer.pre_encoder.lower = do_lower_case
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=transformer_cache)
    return tokenizer


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

class SentencepiecePreTokenizer(object):

    def __init__(self, lower=False):
        self.lower = lower
        self.transl_table = dict( [ (ord(x), ord(y)) for x,y in zip( u"‘’´“”—–-",  u"'''\"\"---") ] )

    def handle_single_quote(self, tokens):
        line = ' '.join(tokens)
        line = re.sub(r"' ([smdSMDtT])\b", r"'\1", line)
        line = re.sub(r"' ll\b", "'ll", line)
        line = re.sub(r"' re\b", "'re", line)
        line = re.sub(r"' ve\b", "'ve", line)
        line = re.sub(r"' LL\b", "'LL ", line)
        line = re.sub(r"' RE\b", "'RE ", line)
        line = re.sub(r"' VE\b", "'VE ", line)
        return line.split()

    def split_on_cont_punc(self, tokens):
        new_tokens = []
        for token in tokens:
            if len(token) > 1:
                last_j = 0
                pre_is_punc = _is_punctuation(token[0])
                for j, ch in enumerate(token):
                    is_punc = _is_punctuation(ch)
                    if is_punc != pre_is_punc:
                        new_tokens.append(token[last_j: j])
                        last_j = j
                    pre_is_punc = is_punc
                if last_j < len(token):
                    new_tokens.append(token[last_j:])
            else:
                new_tokens.append(token)
        return new_tokens

    def split_pre_and_post_punc(self, tokens):
        def pre_punc(token):
            last_j = 0
            for j in range(1, len(token)):
                if not _is_punctuation(token[j]):
                    last_j = j
                    break
            return token[:last_j], token[last_j:]
        def post_punc(token):
            last_j = len(token)
            for j in range(len(token) - 2, -1, -1):
                is_punc = _is_punctuation(token[j])
                if not _is_punctuation(token[j]):
                    last_j = j + 1
                    break
            return token[:last_j], token[last_j:]
        new_tokens = []
        for token in tokens:
            if len(token) > 1 and _is_punctuation(token[0]):
                a, b = pre_punc(token)
                if a:
                    new_tokens.append(a)
                if b:
                    if _is_punctuation(b[-1]):
                        c, d = post_punc(b)
                        if c:
                            new_tokens.append(c)
                        if d:
                            new_tokens.append(d)
                    else:
                        new_tokens.append(b)
            elif len(token) > 1 and _is_punctuation(token[-1]):
                a, b = post_punc(token)
                if a:
                    new_tokens.append(a)
                if b:
                    new_tokens.append(b)
            else:
                new_tokens.append(token)
        return new_tokens

    def encode(self, line):
        line = line.strip()
        line = line.replace("``", '"').replace("''", '"')
        line = line.translate(self.transl_table)
        tokens = line.split()
        tokens = self.split_pre_and_post_punc(tokens)
        tokens = self.handle_single_quote(tokens)
        new_line = " ".join(tokens)
        if self.lower:
            new_line = new_line.lower()
        return new_line
