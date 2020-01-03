import json
import string
import collections
from data_utils.task_def import EncoderModelType

def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

def calc_tokenized_span_range(context, question, answer, answer_start, answer_end, tokenizer, encoderModelType,
                              verbose=False):
    """
    :param context:
    :param question:
    :param answer:
    :param answer_start:
    :param answer_end:
    :param tokenizer:
    :param encoderModelType:
    :param verbose:
    :return: span_start, span_end
    """
    assert encoderModelType == EncoderModelType.BERT
    prefix = context[:answer_start]
    prefix_tokens = tokenizer.tokenize(prefix)
    full = context[:answer_end]
    full_tokens = tokenizer.tokenize(full)
    span_start = len(prefix_tokens)
    span_end = len(full_tokens)
    span_tokens = full_tokens[span_start: span_end]
    recovered_answer = " ".join(span_tokens).replace(" ##", "")
    cleaned_answer = " ".join(tokenizer.basic_tokenizer.tokenize(answer))
    if verbose:
        try:
            assert recovered_answer == cleaned_answer, "answer: %s, recovered_answer: %s, question: %s, select:%s ext_select:%s context: %s" % (
                cleaned_answer, recovered_answer, question, context[answer_start:answer_end],
                context[answer_start - 5:answer_end + 5], context)
        except Exception as e:
            pass
            print(e)
    return span_start, span_end

def is_valid_sample(context, answer_start, answer_end, answer):
    valid = True
    constructed = context[answer_start: answer_end]
    if constructed.lower() != answer.lower():
        valid = False
        return valid
    # check if it is inside of a token
    if answer_start > 0 and answer_end < len(context) - 1:
        prefix = context[answer_start - 1: answer_start]
        suffix = context[answer_end: answer_end + 1]
        if len(remove_punc(prefix)) > 0 or len(remove_punc(suffix)):
            valid = False
    return valid

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def parse_squad_label(label):
    """
    :param label:
    :return: answer_start, answer_end, answer, is_impossible
    """
    answer_start, answer_end, is_impossible, answer = label.split(":::")
    answer_start = int(answer_start)
    answer_end = int(answer_end)
    is_impossible = int(is_impossible)
    return answer_start, answer_end, answer, is_impossible


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    # It is copyed from: https://github.com/google-research/bert/blob/master/run_squad.py
    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # It is copyed from: https://github.com/google-research/bert/blob/master/run_squad.py
    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def doc_split(doc_subwords, doc_stride=180, max_tokens_for_doc=384):
    _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(doc_subwords):
        length = len(doc_subwords) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(doc_subwords):
            break
        start_offset += min(length, doc_stride)
    return doc_spans

def recompute_span(answer, answer_offset, char_to_word_offset):
    answer_length = len(answer)
    start_position = char_to_word_offset[answer_offset]
    end_position = char_to_word_offset[answer_offset + answer_length - 1]
    return start_position, end_position

def is_valid_answer(context, answer_start, answer_end, answer):
    valid = True
    constructed = ' '.join(context[answer_start: answer_end + 1]).lower()
    cleaned_answer_text = ' '.join(answer.split()).lower()
    if constructed.find(cleaned_answer_text) == -1:
        valid = False
    return valid

def token_doc(paragraph_text):
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in paragraph_text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)
    return doc_tokens, char_to_word_offset

class InputFeatures(object):
    def __init__(self,
                unique_id,
                example_index,
                doc_span_index,
                tokens,
                token_to_orig_map,
                token_is_max_context,
                input_ids,
                input_mask,
                segment_ids,
                start_position=None,
                end_position=None,
                is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return json.dumps({
            'unique_id': self.unique_id,
            'example_index': self.example_index,
            'doc_span_index':self.doc_span_index,
            'tokens': self.tokens,
            'token_to_orig_map': self.token_to_orig_map,
            'token_is_max_context': self.token_is_max_context,
            'input_ids' : self.input_ids,
            'input_mask': self.input_mask,
            'segment_ids': self.segment_ids,
            'start_position': self.start_position,
            'end_position': self.end_position,
            'is_impossible': self.is_impossible
            })

def mrc_feature(tokenizer, unique_id, example_index, query, doc_tokens, answer_start_adjusted, answer_end_adjusted, is_impossible, max_seq_len, max_query_len, doc_stride, answer_text=None, is_training=True):
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    query_ids = tokenizer.tokenize(query)
    query_ids = query_ids[0: max_query_len] if len(query_ids) > max_query_len else query_ids
    max_tokens_for_doc = max_seq_len - len(query_ids) - 3
    unique_id_cp = unique_id
    for (i, token) in enumerate(doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
    tok_start_position = None
    tok_end_position = None
    if is_training and is_impossible:
        tok_start_position = -1
        tok_end_position = -1
    if is_training and not is_impossible:
        tok_start_position = orig_to_tok_index[answer_start_adjusted]
        if answer_end_adjusted < len(doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[answer_end_adjusted + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
            answer_text)
 
    doc_spans = doc_split(all_doc_tokens, doc_stride=doc_stride,
                                        max_tokens_for_doc=max_tokens_for_doc)
    feature_list = []
    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = ["[CLS]"] + query_ids + ["[SEP]"]
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = [0 for i in range(len(tokens))]

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                    split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        start_position = None
        end_position = None
        if is_training and not is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False
            if not (tok_start_position >= doc_start and
                    tok_end_position <= doc_end):
                out_of_span = True
            if out_of_span:
                start_position = 0
                end_position = 0
            else:
                doc_offset = len(query_ids) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        if is_training and is_impossible:
            start_position = 0
            end_position = 0
        is_impossible = True if is_impossible else False
        feature = InputFeatures(
          unique_id=unique_id_cp,
          example_index=example_index,
          doc_span_index=doc_span_index,
          tokens=tokens,
          token_to_orig_map=token_to_orig_map,
          token_is_max_context=token_is_max_context,
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
          start_position=start_position,
          end_position=end_position,
          is_impossible=is_impossible)
        feature_list.append(feature)
        unique_id_cp += 1
    return feature_list


