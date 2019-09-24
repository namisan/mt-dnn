from data_utils.task_def import EncoderModelType


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