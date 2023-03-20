from transformers import (
    BertConfig, BertModel, BertTokenizer,
    RobertaConfig, RobertaModel, RobertaTokenizer,
    AlbertConfig, AlbertModel, AlbertTokenizer,
    XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer,
    ElectraConfig, ElectraModel, ElectraTokenizer,
    T5Config, T5EncoderModel, T5ForConditionalGeneration, T5Tokenizer,
    DebertaConfig, DebertaModel, DebertaTokenizer
)
from module.san_model import SanModel
from msrt5.modeling_t5 import MSRT5ForConditionalGeneration, MSRT5EncoderModel
from msrt5.configuration import MSRT5Config
from msrt5.tokenization_t5 import MSRT5Tokenizer

MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer),
    "xlm": (XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer),
    "san": (BertConfig, SanModel, BertTokenizer),
    "electra": (ElectraConfig, ElectraModel, ElectraTokenizer),
    "t5": (T5Config, T5EncoderModel, T5Tokenizer),
    "deberta": (DebertaConfig, DebertaModel, DebertaTokenizer),
    "t5g": (T5Config, T5ForConditionalGeneration, T5Tokenizer),
    "msrt5g": (MSRT5Config, MSRT5ForConditionalGeneration, MSRT5Tokenizer),
    "msrt5": (MSRT5Config, MSRT5EncoderModel, MSRT5Tokenizer),
}
