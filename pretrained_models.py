from transformers import (
    BertConfig, BertModel, BertTokenizer,
    RobertaConfig, RobertaModel, RobertaTokenizer,
    AlbertConfig, AlbertModel, AlbertTokenizer,
    XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer,
    ElectraConfig, ElectraModel, ElectraTokenizer,
    T5Config, T5EncoderModel, T5ForConditionalGeneration, T5Tokenizer,
    DebertaConfig, DebertaModel, DebertaTokenizer,
    MistralConfig, MistralModel, MistralForCausalLM, LlamaTokenizer,
)
from module.san_model import SanModel
from mixtral.modeling_mixtral import MixtralModel, MixtralForCausalLM
from mixtral.configuration_mixtral import MixtralConfig 
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
    # "msrt5g": (MSRT5Config, MSRT5ForConditionalGeneration, MSRT5Tokenizer),
    # "msrt5": (MSRT5Config, MSRT5EncoderModel, MSRT5Tokenizer),
    "mistral": (MistralConfig, MistralForCausalLM, LlamaTokenizer),
    "mixtral": (MixtralConfig, MixtralForCausalLM, LlamaTokenizer)
}
