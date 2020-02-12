from transformers import *
MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetModel, XLNetTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer),
}