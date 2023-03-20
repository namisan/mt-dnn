from typing import Mapping

from transformers.configuration_utils import PretrainedConfig
from transformers.onnx import OnnxSeq2SeqConfigWithPast
from transformers.utils import logging


class MSRLongT5Config(PretrainedConfig):
    model_type = "msrlongt5"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

    def __init__(
        self,
        vocab_size=32128,
        d_model=512,
        d_kv=64,
        d_ff=2048,
        num_layers=6,
        num_decoder_layers=None,
        num_heads=8,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=1,
        eos_token_id=2,
        bos_token_id=0,
        have_bias=False,
        max_position_embeddings=512,
        normalize_before=False,
        decoder_roberta_lm_head=False,
        local_radius=128, # local attention,
        s4_channels=1,
        s4_state_dim=64,
        s4_n_ssm=768,
        s4_dt_min=0.001,
        s4_dt_max=0.1,
        max_seq_length=None,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # default = symmetry
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache
        self.have_bias=have_bias
        self.max_position_embeddings=max_position_embeddings
        self.normalize_before = normalize_before
        self.decoder_roberta_lm_head = decoder_roberta_lm_head
        # s4
        self.local_radius = local_radius
        self.s4_channels = s4_channels
        self.s4_state_dim = s4_state_dim
        self.s4_n_ssm = s4_n_ssm
        self.s4_lr = {"A": 0.0, "B": 0.0, "dt": None}
        self.s4_every_n_layers = 1000
        self.s4_use_residual_dropout = False
        self.s4_dt_min=s4_dt_min
        self.s4_dt_max=s4_dt_max
        self.max_seq_length = max_seq_length if max_seq_length else max_position_embeddings
        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"
        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer."
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )

        # for backwards compatibility
        if feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )


class MSRLongT5OnnxConfig(OnnxSeq2SeqConfigWithPast):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = {
            "input_ids": {0: "batch", 1: "encoder_sequence"},
            "attention_mask": {0: "batch", 1: "encoder_sequence"},
        }
        if self.use_past:
            common_inputs["attention_mask"][1] = "past_encoder_sequence + sequence"
            common_inputs["decoder_input_ids"] = {0: "batch"}
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
        else:
            common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}

        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")

        return common_inputs

    @property
    def default_onnx_opset(self) -> int:
        return 13
