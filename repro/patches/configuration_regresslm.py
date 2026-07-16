"""Configuration for exporting a RegressLM checkpoint to Hugging Face."""

from __future__ import annotations

from typing import Any, Dict, Optional

from transformers import PretrainedConfig


class RegressLMConfig(PretrainedConfig):
    """Configuration that mirrors the minimal knobs required for inference.

    The config keeps track of the underlying T5 Gemma backbone as well as the
    decoder settings that determine the custom numeric vocabulary.
    """

    model_type = "regresslm"
    is_encoder_decoder = True
    auto_map = {
        "AutoConfig": "configuration_regresslm.RegressLMConfig",
        "AutoModelForSeq2SeqLM": "modeling_regresslm.RegressLMForConditionalGeneration",
        "AutoTokenizer": "tokenization_p10.P10Tokenizer",
    }

    def __init__(
        self,
        *,
        backbone_model_name: str = "google/t5gemma-s-s-prefixlm",
        backbone_config: Optional[Dict[str, Any]] = None,
        encoder_vocab_size: Optional[int] = None,
        decoder_vocab_size: Optional[int] = None,
        decoder_tokenizer: str = "P10",
        num_digits: int = 6,
        exponent_range: int = 10,
        ieee_base: int = 10,
        ieee_mantissa_digits: Optional[int] = None,
        ieee_exponent_digits: Optional[int] = None,
        tokenizer_class: Optional[str] = None,
        max_num_objs: int = 1,
        use_regression_head: bool = False,
        pad_token_id: int = 0,
        bos_token_id: int | None = 0,
        eos_token_id: int | None = 0,
        **kwargs: Any,
    ) -> None:
        pad_token_id = kwargs.pop("pad_token_id", pad_token_id or 0)
        bos_token_id = kwargs.pop("bos_token_id", bos_token_id if bos_token_id is not None else pad_token_id)
        eos_token_id = kwargs.pop("eos_token_id", eos_token_id if eos_token_id is not None else pad_token_id)
        decoder_start_token_id = kwargs.pop("decoder_start_token_id", pad_token_id)
        is_encoder_decoder = kwargs.pop("is_encoder_decoder", True)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )
        self.backbone_model_name = backbone_model_name
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.decoder_tokenizer = decoder_tokenizer
        self.max_num_objs = max(1, int(max_num_objs))
        self.use_regression_head = bool(use_regression_head)
        self.ieee_base = int(ieee_base)
        tokenizer_mapping = dict(self.auto_map)
        if tokenizer_class is None:
            tokenizer_class = (
                "tokenization_p10.P10Tokenizer"
                if decoder_tokenizer.upper() == "P10"
                else "tokenization_p10.IEEEFloatTokenizer"
            )
        # Use list-form auto_map (slow, fast)
        tokenizer_mapping["AutoTokenizer"] = [tokenizer_class, None]
        self.auto_map = tokenizer_mapping
        # Keep tokenizer_class as the bare name for broad HF compatibility
        self.tokenizer_class = tokenizer_class.rsplit(".", 1)[-1]

        decoder_spec = decoder_tokenizer.upper()
        if decoder_spec == "P10":
            self.num_digits = int(num_digits)
            self.exponent_range = int(exponent_range)
            self.ieee_mantissa_digits = int(ieee_mantissa_digits) if ieee_mantissa_digits is not None else None
            self.ieee_exponent_digits = int(ieee_exponent_digits) if ieee_exponent_digits is not None else None
            self.num_tokens_per_obj = 2 + self.num_digits
        elif decoder_spec.startswith("IEEE"):
            if ieee_mantissa_digits is None or ieee_exponent_digits is None:
                raise ValueError(
                    "IEEE decoder tokenizer requires both `ieee_mantissa_digits` and `ieee_exponent_digits`."
                )
            self.num_digits = int(num_digits)
            self.exponent_range = int(exponent_range)
            self.ieee_mantissa_digits = int(ieee_mantissa_digits)
            self.ieee_exponent_digits = int(ieee_exponent_digits)
            self.num_tokens_per_obj = 2 + self.ieee_exponent_digits + self.ieee_mantissa_digits
        else:
            raise ValueError(f"Unsupported decoder tokenizer specification: {decoder_tokenizer}")

        if backbone_config is None:
            # REPRO PATCH (autoloop, 2026-07-16): the ONLY caller that reaches here
            # with backbone_config=None is the throwaway default instance transformers
            # builds internally for to_diff_dict()/repr during from_pretrained. The REAL
            # instance always carries a serialized backbone_config in config.json (else
            # branch below). Do NOT auto-download the gated google/t5gemma-s-s-prefixlm
            # for that throwaway default -- leave backbone_config None and continue.
            self.backbone_config = None
        else:
            self.backbone_config = backbone_config

        # Generation defaults – by design we only emit numeric tokens.
        total_decode_tokens = self.max_num_objs * self.num_tokens_per_obj
        self.max_new_tokens = kwargs.get("max_new_tokens", total_decode_tokens)
        self.decoder_start_token_id = decoder_start_token_id

    @property
    def p10_vocab_size(self) -> Optional[int]:
        """Returns the decoder vocab size that corresponds to the numeric tokens."""

        return self.decoder_vocab_size
