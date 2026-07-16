"""Hugging Face wrapper around a T5 Gemma backbone with a numeric decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from transformers import T5GemmaConfig, T5GemmaForConditionalGeneration
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers import PreTrainedModel, GenerationMixin

from .configuration_regresslm import RegressLMConfig
from .tokenization_p10 import IEEEFloatTokenizer, P10Tokenizer


@dataclass
class RegressLMOutput(Seq2SeqLMOutput):
    """Extends the default seq2seq output with optional regression logits."""

    regression_logits: Optional[torch.Tensor] = None


class _NumericConstraintHelper:
    """Utility that mirrors the `DecoderVocab` logic for numeric decoding."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.num_tokens_per_obj = tokenizer.num_tokens_per_obj
        self.pad_token_id = tokenizer.pad_token_id

    def allowed_token_ids(self, prev_token_ids: Sequence[int]) -> list[int]:
        return self.tokenizer.possible_next_token_ids(prev_token_ids)

    def decode(self, token_ids: Sequence[int]) -> list[float]:
        return self.tokenizer.token_ids_to_floats(token_ids)


class _NumericLogitsProcessor(LogitsProcessor):
    """Constrains generation so only valid numeric tokens appear."""

    def __init__(self, helper: _NumericConstraintHelper):
        self.helper = helper

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:  # type: ignore[override]
        batch, _ = input_ids.shape
        updated_scores = scores.clone()
        for row in range(batch):
            prev_ids = input_ids[row].tolist()
            allowed = self.helper.allowed_token_ids(prev_ids)
            vocab = updated_scores.shape[-1]
            if any(i < 0 or i >= vocab for i in allowed):
                raise ValueError(
                    f"Numeric constraint produced out-of-range id(s): "
                    f"max={max(allowed)}, vocab={vocab}. "
                    f"Check tokenizer <-> decoder vocab alignment."
                )
            mask = torch.full_like(updated_scores[row], float("-inf"))
            mask[allowed] = 0.0
            updated_scores[row] = updated_scores[row] + mask
        return updated_scores


class RegressLMForConditionalGeneration(PreTrainedModel, GenerationMixin):
    """Drop-in Hugging Face model that mirrors ``PyTorchModel`` for inference."""

    config_class = RegressLMConfig
    base_model_prefix = "model"

    def __init__(self, config: RegressLMConfig) -> None:
        super().__init__(config)
        backbone_cfg = T5GemmaConfig(**config.backbone_config)
        self.model = T5GemmaForConditionalGeneration(backbone_cfg)

        # Encoder vocabulary: optionally resize the shared embedding.
        if config.encoder_vocab_size is not None:
            cur = self.model.get_input_embeddings().num_embeddings
            if cur != config.encoder_vocab_size:
                self.model.resize_token_embeddings(config.encoder_vocab_size)

        # Decoder vocabulary: always detach from the shared embedding so we can
        # host the numeric tokens.
        if config.decoder_vocab_size is not None:
            self._resize_decoder_vocab(config.decoder_vocab_size)

        hidden_size = getattr(self.model.config.encoder, "d_model", None)
        if hidden_size is None:
            hidden_size = getattr(self.model.config.encoder, "hidden_size")
        if hidden_size is None:
            raise ValueError("Unable to infer hidden size from backbone config.")

        self.use_regression_head = config.use_regression_head
        if self.use_regression_head:
            self.regression_head = nn.Linear(hidden_size, 1)
        else:
            self.regression_head = None

        decoder_spec = getattr(config, "decoder_tokenizer", "P10").upper()
        if decoder_spec.startswith("IEEE"):
            mantissa_digits = getattr(config, "ieee_mantissa_digits", None)
            exponent_digits = getattr(config, "ieee_exponent_digits", None)
            if mantissa_digits is None or exponent_digits is None:
                raise ValueError(
                    "Config missing IEEE tokenizer parameters: `ieee_mantissa_digits` and `ieee_exponent_digits`."
                )
            tokenizer = IEEEFloatTokenizer(
                base=getattr(config, "ieee_base", 10),
                num_mantissa_digits=mantissa_digits,
                num_exponent_digits=exponent_digits,
            )
        else:
            tokenizer = P10Tokenizer(
                num_digits=getattr(config, "num_digits", 6),
                exponent_range=getattr(config, "exponent_range", 10),
            )

        self.constraint_helper = _NumericConstraintHelper(tokenizer)
        # Sanity-check: decoder vocab size must match numeric tokenizer size
        if (config.decoder_vocab_size is not None and
            config.decoder_vocab_size != self.constraint_helper.tokenizer.decoder_vocab_size):
            raise ValueError(
                f"Decoder vocab mismatch: model={config.decoder_vocab_size} "
                f"tokenizer={self.constraint_helper.tokenizer.decoder_vocab_size}. "
                "Make sure the tokenizer does NOT add PAD and preserves training order."
            )
        self.num_tokens_per_obj = config.num_tokens_per_obj
        self.max_num_objs = config.max_num_objs
        self.decoder_start_token_id = config.pad_token_id
        self.post_init()

    # ------------------------------------------------------------------
    # Helpers mirroring PyTorchModel utilities
    # ------------------------------------------------------------------
    def _resize_decoder_vocab(self, vocab_size: int) -> None:
        decoder = self.model.model.decoder.embed_tokens
        if decoder.num_embeddings != vocab_size:
            self.model.model.decoder.embed_tokens = nn.Embedding(
                vocab_size,
                decoder.embedding_dim,
                padding_idx=decoder.padding_idx,
            )
        hidden = getattr(self.model.config.encoder, "d_model", None)
        if hidden is None:
            hidden = getattr(self.model.config.encoder, "hidden_size")
        if hasattr(self.model.lm_head, "out_proj"):
            self.model.lm_head.out_proj = nn.Linear(hidden, vocab_size, bias=False)
        else:
            self.model.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        self.model.register_buffer("final_logits_bias", torch.zeros((1, vocab_size)))
        if hasattr(self.model.config, "tie_word_embeddings"):
            self.model.config.tie_word_embeddings = False

    def _pool_encoder_outputs(self, memory: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        mask = (~pad_mask).unsqueeze(-1).type_as(memory)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (memory * mask).sum(dim=1) / denom

    def get_encoder(self):  # type: ignore[override]
        return self.model.get_encoder()

    def get_decoder(self):  # type: ignore[override]
        return self.model.get_decoder()

    def get_input_embeddings(self):  # type: ignore[override]
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):  # type: ignore[override]
        self.model.set_input_embeddings(new_embeddings)

    def tie_weights(self, **kwargs):  # type: ignore[override]
        # Word embeddings are intentionally untied once the decoder vocab is replaced.
        # REPRO PATCH (autoloop, 2026-07-16): transformers>=5.x calls
        # tie_weights(recompute_mapping=...); accept **kwargs for compatibility.
        pass

    # ------------------------------------------------------------------
    # Forward pass mirroring the training-time behaviour
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        regression_targets: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> RegressLMOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.use_regression_head:
            if input_ids is None:
                raise ValueError("`input_ids` must be provided for regression inference.")
            if attention_mask is None:
                attention_mask = (input_ids != self.config.pad_token_id).long()
            encoder_out = self.model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            memory = encoder_out.last_hidden_state
            pad_mask = attention_mask == 0
            pooled = self._pool_encoder_outputs(memory, pad_mask)
            preds = self.regression_head(pooled).squeeze(-1)
            loss = None
            target = regression_targets if regression_targets is not None else labels
            if target is not None:
                loss = F.mse_loss(preds, target.to(preds.dtype))
            if not return_dict:
                return (loss, preds) if loss is not None else (preds,)
            return RegressLMOutput(
                loss=loss,
                logits=None,
                regression_logits=preds,
                encoder_last_hidden_state=memory,
            )

        proc_labels = None
        if labels is not None:
            proc_labels = labels.clone()
            proc_labels[proc_labels == self.config.pad_token_id] = -100
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=proc_labels,
            return_dict=True,
            **kwargs,
        )
        if not return_dict:
            if outputs.loss is None:
                return (outputs.logits,)
            return (outputs.loss, outputs.logits)
        return RegressLMOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
        )

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------
    def prepare_inputs_for_generation(self, *args, **kwargs):  # type: ignore[override]
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def _get_logits_processor(  # type: ignore[override]
        self,
        generation_config,
        input_ids_seq_length=None,
        encoder_input_ids=None,
        prefix_allowed_tokens_fn=None,
        logits_processor=None,
        device=None,
        model_kwargs=None,
        negative_prompt_ids=None,
        negative_prompt_attention_mask=None,
    ):
        processors = super()._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=encoder_input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        processors.append(_NumericLogitsProcessor(self.constraint_helper))
        return processors

    def generate(self, *args, **kwargs):  # type: ignore[override]
        if "decoder_start_token_id" not in kwargs:
            kwargs["decoder_start_token_id"] = self.config.pad_token_id
        if "max_new_tokens" not in kwargs and "max_length" not in kwargs:
            kwargs["max_new_tokens"] = self.config.max_num_objs * self.num_tokens_per_obj
        return super().generate(*args, **kwargs)

    # ------------------------------------------------------------------
    # Convenience helper used after generation
    # ------------------------------------------------------------------
    def decode_to_floats(self, sequences: torch.Tensor | Sequence[Sequence[int]]) -> list[list[float]]:
        if isinstance(sequences, torch.Tensor):
            iterable = sequences.cpu().tolist()
        else:
            iterable = sequences
        return [self.constraint_helper.decode(seq) for seq in iterable]
