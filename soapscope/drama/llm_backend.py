"""Lazy local-LLM backend for the narrator.

Isolated so the heavy transformers/torch import only happens when the ``llm``
drama backend is actually used. Loads a small HF instruct model (default
Qwen2.5-0.5B-Instruct, CPU-runnable) and generates one short line per call.

Weights download from HuggingFace (which works behind the agent proxy; GitHub
release assets do not — see CLAUDE.md).
"""

from __future__ import annotations


class LocalLLM:
    """Thin wrapper over a HF causal LM with a chat template."""

    def __init__(self, cfg):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.llm_model, torch_dtype="auto")
        self.model.eval()
        torch.manual_seed(cfg.seed)          # run-level reproducibility, still varied per prompt

    def generate(self, system: str, user: str) -> str:
        torch = self._torch
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=int(self.cfg.llm_max_tokens),
                do_sample=True, temperature=float(self.cfg.llm_temperature),
                top_p=0.9, repetition_penalty=1.3, no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


def load_llm(cfg) -> LocalLLM:
    try:
        return LocalLLM(cfg)
    except Exception as e:  # noqa: BLE001 - surface an actionable message
        raise ImportError(
            "LLM narrator unavailable. Install and provide a model:\n"
            "  pip install -e .[llm]   # transformers + accelerate (+ CPU torch)\n"
            f"model={getattr(cfg, 'llm_model', '?')} — {type(e).__name__}: {e}"
        ) from e
