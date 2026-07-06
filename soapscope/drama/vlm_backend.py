"""Lazy local vision-language backend for grounding captions in appearance.

Default is **BLIP** (`Salesforce/blip-image-captioning-base`): a purpose-built
image captioner that is fast on CPU (~0.6 s) and actually describes what it sees
(colour/shape), unlike a tiny instruct-VLM which hallucinates on abstract
microbe blobs and is ~60× slower. ``describe(image)`` returns a short grounded
phrase (e.g. "glowing green"); the captioner styles that into a soap-opera line.

An instruct-VLM path (SmolVLM etc.) is kept for models whose id looks like a
chat VLM, but BLIP is recommended on CPU. Isolated so torch/transformers only
import when the ``vlm`` backend is actually used.
"""

from __future__ import annotations

_BLIP_PREFIX = "a microscopic creature that is"


class LocalVLM:
    def __init__(self, cfg):
        import torch
        self._torch = torch
        self.cfg = cfg
        mid = cfg.vlm_model
        self.is_blip = "blip" in mid.lower()
        if self.is_blip:
            from transformers import BlipForConditionalGeneration, BlipProcessor
            self.processor = BlipProcessor.from_pretrained(mid)
            self.model = BlipForConditionalGeneration.from_pretrained(mid, torch_dtype="auto")
        else:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(mid)
            self.model = AutoModelForImageTextToText.from_pretrained(mid, torch_dtype="auto")
        self.model.eval()
        torch.manual_seed(cfg.seed)

    def describe(self, image) -> str:
        """Return a short phrase grounded in the thumbnail's appearance."""
        torch = self._torch
        if self.is_blip:
            inputs = self.processor(image, _BLIP_PREFIX, return_tensors="pt")
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=20)
            text = self.processor.decode(out[0], skip_special_tokens=True).strip()
            if text.lower().startswith(_BLIP_PREFIX):
                text = text[len(_BLIP_PREFIX):].strip()
            return text
        # Instruct-VLM chat path.
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "In 6 words, describe this creature's appearance."}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=int(self.cfg.vlm_max_tokens),
                                      do_sample=True, temperature=0.7)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(gen, skip_special_tokens=True).strip()


def load_vlm(cfg) -> LocalVLM:
    try:
        return LocalVLM(cfg)
    except Exception as e:  # noqa: BLE001 - surface an actionable message
        raise ImportError(
            "VLM narrator unavailable. Install and provide a model:\n"
            "  pip install -e .[vlm]   # transformers + accelerate (+ CPU torch)\n"
            f"model={getattr(cfg, 'vlm_model', '?')} — {type(e).__name__}: {e}"
        ) from e
