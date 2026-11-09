from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .retrieval import Hit

SYSTEM_PROMPT = (
    "You answer questions about the FastAPI documentation using only the numbered sources provided. "
    "Cite every claim with the source number in square brackets, e.g. [1]. "
    "If the sources do not contain the answer, say you could not find it in the documentation."
)


class Generator(Protocol):
    def generate(self, system: str, user: str) -> str: ...


@dataclass(frozen=True)
class Citation:
    ref: int
    file: str
    section: str
    anchor: str

    def render(self) -> str:
        return f"[{self.ref}] {self.file} § {self.section}"


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    citations: list[Citation]
    cited_refs: list[int] = field(default_factory=list)
    invalid_refs: list[int] = field(default_factory=list)

    def render(self) -> str:
        sources = "\n".join(c.render() for c in self.citations)
        cited = ", ".join(f"[{r}]" for r in self.cited_refs) or "none"
        invalid = ", ".join(f"[{r}]" for r in self.invalid_refs) or "none"
        return f"{self.text}\n\nSources:\n{sources}\n\nCited in answer: {cited}. Invalid citations: {invalid}."


def citations_for(hits: Sequence[Hit]) -> list[Citation]:
    return [
        Citation(ref=i + 1, file=h.chunk.file, section=h.chunk.title, anchor=h.chunk.anchor)
        for i, h in enumerate(hits)
    ]


def build_prompt(question: str, hits: Sequence[Hit], max_chars: int = 1200) -> str:
    blocks = [
        f"[{i + 1}] {h.chunk.file} § {h.chunk.title}\n{h.chunk.text[:max_chars]}" for i, h in enumerate(hits)
    ]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer:"


def parse_refs(text: str, available: int) -> tuple[list[int], list[int]]:
    refs = sorted({int(n) for group in re.findall(r"\[([\d,\s]+)\]", text) for n in re.findall(r"\d+", group)})
    valid = [r for r in refs if 1 <= r <= available]
    invalid = [r for r in refs if not 1 <= r <= available]
    return valid, invalid


def answer(question: str, hits: Sequence[Hit], generator: Generator | None = None) -> Answer:
    citations = citations_for(hits)
    if not hits:
        return Answer(question, "I could not find this in the documentation.", [])
    if generator is None:
        top = hits[0].chunk
        excerpt = " ".join(top.text.split()[:80])
        return Answer(question, f"{excerpt} [1]", citations, [1], [])
    text = generator.generate(SYSTEM_PROMPT, build_prompt(question, hits)).strip()
    valid, invalid = parse_refs(text, len(hits))
    return Answer(question, text, citations, valid, invalid)


class HFGenerator:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens: int = 300) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.max_new_tokens = max_new_tokens

    def generate(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        return str(self.tokenizer.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True))
