from __future__ import annotations

import asyncio
import os
from functools import cached_property
from pathlib import Path

import httpx
import numpy as np

from .config import EmbeddingConfig


class EmbeddingClient:
    def __init__(self, config: EmbeddingConfig):
        self.config = config

    @property
    def model_key(self) -> str:
        return f"{self.config.backend}:{self.config.model}"

    async def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        if not texts:
            return []
        if self.config.backend == "voyage":
            return await self._embed_voyage(texts, input_type)
        if self.config.backend == "custom":
            return await self._embed_openai_compatible(texts)
        if self.config.backend == "local":
            return self._embed_local(texts)
        raise ValueError(f"unknown embedding backend: {self.config.backend}")

    async def _embed_voyage(self, texts: list[str], input_type: str) -> list[list[float]]:
        api_key = (self.config.api_key or os.environ.get(self.config.api_key_env, "")).strip()
        if not api_key:
            raise RuntimeError(f"missing {self.config.api_key_env}")
        last_error = None
        for attempt in range(5):
            async with httpx.AsyncClient(timeout=120) as client:
                try:
                    res = await client.post(
                        "https://api.voyageai.com/v1/embeddings",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={"input": texts, "model": self.config.model, "input_type": input_type},
                    )
                    res.raise_for_status()
                    data = res.json()["data"]
                    return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]
                except httpx.HTTPStatusError as exc:
                    last_error = exc
                    if exc.response.status_code == 429 and attempt < 4:
                        retry_after = exc.response.headers.get("retry-after")
                        wait = float(retry_after) if retry_after else (2 ** attempt)
                        print(f"Voyage rate limited (429). Waiting {wait:.1f}s before retry {attempt + 1}/4...")
                        await asyncio.sleep(wait)
                        continue
                    raise
        raise last_error or RuntimeError("voyage embedding failed")

    async def _embed_openai_compatible(self, texts: list[str]) -> list[list[float]]:
        if not self.config.base_url:
            raise RuntimeError("custom backend requires embedding.base_url")
        api_key = (self.config.api_key or os.environ.get(self.config.api_key_env, "")).strip()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        async with httpx.AsyncClient(timeout=120) as client:
            res = await client.post(
                self.config.base_url.rstrip("/") + "/embeddings",
                headers=headers,
                json={"input": texts, "model": self.config.model},
            )
            res.raise_for_status()
            data = res.json()["data"]
            return [item["embedding"] for item in sorted(data, key=lambda x: x.get("index", 0))]

    @cached_property
    def _local_model(self):
        model_home = Path.home() / ".codex" / "infinite-memory" / "models" / "huggingface"
        model_home.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(model_home))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(model_home))
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(model_home))
        os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
        os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")

        import torch
        from sentence_transformers import SentenceTransformer

        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        if self.config.model == "Qwen/Qwen3-Embedding-0.6B":
            from transformers import AutoModel, AutoTokenizer

            dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model,
                cache_dir=str(model_home),
            )
            model = AutoModel.from_pretrained(
                self.config.model,
                attn_implementation="eager",
                cache_dir=str(model_home),
                torch_dtype=dtype,
            ).to(device)
            model.eval()
            return {
                "backend": "transformers",
                "device": device,
                "model": model,
                "tokenizer": tokenizer,
            }

        return SentenceTransformer(
            self.config.model,
            cache_folder=str(model_home),
            device=device,
        )

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        if isinstance(self._local_model, dict) and self._local_model["backend"] == "transformers":
            vectors = self._embed_local_transformers(texts)
        else:
            vectors = self._local_model.encode(texts, normalize_embeddings=True)
        return [np.asarray(v, dtype=np.float32).tolist() for v in vectors]

    def _embed_local_transformers(self, texts: list[str]) -> list[list[float]]:
        import torch
        import torch.nn.functional as F

        model_info = self._local_model
        tokenizer = model_info["tokenizer"]
        model = model_info["model"]
        device = model_info["device"]

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        attention_mask = encoded["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(token_embeddings.size(0), device=device)
        pooled = token_embeddings[batch_indices, lengths]
        pooled = F.normalize(pooled, p=2, dim=1)

        vectors = pooled.detach().cpu().float().numpy()
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
        return vectors


def normalize(vector: list[float]) -> list[float]:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0:
        return arr.tolist()
    return (arr / norm).tolist()


def cosine(a: list[float], b: list[float]) -> float:
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom == 0:
        return 0.0
    return float(np.dot(av, bv) / denom)
