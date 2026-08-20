"""Core RAG pipeline supporting the full evaluation matrix.

Model axis: Qwen2.5-3B, Qwen2.5-7B, API ceiling (Claude/OpenAI)
RAG axis: on/off
Weights axis: base vs LoRA fine-tuned (Phase 2)

Local inference via llama-cpp-python, API via anthropic/openai SDKs.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).parent.parent
VECTORSTORE_DIR = PROJECT_ROOT / "output" / "vectorstore"
MODELS_DIR = PROJECT_ROOT / "models"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
COLLECTION_NAME = "hospitality_pms"

SYSTEM_PROMPT = """You are a hospitality technology expert specializing in Oracle OPERA Cloud PMS \
and the Oracle Hospitality Integration Platform (OHIP). You help hotel IT teams, \
system integrators, and developers with API orchestration, system configuration, \
and troubleshooting."""

SYSTEM_PROMPT_RAG = SYSTEM_PROMPT + """

Answer based on the provided context. If the context doesn't contain enough \
information, say so rather than guessing. When referencing API endpoints, \
include the HTTP method and full path. When referencing configuration, \
specify the exact OPERA Control or setting name."""

SYSTEM_PROMPT_NO_RAG = SYSTEM_PROMPT + """

Answer from your training knowledge. When referencing API endpoints, \
include the HTTP method and full path where possible. When referencing \
configuration, specify the exact setting name if you know it."""

# Strict-grounding prompt. Phase 1 attributed most errors to the generator
# ignoring retrieved context and inventing API paths, and Phase 2 answered that
# with QLoRA. This tests the cheaper hypothesis first: that the failure is a
# prompting problem, not a training one. If this recovers the fine-tuned model's
# gain, the LoRA contribution needs restating.
SYSTEM_PROMPT_RAG_STRICT = SYSTEM_PROMPT + """

Answer ONLY from the provided context. Follow these rules exactly:

1. Every API path you cite MUST appear verbatim in the context. Never construct,
   guess, complete, or infer a path that is not written there.
2. After each API path, cite the source it came from, e.g. [Source 3].
3. If the context does not contain the endpoint, control, or setting needed to
   answer, say so explicitly and state what is missing. An honest "the provided
   context does not contain this" is correct and preferred over a plausible guess.
4. Do not rely on your own knowledge of Oracle OPERA to fill gaps in the context.
5. Quote exact OPERA Control and setting names as they are spelled in the context."""

# --- Evaluation matrix configs ---

EVAL_CONFIGS = {
    # Phase 1
    "3B-base":       {"backend": "llama_cpp", "model": "qwen2.5-3b-instruct-q4_k_m.gguf", "use_rag": False},
    "3B-RAG":        {"backend": "llama_cpp", "model": "qwen2.5-3b-instruct-q4_k_m.gguf", "use_rag": True},
    "7B-base":       {"backend": "llama_cpp", "model": "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf", "use_rag": False},
    "7B-RAG":        {"backend": "llama_cpp", "model": "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf", "use_rag": True},
    # Frontier ceiling. Uses Grok because that is the only API key available for
    # this project. CAVEAT for the write-up: the judge is also a Grok model, so
    # this config is not judged by a fully independent model. Mitigated by scoring
    # with a *different* Grok model than the one generating (see score_results.py
    # --judge-model), but self-preference bias cannot be ruled out and the ceiling
    # should be read as indicative, not as a precise upper bound.
    "API-ceiling":     {"backend": "grok", "model": "grok-4.5", "use_rag": False},
    "API-ceiling-RAG": {"backend": "grok", "model": "grok-4.5", "use_rag": True},
    # Phase 2 — QLoRA fine-tuned (rank 32, attention projections, 3 epochs)
    "3B-LoRA":       {"backend": "llama_cpp", "model": "qwen2.5-3b-lora-q4_k_m.gguf",     "use_rag": False},
    "3B-LoRA-RAG":   {"backend": "llama_cpp", "model": "qwen2.5-3b-lora-q4_k_m.gguf",     "use_rag": True},
    # Phase 2 controls — the *unmodified* base re-quantized through the same
    # llama.cpp pipeline as the LoRA export. The stock 3B-base GGUF was quantized
    # by Qwen with a different recipe (2.10 GB vs our 1.93 GB), so comparing it
    # against 3B-LoRA would confound fine-tuning with quantization. These configs
    # differ from the LoRA ones by fine-tuning alone.
    "3B-base-req":   {"backend": "llama_cpp", "model": "qwen2.5-3b-base-req-q4_k_m.gguf", "use_rag": False},
    "3B-RAG-req":    {"backend": "llama_cpp", "model": "qwen2.5-3b-base-req-q4_k_m.gguf", "use_rag": True},
    # Retrieval ablations (tuned on dev; gold-path recall in brackets).
    # Baseline dense@5 retrieves only 71% of gold API paths, so roughly a third of
    # orchestration answers cannot be right no matter how good the generator is.
    "3B-LoRA-RAG-k15":    {"backend": "llama_cpp", "model": "qwen2.5-3b-lora-q4_k_m.gguf",
                           "use_rag": True, "top_k": 15},                      # 85%
    "3B-LoRA-RAG-rerank": {"backend": "llama_cpp", "model": "qwen2.5-3b-lora-q4_k_m.gguf",
                           "use_rag": True, "top_k": 5, "retrieval": "rerank"}, # 84% in 5 chunks
    "3B-LoRA-RAG-k15":    {"backend": "llama_cpp", "model": "qwen2.5-3b-lora-q4_k_m.gguf",
                           "use_rag": True, "top_k": 15},
    # Strict-grounding prompt: is the hallucination a training problem or a
    # prompting problem? Same weights as the -req / LoRA configs, prompt only.
    "3B-RAG-req-strict":  {"backend": "llama_cpp", "model": "qwen2.5-3b-base-req-q4_k_m.gguf",
                           "use_rag": True, "prompt_style": "strict"},
    "3B-LoRA-RAG-strict": {"backend": "llama_cpp", "model": "qwen2.5-3b-lora-q4_k_m.gguf",
                           "use_rag": True, "prompt_style": "strict"},
    # Quantisation sensitivity: does Q4 cost a 7B more than it costs a 3B? Both
    # sizes are compared Q4 vs Q8 built from the same f16 export, so the only
    # variable is bit width.
    "3B-RAG-q8":          {"backend": "llama_cpp", "model": "qwen2.5-3b-base-q8_0.gguf",
                           "use_rag": True},
    "7B-RAG-req":         {"backend": "llama_cpp", "model": "qwen2.5-7b-base-req-q4_k_m.gguf",
                           "use_rag": True},
    "7B-RAG-q8":          {"backend": "llama_cpp", "model": "qwen2.5-7b-base-q8_0.gguf",
                           "use_rag": True},
}


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict
    distance: float


@dataclass
class RAGResponse:
    answer: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    model: str = ""
    config_name: str = ""


class RAGPipeline:
    def __init__(
        self,
        backend: str = "llama_cpp",
        model_name: str | None = None,
        use_rag: bool = True,
        top_k: int = 5,
        vectorstore_dir: str | None = None,
        config_name: str = "",
        retrieval: str = "dense",
        rerank_pool: int = 30,
        prompt_style: str = "default",
    ):
        self.prompt_style = prompt_style
        self.backend = backend
        self.use_rag = use_rag
        self.top_k = top_k
        self.config_name = config_name
        # "dense"  - cosine top-k, the Phase 1/2 system
        # "rerank" - fetch rerank_pool candidates, reorder with a cross-encoder,
        #            keep top_k. On the dev split this lifts gold-path recall from
        #            71% to 84% at k=5, i.e. the recall of dense@25 in a fifth of
        #            the context.
        self.retrieval = retrieval
        self.rerank_pool = rerank_pool
        self._reranker = None

        if model_name:
            self.model_name = model_name
        else:
            self.model_name = {
                "anthropic": "claude-sonnet-4-20250514",
                "openai": "gpt-4o-mini",
                "llama_cpp": "qwen2.5-3b-instruct-q4_k_m.gguf",
            }[backend]

        if use_rag:
            vs_path = vectorstore_dir or str(VECTORSTORE_DIR)
            self._chroma = chromadb.PersistentClient(path=vs_path)
            self._collection = self._chroma.get_collection(COLLECTION_NAME)
            self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        else:
            self._chroma = None
            self._collection = None
            self._embedder = None

        self._llm = self._init_llm()

    @classmethod
    def from_config(cls, config_name: str, **overrides) -> "RAGPipeline":
        cfg = EVAL_CONFIGS[config_name].copy()
        cfg.update(overrides)
        if "model" in cfg:
            cfg["model_name"] = cfg.pop("model")
        return cls(config_name=config_name, **cfg)

    def _init_llm(self):
        if self.backend == "llama_cpp":
            from llama_cpp import Llama
            model_path = os.environ.get("LLAMA_MODEL_PATH", str(MODELS_DIR / self.model_name))
            # 16k is enough for top-5, but 15 User Guide chunks can request 23k
            # tokens and abort the run, so the window is overridable for
            # long-context ablations. Qwen2.5 supports 32k natively.
            return Llama(
                model_path=model_path,
                n_ctx=int(os.environ.get("LLAMA_N_CTX", 16384)),
                n_gpu_layers=-1,
                verbose=False,
            )
        elif self.backend == "anthropic":
            from anthropic import Anthropic
            return Anthropic()
        elif self.backend == "openai":
            from openai import OpenAI
            return OpenAI()
        elif self.backend == "grok":
            # xAI is OpenAI-API compatible; only the base_url and key differ.
            from openai import OpenAI
            return OpenAI(
                api_key=os.environ["XAI_API_KEY"],
                base_url="https://api.x.ai/v1",
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANKER_MODEL)
        return self._reranker

    def retrieve(self, query: str, module_filter: str | None = None) -> list[RetrievedChunk]:
        if not self.use_rag:
            return []

        query_embedding = self._embedder.encode(query, normalize_embeddings=True).tolist()

        where_filter = None
        if module_filter:
            where_filter = {"module": module_filter}

        # Over-fetch when reranking; the cross-encoder then picks the final top_k.
        n_results = self.rerank_pool if self.retrieval == "rerank" else self.top_k

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append(RetrievedChunk(text=doc, metadata=meta, distance=dist))

        if self.retrieval == "rerank" and chunks:
            scores = self.reranker.predict([(query, c.text) for c in chunks])
            order = sorted(range(len(chunks)), key=lambda i: -scores[i])
            chunks = [chunks[i] for i in order[:self.top_k]]

        return chunks

    def _build_user_prompt(self, query: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return query

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source_file", "")
            section = chunk.metadata.get("section", chunk.metadata.get("path", ""))
            header = f"[Source {i}: {source} — {section}]"
            context_parts.append(f"{header}\n{chunk.text}")

        context = "\n\n---\n\n".join(context_parts)
        return f"Context:\n{context}\n\nQuestion: {query}"

    def _get_system_prompt(self) -> str:
        if self.use_rag and self.prompt_style == "strict":
            return SYSTEM_PROMPT_RAG_STRICT
        return SYSTEM_PROMPT_RAG if self.use_rag else SYSTEM_PROMPT_NO_RAG

    def query(self, question: str, module_filter: str | None = None) -> RAGResponse:
        chunks = self.retrieve(question, module_filter)
        user_prompt = self._build_user_prompt(question, chunks)
        answer = self._generate(user_prompt)
        return RAGResponse(
            answer=answer,
            chunks=chunks,
            model=self.model_name,
            config_name=self.config_name,
        )

    def query_stream(self, question: str, module_filter: str | None = None):
        chunks = self.retrieve(question, module_filter)
        user_prompt = self._build_user_prompt(question, chunks)
        yield from self._generate_stream(user_prompt, chunks)

    def _generate(self, user_prompt: str) -> str:
        system = self._get_system_prompt()

        if self.backend == "llama_cpp":
            response = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=2048,
            )
            return response["choices"][0]["message"]["content"]

        elif self.backend == "anthropic":
            response = self._llm.messages.create(
                model=self.model_name,
                max_tokens=2048,
                system=system,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text

        elif self.backend in ("openai", "grok"):
            response = self._llm.chat.completions.create(
                model=self.model_name,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content

    def _generate_stream(self, user_prompt: str, chunks: list[RetrievedChunk]):
        system = self._get_system_prompt()

        if self.backend == "llama_cpp":
            stream = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=2048,
                stream=True,
            )
            first = True
            for chunk in stream:
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    if first:
                        yield delta, chunks
                        first = False
                    else:
                        yield delta, None

        elif self.backend == "anthropic":
            with self._llm.messages.stream(
                model=self.model_name,
                max_tokens=2048,
                system=system,
                messages=[{"role": "user", "content": user_prompt}],
            ) as stream:
                first = True
                for text in stream.text_stream:
                    if first:
                        yield text, chunks
                        first = False
                    else:
                        yield text, None

        elif self.backend == "openai":
            stream = self._llm.chat.completions.create(
                model=self.model_name,
                max_tokens=2048,
                stream=True,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
            )
            first = True
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    if first:
                        yield delta, chunks
                        first = False
                    else:
                        yield delta, None
