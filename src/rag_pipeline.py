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

# --- Evaluation matrix configs ---

EVAL_CONFIGS = {
    # Phase 1
    "3B-base":       {"backend": "llama_cpp", "model": "Qwen2.5-3B-Instruct-Q4_K_M.gguf", "use_rag": False},
    "3B-RAG":        {"backend": "llama_cpp", "model": "Qwen2.5-3B-Instruct-Q4_K_M.gguf", "use_rag": True},
    "7B-base":       {"backend": "llama_cpp", "model": "Qwen2.5-7B-Instruct-Q4_K_M.gguf", "use_rag": False},
    "7B-RAG":        {"backend": "llama_cpp", "model": "Qwen2.5-7B-Instruct-Q4_K_M.gguf", "use_rag": True},
    "API-ceiling":   {"backend": "anthropic", "model": "claude-sonnet-4-20250514",         "use_rag": False},
    # Phase 2 — LoRA variants (model filenames TBD after fine-tuning)
    "3B-LoRA":       {"backend": "llama_cpp", "model": "Qwen2.5-3B-LoRA-Q4_K_M.gguf",     "use_rag": False},
    "3B-LoRA-RAG":   {"backend": "llama_cpp", "model": "Qwen2.5-3B-LoRA-Q4_K_M.gguf",     "use_rag": True},
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
    ):
        self.backend = backend
        self.use_rag = use_rag
        self.top_k = top_k
        self.config_name = config_name

        if model_name:
            self.model_name = model_name
        else:
            self.model_name = {
                "anthropic": "claude-sonnet-4-20250514",
                "openai": "gpt-4o-mini",
                "llama_cpp": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
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
        return cls(config_name=config_name, **cfg)

    def _init_llm(self):
        if self.backend == "llama_cpp":
            from llama_cpp import Llama
            model_path = os.environ.get("LLAMA_MODEL_PATH", str(MODELS_DIR / self.model_name))
            return Llama(
                model_path=model_path,
                n_ctx=16384,
                n_gpu_layers=-1,
                verbose=False,
            )
        elif self.backend == "anthropic":
            from anthropic import Anthropic
            return Anthropic()
        elif self.backend == "openai":
            from openai import OpenAI
            return OpenAI()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def retrieve(self, query: str, module_filter: str | None = None) -> list[RetrievedChunk]:
        if not self.use_rag:
            return []

        query_embedding = self._embedder.encode(query, normalize_embeddings=True).tolist()

        where_filter = None
        if module_filter:
            where_filter = {"module": module_filter}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
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

        elif self.backend == "openai":
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
