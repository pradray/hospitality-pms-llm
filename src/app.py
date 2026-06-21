"""Chainlit chat app for the hospitality PMS RAG pipeline."""

import os
import chainlit as cl
from rag_pipeline import RAGPipeline, EVAL_CONFIGS

CONFIG = os.environ.get("RAG_CONFIG", "")
BACKEND = os.environ.get("RAG_BACKEND", "llama_cpp")
MODEL = os.environ.get("RAG_MODEL", None)
USE_RAG = os.environ.get("RAG_USE_RAG", "true").lower() in ("true", "1", "yes")
TOP_K = int(os.environ.get("RAG_TOP_K", "5"))


@cl.on_chat_start
async def start():
    if CONFIG and CONFIG in EVAL_CONFIGS:
        pipeline = RAGPipeline.from_config(CONFIG, top_k=TOP_K)
    else:
        pipeline = RAGPipeline(
            backend=BACKEND, model_name=MODEL, use_rag=USE_RAG, top_k=TOP_K,
        )

    cl.user_session.set("pipeline", pipeline)

    rag_status = "ON" if pipeline.use_rag else "OFF"
    await cl.Message(
        content=(
            "**Hospitality PMS Assistant**\n\n"
            "Ask me about OPERA Cloud APIs, configuration, or troubleshooting.\n\n"
            f"Model: `{pipeline.model_name}` | RAG: `{rag_status}` | Top-K: `{TOP_K}`"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    pipeline: RAGPipeline = cl.user_session.get("pipeline")

    msg = cl.Message(content="")
    chunks_shown = False

    for token, chunks in pipeline.query_stream(message.content):
        if chunks and not chunks_shown:
            sources = []
            for i, c in enumerate(chunks, 1):
                src = c.metadata.get("source_file", "?")
                sec = c.metadata.get("section", c.metadata.get("path", "?"))
                dist = f"{c.distance:.3f}"
                sources.append(f"{i}. **{src}** — {sec} (distance: {dist})")

            msg.elements = [
                cl.Text(
                    name=f"source_{i}",
                    content=c.text[:1000],
                    display="side",
                )
                for i, c in enumerate(chunks, 1)
            ]
            chunks_shown = True

        await msg.stream_token(token)

    if chunks_shown:
        sources_text = "\n".join(sources)
        await msg.stream_token(f"\n\n---\n**Sources:**\n{sources_text}")

    await msg.send()
