from __future__ import annotations

import base64
import io
import os
from typing import List, Tuple

from PIL import Image
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from src.config import settings
from src.logger import logger
from src.utils import chunk_text


class ResearchAssistant:
    def __init__(self, model_name: str | None = None, embedding_model: str | None = None):
        """Initialize the assistant with LLM, embeddings, and vector store."""
        self.model_name = model_name or settings.OLLAMA_MODEL
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL

        try:
            self.llm = OllamaLLM(model=self.model_name, base_url=settings.OLLAMA_BASE_URL)
        except Exception as exc:
            logger.exception(f"Failed to initialize Ollama LLM: {exc}")
            raise

        # Memory for conversational context
        history: BaseChatMessageHistory = InMemoryChatMessageHistory()
        self.memory = ConversationBufferMemory(
            chat_memory=history,
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
            llm=self.llm
        )

        try:
            self.embeddings = OllamaEmbeddings(model=self.embedding_model, base_url=settings.OLLAMA_BASE_URL)
        except Exception as exc:
            logger.exception(f"Failed to initialize embeddings: {exc}")
            raise

        # Vector store (FAISS)
        self.vector_store: FAISS | None = None
        if settings.FAISS_INDEX_PATH:
            index_path = os.path.join(settings.FAISS_INDEX_PATH, "index.faiss")
            if os.path.exists(index_path):
                try:
                    self.vector_store = FAISS.load_local(
                        settings.FAISS_INDEX_PATH, self.embeddings, allow_dangerous_deserialization=True
                    )
                    logger.info(f"Loaded FAISS index from f{settings.FAISS_INDEX_PATH}")
                except Exception as exc:
                    logger.exception(f"Failed to load FAISS index from {settings.FAISS_INDEX_PATH}: {exc}")
            else:
                logger.info(f"No FAISS index found at {index_path}; will create new index on first document")

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template=(
                "Given the following context: {context}\n"
                "And conversation history: {chat_history}\n"
                "Answer the question: {question}\n"
                "If an image is provided, analyze its content to inform the answer.\n"
                "Provide a concise, accurate response and list sources if applicable."
            ),
        )

    def index_document(self, content: str, doc_type: str, doc_name: str):
        """Index text content (chunked) into FAISS with metadata."""
        if doc_type == "image":
            texts = [f"Image description from OCR: {content}"]
            metadatas = [{"source": doc_name, "type": "ocr"}]
        else:
            texts = list(chunk_text(content or "", max_chars=800)) or [content or ""]
            metadatas = [{"source": doc_name, "type": "text"} for _ in texts]

        if not texts:
            logger.warning(f"No content to index for {doc_name}", )
            return

        try:
            if self.vector_store is None:
                self.vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
            else:
                self.vector_store.add_texts(texts, metadatas=metadatas)

            if settings.FAISS_INDEX_PATH:
                os.makedirs(settings.FAISS_INDEX_PATH, exist_ok=True)
                self.vector_store.save_local(settings.FAISS_INDEX_PATH)
                logger.info("Saved FAISS index to {settings.FAISS_INDEX_PATH}")

            logger.info(f"Indexed {len(texts)} texts for {doc_name}")
        except Exception as exc:
            logger.exception(f"Failed to index document {doc_name}: {exc}")
            raise

    def _encode_image_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64 string."""
        try:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as exc:
            logger.exception(f"Failed to encode image: {exc}")
            raise

    def query(self, question: str, image: Image.Image | None = None) -> Tuple[str, List[str]]:
        """Answer a question using retrieved context and optional image analysis.

        Returns: (answer, sources)
        """
        chat_history = self.memory.load_memory_variables({}).get("chat_history", "")
        sources: List[str] = []
        context = ""

        # Retrieve relevant documents
        if self.vector_store:
            try:
                docs = self.vector_store.similarity_search(question, k=3)
                context = "\n\n".join(doc.page_content for doc in docs)
                sources = list({doc.metadata.get("source", "unknown") for doc in docs})
            except Exception as exc:
                logger.exception(f"Vector search failed: {exc}")

        prompt = self.prompt_template.format(context=context, question=question, chat_history=chat_history)

        try:
            if image is not None:
                img_b64 = self._encode_image_base64(image)
                response = self.llm.invoke(prompt, images=[img_b64])
            else:
                response = self.llm.invoke(prompt)
        except Exception as exc:
            logger.exception(f"LLM query failed: {exc}")
            raise

        # Save to memory
        self.memory.save_context(
            inputs={"question": question},
            outputs={"answer": response}
        )

        return response, sources
