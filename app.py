from __future__ import annotations

import os

import streamlit as st

from src.document_processor import DocumentProcessor
from src.logger import logger
from src.research_assistant import ResearchAssistant
from src.utils import image_to_text
from src.web_search_api import WebSearchAPI

st.set_page_config(page_title="Personalized Research Assistant", layout="wide")


@st.cache_resource
def get_web_api() -> WebSearchAPI:
    return WebSearchAPI()


def main():
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
    st.title("📚 Personalized Research Assistant")

    if "assistant" not in st.session_state:
        st.session_state.assistant = ResearchAssistant()

    assistant = st.session_state.assistant
    web_api = get_web_api()

    # File upload
    uploaded_file = st.file_uploader(
        "📂 Upload a PDF, text, or image", type=["pdf", "txt", "png", "jpg", "jpeg"], key="file_uploader"
    )
    if uploaded_file:
        processor = DocumentProcessor()
        try:
            content, doc_type = processor.process_file(uploaded_file)
            assistant.memory.clear()
            assistant.vector_store = None  # Reset FAISS index
            st.session_state["uploaded_file"] = None
            st.session_state["doc_type"] = None

            if doc_type == "image":
                st.image(content, caption=getattr(uploaded_file, "name", "uploaded image"))
                if st.button("Index image (use OCR)"):
                    ocr_text = image_to_text(content)
                    assistant.index_document(ocr_text, "text", getattr(uploaded_file, "name", "image"))
                    st.success("Indexed image via OCR")
                    st.session_state["uploaded_file"] = content
                    st.session_state["doc_type"] = "image"
            else:
                assistant.index_document(content, "text", getattr(uploaded_file, "name", "document"))
                st.success(f"✅ Processed {getattr(uploaded_file, 'name', 'uploaded')}")
                st.session_state["uploaded_file"] = content
                st.session_state["doc_type"] = "text"
        except Exception as exc:
            logger.exception(f"File processing failed for {getattr(uploaded_file, 'name', 'uploaded')}: {exc}")
            st.error("Failed to process the uploaded file. Check logs for details.")

    with st.form(key="query_form"):
        query = st.text_input("💬 Ask a question about your documents or research topic:", key="query_input")
        include_web = st.checkbox("🌐 Include web search results")
        submit_button = st.form_submit_button(label="Ask")

        if submit_button and query:
            with st.spinner("🔍 Processing your request..."):
                image = st.session_state.get("uploaded_file") if st.session_state.get("doc_type") == "image" else None
                try:
                    response, sources = assistant.query(query, image=image)
                    st.subheader("📝 Response")
                    st.write(response)
                    st.subheader("📌 Sources")
                    for s in sources:
                        st.write(f"- {s}")

                    if include_web:
                        results = web_api.search(query)
                        if results:
                            st.subheader("🔗 Web Search Results")
                            for r in results:
                                title = r.get("title", "(no title)")
                                link = r.get("href", "#")
                                snippet = r.get("body", "")
                                st.markdown(f"- **[{title}]({link})**: {snippet}")
                except Exception as exc:
                    logger.exception(f"Query failed for '{query}': {exc}")
                    st.error("Something went wrong while processing the query. Check logs.")


if __name__ == "__main__":
    main()
