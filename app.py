import streamlit as st
import io
from rag_engine import RAGEngine
import time

# Page config
st.set_page_config(
    page_title="📚 DocQ - PDF Q&A Assistant",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def get_rag_engine():
    return RAGEngine()

def main():
    st.title("📚 DocQ - Document Q&A Assistant")
    st.markdown("**Upload any PDF textbook/notes and ask questions answered ONLY from your document**")
    
    # Initialize
    rag = get_rag_engine()
    
    # Sidebar - Document upload
    with st.sidebar:
        st.header("📁 Upload Document")
        uploaded_file = st.file_uploader("Choose PDF file", type="pdf")
        
        doc_status = st.empty()
        
        if uploaded_file is not None:
            if st.button("🔄 **Process Document**", type="primary", use_container_width=True):
                with st.spinner("Processing your PDF..."):
                    try:
                        rag.build_vectorstore(uploaded_file)
                        doc_status.success(f"✅ **{uploaded_file.name}** loaded!\n{len(rag.chunks):,} chunks indexed")
                        st.session_state.doc_ready = True
                        st.session_state.doc_name = uploaded_file.name
                    except Exception as e:
                        doc_status.error(f"❌ Error: {str(e)}")
        
        if st.session_state.get("doc_ready", False):
            if st.button("🗑️ Clear Document", use_container_width=True):
                rag.index = None
                rag.chunks = []
                st.session_state.clear()
                st.rerun()
    
    # Main chat area
    if st.session_state.get("doc_ready", False):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.header("💬 Ask Questions")
        
        with col2:
            st.metric("Chunks Indexed", len(rag.chunks))
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Chat history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    
                    # Sources
                    if "sources" in msg and msg["sources"]:
                        with st.expander(f"📖 Sources ({len(msg['sources'])})", expanded=False):
                            for i, source in enumerate(msg["sources"], 1):
                                with st.container():
                                    col1, col2 = st.columns([4, 1])
                                    with col1:
                                        st.markdown(f"**Chunk {i}:**")
                                        st.caption(source["content"][:300] + "...")
                                    with col2:
                                        st.metric("Score", f"{source['score']:.2f}")
        
        # Chat input
        if prompt := st.chat_input("Ask about your document...", disabled=not st.session_state.get("doc_ready")):
            # User message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Assistant response
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching document..."):
                    result = rag.ask(prompt)
                    
                    response = result["answer"]
                    sources = result.get("sources", [])
                    
                    full_msg = {
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    }
                    st.session_state.messages.append(full_msg)
    
    else:
        # Welcome screen
        st.markdown("### 🚀 **Get Started**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **✅ Features:**
            - PDF upload & processing
            - Semantic search (FAISS)
            - Source citations
            - Out-of-scope detection
            - Gemini-powered answers
            """)
        
        with col2:
            st.info("""
            **📋 Test with:**
            ```
            - What is [topic]?
            - Explain [concept]
            - Current weather? ❌
            ```
            """)
        
        st.balloons()

if __name__ == "__main__":
    main()