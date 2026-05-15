import os
import PyPDF2
import faiss
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

load_dotenv()

class RAGEngine:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.chunks = []
        self.metadata = []
        self.setup_gemini()
    
    def setup_gemini(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ GOOGLE_API_KEY not set in .env")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def extract_pdf_text(self, pdf_file) -> str:
        """Extract text from PDF bytes"""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n\n"
        return text
    
    def smart_chunking(self, text: str) -> List[str]:
        """Advanced chunking with overlap"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "!", "?", " ", ""],
            keep_separator=True
        )
        return splitter.split_text(text)
    
    def build_vectorstore(self, pdf_file):
        """Build FAISS index from PDF"""
        print("📖 Extracting text...")
        text = self.extract_pdf_text(pdf_file)
        
        print("✂️ Chunking...")
        self.chunks = self.smart_chunking(text)
        
        print("🔢 Generating embeddings...")
        embeddings = self.embedding_model.encode(self.chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        print("🏗️ Building FAISS index...")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        
        self.metadata = [{"chunk_id": i, "source": "uploaded_pdf"} for i in range(len(self.chunks))]
        print(f"✅ Index ready: {len(self.chunks)} chunks")
    
    def retrieve(self, query: str, k: int = 4) -> List[Dict]:
        """Retrieve relevant chunks"""
        if self.index is None:
            return []
        
        query_emb = self.embedding_model.encode([query])
        query_emb = np.array(query_emb).astype('float32')
        faiss.normalize_L2(query_emb)
        
        scores, indices = self.index.search(query_emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "content": self.chunks[idx],
                "score": float(score),
                "chunk_id": idx
            })
        return results
    
    def generate_answer(self, query: str, context: List[str]) -> str:
        """Generate concise answer using Gemini"""
        context_str = "\n\n".join(context)
        
        prompt = f"""Answer the question based ONLY on the provided document context. 
Be concise and accurate. If the answer isn't in the context, say "I couldn't find this information in the document."

CONTEXT:
{context_str}

QUESTION: {query}

ANSWER:"""
        
        response = self.model.generate_content(prompt)
        return response.text.strip()
    
    def is_out_of_scope(self, retrieved: List[Dict]) -> bool:
        """Detect out-of-scope queries"""
        if not retrieved:
            return True
        avg_relevance = np.mean([r["score"] for r in retrieved])
        return avg_relevance < 0.65
    
    def ask(self, query: str) -> Dict[str, Any]:
        """Main query pipeline"""
        retrieved = self.retrieve(query)
        
        if self.is_out_of_scope(retrieved):
            return {
                "answer": "❌ **Out of scope** - I couldn't find relevant information in your document.",
                "sources": [],
                "status": "out_of_scope"
            }
        
        top_context = [r["content"] for r in retrieved[:3]]
        answer = self.generate_answer(query, top_context)
        
        return {
            "answer": answer,
            "sources": retrieved[:3],
            "status": "answered"
        }