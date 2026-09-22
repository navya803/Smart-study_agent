"""
Vector Store Manager for Study Assistant Agent.
Manages ChromaDB vector store with clean in-memory vector index fallback.
"""

import os
import sys
import uuid
import numpy as np
from typing import List, Dict, Any, Optional


class InMemoryVectorStore:
    """Fast, stable in-memory vector index using cosine similarity."""

    def __init__(self):
        self.documents: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []

    def add(self, documents: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]], ids: List[str]):
        for doc, emb, meta, doc_id in zip(documents, embeddings, metadatas, ids):
            self.documents.append(doc)
            self.embeddings.append(np.array(emb, dtype=np.float32))
            self.metadatas.append(meta)
            self.ids.append(doc_id)

    def query(self, query_embedding: List[float], top_k: int = 4) -> Dict[str, Any]:
        if not self.embeddings:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scores = []
        for emb in self.embeddings:
            emb_norm = np.linalg.norm(emb)
            if emb_norm > 0:
                score = np.dot(q_vec, emb / emb_norm)
            else:
                score = 0.0
            scores.append(float(score))

        sorted_indices = np.argsort(scores)[::-1][:top_k]

        res_docs = [self.documents[i] for i in sorted_indices]
        res_metas = [self.metadatas[i] for i in sorted_indices]
        res_dists = [round(max(0.0, 1.0 - scores[i]), 4) for i in sorted_indices]
        res_ids = [self.ids[i] for i in sorted_indices]

        return {
            "documents": [res_docs],
            "metadatas": [res_metas],
            "distances": [res_dists],
            "ids": [res_ids]
        }

    def count(self) -> int:
        return len(self.documents)

    def list_documents(self) -> List[Dict[str, Any]]:
        summary = {}
        for meta in self.metadatas:
            filename = meta.get("filename", "Unknown")
            file_type = meta.get("file_type", "Unknown")
            if filename not in summary:
                summary[filename] = {"filename": filename, "file_type": file_type, "chunk_count": 0}
            summary[filename]["chunk_count"] += 1
        return list(summary.values())

    def delete_by_filename(self, filename: str) -> int:
        indices_to_keep = [i for i, meta in enumerate(self.metadatas) if meta.get("filename") != filename]
        deleted_count = len(self.metadatas) - len(indices_to_keep)
        
        self.documents = [self.documents[i] for i in indices_to_keep]
        self.embeddings = [self.embeddings[i] for i in indices_to_keep]
        self.metadatas = [self.metadatas[i] for i in indices_to_keep]
        self.ids = [self.ids[i] for i in indices_to_keep]
        
        return deleted_count

    def clear(self):
        self.documents.clear()
        self.embeddings.clear()
        self.metadatas.clear()
        self.ids.clear()


class VectorStoreManager:
    """Manages Vector Storage for study document chunks."""

    def __init__(self, persist_directory: str = "vectorstore", collection_name: str = "study_materials"):
        self.persist_directory = os.path.abspath(persist_directory)
        os.makedirs(self.persist_directory, exist_ok=True)
        self.collection_name = collection_name
        self.fallback_store = InMemoryVectorStore()
        
        # Check if running under Python 3.14 preview where ChromaDB Rust DLL has C++ access violation
        is_py314 = sys.version_info >= (3, 14)
        self.use_fallback = is_py314

        if not self.use_fallback:
            try:
                import chromadb
                self.client = chromadb.PersistentClient(path=self.persist_directory)
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception:
                self.use_fallback = True

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        if not chunks or not embeddings or len(chunks) != len(embeddings):
            return 0

        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [f"{c['metadata'].get('filename', 'doc')}_{c['metadata'].get('chunk_id', uuid.uuid4().hex[:8])}" for c in chunks]

        clean_metadatas = []
        for meta in metadatas:
            clean_meta = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            clean_metadatas.append(clean_meta)

        if not self.use_fallback:
            try:
                self.collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=clean_metadatas,
                    ids=ids
                )
                return len(chunks)
            except Exception:
                self.use_fallback = True

        self.fallback_store.add(documents, embeddings, clean_metadatas, ids)
        return len(chunks)

    def similarity_search(self, query_embedding: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        if self.use_fallback:
            results = self.fallback_store.query(query_embedding, top_k=top_k)
        else:
            try:
                count = self.collection.count()
                if count == 0:
                    return []
                actual_k = min(top_k, count)
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=actual_k
                )
            except Exception:
                self.use_fallback = True
                results = self.fallback_store.query(query_embedding, top_k=top_k)

        retrieved = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)
            ids = results["ids"][0] if "ids" in results else [""] * len(docs)

            for doc, meta, dist, chunk_id in zip(docs, metas, distances, ids):
                similarity = round(max(0.0, 1.0 - float(dist)), 4)
                retrieved.append({
                    "text": doc,
                    "metadata": meta,
                    "similarity_score": similarity,
                    "distance": round(float(dist), 4),
                    "id": chunk_id
                })

        return retrieved

    def list_indexed_documents(self) -> List[Dict[str, Any]]:
        if self.use_fallback:
            return self.fallback_store.list_documents()

        try:
            count = self.collection.count()
            if count == 0:
                return []
            all_records = self.collection.get()
            metas = all_records.get("metadatas", [])

            doc_summary = {}
            for meta in metas:
                filename = meta.get("filename", "Unknown")
                file_type = meta.get("file_type", "Unknown")
                if filename not in doc_summary:
                    doc_summary[filename] = {"filename": filename, "file_type": file_type, "chunk_count": 0}
                doc_summary[filename]["chunk_count"] += 1

            return list(doc_summary.values())
        except Exception:
            self.use_fallback = True
            return self.fallback_store.list_documents()

    def delete_document_by_filename(self, filename: str) -> int:
        if self.use_fallback:
            return self.fallback_store.delete_by_filename(filename)

        try:
            all_records = self.collection.get()
            ids_to_delete = [chunk_id for chunk_id, meta in zip(all_records["ids"], all_records["metadatas"]) if meta.get("filename") == filename]
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
            return len(ids_to_delete)
        except Exception:
            self.use_fallback = True
            return self.fallback_store.delete_by_filename(filename)

    def clear_all(self):
        self.fallback_store.clear()
        if not self.use_fallback:
            try:
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception:
                self.use_fallback = True

    def get_total_chunks_count(self) -> int:
        if self.use_fallback:
            return self.fallback_store.count()
        try:
            return self.collection.count()
        except Exception:
            self.use_fallback = True
            return self.fallback_store.count()
