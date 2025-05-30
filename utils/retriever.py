import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional


class Retriever:
    def __init__(
        self,
        player_stats_path: str,
        match_conditions_path: str,
        faqs_path: Optional[str] = None,
        embed_model_name: str = 'all-MiniLM-L6-v2'
    ):
        """
        Initialize the Retriever with player stats, match conditions, and optional FAQs.
        Builds a FAISS index over embedded documents for fast similarity search.
        """
        self.documents: List[str] = []
        self.model = SentenceTransformer(embed_model_name)

        # Load data
        self.player_stats = self._load_json(player_stats_path)
        self.match_conditions = self._load_json(match_conditions_path)
        self.faqs = self._load_json(faqs_path) if faqs_path else []

        # Process player stats
        for player in self.player_stats:
            text = (
                f"{player.get('player', 'Unknown player')} is a {player.get('role', 'Unknown role')}.\n"
                f"Recent Form: {player.get('form_last_5_matches', 'N/A')}.\n"
                f"Pitch Performance: {player.get('pitch_performance', 'N/A')}."
            )
            self.documents.append(text)

        # Process match conditions
        if self.match_conditions:
            cond = self.match_conditions
            cond_text = (
                f"Venue: {cond.get('venue', 'Unknown')}. "
                f"Pitch Type: {cond.get('pitch', 'Unknown')}. "
                f"Weather Forecast: {cond.get('weather', 'Unknown')}. "
                f"Opponent: {cond.get('opposition', 'Unknown')}."
            )
            self.documents.append(cond_text)

        # Process FAQs
        for faq in self.faqs:
            question = faq.get("question", "").strip()
            answer = faq.get("answer", "").strip()
            if question and answer:
                self.documents.append(f"Q: {question}\nA: {answer}")

        # Generate embeddings and FAISS index
        self._build_index()

    def _load_json(self, path: str):
        """Load a JSON file and handle missing/corrupted paths."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading {path}: {e}")
            return []

    def _build_index(self):
        """Embeds documents and builds FAISS index."""
        if not self.documents:
            raise ValueError("No documents found for retrieval.")
        self.doc_embeddings = self.model.encode(self.documents, convert_to_tensor=False)
        dim = self.doc_embeddings[0].shape[0]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.doc_embeddings, dtype='float32'))

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Returns top_k relevant documents based on the query."""
        if not query:
            return []
        q_emb = self.model.encode([query], convert_to_tensor=False).astype('float32')
        distances, indices = self.index.search(q_emb, top_k)
        return [self.documents[i] for i in indices[0] if i < len(self.documents)]
