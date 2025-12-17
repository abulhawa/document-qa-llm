from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class QueryRewrite:
    rewritten: Optional[str] = None
    clarify: Optional[str] = None
    raw: Optional[Dict] = None


@dataclass
class RetrievedDocument:
    text: str
    path: str
    chunk_index: Optional[int] = None
    score: Optional[float] = None
    page: Optional[int] = None
    location_percent: Optional[float] = None

    @property
    def source_label(self) -> str:
        if self.page is not None:
            return f"{self.path} (Page {self.page})"
        if self.location_percent is not None:
            return f"{self.path} (~{self.location_percent}%)"
        return self.path


@dataclass
class RetrievalResult:
    query: str
    documents: List[RetrievedDocument] = field(default_factory=list)

    @property
    def context_chunks(self) -> List[str]:
        return [doc.text for doc in self.documents]

    @property
    def sources(self) -> List[str]:
        seen = set()
        unique_sources: List[str] = []
        for doc in self.documents:
            label = doc.source_label
            if label not in seen:
                unique_sources.append(label)
                seen.add(label)
        return unique_sources

    @property
    def summary(self) -> List[str]:
        return [
            f"{doc.path} | idx={doc.chunk_index} | score={doc.score:.4f} | "
            f"page={doc.page} | ~{doc.location_percent}%"
            for doc in self.documents
            if doc.score is not None
        ]


@dataclass
class PromptRequest:
    prompt: Union[str, List[Dict[str, str]]]
    mode: str


@dataclass
class AnswerContext:
    question: str
    mode: str
    temperature: float
    model: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None
    rewritten_question: Optional[str] = None
    clarification: Optional[str] = None
    retrieval: Optional[RetrievalResult] = None
    prompt_request: Optional[PromptRequest] = None
    answer: Optional[str] = None

    @property
    def sources(self) -> List[str]:
        return self.retrieval.sources if self.retrieval else []
