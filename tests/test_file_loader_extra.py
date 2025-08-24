from core.file_loader import _load_txt_with_fallbacks, load_documents
from langchain_core.documents import Document


class AutoLoader:
    def __init__(self, path, autodetect_encoding=False, encoding=None):
        self.autodetect_encoding = autodetect_encoding
        self.encoding = encoding
        self.path = path

    def load(self):
        if self.autodetect_encoding:
            return [Document(page_content="auto", metadata={"source": self.path})]
        if self.encoding == "utf-8":
            return [Document(page_content="utf8", metadata={"source": self.path})]
        raise ValueError("bad encoding")


class FailingLoader(AutoLoader):
    def load(self):
        raise ValueError("fail")


def test_load_txt_autodetect(monkeypatch, tmp_path):
    file_path = tmp_path / "doc.txt"
    file_path.write_text("hello")
    monkeypatch.setattr("langchain_community.document_loaders.TextLoader", AutoLoader)
    docs = _load_txt_with_fallbacks(str(file_path))
    assert docs[0].metadata["encoding"] == "autodetect"


def test_load_txt_fallback_and_salvage(monkeypatch, tmp_path):
    file_path = tmp_path / "doc.txt"
    file_path.write_text("hello")

    class FallbackLoader:
        def __init__(self, path, autodetect_encoding=False, encoding=None):
            self.autodetect_encoding = autodetect_encoding
            self.encoding = encoding
            self.path = path
        def load(self):
            if self.autodetect_encoding:
                raise TypeError("no autodetect")
            raise ValueError("bad")

    monkeypatch.setattr("langchain_community.document_loaders.TextLoader", FallbackLoader)
    docs = _load_txt_with_fallbacks(str(file_path))
    assert docs[0].metadata["encoding"].startswith("utf-8")


def test_load_documents_branches(monkeypatch, tmp_path):
    txtfile = tmp_path / "a.txt"
    txtfile.write_text("hi")
    monkeypatch.setattr("core.file_loader._load_txt_with_fallbacks", lambda p: [Document(page_content="x")])
    docs = load_documents(str(txtfile))
    assert docs and docs[0].page_content == "x"

    monkeypatch.setattr("langchain_community.document_loaders.PyPDFLoader", lambda path: FailingLoader(path))
    assert load_documents(str(tmp_path / "a.pdf")) == []

    class DocxLoader:
        def __init__(self, path):
            self.path = path
        def load(self):
            return [Document(page_content="docx")]
    monkeypatch.setattr("langchain_community.document_loaders.Docx2txtLoader", DocxLoader)
    docs = load_documents(str(tmp_path / "a.docx"))
    assert docs and docs[0].page_content == "docx"

    assert load_documents(str(tmp_path / "a.xyz")) == []
