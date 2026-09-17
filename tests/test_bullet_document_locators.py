"""Offline tests: evidence-bullet payloads say which document each page and quote is in.

bullet.doc_ids is a sorted set of the supporting chunks' documents, so doc_ids[0] is
not the document of the quote or of chunk_ids[0]. The viewer opens a bullet from
chunk_doc_ids / quote_doc_id instead.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.agent import v11_runner  # noqa: E402
from retrieval.agent.v9_types import (  # noqa: E402
    EvidenceBullet,
    EvidenceSummaryUpdate,
    ResearchWorkspace,
    WorkspaceChunk,
)
from retrieval.agent.v9_workspace import (  # noqa: E402
    build_chunk_doc_map,
    bullet_document_locators,
    merge_evidence_summary_update,
)

QUOTE = "the courier delivered the documents to the embassy"


def _workspace():
    ws = ResearchWorkspace(question="who delivered the documents?")
    ws.fulltext_chunks = [
        # Chunk 501 is in the higher-numbered document and listed first.
        WorkspaceChunk(chunk_id=501, text=f"... {QUOTE} ...", doc_id=1080, page="p42"),
        WorkspaceChunk(chunk_id=502, text="a related passage", doc_id=1061, page="p7"),
        WorkspaceChunk(chunk_id=503, text="no document known", doc_id=None, page="p3"),
    ]
    return ws


def _merged_bullet(ws, supporting, quote_chunk_id=None):
    bullet = EvidenceBullet(
        bullet_id="",
        text="A courier delivered the documents.",
        supporting_chunk_ids=supporting,
        support_quote=QUOTE if quote_chunk_id else "",
        quote_chunk_id=quote_chunk_id,
    )
    update = EvidenceSummaryUpdate(
        update_id="u1", generated_from_chunk_ids=supporting,
        summarizer_model="test", created_at="2026-09-16T00:00:00Z", bullets=[bullet],
    )
    merge_evidence_summary_update(ws, update, build_chunk_doc_map(ws))
    return bullet


def test_multi_document_bullet_locates_quote_and_chunks():
    ws = _workspace()
    b = _merged_bullet(ws, [501, 502], quote_chunk_id=501)
    assert b.doc_ids == [1061, 1080]  # sorted set: doc_ids[0] is NOT the quote's document

    loc = bullet_document_locators(b, build_chunk_doc_map(ws))
    assert loc == {"chunk_doc_ids": [1080, 1061], "quote_doc_id": 1080}


def test_unknown_documents_are_none_and_no_quote_means_no_quote_doc():
    ws = _workspace()
    b = _merged_bullet(ws, [503, 502])
    loc = bullet_document_locators(b, build_chunk_doc_map(ws))
    assert loc == {"chunk_doc_ids": [None, 1061]}


def test_v11_bullet_payload_aligns_documents_with_chunks_and_pages(monkeypatch):
    ws = _workspace()
    b = _merged_bullet(ws, [502, 501], quote_chunk_id=501)
    looked_up = []

    def fake_lookup(conn, chunk_id, quote):
        looked_up.append(chunk_id)
        return 43

    monkeypatch.setattr(v11_runner, "_lookup_quote_page", fake_lookup)
    chunk_to_page = {c.chunk_id: v11_runner._parse_page_no(c.page) for c in ws.fulltext_chunks}
    payload = v11_runner._bullet_payload(
        None, b, chunk_to_page, {1061: "Doc A", 1080: "Doc B"}, build_chunk_doc_map(ws),
    )

    assert payload["chunk_ids"] == [502, 501]
    assert payload["pages"] == [7, 42]
    assert payload["chunk_doc_ids"] == [1061, 1080]
    # Unchanged fields older clients read.
    assert payload["doc_ids"] == [1061, 1080]
    assert payload["source_names"] == ["Doc A", "Doc B"]
    # Quote, its page and its document all come from chunk 501.
    assert looked_up == [501]
    assert payload["quote_chunk_id"] == 501
    assert payload["quote_page"] == 43
    assert payload["quote_doc_id"] == 1080
