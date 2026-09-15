"""Tests for the pure chunk<->page mapping in scripts/load_transcript.py.

Pure-function tests only: no network, no DB. The mapper walks pages in
page_seq order and greedily consumes them per chunk (id order) while the
normalized concatenation stays a prefix of the normalized chunk text.
"""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "load_transcript.py"
_spec = importlib.util.spec_from_file_location("load_transcript", SCRIPT)
lt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lt)

map_chunks_to_pages = lt.map_chunks_to_pages
normalize_text = lt.normalize_text


def test_exact_two_page_chunk():
    """A chunk whose text is exactly page 2 + page 3 raw_text (the verified
    chunk-53004 shape) maps to both pages."""
    pages = [
        (2, "Opening garble of page two.\nMore lines here.\n"),
        (3, "Page three body.\nlast line of page three\n"),
    ]
    chunks = [
        (53004,
         "Opening garble of page two.\nMore lines here.\n"
         "Page three body.\nlast line of page three\n"),
    ]
    res = map_chunks_to_pages(pages, chunks)
    assert res.mapped == [(53004, [2, 3])]
    assert res.unmapped == []
    assert res.leftover_pages == []


def test_single_page_chunk():
    pages = [(7, "A lone page of testimony.")]
    chunks = [(900, "A lone page of testimony.")]
    res = map_chunks_to_pages(pages, chunks)
    assert res.mapped == [(900, [7])]
    assert res.unmapped == []
    assert res.leftover_pages == []


def test_multi_chunk_sequence_consumes_all_pages():
    """Consecutive chunks packing 2+1+2 pages consume all five pages exactly
    once, in order."""
    p = {i: f"Page {i} distinctive body text number {i}. " for i in range(1, 6)}
    pages = [(i, p[i]) for i in range(1, 6)]
    chunks = [
        (100, p[1] + p[2]),
        (101, p[3]),
        (102, p[4] + p[5]),
    ]
    res = map_chunks_to_pages(pages, chunks)
    assert res.mapped == [(100, [1, 2]), (101, [3]), (102, [4, 5])]
    assert res.unmapped == []
    assert res.leftover_pages == []


def test_whitespace_only_differences():
    """Mapping is whitespace-insensitive: tabs/newlines/double spaces in the
    chunk text vs the page texts do not matter."""
    pages = [
        (1, "ALPHA  bravo\ncharlie "),
        (2, "\tdelta echo\n\nfoxtrot\n"),
    ]
    chunks = [(11, "ALPHA bravo charlie delta\necho   foxtrot")]
    res = map_chunks_to_pages(pages, chunks)
    assert res.mapped == [(11, [1, 2])]
    assert res.unmapped == []
    assert res.leftover_pages == []


def test_unmapped_chunk_and_resync():
    """A chunk whose text matches no pages is reported UNMAPPED with a
    diagnostic; the walk resyncs on the next chunk's head so it still maps,
    and the bad chunk's page is reported leftover."""
    pages = [
        (1, "alpha bravo charlie"),
        (2, "delta echo foxtrot"),
        (3, "golf hotel india"),
    ]
    chunks = [
        (201, "alpha bravo charlie"),
        (202, "THIS TEXT MATCHES NO PAGE AT ALL"),
        (203, "golf hotel india"),
    ]
    res = map_chunks_to_pages(pages, chunks)
    assert res.mapped == [(201, [1]), (203, [3])]
    assert len(res.unmapped) == 1
    diag = res.unmapped[0]
    assert diag["chunk_id"] == 202
    assert diag["expected_head"] == normalize_text(chunks[1][1])[:60]
    assert diag["expected_tail"] == normalize_text(chunks[1][1])[-60:]
    # The junk-tolerant walk may tentatively consume pages (treating them as
    # deletable junk) before exhausting the budget or the page list, so the
    # failure pointer/tentative list are not pinned to the first page.
    assert diag["reason"]
    assert set(diag["tentative_page_seqs"]) <= {2, 3}
    # Page 2 belonged to the bad chunk and was consumed by no mapped chunk.
    assert res.leftover_pages == [2]


def test_leftover_pages_reported():
    """Trailing pages consumed by no chunk are reported as leftovers."""
    pages = [(1, "one fish"), (2, "two fish"), (3, "red fish")]
    chunks = [(50, "one fish")]
    res = map_chunks_to_pages(pages, chunks)
    assert res.mapped == [(50, [1])]
    assert res.unmapped == []
    assert res.leftover_pages == [2, 3]
