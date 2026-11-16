"""Page text, chunks and BM25 retrieval for the document reader (M21)."""

from __future__ import annotations

from privy_council.search import bm25
from privy_council.search.extract import TABLE_CHARS, chunks, html_blocks, text_blocks

# A made-up filing: its debt table sits after 100,000 characters of other notes,
# past what a web fetch returns.
FILLER = "".join(
    f"<h2>Note {n}: Other matters</h2><p>{'The company discusses revenue recognition. ' * 40}</p>"
    for n in range(1, 60)
)
FILING = f"""<html><head><title>Widget Corp 10-Q</title><style>p {{color: red}}</style>
<script>var tracking = "debt maturities";</script></head><body>
{FILLER}
<h2>Note 60: Debt</h2>
<p>The following table shows principal maturities of debt by year.</p>
<table>
<tr><th>Year</th><th>Principal (in millions)</th></tr>
<tr><td>2026</td><td>1,111</td></tr>
<tr><td>2027</td><td>2,222</td></tr>
<tr><td>Thereafter</td><td>3,333</td></tr>
</table>
<h2>Note 61: Leases</h2><p>Operating leases run to 2035.</p>
</body></html>"""


def test_html_blocks_drop_scripts_and_keep_tables() -> None:
    blocks = html_blocks(FILING)
    assert not any("tracking" in b.text or "color" in b.text for b in blocks)
    table = next(b for b in blocks if b.kind == "table")
    assert table.text.splitlines() == [
        "Year | Principal (in millions)",
        "2026 | 1,111",
        "2027 | 2,222",
        "Thereafter | 3,333",
    ]
    assert any(b.kind == "heading" and b.text == "Note 60: Debt" for b in blocks)


def test_a_table_deep_in_a_long_page_is_the_top_passage() -> None:
    assert len(FILING) > 100_000
    passages = [c.render() for c in chunks(html_blocks(FILING))]
    best = bm25.top("debt principal maturities by year", passages, 6, 8000)
    assert "2027 | 2,222" in passages[best[0]]
    assert passages[best[0]].startswith("Note 60: Debt")
    assert sum(len(passages[i]) for i in best) <= 8000


def test_chunks_stay_near_their_size_and_carry_their_heading() -> None:
    passages = chunks(html_blocks(FILING))
    assert all(len(c.text) <= 1500 * 2 for c in passages)
    assert all(c.heading.startswith("Note") for c in passages)


def test_a_long_table_is_cut_by_rows_with_its_header() -> None:
    rows = "".join(f"<tr><td>Row {i}</td><td>{i * 7}</td></tr>" for i in range(600))
    html = f"<h3>Schedule</h3><table><tr><th>Item</th><th>Amount</th></tr>{rows}</table>"
    parts = chunks(html_blocks(html))
    assert len(parts) > 1
    assert all(p.text.startswith("Item | Amount") for p in parts)
    assert all(len(p.text) <= TABLE_CHARS for p in parts)


def test_plain_text_paragraphs() -> None:
    assert [b.text for b in text_blocks("First  line\nsame para\n\n\nSecond")] == [
        "First line same para",
        "Second",
    ]


def test_bm25_skips_passages_without_a_query_word() -> None:
    assert bm25.top("maturities", ["nothing here", "leases only"], 6, 8000) == []
    assert bm25.top("2027", ["due in 2026", "due in 2027"], 6, 8000) == [1]


def test_a_table_caption_becomes_its_heading() -> None:
    page = "<table><caption>Debt maturities</caption><tr><td>2027</td><td>2,222</td></tr></table>"
    passages = [c.render() for c in chunks(html_blocks(page))]
    assert passages == ["Debt maturities\n2027 | 2,222"]


def test_the_table_without_unasked_qualifiers_wins() -> None:
    rows = (
        "<tr><th>Account</th><th>2025 Oct</th></tr>"
        "<tr><td>Treasury securities</td><td>{v}</td></tr>"
    )
    page = "".join(
        f"<h2>Table {n}. Assets of {who} Banks</h2><p>Seasonally adjusted, billions.</p>"
        f"<table>{rows.format(v=v)}</table>"
        for n, who, v in [
            (1, "Small Regional", "11"),
            (2, "Large Regional", "22"),
            (3, "All Commercial", "33"),
            (4, "Foreign-Related", "44"),
        ]
    )
    cut = chunks(html_blocks(page))
    passages = [c.render() for c in cut]
    labels = [c.label() for c in cut]
    query = "Treasury securities, all commercial banks, seasonally adjusted"
    assert "33" in passages[bm25.top(query, passages, 1, 8000, labels)[0]]
    small = "Treasury securities at small regional banks"
    assert "11" in passages[bm25.top(small, passages, 1, 8000, labels)[0]]
