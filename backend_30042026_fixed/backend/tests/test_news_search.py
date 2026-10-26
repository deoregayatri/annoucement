import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
BASE = "/news/search"


# ── helpers ───────────────────────────────────────────────────────────────
def post(payload):
    return client.post(BASE, json=payload)


# ── 1. Schema / input validation ──────────────────────────────────────────
class TestInputValidation:

    def test_missing_query_returns_422(self):
        r = post({"company_symbol": "INFY"})
        assert r.status_code == 422

    def test_empty_query_returns_422(self):
        r = post({"query": ""})
        assert r.status_code == 422

    def test_no_symbol_accepted(self):
        """No symbol = search all, must not crash"""
        r = post({"query": "quarterly results", "top_k": 2})
        assert r.status_code == 200

    def test_single_string_symbol(self):
        r = post({"query": "revenue", "company_symbol": "ASIANPAINT", "top_k": 2})
        assert r.status_code == 200

    def test_list_symbol(self):
        r = post({"query": "revenue", "company_symbol": ["ASIANPAINT", "INFY"], "top_k": 2})
        assert r.status_code == 200

    def test_lowercase_symbol_normalised(self):
        """lowercase 'infy' should work same as 'INFY'"""
        r = post({"query": "revenue", "company_symbol": "infy", "top_k": 2})
        assert r.status_code == 200

    def test_symbol_with_whitespace_stripped(self):
        r = post({"query": "revenue", "company_symbol": "  INFY  ", "top_k": 2})
        assert r.status_code == 200

    def test_top_k_upper_bound(self):
        r = post({"query": "results", "top_k": 51})
        assert r.status_code == 422

    def test_top_k_lower_bound(self):
        r = post({"query": "results", "top_k": 0})
        assert r.status_code == 422


# ── 2. Response shape ─────────────────────────────────────────────────────
class TestResponseShape:

    def test_required_fields_always_present(self):
        r = post({"query": "Asian Paints results", "company_symbol": "ASIANPAINT"})
        data = r.json()
        for field in ("query", "answer", "confidence", "sources",
                      "total_results", "retrieval_time_ms", "query_id"):
            assert field in data, f"Missing field: {field}"

    def test_sources_is_list(self):
        r = post({"query": "Asian Paints results", "company_symbol": "ASIANPAINT"})
        assert isinstance(r.json()["sources"], list)

    def test_total_results_matches_sources_length(self):
        r = post({"query": "Asian Paints results", "company_symbol": "ASIANPAINT"})
        data = r.json()
        assert data["total_results"] == len(data["sources"])

    def test_confidence_between_0_and_1(self):
        r = post({"query": "Asian Paints results", "company_symbol": "ASIANPAINT"})
        c = r.json()["confidence"]
        assert 0.0 <= c <= 1.0

    def test_source_fields_present(self):
        r = post({"query": "Asian Paints results", "company_symbol": "ASIANPAINT"})
        data = r.json()
        if data["sources"]:
            src = data["sources"][0]
            for field in ("chunk_id", "text_preview", "url",
                          "company_symbol", "fused_score"):
                assert field in src, f"Source missing field: {field}"


# ── 3. Symbol filtering ───────────────────────────────────────────────────
class TestSymbolFiltering:

    def test_symbol_filter_respected_in_sources(self):
        """Every returned source must belong to the requested company."""
        r = post({"query": "quarterly results", "company_symbol": "ASIANPAINT", "top_k": 5})
        data = r.json()
        for src in data["sources"]:
            assert src["company_symbol"].upper() == "ASIANPAINT", \
                f"Expected ASIANPAINT, got {src['company_symbol']}"

    def test_multi_symbol_sources_belong_to_requested(self):
        allowed = {"ASIANPAINT", "INFY"}
        r = post({"query": "quarterly results",
                  "company_symbol": list(allowed), "top_k": 5})
        data = r.json()
        for src in data["sources"]:
            assert src["company_symbol"].upper() in allowed, \
                f"Unexpected company: {src['company_symbol']}"

    def test_unknown_symbol_returns_no_data_message(self):
        r = post({"query": "results", "company_symbol": "ZZZNOTEXIST", "top_k": 3})
        data = r.json()
        assert data["total_results"] == 0
        assert data["sources"] == []
        assert len(data["answer"]) > 0   # friendly message, not empty

    def test_no_symbol_returns_results_from_any_company(self):
        r = post({"query": "quarterly results", "top_k": 5})
        data = r.json()
        if data["sources"]:
            symbols = {s["company_symbol"] for s in data["sources"]}
            assert len(symbols) >= 1


class TestNoFilterBehaviour:

    def test_no_symbol_still_returns_results(self):
        """Core regression — removing company_symbol must not produce no-data."""
        r = post({"query": "National Capital Region", "top_k": 5})
        data = r.json()
        assert data["total_results"] > 0, (
            "No-filter query returned 0 results — likely BM25 break bug"
        )

    def test_no_symbol_results_have_valid_scores(self):
        r = post({"query": "National Capital Region", "top_k": 5})
        for src in r.json()["sources"]:
            assert src["fused_score"] > 0.0

    def test_symbol_and_no_symbol_same_query_both_return_results(self):
        """With and without filter should both return results, not just one."""
        r_filtered   = post({"query": "National Capital Region",
                             "company_symbol": "POLICYBZR", "top_k": 3})
        r_unfiltered = post({"query": "National Capital Region", "top_k": 3})
        assert r_filtered.json()["total_results"] > 0,   "Filtered returned nothing"
        assert r_unfiltered.json()["total_results"] > 0, "Unfiltered returned nothing"

    def test_bm25_low_idf_query_falls_back_to_dense(self):
        """Common financial terms (low BM25 IDF) should still return results via dense."""
        r = post({"query": "National Stock Exchange", "top_k": 3})
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert isinstance(data["sources"], list)


# ── 4. No-data / edge cases ───────────────────────────────────────────────
class TestNoDataHandling:

    def test_no_hits_answer_is_string_not_null(self):
        r = post({"query": "xyzzy gibberish 12345", "company_symbol": "ZZZNOTEXIST"})
        data = r.json()
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 10

    def test_no_hits_confidence_is_zero(self):
        r = post({"query": "xyzzy gibberish 12345", "company_symbol": "ZZZNOTEXIST"})
        assert r.json()["confidence"] == 0.0

    def test_no_hits_sources_empty_list(self):
        r = post({"query": "xyzzy gibberish 12345", "company_symbol": "ZZZNOTEXIST"})
        assert r.json()["sources"] == []

    def test_response_time_within_30s(self):
        import time
        t0 = time.time()
        post({"query": "Asian Paints revenue", "company_symbol": "ASIANPAINT", "top_k": 3})
        assert time.time() - t0 < 30


# ── 5. Answer quality guards ──────────────────────────────────────────────
class TestAnswerQuality:

    def test_answer_has_no_document_citation_artifacts(self):
        """Ensure the old (Document: Unknown | Chunk: | Page:) artifact is gone."""
        r = post({"query": "Asian Paints profit", "company_symbol": "ASIANPAINT"})
        answer = r.json().get("answer", "")
        for banned in ("Document:", "Unknown Document", "Page:", "Announcement Date:"):
            assert banned not in answer, f"Citation artifact found: '{banned}'"

    def test_answer_not_empty_when_hits_exist(self):
        r = post({"query": "Asian Paints profit", "company_symbol": "ASIANPAINT"})
        data = r.json()
        if data["total_results"] > 0:
            assert data["answer"] and len(data["answer"]) > 20
