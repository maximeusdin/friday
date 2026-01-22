import json
import os
from types import SimpleNamespace

import pytest

# Import your module under test
# Adjust this import if your script isn't importable as a module yet.
import scripts.plan_query as plan_query


class _FakeChatCompletions:
    def __init__(self, message):
        self._message = message

    def parse(self, **kwargs):
        # Mimic OpenAI response object shape: resp.choices[0].message
        return SimpleNamespace(choices=[SimpleNamespace(message=self._message)])


class _FakeBeta:
    def __init__(self, message):
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(message))


class FakeOpenAI:
    """
    Fake OpenAI client that returns a fixed message.
    """
    def __init__(self, api_key=None, message=None):
        self.beta = _FakeBeta(message)


def _make_message(parsed, content):
    # Mimic message object with .parsed and .content
    return SimpleNamespace(parsed=parsed, content=content)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    # Ensure the code doesn't fail on missing env vars during tests
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-do-not-use")
    monkeypatch.setenv("OPENAI_MODEL_PLAN", "gpt-5-mini")


def test_call_llm_structured_uses_llm_content_when_parsed_none(monkeypatch):
    """
    If message.parsed is None but message.content is valid JSON,
    call_llm_structured() should return the JSON from message.content.

    This proves that TERM('Treasury') originates from the LLM output JSON,
    not from deterministic code paths like deictic injection.
    """
    llm_json = {
        "query": {
            "raw": "find mentions of Treasury in those results",
            "primitives": [
                {
                    "type": "TERM",
                    "value": "Treasury",
                    "slug": None,
                    "document_id": None,
                    "result_set_id": None,
                    "entity_id": None,
                    "entity_a": None,
                    "entity_b": None,
                    "start": None,
                    "end": None,
                    "window": None,
                    "scope": None,
                    "field": None,
                    "evidence_type": None,
                    "enabled": None,
                    "source_slug": None,
                    "primitives": [],
                }
            ],
        },
        "needs_clarification": False,
        "choices": [],
    }

    msg = _make_message(parsed=None, content=json.dumps(llm_json))

    # Patch the OpenAI class used inside plan_query to our fake
    def _fake_openai_ctor(api_key=None):
        return FakeOpenAI(api_key=api_key, message=msg)

    monkeypatch.setattr(plan_query, "OpenAI", _fake_openai_ctor)

    out = plan_query.call_llm_structured("PROMPT DOES NOT MATTER HERE", model="gpt-5-mini")

    assert out["query"]["raw"] == "find mentions of Treasury in those results"
    prims = out["query"]["primitives"]
    assert any(p.get("type") == "TERM" and p.get("value") == "Treasury" for p in prims), (
        "Expected TERM('Treasury') to come from LLM JSON content fallback path"
    )


def test_deictic_injection_does_not_invent_terms():
    """
    Deterministic deictic injection should add WITHIN_RESULT_SET if needed,
    but must not fabricate query terms.
    """
    base_plan = {
        "query": {
            "raw": "find mentions of Treasury in those results",
            "primitives": [],  # no terms present
        },
        "needs_clarification": False,
        "choices": [],
    }

    injected = plan_query.inject_within_result_set_if_needed(base_plan, rs_id=11)
    prims = injected["query"]["primitives"]

    assert any(p.get("type") == "WITHIN_RESULT_SET" and p.get("result_set_id") == 11 for p in prims)
    assert not any(p.get("type") == "TERM" for p in prims), (
        "Deictic injection must not fabricate TERM primitives"
    )


def test_normalization_preserves_llm_term(monkeypatch):
    """
    After normalization, the TERM value should still be present and compacted.
    """
    llm_plan = {
        "query": {
            "raw": "find mentions of Treasury in those results",
            "primitives": [
                {
                    "type": "TERM",
                    "value": "Treasury",
                    "slug": None,
                    "document_id": None,
                    "result_set_id": None,
                    "entity_id": None,
                    "entity_a": None,
                    "entity_b": None,
                    "start": None,
                    "end": None,
                    "window": None,
                    "scope": None,
                    "field": None,
                    "evidence_type": None,
                    "enabled": None,
                    "source_slug": None,
                    "primitives": [],
                }
            ],
        },
        "needs_clarification": False,
        "choices": [],
    }

    normalized = plan_query.normalize_plan_dict(llm_plan)
    prims = normalized["query"]["primitives"]

    assert prims == [{"type": "TERM", "value": "Treasury"}], (
        f"Expected compact normalized primitive, got: {prims}"
    )
