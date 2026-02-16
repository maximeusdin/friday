#!/usr/bin/env python3
"""
Quick test: Can GPT-4.1-mini produce structured JSON output?

Run from a terminal where OPENAI_API_KEY is set:
    python scripts/test_gpt41_mini_structured.py

Compares gpt-4.1-mini-2025-04-14 vs gpt-4o for structured output compatibility.
"""

import json
import os
import sys


def test_model(client, model: str, test_name: str) -> dict | None:
    """Test a model with response_format json_object. Returns parsed JSON or None on error."""
    print(f"\n--- {test_name} ({model}) ---")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Extract entities from this sentence as JSON: 'John Smith met with Maria Garcia in Paris on March 15.' "
                    "Return a JSON object with keys: people (list of names), location (string), date (string).",
                }
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        text = resp.choices[0].message.content
        parsed = json.loads(text)
        print(f"Raw: {text[:200]}...")
        print(f"Parsed: {json.dumps(parsed, indent=2)}")
        return parsed
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_json_schema(client, model: str) -> dict | None:
    """Test structured output with JSON schema (stricter than json_object)."""
    print(f"\n--- JSON schema structured output ({model}) ---")
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "entity_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "people": {"type": "array", "items": {"type": "string"}},
                    "location": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["people", "location", "date"],
                "additionalProperties": False,
            },
        },
    }
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Extract: 'Alice and Bob met in London on 2024-01-10'. Return JSON with people, location, date.",
                }
            ],
            response_format=schema,
            temperature=0,
        )
        text = resp.choices[0].message.content
        parsed = json.loads(text)
        print(f"Raw: {text}")
        print(f"Parsed: {json.dumps(parsed, indent=2)}")
        return parsed
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set. Set it and run again.")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("openai package not installed. Run: pip install openai")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print("=" * 60)
    print("GPT-4.1-mini vs gpt-4o: Structured JSON output test")
    print("=" * 60)

    # Test 1: json_object format
    r1 = test_model(client, "gpt-4.1-mini-2025-04-14", "gpt-4.1-mini (json_object)")
    r2 = test_model(client, "gpt-4o", "gpt-4o (json_object)")

    # Test 2: JSON schema (stricter)
    r3 = test_json_schema(client, "gpt-4.1-mini-2025-04-14")
    r4 = test_json_schema(client, "gpt-4o")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    results = [
        ("gpt-4.1-mini json_object", r1 is not None),
        ("gpt-4o json_object", r2 is not None),
        ("gpt-4.1-mini json_schema", r3 is not None),
        ("gpt-4o json_schema", r4 is not None),
    ]
    for name, ok in results:
        status = "OK" if ok else "FAILED"
        print(f"  {name}: {status}")

    all_ok = all(ok for _, ok in results)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
