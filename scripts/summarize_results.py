#!/usr/bin/env python3
"""
summarize_results.py --result-set <id> --type brief|detailed|thematic

Generate LLM summary of result set.

Summary types:
    brief    - 2-3 sentence overview
    detailed - Comprehensive summary with key findings
    thematic - Organized by themes/topics found in results

Usage:
    python scripts/summarize_results.py --result-set 123 --type brief
    python scripts/summarize_results.py --result-set 123 --type detailed --save
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from psycopg2.extras import Json
from openai import OpenAI


def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Missing DATABASE_URL environment variable")
    return psycopg2.connect(dsn)


def get_result_set(conn, result_set_id: int) -> Dict[str, Any]:
    """Load result set metadata and chunk IDs."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, name, retrieval_run_id, chunk_ids, created_at
            FROM result_sets
            WHERE id = %s
            """,
            (result_set_id,),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Result set not found: {result_set_id}")
        
        return {
            "id": row[0],
            "name": row[1],
            "retrieval_run_id": row[2],
            "chunk_ids": row[3] or [],
            "created_at": row[4],
        }


def get_chunks_text(conn, chunk_ids: List[int], limit: int = 50) -> List[Dict[str, Any]]:
    """Load chunk text for summarization."""
    if not chunk_ids:
        return []
    
    # Limit to avoid exceeding context window
    chunk_ids_limited = chunk_ids[:limit]
    
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 
                c.id,
                c.text,
                cm.document_id,
                d.source_name as doc_title,
                col.slug as collection_slug
            FROM chunks c
            LEFT JOIN chunk_metadata cm ON cm.chunk_id = c.id
            LEFT JOIN documents d ON d.id = cm.document_id
            LEFT JOIN collections col ON col.id = d.collection_id
            WHERE c.id = ANY(%s)
            ORDER BY array_position(%s, c.id)
            """,
            (chunk_ids_limited, chunk_ids_limited),
        )
        
        return [
            {
                "chunk_id": row[0],
                "text": row[1][:1000] if row[1] else "",  # Truncate long chunks
                "document_id": row[2],
                "document_title": row[3] or "Unknown",
                "collection": row[4] or "unknown",
            }
            for row in cur.fetchall()
        ]


def get_retrieval_context(conn, retrieval_run_id: int) -> Optional[Dict[str, Any]]:
    """Get context about the retrieval run that produced these results."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 
                rr.id,
                rr.search_type,
                rr.query_text,
                rp.user_utterance
            FROM retrieval_runs rr
            LEFT JOIN research_plans rp ON rp.retrieval_run_id = rr.id
            WHERE rr.id = %s
            """,
            (retrieval_run_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        
        return {
            "run_id": row[0],
            "search_type": row[1],
            "query_text": row[2],
            "user_utterance": row[3],
        }


def build_summarization_prompt(
    summary_type: str,
    result_set: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    retrieval_context: Optional[Dict[str, Any]],
) -> str:
    """Build the prompt for LLM summarization."""
    
    # Build context section
    context_parts = []
    if retrieval_context:
        if retrieval_context.get("user_utterance"):
            context_parts.append(f"User Query: {retrieval_context['user_utterance']}")
        if retrieval_context.get("query_text"):
            context_parts.append(f"Search Terms: {retrieval_context['query_text']}")
    
    context_section = "\n".join(context_parts) if context_parts else "No query context available."
    
    # Build chunks section
    chunks_text = []
    for i, chunk in enumerate(chunks, 1):
        chunk_header = f"[{i}] From: {chunk['document_title']} ({chunk['collection']})"
        chunks_text.append(f"{chunk_header}\n{chunk['text']}\n")
    
    chunks_section = "\n---\n".join(chunks_text)
    
    # Build type-specific instructions
    if summary_type == "brief":
        type_instructions = """
Generate a brief summary (2-3 sentences) that:
- Captures the main topic/theme across these results
- Highlights key findings or patterns
- Is concise and actionable for a researcher
"""
    elif summary_type == "detailed":
        type_instructions = """
Generate a detailed summary (3-5 paragraphs) that:
- Provides a comprehensive overview of the content
- Identifies key findings, patterns, and themes
- Notes any significant documents, entities, or events mentioned
- Highlights connections between different pieces of evidence
- Suggests areas for further investigation
"""
    elif summary_type == "thematic":
        type_instructions = """
Generate a thematic summary organized by topic:
- Identify 3-5 main themes or topics in the results
- For each theme, provide:
  - A descriptive heading
  - Key evidence supporting this theme
  - Relevant documents or sources
- Conclude with connections between themes
"""
    else:
        type_instructions = "Generate a summary of the following search results."
    
    return f"""You are a research assistant summarizing search results from historical archives.

SEARCH CONTEXT:
{context_section}

RESULT SET: {result_set['name']}
Total Results: {len(result_set['chunk_ids'])} chunks
Showing: {len(chunks)} top-ranked results

INSTRUCTIONS:
{type_instructions}

SEARCH RESULTS:
{chunks_section}

SUMMARY:"""


def call_llm_summarize(prompt: str, model: Optional[str] = None) -> str:
    """Call LLM to generate summary."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable")
    
    if model is None:
        model = os.getenv("OPENAI_MODEL_SUMMARY", "gpt-4o-mini")
    
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a research assistant helping historians analyze archival documents. "
                           "Provide clear, factual summaries based only on the evidence provided."
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,  # Lower temperature for more consistent summaries
        max_tokens=1500,
    )
    
    return response.choices[0].message.content or ""


def save_summary(
    conn,
    result_set_id: int,
    summary_type: str,
    summary_text: str,
) -> int:
    """Save summary to database (as a system message in the session)."""
    # Get session_id from result_set
    with conn.cursor() as cur:
        cur.execute(
            "SELECT session_id FROM result_sets WHERE id = %s",
            (result_set_id,),
        )
        row = cur.fetchone()
        session_id = row[0] if row else None
        
        if session_id:
            # Save as a system message
            cur.execute(
                """
                INSERT INTO research_messages (session_id, role, content, result_set_id, metadata)
                VALUES (%s, 'system', %s, %s, %s)
                RETURNING id
                """,
                (
                    session_id,
                    f"Summary ({summary_type}): {summary_text}",
                    result_set_id,
                    Json({
                        "summary_type": summary_type,
                        "generated_at": datetime.now().isoformat(),
                    }),
                ),
            )
            message_id = cur.fetchone()[0]
            conn.commit()
            return message_id
        
        return 0


def summarize_result_set(
    conn,
    result_set_id: int,
    summary_type: str = "brief",
    model: Optional[str] = None,
    save: bool = False,
) -> Dict[str, Any]:
    """
    Generate a summary of a result set.
    
    Args:
        conn: Database connection
        result_set_id: ID of the result set to summarize
        summary_type: 'brief', 'detailed', or 'thematic'
        model: LLM model to use (optional)
        save: Whether to save the summary to the database
    
    Returns:
        Dict with summary text and metadata
    """
    # Load result set
    result_set = get_result_set(conn, result_set_id)
    
    if not result_set["chunk_ids"]:
        return {
            "result_set_id": result_set_id,
            "summary_type": summary_type,
            "summary": "No results to summarize.",
            "chunk_count": 0,
        }
    
    # Load chunks
    chunks = get_chunks_text(conn, result_set["chunk_ids"], limit=50)
    
    # Get retrieval context
    retrieval_context = None
    if result_set.get("retrieval_run_id"):
        retrieval_context = get_retrieval_context(conn, result_set["retrieval_run_id"])
    
    # Build prompt
    prompt = build_summarization_prompt(
        summary_type,
        result_set,
        chunks,
        retrieval_context,
    )
    
    # Call LLM
    print(f"Generating {summary_type} summary for {len(chunks)} chunks...", file=sys.stderr)
    summary_text = call_llm_summarize(prompt, model)
    
    result = {
        "result_set_id": result_set_id,
        "result_set_name": result_set["name"],
        "summary_type": summary_type,
        "summary": summary_text,
        "chunk_count": len(result_set["chunk_ids"]),
        "summarized_count": len(chunks),
    }
    
    # Save if requested
    if save:
        message_id = save_summary(conn, result_set_id, summary_type, summary_text)
        result["saved_message_id"] = message_id
        print(f"Summary saved as message ID: {message_id}", file=sys.stderr)
    
    return result


def main():
    ap = argparse.ArgumentParser(
        description="Generate LLM summary of result set"
    )
    ap.add_argument(
        "--result-set", type=int, required=True,
        help="Result set ID to summarize"
    )
    ap.add_argument(
        "--type", type=str, default="brief",
        choices=["brief", "detailed", "thematic"],
        help="Summary type: brief (2-3 sentences), detailed (comprehensive), thematic (by topic)"
    )
    ap.add_argument(
        "--model", type=str, default=None,
        help="LLM model to use (default: OPENAI_MODEL_SUMMARY or gpt-4o-mini)"
    )
    ap.add_argument(
        "--save", action="store_true",
        help="Save summary to database as a system message"
    )
    ap.add_argument(
        "--json", action="store_true",
        help="Output as JSON"
    )
    args = ap.parse_args()
    
    conn = get_conn()
    try:
        result = summarize_result_set(
            conn,
            args.result_set,
            summary_type=args.type,
            model=args.model,
            save=args.save,
        )
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n{'='*60}")
            print(f"SUMMARY ({result['summary_type'].upper()})")
            print(f"Result Set: {result['result_set_name']} ({result['chunk_count']} chunks)")
            print(f"{'='*60}\n")
            print(result["summary"])
            print()
    
    finally:
        conn.close()


if __name__ == "__main__":
    main()
