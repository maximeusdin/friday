# Quick Start: Plan Query Workflow

## Step-by-Step Example

### 1. Create Session
```bash
python scripts/new_session.py --label "test session"
# Output: 1  (save this ID)
```

### 2. Run Some Queries & Save Result Sets

```bash
# Query 1
python scripts/run_query.py --session 1 "Oppenheimer atomic bomb" --k 20
# Note the Run ID from output (e.g., "Run ID: 42")

python scripts/save_result_set.py --run-id 42 --name "Oppenheimer search"

# Query 2  
python scripts/run_query.py --session 1 "Alger Hiss" --k 15
# Note the Run ID (e.g., "Run ID: 43")

python scripts/save_result_set.py --run-id 43 --name "Hiss search"
```

### 3. Create Plan with Deictic Reference

```bash
# This will detect "those results" and resolve to most recent result_set
python scripts/plan_query.py --session 1 --text "find mentions of Treasury in those results"
```

**What you'll see:**
- Deictic detection: "those results" → automatically resolved to result_set_id 2 (most recent)
- Plan summary showing resolved references
- Plan saved with status='proposed'

### 4. Verify

```bash
# Check saved plan
psql $DATABASE_URL -c "SELECT id, raw_utterance, resolved_deictics::text, status FROM research_plans WHERE session_id = 1;"
```

The `resolved_deictics` JSON will show:
```json
{"detected": {"those_results": true}, "resolved_to": 2, "resolved_result_set_id": 2}
```

### 5. Try Another Deictic

```bash
python scripts/plan_query.py --session 1 --text "search for Rosenberg in earlier results"
```

This detects "earlier results" and resolves to the most recent result_set.

## Environment Variables

```bash
export DATABASE_URL="postgresql://user:pass@localhost/dbname"
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL_PLAN="gpt-4o-mini"  # or gpt-5-mini
```

## Notes

- Deictic patterns detected: "those results", "that result", "earlier results", "previous results", "above results", "last results"
- Resolution: Always resolves to most recent result_set in session
- Plan status: Always saved as 'proposed' (requires approval before execution)
- Dry run: Add `--dry-run` to test without saving
