# Research Console - Setup & Development Guide

A minimal "research console" UI for the Friday research assistant. Enables the workflow:
question → plan proposal → approve → execute → results with click-to-evidence.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Next.js)                       │
│  ┌─────────────┬──────────────────────┬────────────────────────┐ │
│  │  Sessions   │    Conversation      │  Plan / Results /      │ │
│  │  List       │    (messages)        │  Evidence Viewer       │ │
│  └─────────────┴──────────────────────┴────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ HTTP API
┌─────────────────────────────────────────────────────────────────┐
│                       Backend (FastAPI)                          │
│  /api/sessions, /api/plans, /api/result-sets, /api/documents     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ subprocess / import
┌─────────────────────────────────────────────────────────────────┐
│                    Existing Scripts & DB                         │
│  plan_query.py, execute_plan.py, approve_plan.py                 │
│  PostgreSQL: research_sessions, research_plans, result_sets      │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL with existing Friday schema
- `DATABASE_URL` environment variable set

### Backend

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Run migration for messages table
psql $DATABASE_URL -f ../migrations/0039_research_messages.sql

# Set required env (or put it in ../.env)
export DATABASE_URL="postgresql://..."

# Start server
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

Open http://localhost:3000

## API Contract

See [docs/v1_contract.md](./v1_contract.md) for the complete API contract.

Key endpoints:
- `POST /api/sessions` - Create session
- `POST /api/sessions/:id/messages` - Send message (triggers plan proposal)
- `POST /api/plans/:id/approve` - Approve plan
- `POST /api/plans/:id/execute` - Execute plan
- `GET /api/result-sets/:id` - Get results
- `GET /api/documents/:id/pdf` - Serve PDF

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `PDF_ROOT` | Root directory for PDF files | `./data` |
| `OPENAI_API_KEY` | For plan_query.py LLM calls | Required for planning |
| `OPENAI_MODEL_PLAN` | Model for plan generation | `gpt-5-mini` |

### Frontend Configuration

Edit `frontend/next.config.js` to configure:
- `API_URL` - Backend URL for API proxy (default: `http://localhost:8000`)

## Development

### Project Structure

```
friday/
├── backend/
│   └── app/
│       ├── main.py              # FastAPI app
│       ├── routes/              # API routes
│       │   ├── sessions.py      # Sessions & messages
│       │   ├── plans.py         # Plan CRUD
│       │   ├── results.py       # Result sets
│       │   └── documents.py     # PDFs & evidence
│       └── services/            # Business logic
│           ├── planner.py       # Wraps plan_query.py
│           ├── executor.py      # Wraps execute_plan.py
│           └── evidence.py      # Evidence assembly
├── frontend/
│   ├── src/
│   │   ├── app/                 # Next.js app router
│   │   ├── components/          # React components
│   │   ├── lib/api.ts           # API client
│   │   └── types/api.ts         # TypeScript types
│   └── package.json
├── docs/
│   ├── v1_contract.md           # API contract (LOCKED)
│   └── RESEARCH_CONSOLE.md      # This file
└── migrations/
    └── 0039_research_messages.sql
```

### Adding New Features

1. Update `docs/v1_contract.md` if API changes are needed
2. Add/update types in `frontend/src/types/api.ts`
3. Implement backend route in `backend/app/routes/`
4. Add frontend component in `frontend/src/components/`

### Testing

```bash
# Backend
cd backend
pytest

# Frontend
cd frontend
npm run lint
```

## Deployment

### Docker (recommended for production)

```dockerfile
# See docker-compose.yml for full configuration
docker-compose up -d
```

### Manual

1. Build frontend: `cd frontend && npm run build`
2. Serve frontend with nginx or `next start`
3. Run backend with gunicorn: `gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker`

## Troubleshooting

### "Plan proposal failed"

- Check `OPENAI_API_KEY` is set
- Check plan_query.py works standalone: `python scripts/plan_query.py --session 1 --text "test"`

### "PDF not found"

- Check `PDF_ROOT` environment variable
- Verify document's `source_ref` path exists

### "Database connection failed"

- Verify `DATABASE_URL` is correct
- Run migrations: `psql $DATABASE_URL -f migrations/0039_research_messages.sql`
