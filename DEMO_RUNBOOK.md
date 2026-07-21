# Friday — Live Demo Runbook

## Before the meeting (15 minutes)

1. **Deploy** (if not already done): `bash deploy_all.sh` from the repo root — deploys API + frontend from `v13-chat-engine`. Migrations 0073/0074 are already applied to prod; RDS is already downsized and verified.
2. **Prod smoke** (2 min): open https://fridayarchive.org — check: sign-in works, `/health` is green, Search tab loads, About modal opens.
3. **Pre-warm the showpiece session**: create a session named "Demo", and run these queries in Chat (they persist — the demo shows the *results*, not the wait):
   - `Names of NKVD spies who infiltrated the OSS`
   - `Who were the members of the Perlo group?`
   - `When did the FBI recruit Morris Childs as an informer?`

## The demo flow (suggested ~15 min)

**Act 1 — the answer** (pre-warmed session): open the OSS question. Show the roster with citations. Click a citation → the document opens on the right page with the supporting passage **highlighted in amber**. That's the trust loop: every claim → its exact sentence on the scanned page.

**Act 2 — the worksheet**: flip to the Search tab → open **⚡ Chat's searches** → show the boolean pools Chat ran while answering (e.g. `OSS AND (NKVD OR Soviet OR agent OR espionage)` with its hit count). Open one — it's a normal result set: numbered hits, prune (✕ / restore), export CSV. *"Chat is a researcher whose worksheet you can inspect and continue."*

**Act 3 — live fire** (fast, reliable queries):
- `how many engineers and journalists were recruited by soviet intelligence in the 1930s` scoped to Vassiliev (~1–2 min) → "49 engineers, 22 journalists" [p172]
- `Who is Jurist?` (~2–6 min) → Harry Dexter White via codename resolution
- One live boolean search in the Search tab: `(OSS OR "Office of Strategic Services") AND (Soviet OR NKVD)` → ~220 hits with per-collection counts

**Act 4 — the researcher features**: numbered results (resume at #40 next week), tabbed searches, scope panel (click any checkbox — it engages), Concordance Index on the home page (view + CSV download), copy-answer button with source links.

## If something goes wrong

- **A live query runs long**: switch to the Search tab and watch ⚡ tabs appear in real time — the investigation is visible; narrate what it's searching. Or fall back to the pre-warmed session.
- **A feature misbehaves** — flip its kill-switch (ECS task env var, then restart service):

| Layer | Switch | Off = |
|---|---|---|
| Roster pool mining | `FRIDAY_POOL_MINING=0` | sampled roster (still good) |
| Insufficiency mine | `FRIDAY_INSUFFICIENCY_MINE=0` | exhortation-only pushback |
| Answer-term chase | `FRIDAY_ANSWER_TERMS=0` | (default; enable=1) |
| Records rewrite | `FRIDAY_RECORDS_REWRITE=0` | pre-rewrite retrieval |
| Context budgets | `V9_SUMMARIZER_INPUT_CHARS=4000` etc. | old budgets |
| Whole engine | `FRIDAY_CHAT_ENGINE=v13` | last week's proven engine |

## Expected timings (set expectations out loud)

- Scoped needle/count queries: **1–2 min**
- Full-archive lookups: **5–8 min** (deep budget: 20 investigation turns)
- Rosters/enumerations: **6–10 min** (includes the exhaustive pool mine)
- The latency IS the depth — every query runs at what used to be "Think Deeper" level, and Think Deeper still extends beyond it.

## The one-slide architecture story

*Translate → Bound → Read → Prove:* the question is translated into the archive's own record language (rewrites + codename resolution), a deterministic boolean pool bounds the complete match set (visible as ⚡ tabs), every page in the pool gets read by the miner (nothing sampled away), and every claim carries a verbatim quote that highlights on the scanned page. Search finds pages; Chat answers questions; the ⚡ tabs are where they meet.
