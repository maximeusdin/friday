from fastapi import FastAPI

app = FastAPI(
    title="Friday",
    description="AI research assistant for Cold War archival materials",
    version="0.1.0",
)

@app.get("/health")
def health():
    return {"status": "ok"}
