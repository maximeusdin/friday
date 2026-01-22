import os
from retrieval.ops import get_conn, concordance_expand_terms

os.environ["DATABASE_URL"] = os.environ.get("DATABASE_URL", "postgresql://neh:neh@localhost:5432/neh")

conn = get_conn()
print("AAF ->", concordance_expand_terms(conn, "AAF"))
print("Army Air Force, U.S. ->", concordance_expand_terms(conn, "Army Air Force, U.S."))
