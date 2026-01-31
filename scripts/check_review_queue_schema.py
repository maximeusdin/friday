import psycopg2
c = psycopg2.connect(host='localhost', port=5432, dbname='neh', user='neh', password='neh')
r = c.cursor()
r.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'mention_review_queue'")
print("mention_review_queue columns:")
for x in r.fetchall():
    print(f"  {x[0]}")

# Get a sample row
r.execute("SELECT * FROM mention_review_queue LIMIT 1")
row = r.fetchone()
if row:
    r.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'mention_review_queue' ORDER BY ordinal_position")
    cols = [x[0] for x in r.fetchall()]
    print("\nSample row:")
    for col, val in zip(cols, row):
        val_str = str(val)[:60] if val else 'NULL'
        print(f"  {col}: {val_str}")

# Check collections
r.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'collections'")
print("\ncollections columns:")
for x in r.fetchall():
    print(f"  {x[0]}")
c.close()
