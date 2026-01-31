import psycopg2
c = psycopg2.connect(host='localhost', port=5432, dbname='neh', user='neh', password='neh')
r = c.cursor()
r.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'documents'")
print("documents columns:")
for x in r.fetchall():
    print(f"  {x[0]}")
r.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'chunks'")
print("\nchunks columns:")
for x in r.fetchall():
    print(f"  {x[0]}")
c.close()
