r"""Interactive questions for the running native RAG service.

    python scripts\ask.py            (course 1)
    python scripts\ask.py --course 2

Type a question and press Enter. Type q (or press Ctrl+C) to quit.
Reads SERVICE_API_KEY from ..\.env, so you never type the key.
"""
import argparse
import os
from pathlib import Path

import httpx
from dotenv import dotenv_values

ap = argparse.ArgumentParser()
ap.add_argument("--course", default="1")
ap.add_argument("--url", default="http://127.0.0.1:8003")
a = ap.parse_args()

key = dotenv_values(Path(__file__).resolve().parent.parent / ".env").get("SERVICE_API_KEY") or os.getenv("SERVICE_API_KEY", "")
if not key:
    raise SystemExit("SERVICE_API_KEY not found in .env")

print(f"Course {a.course}. Type q to quit.\n")
with httpx.Client(timeout=600, headers={"X-API-Key": key}) as c:
    while True:
        try:
            q = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"q", "quit", "exit"}:
            break
        if not q:
            continue
        r = c.post(f"{a.url}/query", json={"question": q, "course_id": a.course})
        if r.status_code != 200:
            print(f"Error {r.status_code}: {r.text}\n")
            continue
        j = r.json()
        print("\n" + (j.get("answer") or "(no answer)"))
        cited = sorted({f"{x['source']} p.{x['page']}" if x.get("page") else x["source"]
                        for x in j.get("references", []) if x.get("cited")})
        print(f"\nCited: {', '.join(cited) or 'none'}   |   {j.get('seconds')}s\n")


        
