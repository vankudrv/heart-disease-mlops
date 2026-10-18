"""Send realistic traffic to the running API so the Grafana dashboard has something to show.

Usage (with `minikube tunnel` running):
    python scripts/generate_traffic.py --url http://127.0.0.1:8000 --requests 300 --delay 0.1

Mix: mostly valid patients (randomly varied around two sample patients), plus a few deliberately
invalid ones (HTTP 422) so the error panels move too.
"""
import argparse
import json
import random
import time
from pathlib import Path

import requests

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def vary(record, rng):
    r = dict(record)
    r["age"] = int(min(77, max(29, r["age"] + rng.randint(-8, 8))))
    r["trestbps"] = int(min(200, max(94, r["trestbps"] + rng.randint(-15, 15))))
    r["chol"] = int(min(564, max(126, r["chol"] + rng.randint(-40, 40))))
    r["thalach"] = int(min(202, max(71, r["thalach"] + rng.randint(-20, 20))))
    r["oldpeak"] = round(min(6.2, max(0.0, r["oldpeak"] + rng.uniform(-0.8, 0.8))), 1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--delay", type=float, default=0.1, help="seconds between requests")
    ap.add_argument("--invalid-rate", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    bases = [json.loads((SAMPLES / n).read_text()) for n in ("sample_patient.json", "high_risk_patient.json")]
    counts = {}
    for i in range(a.requests):
        rec = vary(rng.choice(bases), rng)
        if rng.random() < a.invalid_rate:
            rec["age"] = 400  # out of range -> 422
        try:
            resp = requests.post(f"{a.url}/predict", json=rec, timeout=10)
            counts[resp.status_code] = counts.get(resp.status_code, 0) + 1
        except requests.RequestException as exc:
            print("request failed:", exc)
            counts["error"] = counts.get("error", 0) + 1
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{a.requests} sent, status counts: {counts}")
        time.sleep(a.delay)
    print("done:", counts)


if __name__ == "__main__":
    main()
