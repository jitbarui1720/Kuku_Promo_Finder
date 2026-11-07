#!/usr/bin/env python3
"""validate.py - hard-constraint checker for any promos.json (code, no AI).
Checks every promo against the brief: 20-60 s total, timestamps inside the episode, segments in order,
no spoken line cut in half at a boundary, hook/cliffhanger text really present in the clip.
  python validate.py --input screenplays --results results_rules/promos.json
"""
import argparse, json, re, sys
from parse import load_show
from candidates import events_in, straddles

ap = argparse.ArgumentParser()
ap.add_argument("--input", default="screenplays")
ap.add_argument("--results", default="results_rules/promos.json")
a = ap.parse_args()
eps = {e["episode"]: e for e in load_show(a.input)}
promos = json.load(open(a.results))["promos"]
problems, norm = [], lambda s: re.sub(r"\s+", " ", s.lower()).strip(" .\"'…")
for p in promos:
    tag = f"#{p['rank']} {p['promo_id']}"
    tot, prev = 0.0, None
    texts = []
    for s in p["segments"]:
        ep = eps.get(s["episode"])
        if not ep: problems.append(f"{tag}: unknown episode {s['episode']}"); continue
        if not (0 <= s["start"] < s["end"] <= ep["duration"] + 0.01):
            problems.append(f"{tag}: segment outside episode {s['episode']} ({s['start_ts']}-{s['end_ts']})")
        if prev and (s["episode"], s["start"]) < prev:
            problems.append(f"{tag}: segments out of order")
        prev = (s["episode"], s["end"]); tot += s["end"] - s["start"]
        for t in (s["start"], s["end"]):
            if straddles(ep, t): problems.append(f"{tag}: cut at {t:.1f}s chops a spoken line in ep {s['episode']}")
        texts += [norm(e["text"]) for e in events_in(ep, s["start"], s["end"])]
    if not (20 <= tot <= 60.05): problems.append(f"{tag}: duration {tot:.1f}s outside 20-60 s")
    blob = " ".join(texts)
    for k in ("hook", "cliffhanger"):
        if p[k] and norm(p[k])[:40] not in blob:
            problems.append(f"{tag}: {k} text not found in clip: {p[k][:50]!r}")
print(f"checked {len(promos)} promos: {len(problems)} problem(s)")
for x in problems: print(" -", x)
sys.exit(1 if problems else 0)
