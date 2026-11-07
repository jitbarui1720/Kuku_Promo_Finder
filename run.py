#!/usr/bin/env python3
"""run.py - find and rank promos in a micro-drama screenplay folder.

  python run.py --input screenplays --out results                       # rules only, no key needed
  python run.py --input screenplays --out results --provider gemini     # + LLM judge (GEMINI_API_KEY)
  python run.py --input NEW_SHOW_FOLDER --out out2 --provider openai --model gpt-4o-mini
"""
import argparse
import csv
import json
import os
import re
import time

import candidates as C
import judge as J
from parse import load_show, fmt

W_LLM = (0.35, 0.35, 0.30)       # hook, cliffhanger, standalone weights for the LLM score
BLEND_LLM, BLEND_RULES = 0.70, 0.30
REJECT_PENALTY = 25.0
ENTRY_BONUS = [(5, 3.0), (15, 1.5)]   # assumption: promos from early eps are better entry points


def llm_score(l):
    return 100 * (W_LLM[0] * l["hook"] + W_LLM[1] * l["cliff"] + W_LLM[2] * l["standalone"]) / 10


def entry_bonus(ep_no):
    for lim, b in ENTRY_BONUS:
        if ep_no <= lim:
            return b
    return 0.0


def finalize(c, use_bonus=True):
    """Compute final score + display fields for a candidate dict."""
    l = c.get("llm")
    rules = c.get("score")
    if l:
        base = BLEND_LLM * llm_score(l) + BLEND_RULES * (rules if rules is not None else llm_score(l))
        if l["verdict"] == "reject":
            base -= REJECT_PENALTY
    else:
        base = rules or 0.0
    c["final"] = round(base + (entry_bonus(c["episode"]) if use_bonus else 0.0), 1)
    c["hook_final"] = (l or {}).get("hook_quote") or c.get("hook_text", "")
    c["cliff_final"] = (l or {}).get("cliff_text") or c.get("cliff_text", "")
    return c


def why_text(c):
    l = c.get("llm")
    if l and l.get("why"):
        return l["why"]
    spk = c.get("standalone", 0)
    return (f"Opens on \"{c['hook_final'][:70]}\" (hook {c['hook']}/10) and cuts on \"{c['cliff_final'][:70]}\" "
            f"(cliffhanger {c['cliff']}/10); standalone proxy {spk}/10.")


def rank_reason(c, rank):
    parts = []
    l = c.get("llm")
    dims = {"hook": (l or c).get("hook"), "cliffhanger": (l or c).get("cliff"),
            "standalone": (l or c).get("standalone")}
    best = max(("hook", "cliffhanger"), key=lambda k: dims[k] or 0)
    parts.append(f"strongest on {best} ({dims[best]}/10)")
    if (dims["standalone"] or 0) < 7:
        parts.append(f"weaker standalone ({dims['standalone']}/10)")
    if entry_bonus(c["episode"]):
        parts.append(f"small early-episode bonus (episode {c['episode']})")
    if c.get("at_episode_end"):
        parts.append("ends on the episode's own closing beat")
    if c["kind"] != "continuous":
        parts.append(f"{c['kind']} edit")
    return f"#{rank}: " + "; ".join(parts)


def flags(c):
    f = []
    l = c.get("llm")
    if c.get("unseen_refs"):
        f.append("mentions characters not shown in clip: " + ", ".join(c["unseen_refs"]))
    if l and not l.get("quotes_verified"):
        f.append("LLM quote not found verbatim in clip - used rule-based text")
    if l and l.get("verdict") == "reject":
        f.append("LLM verdict: reject")
    if c["kind"] == "montage" and not c.get("needs_human_check"):
        f.append("joins two moments - rules cannot verify the join plays coherently; editor must check")
    if c.get("needs_human_check"):
        f.append("composed across moments - editor must check it plays well")
    if c.get("llm") is None and c.get("_llm_run"):
        f.append("no LLM judgement for this clip - rule-based score only")
    return f


def seg_dicts(c):
    if "segment_list" in c:
        return [{"episode": e, "start": a, "end": b, "start_ts": fmt(a), "end_ts": fmt(b)}
                for e, (a, b) in c["segment_list"]]
    return [{"episode": c["episode"], "start": a, "end": b, "start_ts": fmt(a), "end_ts": fmt(b)}
            for a, b in c["segments"]]


def near(a, b, gap=5.0):
    """True if any segment of a is within `gap` seconds of (or overlaps) any segment of b."""
    sa = [(x, y) for x, y in (a["segments"] if "segments" in a else [s for _, s in a["segment_list"]])]
    sb = [(x, y) for x, y in (b["segments"] if "segments" in b else [s for _, s in b["segment_list"]])]
    return any(x1 < y2 + gap and x2 < y1 + gap for x1, y1 in sa for x2, y2 in sb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="screenplays")
    ap.add_argument("--out", default="results")
    ap.add_argument("--provider", default=None, help="gemini|openai|anthropic|mock (omit = rules only)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None, help="for OpenAI-compatible providers (Groq, OpenRouter, Ollama)")
    ap.add_argument("--top-k", type=int, default=8, help="candidates per episode sent to the judge")
    ap.add_argument("--final-n", type=int, default=30, help="promos in the final ranked list")
    ap.add_argument("--per-episode-cap", type=int, default=2)
    ap.add_argument("--min-score", type=float, default=40.0)
    ap.add_argument("--max-failures", type=int, default=2,
                    help="stop sending requests after this many failed ones (protects free-tier quota)")
    ap.add_argument("--no-global", action="store_true", help="skip the final show-level request")
    ap.add_argument("--keep-rejected", action="store_true",
                    help="keep clips the LLM judged 'reject' (default: drop them from the final list)")
    ap.add_argument("--no-entry-bonus", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="parallel LLM calls (keep 1 on free tiers)")
    ap.add_argument("--batch", type=int, default=1, help="episodes judged per LLM request (5 = ~5x fewer requests)")
    ap.add_argument("--min-interval", type=float, default=None, help="seconds between requests (free Gemini = 5 RPM -> 13)")
    ap.add_argument("--episodes", default=None, help="e.g. 1-10 to test on a subset")
    a = ap.parse_args()

    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    eps = load_show(a.input)
    if not eps:
        raise SystemExit(f"No screenplays found in {a.input}")
    by_no = {e["episode"]: e for e in eps}
    cast = C.cast_of(eps)
    sel = None
    if a.episodes:
        lo, hi = [int(x) for x in a.episodes.split("-")]
        sel = set(range(lo, hi + 1))

    llm = None
    if a.provider:
        from llm import LLM
        llm = LLM(a.provider, a.model, base_url=a.base_url, min_interval=a.min_interval)
    log = {"mode": llm.label if llm else "rules-only", "episodes": len(eps), "cast_detected": cast,
           "prompt_chars": 0, "errors": [], "missing_judgements": 0, "rejected_windows": []}
    if llm and llm.provider == "mock":
        log["WARNING"] = "MOCK provider: judgements are placeholders, for plumbing tests only"

    pool, jobs = [], []
    for i, ep in enumerate(eps):
        if sel and ep["episode"] not in sel:
            continue
        nxt = eps[i + 1] if i + 1 < len(eps) else None
        cont = bool(nxt and re.match(r"(picks up|directly|immediately|continues|follows)", nxt["continuity"], re.I))
        jobs.append({"i": i, "ep": ep, "cont": cont, "cands": C.generate(ep, cast, cont, top_k=a.top_k)})

    done = {}                                   # episode number -> judged candidates
    state = {"fails": 0, "stopped": False}

    def new_local():
        return {"prompt_chars": 0, "errors": [], "missing_judgements": 0, "rejected_windows": []}

    def merge(local):
        for k in ("prompt_chars", "missing_judgements"):
            log[k] += local[k]
        for k in ("errors", "rejected_windows"):
            log[k] += local[k]

    def report(ep, out, failed, skipped=False):
        for c in out:
            c["_llm_run"] = True
        status = "SKIPPED (stopped early to protect quota) - rules-only" if skipped else (
            "LLM FAILED - rules-only for this episode" if failed else "ok")
        print(f"  judged episode {ep['episode']:>2} ({len(out)} clips) {status}", flush=True)

    def after(local):
        """Count a failed request; print WHY right now; trip the breaker when too many fail."""
        if local["errors"]:
            state["fails"] += 1
            print(f"    REASON: {local['errors'][0][:400]}", flush=True)
            if state["fails"] >= a.max_failures and not state["stopped"]:
                state["stopped"] = True
                print(f"    -> {state['fails']} requests failed: no more requests will be sent this run.", flush=True)

    def run_single(job):
        local = new_local()
        ep = job["ep"]
        if state["stopped"]:
            report(ep, job["cands"], True, skipped=True)
            return {ep["episode"]: job["cands"]}, local
        out = J.judge_episode(llm, ep, eps[job["i"] - 1] if job["i"] else None, job["cands"], cast, job["cont"], local)
        report(ep, out, bool(local["errors"]))
        after(local)
        return {ep["episode"]: out}, local

    def run_batch(group):
        local = new_local()
        if state["stopped"]:
            for j in group:
                report(j["ep"], j["cands"], True, skipped=True)
            return {j["ep"]["episode"]: j["cands"] for j in group}, local
        items = [(j["ep"], eps[j["i"] - 1] if j["i"] else None, j["cands"], j["cont"]) for j in group]
        res = J.judge_batch(llm, items, cast, local)
        for j in group:
            failed = any(f"episode {j['ep']['episode']}" in e or f"episodes [{j['ep']['episode']}" in e
                         or f", {j['ep']['episode']}," in e or f", {j['ep']['episode']}]" in e for e in local["errors"])
            report(j["ep"], res[j["ep"]["episode"]], failed)
        after(local)
        return res, local

    if not llm:
        for j in jobs:
            J.assign_ids(j["ep"], j["cands"])
            done[j["ep"]["episode"]] = j["cands"]
    else:
        # episodes whose single-episode answer is already cached cost no request: finish those first
        todo = []
        for j in jobs:
            prompt = J.episode_prompt(j["ep"], eps[j["i"] - 1] if j["i"] else None, j["cands"])
            if llm.peek(J.SYSTEM, prompt) is not None:
                res, local = run_single(j)
                done.update(res)
                merge(local)
            else:
                todo.append(j)
        tasks = []
        if a.batch > 1:
            tasks = [(run_batch, todo[k:k + a.batch]) for k in range(0, len(todo), a.batch)]
        else:
            tasks = [(run_single, j) for j in todo]
        print(f"  {len(jobs) - len(todo)} episode(s) from cache, {len(todo)} to judge in {len(tasks)} request(s)", flush=True)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
            for res, local in ex.map(lambda t: t[0](t[1]), tasks):
                done.update(res)
                merge(local)
    for j in jobs:
        pool += done[j["ep"]["episode"]]
    for c in pool:
        finalize(c, not a.no_entry_bonus)

    # ---- show-level stage (LLM only): order for posting + cross-episode compositions
    order, post_first, comps = None, "", []
    if llm and not state["stopped"] and not a.no_global:
        top = sorted(pool, key=lambda c: -c["final"])[:30]
        order, post_first, comps = J.judge_global(llm, by_no, top, log)
        for comp in comps:
            comp.update({"hook_text": comp["llm"]["hook_quote"] or "", "cliff_text": comp["llm"]["cliff_text"] or "",
                         "hook": comp["llm"]["hook"], "cliff": comp["llm"]["cliff"],
                         "standalone": comp["llm"]["standalone"], "score": None, "_llm_run": True})
            finalize(comp, not a.no_entry_bonus)
        pool += comps

    # ---- rank: score, per-episode cap, threshold
    ranked, per_ep, dropped_rejected = [], {}, 0
    for c in sorted(pool, key=lambda c: -c["final"]):
        if c["final"] < a.min_score:
            continue
        if not a.keep_rejected and (c.get("llm") or {}).get("verdict") == "reject":
            dropped_rejected += 1                # the AI judge would not post this one
            continue
        if per_ep.get(c["episode"], 0) >= a.per_episode_cap:
            continue
        if any(near(c, k) for k in ranked if k["episode"] == c["episode"]):
            continue                     # same moment posted twice (overlapping or touching windows)
        ranked.append(c)
        per_ep[c["episode"]] = per_ep.get(c["episode"], 0) + 1
        if len(ranked) >= a.final_n:
            break
    if order:  # nudge: honour the model's posting order only among near-ties (within 3 points)
        pos = {i: n for n, i in enumerate(order)}
        ranked.sort(key=lambda c: (-round(c["final"] / 3), pos.get(c["id"], 999), -c["final"]))

    promos = []
    for r, c in enumerate(ranked, 1):
        segs = seg_dicts(c)
        promos.append({
            "rank": r, "promo_id": c["id"], "type": c["kind"], "source": c.get("source", "rules"),
            "segments": segs, "total_duration_s": round(sum(s["end"] - s["start"] for s in segs), 1),
            "hook": c["hook_final"], "cliffhanger": c["cliff_final"], "why_it_works": why_text(c),
            "scores": {"final": c["final"], "rules": c.get("score"),
                       "hook": (c.get("llm") or c).get("hook"), "cliffhanger": (c.get("llm") or c).get("cliff"),
                       "standalone": (c.get("llm") or c).get("standalone"),
                       "llm_verdict": (c.get("llm") or {}).get("verdict")},
            "ai_context_note": (c.get("llm") or {}).get("missing_context", ""),
            "rank_reason": rank_reason(c, r), "flags": flags(c)})

    # ---- write everything
    json.dump({"show_input": a.input, "mode": log["mode"], "post_first_note": post_first, "promos": promos},
              open(os.path.join(a.out, "promos.json"), "w"), indent=2)
    with open(os.path.join(a.out, "promos.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "promo_id", "type", "episode(s)", "segments (episode start-end)", "duration_s",
                    "hook", "cliffhanger", "why_it_works", "final_score", "ai_note_on_context", "flags", "rank_reason"])
        for p in promos:
            w.writerow([p["rank"], p["promo_id"], p["type"], "+".join(sorted({str(s["episode"]) for s in p["segments"]})),
                        " | ".join(f"ep{s['episode']} {s['start_ts']}-{s['end_ts']}" for s in p["segments"]),
                        p["total_duration_s"], p["hook"], p["cliffhanger"], p["why_it_works"],
                        p["scores"]["final"], p["ai_context_note"], "; ".join(p["flags"]), p["rank_reason"]])
    with open(os.path.join(a.out, "candidates_all.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "episode", "type", "segments", "duration_s", "final", "rules_score", "hook_r", "cliff_r",
                    "standalone_r", "llm_hook", "llm_cliff", "llm_standalone", "llm_verdict"])
        for c in sorted(pool, key=lambda c: (c["episode"], -c["final"])):
            l = c.get("llm") or {}
            w.writerow([c["id"], c["episode"], c["kind"],
                        " | ".join(f"{s['start_ts']}-{s['end_ts']}" for s in seg_dicts(c)),
                        c.get("duration"), c["final"], c.get("score"), c.get("hook"), c.get("cliff"),
                        c.get("standalone"), l.get("hook"), l.get("cliff"), l.get("standalone"), l.get("verdict")])
    with open(os.path.join(a.out, "promos.md"), "w", encoding="utf-8") as f:
        f.write(f"# Promos found ({log['mode']})\n\n")
        if post_first:
            f.write(f"**Post first - model's note:** {post_first}\n\n")
        for p in promos:
            f.write(f"## #{p['rank']}  {p['type']}  ·  {p['total_duration_s']} s  ·  score {p['scores']['final']}\n")
            for s in p["segments"]:
                f.write(f"- Episode {s['episode']}: **{s['start_ts']} -> {s['end_ts']}**\n")
            f.write(f"- **Hook:** {p['hook']}\n- **Cliffhanger:** {p['cliffhanger']}\n"
                    f"- **Why it works:** {p['why_it_works']}\n- **AI note on context:** {p['ai_context_note']}\n"
                    f"- **Ranking:** {p['rank_reason']}\n")
            if p["flags"]:
                f.write("- **Flags:** " + "; ".join(p["flags"]) + "\n")
            f.write("\n")
    n_calls = (llm.calls, llm.cache_hits) if llm else (0, 0)
    log.update({"dropped_llm_rejected": dropped_rejected, "candidates_total": len(pool), "promos_final": len(promos), "seconds": round(time.time() - t0, 1),
                "llm_calls": n_calls[0], "llm_cache_hits": n_calls[1],
                "approx_input_tokens": log["prompt_chars"] // 4,
                "settings": {k: v for k, v in vars(a).items()}})
    json.dump(log, open(os.path.join(a.out, "run_log.json"), "w"), indent=2)
    if state["stopped"]:
        print("\nSTOPPED EARLY: results below are PARTIAL (some episodes rules-only). Fix the reason above, then run "
              "the SAME command again: cached episodes are free.")
    if log["errors"]:
        print(f"\nWARNING: {len(log['errors'])} LLM call(s) failed (see run_log.json). Run the SAME command again: "
              f"successful calls are cached, only the failed ones are retried.")
    if dropped_rejected:
        print(f"dropped {dropped_rejected} clip(s) the LLM judged 'reject' (use --keep-rejected to keep them)")
    print(f"done: {len(promos)} promos from {len(pool)} candidates in {log['seconds']}s -> {a.out}/")


if __name__ == "__main__":
    main()
