"""judge.py - the AI half. The model JUDGES quality; code VALIDATES everything it returns.

Guards against model mistakes:
  * every timestamp the model proposes is snapped to a real cut point and re-checked for 20-60 s
  * every quoted hook / cliffhanger must literally appear in the clip text, else it is replaced
  * scores are clamped to 0-10; ids the model forgot fall back to the rule-based score
"""
import re
from candidates import (MIN_LEN, MAX_LEN, beats, events_in, snap_start, snap_end, straddles,
                        score_segments)
from parse import fmt

SYSTEM = """You are a senior promo editor for vertical micro-drama (1-2 minute episodes) on TikTok / \
Instagram / YouTube Shorts, like ReelShort and DramaBox. You choose clips that make a scrolling stranger \
tap 'watch the show'. You judge only from the screenplay text given. Never invent events, names or \
quotes - quote the screenplay exactly. Reply with JSON only."""

RUBRIC = """RUBRIC (score each 0-10, be harsh - most clips are 3-6):
- hook_strength: in the FIRST 3 SECONDS of the clip is there something that stops a scroll (slap, reveal, \
shocking or cruel line, sudden danger)? 0 = quiet scene-setting.
- cliffhanger_strength: does the LAST few seconds cut at an UNRESOLVED moment that forces 'what happens next'? \
A resolved or merely sad ending = low. A reveal, threat, arrival or unanswered question = high.
- standalone: would someone who has NEVER seen the show understand who is fighting over what? \
Deduct for references to unseen characters/events, and for needing earlier episodes.
Verdict: 'keep' only if it would be a promo you'd post; else 'reject'."""


def transcript(ep, lo=None, hi=None):
    """Compact timestamped text, camera directions dropped (they are noise for judging)."""
    rows = []
    for e in ep["events"]:
        if e["type"] == "camera":
            continue
        if lo is not None and (e["end"] < lo or e["start"] > hi):
            continue
        if e["type"] == "dialogue":
            tone = f" ({e['tone']})" if e["tone"] else ""
            rows.append(f"[{fmt(e['start'])}-{fmt(e['end'])}] {e['speaker']}{tone}: \"{e['text']}\"")
        elif e["type"] == "action":
            rows.append(f"[{fmt(e['start'])}-{fmt(e['end'])}] ACTION: {e['text']}")
        else:
            rows.append(f"[{fmt(e['start'])}-{fmt(e['end'])}] SFX: {e['text']}")
    return "\n".join(rows)


def clip_text(ep, segs):
    return " ".join(e["text"].lower() for a, b in segs for e in events_in(ep, a, b))


def verify_quote(q, text):
    q = re.sub(r"\s+", " ", (q or "").lower()).strip(" \"'.…")
    return len(q) >= 6 and q in re.sub(r"\s+", " ", text)


def clamp(x, lo=0.0, hi=10.0):
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ stage A: one call per episode
def build_episode_prompt(ep, prev_ep, cands):
    ctx = f"EPISODE {ep['episode']} logline: {ep['logline']}\nContinuity: {ep['continuity']}"
    if prev_ep:
        ctx += f"\n(Previous episode {prev_ep['episode']} logline: {prev_ep['logline']})"
    lst = "\n".join(
        '{"id": "%s", "segments": %s}' % (c["id"], [[fmt(a), fmt(b)] for a, b in c["segments"]]) for c in cands)
    return f"""TASK: EPISODE
{ctx}

SCREENPLAY (timestamps are mm:ss.s from episode start):
{transcript(ep)}

CANDIDATE CLIPS found by rules (segments are played in order and joined):
{lst}

{RUBRIC}

For EVERY candidate return a judgement. Also, if you see a clearly better 20-60 s promo that is NOT \
covered by a candidate (max 2), return it in extra_windows. A window must START at the start time of a \
listed line and END at the end time of a listed line, and be 20-60 seconds long in total. hook_quote and \
cliffhanger_text must be copied EXACTLY from the screenplay lines inside the clip.

Reply with JSON:
{{"judgements": [{{"id": "...", "hook_strength": 0-10, "hook_quote": "exact words from first 3 s", \
"cliffhanger_strength": 0-10, "cliffhanger_text": "exact last line/action", "standalone": 0-10, \
"missing_context": "what a new viewer would not know, or 'none'", "verdict": "keep|reject", \
"why": "one line, max 25 words"}}],
"extra_windows": [{{"start": "mm:ss.s", "end": "mm:ss.s", "hook_strength": 0-10, "hook_quote": "...", \
"cliffhanger_strength": 0-10, "cliffhanger_text": "...", "standalone": 0-10, "missing_context": "...", \
"verdict": "keep|reject", "why": "..."}}]}}"""


def parse_ts(s):
    m = re.match(r"^\s*(\d+):(\d+(?:\.\d+)?)\s*$", str(s))
    return int(m.group(1)) * 60 + float(m.group(2)) if m else None


def snap_window(ep, s, e, tol=1.5):
    """Snap a model-proposed window to real cut points; None if impossible/invalid."""
    bs = beats(ep)
    starts = [x["start"] for x in bs if x["type"] in ("dialogue", "action")]
    ends = [x["end"] for x in bs]
    if s is None or e is None or not starts:
        return None
    s0 = min(starts, key=lambda t: abs(t - s))
    e0 = min(ends, key=lambda t: abs(t - e))
    if abs(s0 - s) > tol or abs(e0 - e) > tol or straddles(ep, s0):
        return None
    s1, e1 = snap_start(ep, s0), snap_end(ep, e0)
    if straddles(ep, e1) or not (MIN_LEN <= e1 - s1 <= MAX_LEN):
        return None
    return round(s1, 2), round(e1, 2)


def apply_judgement(c, ep, j):
    """Merge a validated model judgement into a candidate dict."""
    h, cl, st = clamp(j.get("hook_strength")), clamp(j.get("cliffhanger_strength")), clamp(j.get("standalone"))
    if None in (h, cl, st):
        c["llm"] = None
        return
    text = clip_text(ep, c["segments"])
    hq, ct = j.get("hook_quote", ""), j.get("cliffhanger_text", "")
    c["llm"] = {
        "hook": h, "cliff": cl, "standalone": st,
        "hook_quote": hq if verify_quote(hq, text) else None,
        "cliff_text": ct if verify_quote(ct, text) else None,
        "missing_context": str(j.get("missing_context", ""))[:200],
        "verdict": "keep" if str(j.get("verdict", "")).lower().startswith("keep") else "reject",
        "why": str(j.get("why", ""))[:220],
    }
    c["llm"]["quotes_verified"] = bool(c["llm"]["hook_quote"] and c["llm"]["cliff_text"])


def assign_ids(ep, cands):
    for k, c in enumerate(cands):
        c["id"] = f"E{ep['episode']:02d}-{k + 1}"


def episode_prompt(ep, prev_ep, cands):
    assign_ids(ep, cands)
    return build_episode_prompt(ep, prev_ep, cands)


def apply_output(out, ep, cands, cast, next_continues, log):
    """Merge one episode's parsed model output into its candidates (validating everything)."""
    if not isinstance(out, dict):
        out = {}
    by_id = {j.get("id"): j for j in (out.get("judgements") or []) if isinstance(j, dict)}
    for c in cands:
        if c["id"] in by_id:
            apply_judgement(c, ep, by_id[c["id"]])
        else:
            c["llm"] = None
            log["missing_judgements"] += 1
    extras = []
    for n, w in enumerate(out.get("extra_windows", [])[:2]):
        if not isinstance(w, dict):
            continue
        win = snap_window(ep, parse_ts(w.get("start")), parse_ts(w.get("end")))
        if not win:
            log["rejected_windows"].append({"episode": ep["episode"], "window": [w.get("start"), w.get("end")],
                                            "reason": "could not snap to valid cut points / 20-60 s"})
            continue
        sc = score_segments(ep, [win], cast, next_continues)
        sc.update({"episode": ep["episode"], "segments": [win], "kind": "continuous",
                   "id": f"E{ep['episode']:02d}-X{n + 1}", "source": "llm-extra"})
        apply_judgement(sc, ep, w)
        extras.append(sc)
    return cands + extras


def judge_episode(llm, ep, prev_ep, cands, cast, next_continues, log):
    prompt = episode_prompt(ep, prev_ep, cands)
    log["prompt_chars"] += len(prompt) + len(SYSTEM)
    try:
        out = llm.complete_json(SYSTEM, prompt)
    except Exception as ex:                                  # never crash the whole show on one episode
        log["errors"].append(f"episode {ep['episode']}: {ex}")
        return cands
    return apply_output(out, ep, cands, cast, next_continues, log)


def build_batch_prompt(items):
    """items: list of (ep, prev_ep, cands, next_continues). One call judges several episodes."""
    blocks = []
    for ep, prev_ep, cands, _ in items:
        ctx = f"EPISODE {ep['episode']} logline: {ep['logline']}\nContinuity: {ep['continuity']}"
        if prev_ep:
            ctx += f"\n(Previous episode {prev_ep['episode']} logline: {prev_ep['logline']})"
        lst = "\n".join('{"id": "%s", "segments": %s}' % (c["id"], [[fmt(a), fmt(b)] for a, b in c["segments"]])
                         for c in cands)
        blocks.append(f"=== EPISODE {ep['episode']} ===\n{ctx}\n\nSCREENPLAY (mm:ss.s from episode start):\n"
                      f"{transcript(ep)}\n\nCANDIDATE CLIPS for episode {ep['episode']}:\n{lst}")
    return f"""TASK: EPISODES
Judge each episode below separately. Clips never span episodes here.

{chr(10).join(blocks)}

{RUBRIC}

For EVERY candidate of EVERY episode return a judgement. Per episode you may also return up to 2 extra_windows \
(clearly better 20-60 s promos not covered by a candidate; START at a listed line's start time, END at a listed \
line's end time). hook_quote and cliffhanger_text must be copied EXACTLY from that episode's screenplay lines.

Reply with JSON: {{"episodes": [{{"episode": N, "judgements": [{{"id": "...", "hook_strength": 0-10, \
"hook_quote": "...", "cliffhanger_strength": 0-10, "cliffhanger_text": "...", "standalone": 0-10, \
"missing_context": "...", "verdict": "keep|reject", "why": "max 25 words"}}], \
"extra_windows": [{{"start": "mm:ss.s", "end": "mm:ss.s", "hook_strength": 0-10, "hook_quote": "...", \
"cliffhanger_strength": 0-10, "cliffhanger_text": "...", "standalone": 0-10, "missing_context": "...", \
"verdict": "keep|reject", "why": "..."}}]}}]}}"""


def judge_batch(llm, items, cast, log):
    """Returns {episode_no: cands}. On total failure every episode falls back to rules-only."""
    prompt = build_batch_prompt(items)
    log["prompt_chars"] += len(prompt) + len(SYSTEM)
    res = {ep["episode"]: cands for ep, _, cands, _ in items}
    try:
        out = llm.complete_json(SYSTEM, prompt, max_tokens=min(16000, 2500 * len(items) + 2000))
    except Exception as ex:
        log["errors"].append(f"batch of episodes {[i[0]['episode'] for i in items]}: {ex}")
        return res
    by_ep = {}
    if isinstance(out, list):
        out = {"episodes": out}
    for e in out.get("episodes", []) if isinstance(out, dict) else []:
        try:
            by_ep[int(e.get("episode"))] = e
        except (TypeError, ValueError):
            pass
    for ep, _, cands, cont in items:
        if ep["episode"] in by_ep:
            res[ep["episode"]] = apply_output(by_ep[ep["episode"]], ep, cands, cast, cont, log)
        else:
            log["errors"].append(f"episode {ep['episode']}: missing from batch reply")
    return res


# ------------------------------------------------------------------ stage B: show-level ranking + cross-episode
def build_global_prompt(eps_by_no, top):
    rows = []
    for c in top:
        rows.append('{"id": "%s", "episode": %d, "segments": %s, "hook": %s, "cliffhanger": %s, "why": %s}' % (
            c["id"], c["episode"], [[fmt(a), fmt(b)] for a, b in c["segments"]],
            repr(c.get("hook_final", ""))[:140].replace("'", '"'),
            repr(c.get("cliff_final", ""))[:140].replace("'", '"'),
            repr((c.get("llm") or {}).get("why", ""))[:160].replace("'", '"')))
    lines = "\n".join(f"Ep {n}: {e['logline']}" for n, e in sorted(eps_by_no.items()) if n <= 50)
    return f"""TASK: GLOBAL
You are choosing which promos to post first for this show. Episode loglines (story so far):
{lines}

Shortlisted single-episode promos:
{chr(10).join(rows)}

1) "order": ids best-first for POSTING. First posts should work as an entry point for brand-new viewers \
(clear premise, little spoiler). Give "post_first_reason" (max 40 words).
2) "compositions": up to 3 NEW promos that JOIN moments (2-3 segments, possibly from different episodes) when \
the join clearly beats any single clip - e.g. a shocking line from one episode + the payoff and cliffhanger \
from another. Each segment must start/end exactly at times in that episode's screenplay; total 20-60 s. \
Only propose if you are confident it makes sense to a stranger; otherwise return [].
Reply with JSON: {{"order": ["id", ...], "post_first_reason": "...", "compositions": [{{"segments": \
[{{"episode": n, "start": "mm:ss.s", "end": "mm:ss.s"}}], "hook": "...", "cliffhanger": "...", \
"why": "...", "hook_strength": 0-10, "cliffhanger_strength": 0-10, "standalone": 0-10}}]}}"""


def judge_global(llm, eps_by_no, top, log):
    prompt = build_global_prompt(eps_by_no, top)
    log["prompt_chars"] += len(prompt) + len(SYSTEM)
    try:
        out = llm.complete_json(SYSTEM, prompt, max_tokens=3000)
    except Exception as ex:
        log["errors"].append(f"global stage: {ex}")
        return None, "", []
    if isinstance(out, list):                   # model sometimes returns a bare list
        if out and isinstance(out[0], dict):
            out = out[0]
        else:
            out = {"order": [x for x in out if isinstance(x, str)], "compositions": []}
    if not isinstance(out, dict):
        log["errors"].append(f"global stage: unexpected reply type {type(out).__name__}; kept score order")
        return None, "", []
    valid_ids = {c["id"] for c in top}
    order = [i for i in (out.get("order") or []) if isinstance(i, str) and i in valid_ids]
    comps = []
    raw_comps = out.get("compositions") or []
    for n, comp in enumerate(raw_comps[:3] if isinstance(raw_comps, list) else []):
        try:
            segs, eps_used, ok = [], set(), True
            for sg in comp["segments"][:3]:
                ep = eps_by_no.get(int(sg["episode"]))
                win = snap_window(ep, parse_ts(sg["start"]), parse_ts(sg["end"]), tol=1.5) if ep else None
                if not win:
                    # allow short segments (<20 s) when part of a join; re-snap loosely
                    ok = False
                    break
                segs.append((ep["episode"], win))
                eps_used.add(ep["episode"])
            total = sum(b - a for _, (a, b) in segs)
            if not ok or not (MIN_LEN <= total <= MAX_LEN):
                log["rejected_windows"].append({"composition": n + 1, "reason": "segment did not snap or total not 20-60 s"})
                continue
            comps.append({"id": f"C{n + 1}", "kind": "cross-episode" if len(eps_used) > 1 else "montage",
                          "episode": segs[0][0], "episodes": sorted(eps_used),
                          "segment_list": segs, "duration": round(total, 1),
                          "source": "llm-composed", "needs_human_check": True,
                          "llm": {"hook": clamp(comp.get("hook_strength")) or 0,
                                  "cliff": clamp(comp.get("cliffhanger_strength")) or 0,
                                  "standalone": clamp(comp.get("standalone")) or 0,
                                  "hook_quote": comp.get("hook"), "cliff_text": comp.get("cliffhanger"),
                                  "missing_context": "unchecked - composed across moments", "verdict": "keep",
                                  "why": str(comp.get("why", ""))[:220], "quotes_verified": False}})
        except (KeyError, TypeError, ValueError, AttributeError):
            log["rejected_windows"].append({"composition": n + 1, "reason": "malformed"})
    return order, str(out.get("post_first_reason", ""))[:400], comps
