"""candidates.py - CODE decides what is a *valid* promo clip; simple rules give a first score.

Valid = 20-60 s, starts on a dialogue/action beat, ends on a beat boundary, never cuts a line
in half, cuts snapped to the show's own camera shots where possible.
The heuristic score is a cheap pre-filter. The LLM judge (judge.py) decides what is *good*.
"""
import re
from parse import fmt

MIN_LEN, MAX_LEN = 20.0, 60.0

# ---- small, readable lexicons (the "simple rules" half of the system) ----
SHOCK = r"""betray|divorce|pregnan|dead|death|die\b|dying|kill|murder|poison|lie\b|lied|liar|secret|truth|
traitor|bastard|slave|whore|guilty|innocent|scandal|ruin|destroy|revenge|blood|forgive|never|how dare|
get out|leave|marry|wife|husband|child|baby|duke|king|queen|prison|arrest|sword|knife|dagger|
who are you|what did you|you\s+will|you\s+are\s+(?:nothing|mine|dead)|mine\b|dare\b|beg\b|gone\b|missing"""
SHOCK_RE = re.compile(SHOCK.replace("\n", ""), re.I)
HOOK_ACT = re.compile(r"slap|strike|struck|shove|grab|seiz|collaps|drops?\b|kneel|burst|gasp|stab|"
                      r"throws?|smash|shatter|storm|tears? (?:off|at)|rips?|pulls? (?:out|a)|"
                      r"points?|draws?\b|unsheath|faint|scream|flinch|recoil|reveal|lunges?|"
                      r"slam|pins?\b|drag", re.I)
SOUND_HOT = re.compile(r"gasp|crash|thunder|slam|sting|shatter|scream|silence|ominous|swell|"
                       r"cliffhanger|dramatic|sharp|heartbeat|impact|cut", re.I)
HOT_TONE = re.compile(r"shock|furious|cold|menac|venom|scream|gasp|trembl|desperate|stunned|"
                      r"horrif|seething|icy|threat|broken|enrag|sneer|vicious|dangerous", re.I)
CLIFF_ACT = re.compile(r"freez|gasp|appears?|steps? (?:in|out|forward)|door (?:bursts|opens|swings)|"
                       r"stares?|realiz|eyes widen|stunned|turns? to (?:see|find)|horror|"
                       r"pale|drops?\b|collaps|shadow|figure|unseen|watch|overheard|witness|"
                       r"walked in|discover|notice", re.I)
RESOLVED = re.compile(r"thank you|smiles? (?:warmly|softly|gently)|embrac|agree|forgiv|peace|relief|"
                      r"laughs?|at last|safe\b|home\b|goodnight|good night|sleep|content", re.I)
CONTINUER = re.compile(r"^(and|but|so|then|because|which|that|as i|like i|still|also|anyway)\b", re.I)
PRONOUN_OPEN = re.compile(r"^(he|she|they|him|her|it|this|these|those)\b", re.I)


def words(s):
    return len(re.findall(r"[A-Za-z']+", s))


# ------------------------------------------------------------------ timeline helpers
def beats(ep):
    return [e for e in ep["events"] if e["type"] != "camera"]


def shots(ep):
    return [e for e in ep["events"] if e["type"] == "camera"]


def straddles(ep, t):
    """True if a dialogue/action line is mid-way at time t (cutting here would chop a line)."""
    return any(e["start"] + 0.05 < t < e["end"] - 0.05 and e["type"] in ("dialogue",)
               for e in ep["events"])


def snap_start(ep, t):
    for s in shots(ep):
        if s["start"] <= t + 1e-6 < s["end"] and 0 < t - s["start"] <= 1.5 and not straddles(ep, s["start"]):
            return s["start"]
    return t


def snap_end(ep, t):
    for s in shots(ep):
        if s["start"] < t - 1e-6 <= s["end"] + 1e-6 and 0 < s["end"] - t <= 1.0 and not straddles(ep, s["end"]):
            return s["end"]
    return min(t, ep["duration"])


def events_in(ep, a, b, kinds=("dialogue", "action", "sound")):
    out = []
    for e in ep["events"]:
        if e["type"] not in kinds:
            continue
        ov = min(e["end"], b) - max(e["start"], a)
        if ov > 0.5 * max(e["end"] - e["start"], 0.01):
            out.append(e)
    return out


# ------------------------------------------------------------------ scoring pieces (each 0-10)
def hook_score(evs):
    """What happens in the first ~3 s of the clip."""
    if not evs:
        return 0.0, ""
    pts, best = 0.0, None
    for e in evs:
        t = e["text"]
        p = 0.0
        if e["type"] == "dialogue":
            p += 1.0
            p += min(3.0, 1.2 * len(SHOCK_RE.findall(t)))
            if "!" in t or "?" in t:
                p += 1.2
            if words(t) <= 8:
                p += 1.0
            if HOT_TONE.search(e["tone"]):
                p += 1.5
        elif e["type"] == "action":
            if HOOK_ACT.search(t):
                p += 3.5
            p += min(1.5, 0.75 * len(SHOCK_RE.findall(t)))
        elif e["type"] == "sound" and SOUND_HOT.search(t):
            p += 1.5
        if best is None or p > best[0]:
            best = (p, e)
        pts += p
    score = min(10.0, 0.6 * best[0] + 0.15 * pts)
    return round(score, 2), best[1]["text"] if best else ""


def cliff_score(evs, next_ep_continues, at_episode_end, ends_mid_scene):
    """What the last ~4 s of the clip leave hanging."""
    if not evs:
        return 0.0, ""
    last = evs[-1]
    tail_txt = " ".join(e["text"] for e in evs)
    pts = 0.0
    lastd = [e for e in evs if e["type"] == "dialogue"]
    if lastd:
        t = lastd[-1]["text"]
        if t.rstrip().endswith("?"):
            pts += 2.5
        if t.rstrip().endswith(("...", "—", "-")):
            pts += 1.5
        if "!" in t:
            pts += 1.0
        pts += min(2.0, 1.0 * len(SHOCK_RE.findall(t)))
        if HOT_TONE.search(lastd[-1]["tone"]):
            pts += 1.0
    if any(e["type"] == "sound" and re.search(r"cliffhanger|sting|ominous|swell|silence|gasp", e["text"], re.I)
           for e in evs):
        pts += 2.0
    if CLIFF_ACT.search(tail_txt):
        pts += 2.0
    if last["type"] in ("action", "sound") and CLIFF_ACT.search(last["text"]):
        pts += 1.0
    if at_episode_end:
        pts += 0.5 + (1.0 if next_ep_continues else 0.0)   # weak prior: episode ends are often cliffhangers
    if ends_mid_scene:
        pts += 1.0                                        # cut while the conflict is still live
    if RESOLVED.search(tail_txt):
        pts -= 3.0
    return round(max(0.0, min(10.0, pts)), 2), last["text"]


def standalone_score(evs, cast_names, start_scene_continuous, first_text):
    """Would a stranger follow this? Proxy rules only; the LLM judge checks the real thing."""
    s = 10.0
    speakers = {e["speaker"] for e in evs if e["type"] == "dialogue"}
    if len(speakers) > 4:
        s -= 1.0 * (len(speakers) - 4)
    present = " ".join(sp.lower() for sp in speakers) + " " + " ".join(
        e["text"].lower() for e in evs if e["type"] == "action")
    spoken = " ".join(e["text"] for e in evs if e["type"] == "dialogue")
    unseen = [n for n in cast_names if re.search(rf"\b{re.escape(n)}\b", spoken, re.I)
              and n.lower() not in present]
    s -= min(4.0, 1.5 * len(unseen))
    if first_text:
        if CONTINUER.search(first_text):
            s -= 1.5
        elif PRONOUN_OPEN.search(first_text):
            s -= 1.0
    if start_scene_continuous:
        s -= 1.0
    return round(max(0.0, s), 2), unseen


def density_score(evs, dur):
    w = sum(words(e["text"]) for e in evs if e["type"] == "dialogue")
    wps = w / max(dur, 1)
    if wps < 0.6:
        return round(wps / 0.6 * 6, 2)
    if wps <= 3.5:
        return 10.0
    return round(max(4.0, 10 - (wps - 3.5) * 3), 2)


def duration_penalty(dur):
    if 25 <= dur <= 50:
        return 0.0
    return 3.0 if dur < 25 else min(5.0, 0.5 * (dur - 50))


# ------------------------------------------------------------------ main entry points
ROLE_WORDS = {"Judge", "Blond", "Dowager", "Maid", "Butler", "Lord", "Lady", "Noble", "Priest", "High"}


def cast_of(eps):
    """Names = tokens from speaker labels that appear Capitalised in the show's text and never
    in lowercase (so 'Butler', 'Father', 'Crown' are rejected; proper first names are kept)."""
    text = " ".join(e["text"] for ep in eps for e in ep["events"] if e["type"] != "camera")
    text += " " + " ".join(ep["logline"] + " " + ep["continuity"] for ep in eps)
    cand = set()
    for ep in eps:
        for e in ep["events"]:
            if e["type"] == "dialogue":
                cand.update(t.title() for t in re.findall(r"[A-Z]{4,}", e["speaker"]))
    names = []
    for tok in sorted(cand):
        cap = len(re.findall(rf"\b{tok}\b", text))
        low = len(re.findall(rf"\b{tok.lower()}\b", text))
        if cap >= 3 and low == 0 and tok not in ROLE_WORDS:
            names.append(tok)
    return names


def score_segments(ep, segs, cast, next_continues):
    """segs = [(start, end), ...] inside ONE episode. Returns a dict of sub-scores."""
    first_a, _ = segs[0]
    _, last_b = segs[-1]
    first_evs = events_in(ep, first_a, first_a + 3.0)
    last_evs = events_in(ep, last_b - 4.0, last_b)
    all_evs = []
    for a, b in segs:
        all_evs += events_in(ep, a, b)
    dur = sum(b - a for a, b in segs)
    h, hook_txt = hook_score(first_evs)
    at_end = last_b >= ep["duration"] - 1.0
    scene_end = max((s["end"] or 0) for s in ep["scenes"] if s["start"] <= last_b - 0.01) \
        if ep["scenes"] else ep["duration"]
    c, cliff_txt = cliff_score(last_evs, next_continues, at_end, last_b < scene_end - 1.0)
    first_dlg = next((e["text"] for e in all_evs if e["type"] == "dialogue"), "")
    sc = next((s for s in ep["scenes"] if s["start"] <= first_a < (s["end"] or 1e9)), None)
    cont = bool(sc and "CONTINUOUS" in sc["heading"].upper() and first_a - sc["start"] < 5)
    st, unseen = standalone_score(all_evs, cast, cont, first_dlg)
    dn = density_score(all_evs, dur)
    acts = sum(1 for e in all_evs if e["type"] in ("action", "sound"))
    static = 6.0 if acts == 0 else (3.0 if acts / max(dur, 1) * 30 < 1.5 else 0.0)   # all-talk = exposition risk
    total = 100 * (0.35 * h + 0.35 * c + 0.20 * st + 0.10 * dn) / 10 - duration_penalty(dur) - static
    return {"score": round(total, 1), "hook": h, "cliff": c, "standalone": st, "density": dn,
            "duration": round(dur, 1), "hook_text": hook_txt, "cliff_text": cliff_txt,
            "unseen_refs": unseen, "at_episode_end": at_end, "static_penalty": static}


def generate(ep, cast, next_continues, top_k=8):
    b = beats(ep)
    starts = sorted({e["start"] for e in b if e["type"] in ("dialogue", "action")})
    ends = sorted({e["end"] for e in b} | {ep["duration"]})
    pool = []
    for s0 in starts:
        if straddles(ep, s0):
            continue
        s = snap_start(ep, s0)
        for e0 in ends:
            if e0 - s0 < MIN_LEN - 1.5 or e0 - s0 > MAX_LEN:
                continue
            e = snap_end(ep, e0)
            if not (MIN_LEN <= e - s <= MAX_LEN) or straddles(ep, e):
                continue
            sc = score_segments(ep, [(s, e)], cast, next_continues)
            sc.update({"episode": ep["episode"], "segments": [(round(s, 2), round(e, 2))], "kind": "continuous"})
            pool.append(sc)
    # within-episode montage: strongest hook beat (6-14 s) + a cliffhanger-ending run (14-45 s)
    pool += montages(ep, cast, next_continues, pool)
    return nms(pool, top_k)


def montages(ep, cast, next_continues, pool):
    """Join an earlier punchy beat to a cliffhanger ending when no single window has both."""
    out = []
    if not pool:
        return out
    best_cont = max(pool, key=lambda c: c["score"])
    # candidate hook beats
    hooks = []
    for s0 in sorted({e["start"] for e in beats(ep) if e["type"] in ("dialogue", "action")}):
        for e0 in sorted({e["end"] for e in beats(ep)}):
            if 6 <= e0 - s0 <= 14 and not straddles(ep, s0) and not straddles(ep, e0):
                hooks.append((s0, e0))
    hooks.sort(key=lambda w: -hook_score(events_in(ep, w[0], w[0] + 3.0))[0])
    hooks = hooks[:6]
    tails = sorted([c for c in pool if c["kind"] == "continuous"], key=lambda c: -c["cliff"])[:12]
    for t in tails:
        ts, te = t["segments"][0]
        for hs, he in hooks:
            if he > ts - 1.0:
                continue
            # trim the tail so total length fits
            for new_ts in sorted({e["start"] for e in beats(ep) if ts <= e["start"] < te - 8
                                  and e["type"] in ("dialogue", "action")}):
                total = (he - hs) + (te - new_ts)
                if MIN_LEN <= total <= MAX_LEN and not straddles(ep, new_ts):
                    sc = score_segments(ep, [(hs, he), (new_ts, te)], cast, next_continues)
                    if sc["score"] > best_cont["score"] + 4:
                        sc.update({"episode": ep["episode"], "kind": "montage",
                                   "segments": [(round(hs, 2), round(he, 2)), (round(new_ts, 2), round(te, 2))]})
                        out.append(sc)
                    break
    return out


def overlap(a, b):
    tot = 0.0
    for s1, e1 in a["segments"]:
        for s2, e2 in b["segments"]:
            tot += max(0.0, min(e1, e2) - max(s1, s2))
    return tot / max(1e-6, min(a["duration"], b["duration"]))


def nms(pool, k):
    """Non-max suppression: keep the best, drop near-duplicates of it, repeat."""
    pool = sorted(pool, key=lambda c: -c["score"])
    keep = []
    for c in pool:
        if all(overlap(c, kept) < 0.35 for kept in keep):
            keep.append(c)
        if len(keep) >= k:
            break
    return keep


def ts_pair(seg):
    return fmt(seg[0]), fmt(seg[1])
