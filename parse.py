"""parse.py - turn Kuku-style screenplay .txt files into structured episodes.

Format handled (observed in the input pack):
  EPISODE NUMBER / EPISODE LOGLINE / CONTINUITY NOTES header
  SCENE NN, Scene heading, Scene start, Scene end   (timestamps are episode-global)
  blocks:  **LABEL (tone):**  \n  Start: (mm:ss.ff) | End: (mm:ss.ff)  \n  text...
  LABEL is CAMERA, ACTION, SOUND FX, or a character name (= dialogue).
"""
import glob
import os
import re

TS = r"(\d+):(\d+(?:\.\d+)?)"
BLOCK_HEAD = re.compile(r"^\*\*(.+?)\*\*:?\s*$")
TIME_LINE = re.compile(rf"^Start:\s*\({TS}\)\s*\|\s*End:\s*\({TS}\)")


def to_sec(m, i=1):
    return int(m.group(i)) * 60 + float(m.group(i + 1))


def fmt(t):
    """seconds -> mm:ss.s"""
    m, s = divmod(max(t, 0), 60)
    return f"{int(m):02d}:{s:04.1f}"


def parse_episode(path):
    raw = open(path, encoding="utf-8").read()
    ep = {"file": os.path.basename(path), "scenes": [], "events": []}

    m = re.search(r"EPISODE NUMBER:\s*Episode\s*(\d+)", raw)
    ep["episode"] = int(m.group(1)) if m else None
    m = re.search(r"EPISODE LOGLINE:\s*(.+)", raw)
    ep["logline"] = m.group(1).strip() if m else ""
    m = re.search(r"CONTINUITY NOTES:\s*(.+)", raw)
    ep["continuity"] = m.group(1).strip() if m else ""

    body = raw.split("--- SCRIPT START ---")[-1].split("--- SCRIPT END ---")[0]
    lines = body.splitlines()

    scene_idx, heading = 0, ""
    i = 0
    cur = None  # current open block
    def close():
        nonlocal cur
        if cur and cur.get("start") is not None:
            cur["text"] = " ".join(cur["text"]).strip()
            ep["events"].append(cur)
        cur = None

    while i < len(lines):
        ln = lines[i].rstrip()
        sm = re.match(r"^SCENE\s+(\d+)", ln)
        if sm:
            close()
            scene_idx = int(sm.group(1))
            heading = ""
            i += 1
            continue
        if ln.startswith("Scene heading:"):
            heading = ln.split(":", 1)[1].strip()
            i += 1
            continue
        if ln.startswith("Scene start:") or ln.startswith("Scene end:"):
            m = re.search(TS, ln)
            if m:
                key = "start" if "start" in ln.split(":")[0].lower() else "end"
                if key == "start":
                    ep["scenes"].append({"scene": scene_idx, "heading": heading,
                                         "start": to_sec(m), "end": None})
                else:
                    ep["scenes"][-1]["end"] = to_sec(m)
            i += 1
            continue
        hm = BLOCK_HEAD.match(ln)
        if hm:
            close()
            label = hm.group(1)
            tone = ""
            tm = re.match(r"^(.*?)\s*\((.*)\)\s*:?$", label)
            if tm:
                label, tone = tm.group(1).strip(), tm.group(2).strip()
            label = label.rstrip(":").strip()
            kind = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sound"}.get(label, "dialogue")
            cur = {"type": kind, "speaker": label if kind == "dialogue" else None,
                   "tone": tone, "start": None, "end": None, "text": [],
                   "scene": scene_idx, "heading": heading}
            i += 1
            continue
        if cur is not None:
            tl = TIME_LINE.match(ln)
            if tl and cur["start"] is None:
                cur["start"], cur["end"] = to_sec(tl, 1), to_sec(tl, 3)
            elif ln.strip():
                cur["text"].append(ln.strip().strip('"'))
        i += 1
    close()
    ep["events"].sort(key=lambda e: (e["start"], e["end"]))
    ep["duration"] = max([e["end"] for e in ep["events"]] + [s["end"] or 0 for s in ep["scenes"]])
    return ep


def load_show(folder):
    files = sorted(glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True))
    eps = [parse_episode(f) for f in files]
    eps = [e for e in eps if e["episode"] is not None and e["events"]]
    eps.sort(key=lambda e: e["episode"])
    return eps


if __name__ == "__main__":
    import sys
    eps = load_show(sys.argv[1] if len(sys.argv) > 1 else "screenplays")
    print(f"{len(eps)} episodes parsed")
    tot = {"dialogue": 0, "action": 0, "camera": 0, "sound": 0}
    bad = 0
    for e in eps:
        for ev in e["events"]:
            tot[ev["type"]] += 1
            if ev["end"] < ev["start"]:
                bad += 1
    print("events:", tot, "| negative-duration events:", bad)
    print("durations (s): min %.0f max %.0f avg %.0f" % (
        min(e["duration"] for e in eps), max(e["duration"] for e in eps),
        sum(e["duration"] for e in eps) / len(eps)))
