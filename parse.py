"""parse.py - turn Kuku-style screenplay .txt files into structured episodes.

Format handled (observed in the input pack):
  EPISODE NUMBER / EPISODE LOGLINE / CONTINUITY NOTES header
  SCENE NN, Scene heading, Scene start, Scene end   (timestamps are episode-global)
  blocks:  **LABEL (tone):**  \n  Start: (mm:ss.ff) | End: (mm:ss.ff)  \n  text...
  LABEL is CAMERA, ACTION, SOUND FX, or a character name (= dialogue).

Input can be a folder (subfolders are searched), a .zip, or a single .txt file.
Timestamps may be mm:ss.ff or hh:mm:ss.ff. A file that cannot be read is skipped with a warning
instead of stopping the run.
"""
import os
import re
import sys
import tempfile
import zipfile

TSTR = r"(\d+(?::\d+){1,2}(?:\.\d+)?)"          # mm:ss(.ff) or hh:mm:ss(.ff)
BLOCK_HEAD = re.compile(r"^\*\*(.+?)\*\*:?\s*$")
TIME_LINE = re.compile(rf"^Start:\s*\({TSTR}\)\s*\|\s*End:\s*\({TSTR}\)")


def ts_sec(s):
    """'01:23.50' or '00:01:23.50' -> seconds"""
    secs = 0.0
    for part in s.split(":"):
        secs = secs * 60 + float(part)
    return secs


def fmt(t):
    """seconds -> mm:ss.s"""
    m, s = divmod(max(t, 0), 60)
    return f"{int(m):02d}:{s:04.1f}"


def parse_episode(path):
    raw = open(path, encoding="utf-8-sig", errors="replace").read()
    ep = {"file": os.path.basename(path), "scenes": [], "events": []}

    m = (re.search(r"EPISODE NUMBER:\s*(?:Episode\s*)?#?(\d+)", raw, re.I)
         or re.search(r"^\s*Episode\s*[:#-]?\s*(\d+)\s*$", raw, re.I | re.M))
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
            m = re.search(TSTR, ln)
            if m:
                key = "start" if "start" in ln.split(":")[0].lower() else "end"
                if key == "start":
                    ep["scenes"].append({"scene": scene_idx, "heading": heading,
                                         "start": ts_sec(m.group(1)), "end": None})
                elif ep["scenes"]:
                    ep["scenes"][-1]["end"] = ts_sec(m.group(1))
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
                cur["start"], cur["end"] = ts_sec(tl.group(1)), ts_sec(tl.group(2))
            elif ln.strip():
                cur["text"].append(ln.strip().strip('"'))
        i += 1
    close()
    ep["events"].sort(key=lambda e: (e["start"], e["end"]))
    ep["duration"] = max([e["end"] for e in ep["events"]] + [s["end"] or 0 for s in ep["scenes"]] + [0.0])
    return ep


def _collect_files(path):
    """folder (subfolders searched), .zip, or one .txt file -> sorted list of .txt paths"""
    if os.path.isfile(path) and path.lower().endswith(".zip"):
        tmp = tempfile.mkdtemp(prefix="promo_input_")
        with zipfile.ZipFile(path) as z:
            z.extractall(tmp)
        path = tmp
    if os.path.isfile(path):
        return [path]
    out = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d != "__MACOSX" and not d.startswith(".")]
        for f in files:
            if f.lower().endswith(".txt") and not f.startswith("._"):
                out.append(os.path.join(root, f))
    return sorted(out)


def load_show(path):
    files = _collect_files(path)
    eps, skipped, notes = [], [], []
    for f in files:
        name = os.path.basename(f)
        try:
            ep = parse_episode(f)
        except Exception as ex:                                  # one bad file must not stop the show
            skipped.append((name, f"unreadable ({ex})"))
            continue
        if not ep["events"]:
            skipped.append((name, "no timestamped blocks found (not this screenplay format?)"))
            continue
        if ep["episode"] is None:                                # header missing: try the file name
            m = re.search(r"(?:episode|ep)[-_ ]*0*(\d+)", name, re.I)
            if m:
                ep["episode"] = int(m.group(1))
                notes.append(f"{name}: episode number taken from the file name ({ep['episode']})")
        eps.append(ep)
    taken = set()
    nxt = max([e["episode"] for e in eps if e["episode"] is not None] + [0]) + 1
    for e in eps:                                                # fill gaps, keep numbers unique
        if e["episode"] is None or e["episode"] in taken:
            notes.append(f"{e['file']}: no usable unique episode number, numbered {nxt}")
            e["episode"], nxt = nxt, nxt + 1
        taken.add(e["episode"])
    if skipped:
        print(f"WARNING: skipped {len(skipped)} of {len(files)} file(s):", file=sys.stderr)
        for name, why in skipped[:10]:
            print(f"  - {name}: {why}", file=sys.stderr)
    for n in notes[:10]:
        print(f"NOTE: {n}", file=sys.stderr)
    eps.sort(key=lambda e: e["episode"])
    return eps


if __name__ == "__main__":
    eps = load_show(sys.argv[1] if len(sys.argv) > 1 else "screenplays")
    print(f"{len(eps)} episodes parsed")
    if not eps:
        raise SystemExit("Nothing parsed: check the folder and the screenplay format.")
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
