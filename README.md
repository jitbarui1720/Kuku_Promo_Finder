# Promo Finder

Finds 20-60 s promo clips in a micro-drama screenplay.

**Code decides what is valid** (timestamps, 20-60 s, cut points, never chop a spoken line).
**AI decides what is good** (hook, cliffhanger, standalone). The model is swappable with one flag.

Input: the show's screenplay `.txt` files (not included here: they are Kuku's content).
Requirements: Python 3.9+, standard library only. Nothing to install.

## Run

Put the screenplay `.txt` files in a folder called `screenplays`.

### 1. Rules only (no key, about 6 s for 50 episodes)

```
python run.py --input screenplays --out results_rules
```

### 2. With the AI judge (the command used for the submitted outputs)

Windows PowerShell:

```
$env:GEMINI_API_KEY="your key"
$env:PROMO_TIMEOUT="900"
python run.py --input screenplays --out results_llm --provider gemini --model gemini-3.6-flash --final-n 30 --batch 5 --min-interval 13
```

Mac / Linux:

```
export GEMINI_API_KEY=your_key
python run.py --input screenplays --out results_llm --provider gemini --model gemini-3.6-flash --final-n 30 --batch 5 --min-interval 13
```

### 3. Check the hard constraints on any output

```
python validate.py --input screenplays --results results_llm/promos.json
```

### A new show

Same command with a new input folder and a new output folder:

```
python run.py --input NEW_FOLDER --out out_new --provider gemini --model gemini-3.6-flash --batch 5 --min-interval 13
python validate.py --input NEW_FOLDER --results out_new/promos.json
```

## Switching the model (one flag, no code change)

```
--provider gemini | openai | anthropic
--model <model name>
```

Any OpenAI-compatible host (Groq, OpenRouter, Ollama):

```
--provider openai --base-url https://api.groq.com/openai/v1 --model <model name>
```

Keys: `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` (or `PROMO_API_KEY`).
A model name must be one your key can use today; a wrong name gives HTTP 404 and the run stops early.

## Free-tier notes (what happened on my account)

- Gemini's free tier allowed 5 requests/minute and 20 requests/day per model, so `--batch 5`
  (5 episodes per request) and `--min-interval 13` (seconds between requests) are used.
  A 50-episode show took 11 requests.
- Every reply is cached in `.llm_cache/`, so a stopped or rate-limited run resumes without repeating requests.
- The run stops by itself after 2 failed requests (`--max-failures`) and prints the reason.
  A stopped run is PARTIAL (some episodes rules-only): fix the reason and run the same command again.

## Pipeline

1. `parse.py`: screenplay to events with timestamps (dialogue / action / sound / camera).
2. `candidates.py`: every valid window with a rule-based score (shock words, trailing question,
   action and sound cues, all-talk penalty), cuts snapped to camera shots, near-duplicates removed,
   up to 8 per episode.
3. `judge.py`: the model scores each clip 0-10 for hook, cliffhanger and standalone, and may propose
   2 extra windows per episode. Everything it returns is re-validated: timestamps re-snapped and
   re-checked for 20-60 s, and a quoted hook or cliffhanger must appear verbatim in the clip.
4. `run.py`: 70% AI + 30% rules score, small early-episode bonus, max 2 per episode, touching windows
   merged, clips the AI rejected dropped, outputs written.

## Outputs of a run

- `promos.md`, `promos.csv`, `promos.json`: ranked promos with episode, every segment's start and end,
  hook, cliffhanger, why it works, rank reason, AI note on context, flags.
- `candidates_all.csv`: every judged clip with rule scores and AI scores (including rejected).
- `run_log.json`: settings, requests made, approximate tokens, errors, rejected windows.

## Known limits

- Works from screenplay text, not video: it cannot judge acting, music or visual shock.
- The "standalone" rule is only a proxy; the AI judges it better but is not ground truth.
- The model grades its own proposed windows leniently; cross-episode compositions were not produced in the real run.
- One show tested. No human-labelled ground truth.
