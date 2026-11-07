"""llm.py - one function, `complete(system, prompt) -> str`, behind a swappable provider.

Swap the model with env vars or CLI flags - nothing else in the code changes:
  PROMO_PROVIDER = gemini | openai | anthropic | mock
  PROMO_MODEL    = e.g. gemini-3.8-flash | gpt-4o-mini | claude-sonnet-4-5 | llama-3.3-70b-versatile
  PROMO_API_KEY  = your key (or GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY)
  PROMO_BASE_URL = only for 'openai' provider pointing at Groq / OpenRouter / Ollama / etc.
Standard library only (urllib) so it runs anywhere, including Colab, with no installs.
"""
import hashlib
import json
import threading
import os
import re
import time
import urllib.error
import urllib.request

CACHE_DIR = os.environ.get("PROMO_CACHE", ".llm_cache")

DEFAULTS = {
    "gemini": ("gemini-3.8-flash", "GEMINI_API_KEY"),
    "openai": ("gpt-4o-mini", "OPENAI_API_KEY"),
    "anthropic": ("claude-sonnet-4-5", "ANTHROPIC_API_KEY"),
}


class LLM:
    def __init__(self, provider=None, model=None, api_key=None, base_url=None, use_cache=True, min_interval=None):
        self.provider = (provider or os.environ.get("PROMO_PROVIDER") or "").lower()
        if self.provider not in ("gemini", "openai", "anthropic", "mock"):
            raise ValueError("provider must be gemini | openai | anthropic | mock")
        dm, keyenv = DEFAULTS.get(self.provider, ("mock", ""))
        self.model = model or os.environ.get("PROMO_MODEL") or dm
        self.key = api_key or os.environ.get("PROMO_API_KEY") or os.environ.get(keyenv, "")
        self.base_url = (base_url or os.environ.get("PROMO_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.use_cache = use_cache
        self.timeout = float(os.environ.get("PROMO_TIMEOUT", 300))
        self.min_interval = float(min_interval if min_interval is not None
                                  else os.environ.get("PROMO_MIN_INTERVAL", 0))   # seconds between calls (free-tier RPM)
        self._lock, self._last = threading.Lock(), 0.0   # slow models can take >2 min
        self.calls = self.cache_hits = 0
        if self.provider != "mock" and not self.key:
            raise RuntimeError(f"No API key for provider '{self.provider}'. Set PROMO_API_KEY or {keyenv}.")
        os.makedirs(CACHE_DIR, exist_ok=True)

    @property
    def label(self):
        return f"{self.provider}:{self.model}"

    # ---------------------------------------------------------------- public
    def _path(self, system, prompt):
        h = hashlib.sha256(f"{self.label}\n{system}\n{prompt}".encode()).hexdigest()[:24]
        return os.path.join(CACHE_DIR, h + ".txt")

    def peek(self, system, prompt):
        """Cached reply if we already have one (costs no API request), else None."""
        path = self._path(system, prompt)
        return open(path, encoding="utf-8").read() if self.use_cache and os.path.exists(path) else None

    def _throttle(self):
        if self.min_interval <= 0:
            return
        with self._lock:
            wait = self._last + self.min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()

    def complete(self, system, prompt, max_tokens=4000):
        path = self._path(system, prompt)
        if self.use_cache and os.path.exists(path):
            self.cache_hits += 1
            return open(path, encoding="utf-8").read()
        out = self._with_retries(system, prompt, max_tokens)
        self.calls += 1
        if self.use_cache:
            open(path, "w", encoding="utf-8").write(out)
        return out

    def complete_json(self, system, prompt, max_tokens=4000):
        """Ask for JSON; parse tolerantly; one repair retry. Raises ValueError if still bad."""
        txt = self.complete(system, prompt, max_tokens)
        try:
            return extract_json(txt)
        except ValueError:
            try:                                   # never keep an unusable reply in the cache
                os.remove(self._path(system, prompt))
            except OSError:
                pass
            txt = self.complete(system, prompt + "\n\nYour previous reply was not valid JSON. "
                                "Reply with ONLY the JSON, no prose, no code fences.", max_tokens)
            try:
                return extract_json(txt)
            except ValueError:
                try:
                    os.remove(self._path(system, prompt + "\n\nYour previous reply was not valid JSON. "
                                         "Reply with ONLY the JSON, no prose, no code fences."))
                except OSError:
                    pass
                raise ValueError(f"model reply was not valid JSON (length {len(txt)} chars; likely cut off or "
                                 f"wrapped in prose). Start: {txt[:120]!r} ... End: {txt[-120:]!r}")

    # ---------------------------------------------------------------- providers
    def _with_retries(self, system, prompt, max_tokens, tries=5):
        wait = 4
        for k in range(tries):
            try:
                self._throttle()
                return self._call(system, prompt, max_tokens)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "ignore")[:1500]
                if e.code == 429 and re.search(r"PerDay|per day|daily", body, re.I):
                    # retrying would only burn more of today's quota
                    raise RuntimeError(f"{self.label} DAILY QUOTA exhausted (HTTP 429). Wait for the reset "
                                       f"or switch model/provider. {body[:200]}")
                limit = 3 if e.code == 429 else tries
                if e.code in (429, 500, 502, 503) and k < limit - 1:     # free tiers rate-limit
                    time.sleep(30 if e.code == 429 else wait)
                    wait *= 2
                    continue
                raise RuntimeError(f"{self.label} HTTP {e.code}: {body}")
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:   # incl. read timeouts
                if k < tries - 1:
                    time.sleep(wait)
                    wait *= 2
                    continue
                raise RuntimeError(f"{self.label} network/timeout error after {tries} tries: {e}")

    def _post(self, url, headers, payload):
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def _call(self, system, prompt, max_tokens):
        if self.provider == "mock":
            return mock_reply(prompt)
        if self.provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.key}"
            d = self._post(url, {"Content-Type": "application/json"}, {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens,
                                     "responseMimeType": "application/json"}})
            return d["candidates"][0]["content"]["parts"][0]["text"]
        if self.provider == "anthropic":
            d = self._post("https://api.anthropic.com/v1/messages",
                           {"Content-Type": "application/json", "x-api-key": self.key,
                            "anthropic-version": "2023-06-01"},
                           {"model": self.model, "max_tokens": max_tokens, "temperature": 0.2,
                            "system": system, "messages": [{"role": "user", "content": prompt}]})
            return "".join(b.get("text", "") for b in d["content"])
        # openai-compatible (OpenAI, Groq, OpenRouter, Together, Ollama ...)
        payload = {"model": self.model, "temperature": 0.2, "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        hdr = {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"}
        try:
            d = self._post(self.base_url + "/chat/completions", hdr,
                           dict(payload, response_format={"type": "json_object"}))
        except urllib.error.HTTPError as e:
            if e.code != 400:
                raise
            d = self._post(self.base_url + "/chat/completions", hdr, payload)   # provider has no JSON mode
        return d["choices"][0]["message"]["content"]


def extract_json(txt):
    txt = txt.strip()
    txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        a, b = txt.find(opener), txt.rfind(closer)
        if a != -1 and b > a:
            try:
                return json.loads(txt[a:b + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no valid JSON in model reply")


def mock_reply(prompt):
    """PLUMBING TEST ONLY. Returns well-formed but meaningless judgements so the pipeline's parsing,
    validation and blending can be exercised without an API key. Never used for deliverable outputs."""
    ids = re.findall(r'"id":\s*"([^"]+)"', prompt)
    if "TASK: EPISODES" in prompt:
        eps = {}
        for i in ids:
            m = re.match(r"E(\d+)-", i)
            if m:                                   # skip the schema example's placeholder "..."
                eps.setdefault(int(m.group(1)), []).append(i)
        j = lambda i: {"id": i, "hook_strength": 5, "hook_quote": "MOCK", "cliffhanger_strength": 5,
                       "cliffhanger_text": "MOCK", "standalone": 5, "missing_context": "MOCK",
                       "verdict": "keep", "why": "MOCK judgement - not a real model output"}
        return json.dumps({"episodes": [{"episode": n, "judgements": [j(i) for i in v], "extra_windows": []}
                                        for n, v in eps.items()]})
    if "TASK: GLOBAL" in prompt:
        return json.dumps({"order": ids[:10], "post_first_reason": "MOCK", "compositions": []})
    return json.dumps({"judgements": [
        {"id": i, "hook_strength": 5, "hook_quote": "MOCK", "cliffhanger_strength": 5,
         "cliffhanger_text": "MOCK", "standalone": 5, "missing_context": "MOCK",
         "verdict": "keep", "why": "MOCK judgement - not a real model output"} for i in ids], "extra_windows": []})
