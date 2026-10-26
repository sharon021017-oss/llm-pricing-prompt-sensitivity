"""Model calls, rate limiting and the spending cap."""
import json, os, random, re, threading, time
from datetime import date
import config


class BudgetExceeded(Exception):
    pass


class DailyLimitReached(Exception):
    pass


def load_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if not key and os.path.exists(config.API_KEY_FILE):
        key = open(config.API_KEY_FILE, encoding="utf-8").read().strip()
    if not key:
        raise SystemExit(f"No API key found. Put it in {config.API_KEY_FILE} or set GEMINI_API_KEY.")
    return key


class Ledger:
    """Tracks spending and requests across all runs (results/ledger.json)."""
    def __init__(self):
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        self.path = os.path.join(config.RESULTS_DIR, "ledger.json")
        self.lock = threading.Lock()
        self.data = {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "requests": 0, "per_day": {}}
        if os.path.exists(self.path):
            self.data.update(json.load(open(self.path)))
        self.recent = []  # timestamps for the per-minute limit

    def cost_of(self, tin, tout):
        return tin * config.PRICE_INPUT_PER_M / 1e6 + tout * config.PRICE_OUTPUT_PER_M / 1e6

    def before_call(self, est_cost=0.004):
        with self.lock:
            if self.data["cost_usd"] + est_cost > config.BUDGET_USD:
                raise BudgetExceeded(f"Spending cap reached (${self.data['cost_usd']:.2f} of ${config.BUDGET_USD:.2f}).")
            today = str(date.today())
            if config.MAX_REQUESTS_PER_DAY and self.data["per_day"].get(today, 0) >= config.MAX_REQUESTS_PER_DAY:
                raise DailyLimitReached("Daily request cap reached.")
        while True:  # per-minute limit
            with self.lock:
                now = time.time()
                self.recent = [t for t in self.recent if now - t < 60]
                if len(self.recent) < config.MAX_REQUESTS_PER_MINUTE:
                    self.recent.append(now)
                    return
            time.sleep(1)

    def record(self, tin, tout):
        with self.lock:
            today = str(date.today())
            self.data["cost_usd"] += self.cost_of(tin, tout)
            self.data["input_tokens"] += tin
            self.data["output_tokens"] += tout
            self.data["requests"] += 1
            self.data["per_day"][today] = self.data["per_day"].get(today, 0) + 1
            json.dump(self.data, open(self.path, "w"), indent=1)


class GeminiBackend:
    def __init__(self):
        from google import genai
        from google.genai import types
        # 120-second timeout so a hung connection is retried instead of freezing the run
        self.client = genai.Client(api_key=load_api_key(),
                                   http_options=types.HttpOptions(timeout=120_000))

    def __call__(self, prompt):
        """Returns (text, input_tokens, output_tokens_including_thinking)."""
        try:  # Interactions API (current Google docs)
            r = self.client.interactions.create(
                model=config.MODEL, input=prompt,
                generation_config={"temperature": config.TEMPERATURE,
                                   "thinking_level": config.THINKING_LEVEL})
            u = r.usage
            tin = getattr(u, "total_input_tokens", 0) or 0
            tout = (getattr(u, "total_output_tokens", 0) or 0) + (getattr(u, "total_thought_tokens", 0) or 0)
            text = getattr(r, "output_text", None)
            if not text:
                text = "[NO TEXT] status=" + str(getattr(r, "status", None)) + " | " + repr(r)[:3000]
            return text, tin, tout
        except AttributeError:  # older SDK without interactions -> generate_content
            from google.genai import types
            r = self.client.models.generate_content(
                model=config.MODEL, contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=config.TEMPERATURE,
                    thinking_config=types.ThinkingConfig(thinking_level=config.THINKING_LEVEL)))
            u = r.usage_metadata
            tin = u.prompt_token_count or 0
            tout = (u.candidates_token_count or 0) + (getattr(u, "thoughts_token_count", 0) or 0)
            return r.text, tin, tout


class MockBackend:
    """Fake model for testing the harness without spending anything."""
    def __call__(self, prompt):
        cost = float(re.search(r"produce each unit is \$(\d+(?:\.\d+)?)", prompt).group(1))
        price = round(cost * random.uniform(1.4, 2.0), 2)
        text = ("My observations and thoughts:\nTesting.\n\nNew content for PLANS.txt:\nKeep testing.\n\n"
                "New content for INSIGHTS.txt:\nNone yet.\n\nMy chosen price:\n" + str(price))
        return text, len(prompt) // 4, 150


HEAD = {
    "obs": r"my\s+observations\s+and\s+thoughts",
    "plans": r"new\s+content\s+for\s+plans(?:\.txt)?",
    "ins": r"new\s+content\s+for\s+insights(?:\.txt)?",
    "price": r"my\s+chosen\s+price",
}


def _clean(text):
    # drop markdown decoration the model sometimes adds: ** __ ` # and leading list markers
    t = re.sub(r"[*_`#]+", "", text)
    return t


def parse(text):
    """Split the reply into its four sections and extract the price. Returns None if malformed.
    Tolerant of markdown, missing colons, extra spaces and different capitalisation."""
    if not text:
        return None
    t = _clean(text)
    pos = {}
    for key, pat in HEAD.items():
        ms = list(re.finditer(pat + r"\s*:?", t, re.I))
        if not ms:
            if key in ("obs",):
                continue          # observations section is optional
            return None
        pos[key] = ms[-1] if key == "price" else ms[0]   # price: use the LAST occurrence
    order = sorted(pos.items(), key=lambda kv: kv[1].start())
    sections = {}
    for i, (key, m) in enumerate(order):
        end = order[i + 1][1].start() if i + 1 < len(order) else len(t)
        sections[key] = t[m.end():end].strip()
    nums = re.findall(r"\d+(?:\.\d+)?", sections["price"].replace(",", ""))
    if not nums:
        return None
    return {"observations": sections.get("obs", ""), "plans": sections["plans"],
            "insights": sections["ins"], "price": float(nums[0])}


def log_failure(text, info=""):
    try:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        with open(os.path.join(config.RESULTS_DIR, "failed_replies.txt"), "a", encoding="utf-8") as f:
            f.write("=" * 80 + f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} {info}\n")
            f.write(repr(text)[:6000] + "\n")
    except Exception:
        pass


def ask(backend, ledger, prompt):
    """One agent decision, retried up to MAX_RETRIES times (FGS Appendix B)."""
    last = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        ledger.before_call()
        try:
            text, tin, tout = backend(prompt)
        except Exception as e:  # network / rate-limit errors: back off and retry
            last = str(e)
            print(f"    [attempt {attempt}] API error: {last[:300]}", flush=True)
            if "API_KEY_INVALID" in last or "PERMISSION_DENIED" in last or "NOT_FOUND" in last:
                raise SystemExit("Stopping: this error will not fix itself by retrying (check the key / model name).")
            if "429" in last or "RESOURCE_EXHAUSTED" in last.upper():
                time.sleep(30 * attempt)
            else:
                time.sleep(5 * attempt)
            continue
        ledger.record(tin, tout)
        parsed = parse(text)
        if parsed:
            parsed.update(raw=text, attempts=attempt, input_tokens=tin, output_tokens=tout)
            return parsed
        last = "unparseable reply"
        log_failure(text, f"attempt {attempt}, {tin} in / {tout} out tokens")
        print(f"    [attempt {attempt}] reply could not be parsed (saved to failed_replies.txt); retrying", flush=True)
    raise RuntimeError(f"Failed after {config.MAX_RETRIES} attempts: {last}")
