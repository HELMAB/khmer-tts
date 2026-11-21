"""Turn raw Khmer text (an article, a script) into the one-short-phrase-per-line text `read` expects.

VoxPocket reads long paragraphs badly and cannot read digits or symbols, so:
    - numbers, dates, times, % and $ become Khmer words (២៩ → ម្ភៃប្រាំបួន, 21,46 → ម្ភៃមួយក្បៀសសែសិបប្រាំមួយ)
    - ៗ is written out (ថ្មីៗ → ថ្មីថ្មី)
    - URLs, source lines, bullets, quote marks, emoji and Latin-only brackets are dropped
    - sentences are cut into short lines, preferably before linking words (និង, ដែល, ដើម្បី…)
English words are kept as they are (never translated or transliterated) and listed, so you can listen to them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

KH_DIGITS = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")
ONES = ["សូន្យ", "មួយ", "ពីរ", "បី", "បួន", "ប្រាំ", "ប្រាំមួយ", "ប្រាំពីរ", "ប្រាំបី", "ប្រាំបួន"]
TENS = ["", "ដប់", "ម្ភៃ", "សាមសិប", "សែសិប", "ហាសិប", "ហុកសិប", "ចិតសិប", "ប៉ែតសិប", "កៅសិប"]
MONTHS = ["មករា", "កុម្ភៈ", "មីនា", "មេសា", "ឧសភា", "មិថុនា", "កក្កដា", "សីហា", "កញ្ញា", "តុលា", "វិច្ឆិកា", "ធ្នូ"]
SENTENCE_END = "។!?៕"
# linking words a line may start with; breaking before them keeps each phrase whole
CONNECTORS = ("បន្ទាប់ពី", "ចាប់ពី", "កាលពី", "ដើម្បី", "ប៉ុន្តែ", "ពីព្រោះ", "ព្រោះ", "ក្រោម", "ខណៈ", "ដែល", "និង",
              "ហើយ", "ដោយ", "ក្នុង", "របស់", "នៅ", "ឬ", "តែ")

D = r"[0-9០-៩]"
NUM = rf"{D}+(?:[.,]{D}+)*"


# ---------------------------------------------------------------- numbers

def below_1000(n: int) -> str:
    h, r = divmod(n, 100)
    t, o = divmod(r, 10)
    return (ONES[h] + "រយ" if h else "") + TENS[t] + (ONES[o] if o else "")


def number_words(n: int) -> str:
    """2012 → ពីរពាន់ដប់ពីរ, 625000 → ប្រាំមួយរយម្ភៃប្រាំពាន់ (grouped by thousands, as news readers do)."""
    if n == 0:
        return ONES[0]
    out = ""
    for value, name in ((10 ** 9, "ប៊ីលាន"), (10 ** 6, "លាន"), (1000, "ពាន់")):
        q, n = divmod(n, value)
        if q:
            out += number_words(q) + name
    return out + (below_1000(n) if n else "")


def digits_one_by_one(s: str) -> str:
    return "".join(ONES[int(c)] for c in s)


def read_number(raw: str) -> str:
    s = raw.translate(KH_DIGITS)
    groups = re.split(r"[.,]", s)
    if len(groups) == 1:
        # phone numbers, codes, leading zeros: digit by digit
        return digits_one_by_one(s) if len(s) >= 8 or (len(s) > 1 and s[0] == "0") else number_words(int(s))
    grouped = 1 <= len(groups[0]) <= 3 and all(len(g) == 3 for g in groups[1:])
    if grouped and (len(groups) > 2 or "," in s):
        return number_words(int("".join(groups)))  # 625,000 / 1.250.000: thousands separators
    if len(groups) == 2:
        whole, frac = groups
        frac_words = digits_one_by_one(frac) if frac.startswith("0") else number_words(int(frac))
        return number_words(int(whole)) + "ក្បៀស" + frac_words  # 21,46 / 3.5: decimals
    return " ".join(read_number(g) for g in groups)  # 18.11.2026 and the like


def expand_numbers(text: str) -> str:
    def date(m):
        d, mo, y = (int(x.translate(KH_DIGITS)) for x in m.groups())
        if not 1 <= mo <= 12:
            return m.group(0)
        return f"ថ្ងៃទី{number_words(d)} ខែ{MONTHS[mo - 1]} ឆ្នាំ{number_words(y)}"

    def time(m):
        h, mi = (int(x.translate(KH_DIGITS)) for x in m.groups())
        return number_words(h) + (f" និង{number_words(mi)}នាទី" if mi else "")

    text = re.sub(rf"({D}{{1,2}})/({D}{{1,2}})/({D}{{4}})", date, text)
    text = re.sub(rf"(?<![0-9០-៩])({D}{{1,2}}):({D}{{2}})(?![0-9០-៩])", time, text)
    text = re.sub(rf"\$\s?({NUM})", lambda m: read_number(m.group(1)) + "ដុល្លារ", text)
    text = re.sub(rf"({NUM})\s?(?:%|ភាគរយ)", lambda m: read_number(m.group(1)) + "ភាគរយ", text)
    text = re.sub(rf"({NUM})\s?-\s?({NUM})", lambda m: f"{read_number(m.group(1))} ដល់ {read_number(m.group(2))}", text)
    return re.sub(NUM, lambda m: read_number(m.group(0)), text)


# ---------------------------------------------------------------- words

@lru_cache(maxsize=1)
def segmenter():
    """khmer-nltk's word_tokenize (Khmer has no spaces between words), or None if it isn't installed."""
    import logging
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import khmernltk.utils.log_utils  # noqa: F401  (creates the logger, so it can be quietened)
            logging.getLogger("khmer-nltk").setLevel(logging.WARNING)
            from khmernltk import word_tokenize
    except ImportError:
        return None
    return word_tokenize


def khmer_words(run: str) -> list[str]:
    seg = segmenter()
    if seg and run:
        words = [w for w in seg(run, return_tokens=True) if w]
        if "".join(words) == run:
            return words
    return [run]


def expand_repeat(text: str) -> str:
    """ថ្មីៗ → ថ្មីថ្មី: ៗ repeats the word before it."""
    def repl(m):
        words = khmer_words(m.group(1))
        return m.group(1) + words[-1]
    return re.sub(r"([ក-៓]+)\s?ៗ", repl, text)


# ---------------------------------------------------------------- cleaning

URL = re.compile(r"(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|org|net|gov|kh|info)(?:/\S*)?", re.I)
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")


def clean_line(line: str) -> str:
    line = re.sub(r"^\s*[\u1780-\u17ff]{1,15}\s*៖\s*", "", line)  # dateline: "ភ្នំពេញ៖ …"
    line = URL.sub("", line)
    line = EMOJI.sub("", line)
    line = re.sub(r"\(([^()ក-៿]*)\)", "", line)  # (Oxfam), (UNESCO): Latin-only brackets
    line = re.sub(r"[()\[\]«»“”\"'‘’]", "", line)
    line = re.sub(r"^\s*(?:[-*•·▪●○►✓✔]+|\d+[.)]|[០-៩]+[.)])\s*", lambda m: _list_marker(m.group(0)), line)
    line = line.replace("៖", " ").replace("—", " ").replace("–", " ")
    line = re.sub(r"\s*\.{3}|…", " ", line)
    line = re.sub(rf"\s+([{SENTENCE_END}])", r"\1", line)
    return re.sub(r"\s+", " ", line).strip(" ,")


def _list_marker(marker: str) -> str:
    """'១. ' → 'ចំណុចទីមួយ ' (a spoken list number); bullets just go."""
    m = re.match(rf"\s*({D}+)[.)]", marker)
    return f"ចំណុចទី{number_words(int(m.group(1).translate(KH_DIGITS)))} " if m else ""


def is_meta(line: str) -> bool:
    """Source/credit lines that should not be read out."""
    return bool(re.match(r"^\s*(?:ប្រភព|Source|Photo|រូបភាព|អត្ថបទ)\s*[៖:]", line, re.I)) or \
        (URL.search(line) is not None and len(URL.sub("", line).strip()) < 25)


# ---------------------------------------------------------------- line splitting

def split_sentences(paragraph: str) -> list[str]:
    parts = re.split(rf"(?<=[{SENTENCE_END}])\s*", paragraph)
    return [p.strip() for p in parts if p.strip()]


def pieces(sentence: str) -> list[tuple[str, bool]]:
    """(word, space before?) for every word; unspaced Khmer runs are segmented into words."""
    out = []
    for i, tok in enumerate(sentence.split(" ")):
        for j, w in enumerate(khmer_words(tok) if re.search(r"[ក-៿]", tok) else [tok]):
            out.append((w, i > 0 and j == 0))
    return out


def join(ws: list[tuple[str, bool]]) -> str:
    return "".join((" " if sp and k else "") + w for k, (w, sp) in enumerate(ws))


NUMBER_WORD = re.compile(r"^(?:សូន្យ|មួយ|ពីរ|បី|បួន|ប្រាំ|ដប់|ម្ភៃ|សាម|សែ|ហា|ហុក|ចិត|ប៉ែត|កៅ|សិប|រយ|ពាន់|ម៉ឺន|សែន|លាន|ប៊ីលាន|ក្បៀស)+$")
# words that need the next word on the same line (ថ្ងៃទី | ដប់ប្រាំពីរ, លោក | ស សុខា)
LEADS_IN = ("ទី", "ថ្ងៃទី", "ខែ", "ឆ្នាំ", "លោក", "លោកស្រី", "អ្នក", "ឯកឧត្តម", "សម្តេច", "ប្រមាណ", "ចំនួន", "ជាង")


def breakable(ws: list[tuple[str, bool]], k: int) -> bool:
    prev, (word, _) = ws[k - 1][0], ws[k]
    if not re.search(r"[\u1780-\u17d3A-Za-z0-9]", word):  # never start a line with punctuation
        return False
    if NUMBER_WORD.match(prev) and NUMBER_WORD.match(word):  # inside a spelled-out number
        return False
    return not prev.endswith(LEADS_IN)


def split_long(sentence: str, max_len: int, min_len: int) -> list[str]:
    """Cut a sentence into lines of at most ~max_len characters, preferring breaks before linking words,
    then at spaces, then between words."""
    if len(sentence) <= max_len:
        return [sentence]
    ws, lines = pieces(sentence), []
    while ws:
        if len(join(ws)) <= max_len:
            lines.append(ws)
            break
        best, best_score, fallback = None, -1.0, None
        for k in range(1, len(ws)):
            head = len(join(ws[:k]))
            if head > max_len:
                break
            if not breakable(ws, k):
                continue
            fallback = k
            if head < min_len:
                continue
            word, space = ws[k]
            score = head / max_len + (1.0 if word.startswith(CONNECTORS) else 0) + (0.6 if space else 0)
            if ws[k - 1][0].endswith(","):
                score += 1.2
            if score >= best_score:
                best, best_score = k, score
        k = best or fallback
        if k is None:  # no allowed break before max_len: take the first allowed one after it
            k = next((i for i in range(1, len(ws)) if breakable(ws, i)), len(ws))
        lines.append(ws[:k])
        ws = [(ws[k][0], False)] + ws[k + 1:] if k < len(ws) else []
    if len(lines) > 1 and len(join(lines[-1])) < min_len and \
            len(join(lines[-2])) + len(join(lines[-1])) <= max_len * 1.25:
        tail = lines.pop()
        lines[-1] = lines[-1] + [(tail[0][0], True)] + tail[1:]
    return [join(ln).strip(" ,") for ln in lines]


# ---------------------------------------------------------------- whole text

@dataclass
class Prepared:
    lines: list[str]
    english: list[str] = field(default_factory=list)  # Latin words left in; check how they sound
    dropped: list[str] = field(default_factory=list)  # meta lines not read out
    segmented: bool = True


def prepare_text(raw: str, max_len: int = 60, min_len: int = 14) -> Prepared:
    out, dropped = [], []
    for para in raw.splitlines():
        if not para.strip():
            continue
        if is_meta(para):
            dropped.append(para.strip())
            continue
        text = expand_repeat(expand_numbers(clean_line(para)))
        for sent in split_sentences(text):
            out += [ln for ln in split_long(sent, max_len, min_len) if re.search(r"[ក-៿A-Za-z]", ln)]
    english = sorted({w for ln in out for w in re.findall(r"[A-Za-z][A-Za-z'.-]*", ln)}, key=str.lower)
    return Prepared(out, english, dropped, segmenter() is not None)
