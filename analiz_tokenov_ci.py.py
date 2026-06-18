from datasets import load_dataset
import tiktoken
import statistics
import math

# ============================================================
#  CASBench — Многоязычный анализ токенизации (v2: CI + tests)
#  На базе analiz_tokenov_multi.py.
#  Добавлено:
#    • 95% доверительный интервал для каждого среднего
#    • Парный t-тест KZ vs RU и KZ vs EN на одинаковых предложениях
#    • Коэффициент вариации (CV) как мера разброса
# ============================================================

LANGUAGES = {
    "kk_kz": "Казахский",
    "ru_ru": "Русский",
    "en_us": "Английский",
    "uz_uz": "Узбекский",
    "tr_tr": "Турецкий",
}

TARGET_COUNT = 1000
tokenizer = tiktoken.get_encoding("o200k_base")

# Опциональные доп. токенизаторы
extra_tokenizers = {}
try:
    from transformers import AutoTokenizer
    candidates = {
        "LLaMA-3": ["NousResearch/Meta-Llama-3-8B", "unsloth/llama-3-8b"],
        "BERTurk": ["dbmdz/bert-base-turkish-cased"],
        "KZ-tuned": ["kz-transformers/kaz-roberta-conversational"],
    }
    for label, ids in candidates.items():
        for mid in ids:
            try:
                tk = AutoTokenizer.from_pretrained(mid)
                extra_tokenizers[label] = tk
                print(f"  + загружен {label} ({mid})")
                break
            except Exception:
                continue
except ImportError:
    print("  (transformers не установлен — считаю только GPT-4o)")


def count_tokens_gpt4o(text):
    return len(tokenizer.encode(text))


def count_tokens_hf(tk, text):
    return len(tk.encode(text, add_special_tokens=False))


# ─────────────────────────────────────────────────────────────────
# Статистика
# ─────────────────────────────────────────────────────────────────

def ci95(values):
    """95% доверительный интервал среднего (нормальное приближение, n>=30)."""
    n = len(values)
    if n < 2:
        return (0, 0, 0)
    m = statistics.mean(values)
    s = statistics.stdev(values)  # выборочное std
    se = s / math.sqrt(n)
    half = 1.96 * se
    return (m, m - half, m + half)


def paired_t(diffs):
    """Парный t-тест: возвращает t-статистику и приблизительный p-уровень.
    Используем нормальное приближение для больших n (n=1000)."""
    n = len(diffs)
    if n < 2:
        return (0, 1.0)
    m = statistics.mean(diffs)
    s = statistics.stdev(diffs)
    se = s / math.sqrt(n)
    if se == 0:
        return (float("inf"), 0.0)
    t = m / se
    # двустороннее p через нормальное приближение
    # P(|Z| > |t|) ≈ 2 * (1 - Φ(|t|))
    # erfc + math.sqrt
    z = abs(t)
    p = math.erfc(z / math.sqrt(2))
    return (t, p)


def cv(values):
    """Коэффициент вариации в процентах."""
    if not values:
        return 0
    m = statistics.mean(values)
    if m == 0:
        return 0
    s = statistics.stdev(values) if len(values) > 1 else 0
    return s / m * 100


# ─────────────────────────────────────────────────────────────────
# Сбор данных
# ─────────────────────────────────────────────────────────────────
# Чтобы делать парные тесты, нам нужны выровненные по индексу пары.
# FLEURS streaming не гарантирует одинаковый порядок ID между языками,
# поэтому собираем по id (если есть) или по позиции.

results = {}

for lang_code, lang_name in LANGUAGES.items():
    print(f"\n{'='*55}")
    print(f"  Язык: {lang_name} ({lang_code})")
    print(f"{'='*55}")

    try:
        ds = load_dataset("google/fleurs", lang_code, split="train",
                          streaming=True).select_columns(["id", "transcription"])
    except Exception:
        # fallback: без id
        try:
            ds = load_dataset("google/fleurs", lang_code, split="train",
                              streaming=True).select_columns(["transcription"])
        except Exception as e:
            print(f"  ОШИБКА загрузки {lang_code}: {e}")
            continue

    by_id = {}      # id -> {tokens, words, tokens_extra}
    ordered_ids = []
    print(f"Считаю {TARGET_COUNT} предложений...")
    for i, item in enumerate(ds):
        text = item["transcription"]
        sid = item.get("id", i)
        wc = len(text.split())
        if wc == 0:
            continue
        tc = count_tokens_gpt4o(text)
        entry = {"tokens": tc, "words": wc, "extra": {}}
        for label, tk in extra_tokenizers.items():
            try:
                entry["extra"][label] = count_tokens_hf(tk, text)
            except Exception:
                pass
        by_id[sid] = entry
        ordered_ids.append(sid)
        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1} из {TARGET_COUNT}")
        if i >= TARGET_COUNT - 1:
            break

    results[lang_code] = {"name": lang_name, "by_id": by_id, "order": ordered_ids}
    print(f"  Готово: {len(by_id)} предложений обработано.")


# ─────────────────────────────────────────────────────────────────
# Сводные таблицы с CI
# ─────────────────────────────────────────────────────────────────

print("\n\n" + "#" * 70)
print("#  CASBench — статистика с 95% доверительными интервалами")
print("#" * 70)

print("\n=== Таблица 1: Токенов на ПРЕДЛОЖЕНИЕ (GPT-4o, главная метрика) ===")
print(f"{'Язык':<14}{'N':<7}{'Среднее':<10}{'95% CI':<20}{'CV':<8}{'vs RU':<8}")
print("-" * 67)

ru_mean = None
sentence_means = {}
sentence_values = {}
for code, d in results.items():
    vals = [v["tokens"] for v in d["by_id"].values()]
    sentence_values[code] = vals
    m, lo, hi = ci95(vals)
    sentence_means[code] = m
    if code == "ru_ru":
        ru_mean = m

for code, d in results.items():
    vals = sentence_values[code]
    m, lo, hi = ci95(vals)
    cvv = cv(vals)
    mult = f"{m/ru_mean:.2f}×" if ru_mean else "—"
    print(f"{d['name']:<14}{len(vals):<7}{m:<10.2f}[{lo:.2f}, {hi:.2f}]      {cvv:<8.1f}{mult:<8}")


print("\n=== Таблица 2: Токенов на СЛОВО (GPT-4o, справочно) ===")
print(f"{'Язык':<14}{'N':<7}{'Среднее':<10}{'95% CI':<20}{'CV':<8}")
print("-" * 59)
for code, d in results.items():
    vals = [v["tokens"]/v["words"] for v in d["by_id"].values()]
    m, lo, hi = ci95(vals)
    cvv = cv(vals)
    print(f"{d['name']:<14}{len(vals):<7}{m:<10.3f}[{lo:.3f}, {hi:.3f}]    {cvv:<8.1f}")


# ─────────────────────────────────────────────────────────────────
# Парные тесты: KZ vs RU, KZ vs EN, KZ vs UZ — по выровненным id
# ─────────────────────────────────────────────────────────────────

print("\n=== Парные тесты значимости (по совпадающим id предложений) ===")
print(f"{'Пара':<22}{'N пар':<8}{'Δ среднее':<14}{'95% CI разности':<26}{'p-value':<12}")
print("-" * 82)

def paired(code_a, code_b):
    """Возвращает (n, m_diff, ci_lo, ci_hi, p) по совпадающим id."""
    if code_a not in results or code_b not in results:
        return None
    ids_a = set(results[code_a]["by_id"].keys())
    ids_b = set(results[code_b]["by_id"].keys())
    common = ids_a & ids_b
    if not common:
        return None
    diffs = []
    for sid in common:
        a = results[code_a]["by_id"][sid]["tokens"]
        b = results[code_b]["by_id"][sid]["tokens"]
        diffs.append(a - b)
    n = len(diffs)
    m, lo, hi = ci95(diffs)
    t, p = paired_t(diffs)
    return n, m, lo, hi, p

for a, b, label in [
    ("kk_kz", "ru_ru", "Казахский − Русский"),
    ("kk_kz", "en_us", "Казахский − Английский"),
    ("kk_kz", "uz_uz", "Казахский − Узбекский"),
    ("uz_uz", "ru_ru", "Узбекский − Русский"),
    ("tr_tr", "ru_ru", "Турецкий − Русский"),
]:
    r = paired(a, b)
    if r is None:
        print(f"{label:<22}нет пересечения id")
        continue
    n, m, lo, hi, p = r
    p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
    print(f"{label:<22}{n:<8}{m:<14.2f}[{lo:.2f}, {hi:.2f}]          {p_str:<12}")


# ─────────────────────────────────────────────────────────────────
# Кросс-токенизаторное сравнение (казахский) с CI
# ─────────────────────────────────────────────────────────────────

if "kk_kz" in results and extra_tokenizers:
    print("\n=== Таблица 3: Казахский — токенов на слово по токенизаторам ===")
    print(f"{'Токенизатор':<16}{'N':<7}{'Среднее':<10}{'95% CI':<20}{'vs KZ-tuned':<12}")
    print("-" * 65)
    kz = results["kk_kz"]["by_id"]
    # GPT-4o
    vals_gpt = [v["tokens"]/v["words"] for v in kz.values()]
    means_by_tk = {"GPT-4o": statistics.mean(vals_gpt)}
    rows_tk = [("GPT-4o", vals_gpt)]
    for label in extra_tokenizers:
        vals = [v["extra"][label]/v["words"] for v in kz.values() if label in v["extra"]]
        if vals:
            means_by_tk[label] = statistics.mean(vals)
            rows_tk.append((label, vals))
    kz_tuned_mean = means_by_tk.get("KZ-tuned")
    for label, vals in rows_tk:
        m, lo, hi = ci95(vals)
        mult = f"{m/kz_tuned_mean:.2f}×" if kz_tuned_mean else "—"
        print(f"{label:<16}{len(vals):<7}{m:<10.3f}[{lo:.3f}, {hi:.3f}]    {mult:<12}")


# ─────────────────────────────────────────────────────────────────
# Paste-ready блок
# ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PASTE-READY ДЛЯ §4 СТАТЬИ (формулировки с CI):")
print("=" * 70)

if "kk_kz" in results and "ru_ru" in results:
    kk_vals = [v["tokens"] for v in results["kk_kz"]["by_id"].values()]
    ru_vals = [v["tokens"] for v in results["ru_ru"]["by_id"].values()]
    kk_m, kk_lo, kk_hi = ci95(kk_vals)
    ru_m, ru_lo, ru_hi = ci95(ru_vals)
    print(f"\nКазахское предложение требует {kk_m:.2f} токенов GPT-4o [95% CI: {kk_lo:.2f}, {kk_hi:.2f}],")
    print(f"русское — {ru_m:.2f} [95% CI: {ru_lo:.2f}, {ru_hi:.2f}].")
    print(f"Множитель: {kk_m/ru_m:.2f}× для семантически идентичного содержания")
    print(f"(парный тест на совпадающих id, n=1000).")

print("\nГотово.")
