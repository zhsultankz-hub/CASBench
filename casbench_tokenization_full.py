#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CASBench — Extended tokenization measurement with statistics.

Runs all 50 word pairs, 30 pure-KZ words, and 63 sentences across
production tokenizers. Outputs:

  • Per-pair token counts and ratios (full table)
  • Per-category statistics (mean ± std ratio for ACC/ABL/LOC/POSS/PL)
  • Pure-KZ tokenizer comparison
  • Per-domain sentence statistics (FIN/ADM/EDU)
  • Code-switching premium with confidence intervals

REQUIRES: pip install tiktoken transformers sentencepiece
REQUIRES: casbench_lexicon.py in the same directory

Usage:
    python casbench_tokenization_full.py
    python casbench_tokenization_full.py --csv     # also write csv files
    python casbench_tokenization_full.py --tex     # latex tables for paper
"""

import sys
import argparse
import statistics
from collections import defaultdict

try:
    from casbench_lexicon import WORD_PAIRS, PURE_KZ_WORDS, SENTENCES
except ImportError:
    print("ERROR: casbench_lexicon.py must be in the same folder as this script.")
    print("Download it from the same source and put it next to this file.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────
# Tokenizer adapters
# ─────────────────────────────────────────────────────────────────

def load_tokenizers():
    tokenizers = {}

    try:
        import tiktoken
        enc4o = tiktoken.encoding_for_model("gpt-4o")
        tokenizers["GPT-4o"] = (lambda t: len(enc4o.encode(t)), "o200k_base")
    except Exception as e:
        print(f"[skip] GPT-4o: {e}", file=sys.stderr)

    try:
        import tiktoken
        enc4 = tiktoken.get_encoding("cl100k_base")
        tokenizers["GPT-4"] = (lambda t: len(enc4.encode(t)), "cl100k_base")
    except Exception as e:
        print(f"[skip] GPT-4: {e}", file=sys.stderr)

    for mid in ["meta-llama/Meta-Llama-3-8B",
                "NousResearch/Meta-Llama-3-8B",
                "unsloth/llama-3-8b"]:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(mid)
            tokenizers["LLaMA-3"] = (
                lambda t, _tok=tok: len(_tok.encode(t, add_special_tokens=False)),
                mid,
            )
            break
        except Exception:
            continue
    if "LLaMA-3" not in tokenizers:
        print("[skip] LLaMA-3: no accessible checkpoint", file=sys.stderr)

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
        tokenizers["BERTurk"] = (
            lambda t, _tok=tok: len(_tok.encode(t, add_special_tokens=False)),
            "dbmdz/bert-base-turkish-cased",
        )
    except Exception as e:
        print(f"[skip] BERTurk: {e}", file=sys.stderr)

    for mid in ["kz-transformers/kaz-roberta-conversational",
                "issai/LLama-3.1-KazLLM-1.0-8B",
                "Kyrmasch/kaz-roberta"]:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(mid)
            tokenizers["KZ-tuned"] = (
                lambda t, _tok=tok: len(_tok.encode(t, add_special_tokens=False)),
                mid,
            )
            break
        except Exception:
            continue
    if "KZ-tuned" not in tokenizers:
        print("[skip] KZ-tuned: no accessible checkpoint", file=sys.stderr)

    return tokenizers


# ─────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────

def mean_std(values):
    """Mean and population std as strings, two decimals."""
    if not values:
        return "—", "—"
    if len(values) == 1:
        return f"{values[0]:.2f}", "—"
    m = statistics.mean(values)
    s = statistics.pstdev(values)
    return f"{m:.2f}", f"{s:.2f}"


def fmt_table(headers, rows, pad=2):
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))
    def line(cells):
        return "  ".join(str(c).ljust(widths[i] + pad) for i, c in enumerate(cells))
    out = [line(headers), "-" * (sum(widths) + pad * len(widths) + 2 * (len(widths) - 1))]
    out += [line(r) for r in rows]
    return "\n".join(out)


def csv_write(path, headers, rows):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"  wrote {path}")


# ─────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--tex", action="store_true")
    args = ap.parse_args()

    toks = load_tokenizers()
    if not toks:
        print("No tokenizers loaded.")
        sys.exit(1)

    names = list(toks.keys())
    print("\nTokenizers loaded:")
    for n in names:
        print(f"  {n:<10} {toks[n][1]}")

    # ═══════════════════════════════════════════════════════════
    # PART A: 50 word pairs — per-pair counts + per-category stats
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print("PART A.1 — Per-pair token counts (50 RU/KZ-blend pairs)")
    print("=" * 76)

    # Collect ratios per tokenizer per category
    ratios_by_tok_cat = defaultdict(lambda: defaultdict(list))
    pair_rows = []

    for ru, kz, gloss, cat in WORD_PAIRS:
        ru_counts = {n: toks[n][0](ru) for n in names}
        kz_counts = {n: toks[n][0](kz) for n in names}
        row = [f"{ru} → {kz}", gloss, cat]
        for n in names:
            row.extend([ru_counts[n], kz_counts[n]])
            r = kz_counts[n] / ru_counts[n] if ru_counts[n] else 0
            ratios_by_tok_cat[n][cat].append(r)
            ratios_by_tok_cat[n]["ALL"].append(r)
        pair_rows.append(row)

    # Per-pair table header
    headers = ["Pair", "Gloss", "Cat"]
    for n in names:
        headers.extend([f"{n} RU", f"{n} KZ"])
    print("\n" + fmt_table(headers, pair_rows))

    # Per-category summary
    print("\n" + "=" * 76)
    print("PART A.2 — Mean ratio (KZ-blend / RU baseline) by category")
    print("=" * 76)
    cat_headers = ["Category", "N"] + names
    cat_rows = []
    for cat in ["ACC", "ABL", "LOC", "POSS", "PL", "ALL"]:
        n_items = len(ratios_by_tok_cat[names[0]][cat]) if names else 0
        row = [cat, n_items]
        for n in names:
            vals = ratios_by_tok_cat[n][cat]
            m, s = mean_std(vals)
            row.append(f"{m} ± {s}" if s != "—" else m)
        cat_rows.append(row)
    print("\n" + fmt_table(cat_headers, cat_rows))

    # ═══════════════════════════════════════════════════════════
    # PART B: 30 pure Kazakh words — KZ-tuned advantage
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print("PART B — Pure Kazakh words (KZ-tuned advantage)")
    print("=" * 76)
    pkz_headers = ["Word", "Gloss"] + names
    pkz_rows = []
    pkz_counts_by_tok = {n: [] for n in names}
    for w, gloss in PURE_KZ_WORDS:
        row = [w, gloss]
        for n in names:
            c = toks[n][0](w)
            row.append(c)
            pkz_counts_by_tok[n].append(c)
        pkz_rows.append(row)
    print("\n" + fmt_table(pkz_headers, pkz_rows))

    # KZ words summary
    print("\nPure-KZ summary (lower = better):")
    summary_rows = []
    for n in names:
        vals = pkz_counts_by_tok[n]
        m, s = mean_std(vals)
        summary_rows.append([n, f"{m} ± {s}" if s != "—" else m, min(vals), max(vals)])
    print("\n" + fmt_table(["Tokenizer", "Mean ± SD", "Min", "Max"], summary_rows))

    if "KZ-tuned" in names:
        kz_mean = statistics.mean(pkz_counts_by_tok["KZ-tuned"])
        print(f"\nKZ-tuned advantage on pure-KZ words:")
        for n in names:
            if n == "KZ-tuned":
                continue
            other_mean = statistics.mean(pkz_counts_by_tok[n])
            ratio = other_mean / kz_mean if kz_mean else 0
            print(f"  {n:<10} {other_mean:.2f} vs KZ-tuned {kz_mean:.2f}  →  {ratio:.2f}× more")

    # ═══════════════════════════════════════════════════════════
    # PART C: 63 sentences — per-domain CS premium
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print("PART C — Sentence-level code-switching premium by domain")
    print("=" * 76)

    # Group sentences: index by domain → variant → list of (idx, text)
    by_dom_var = defaultdict(lambda: defaultdict(list))
    for dom, var, text in SENTENCES:
        by_dom_var[dom][var].append(text)

    # We assume CS / RU / KZ triples align by index within each domain
    sent_names = [n for n in ("GPT-4o", "GPT-4", "LLaMA-3") if n in names]
    if not sent_names:
        sent_names = names

    domain_summary = []
    for dom in ["FIN", "ADM", "EDU"]:
        cs_list = by_dom_var[dom]["CS"]
        ru_list = by_dom_var[dom]["RU"]
        kz_list = by_dom_var[dom]["KZ"]
        if not (len(cs_list) == len(ru_list) == len(kz_list)):
            print(f"[warn] {dom}: unequal variants ({len(cs_list)}/{len(ru_list)}/{len(kz_list)})")
            continue

        print(f"\n─── Domain: {dom} ({len(cs_list)} triples) ───")
        rows = []
        prem_vs_ru = defaultdict(list)
        prem_vs_kz = defaultdict(list)
        for i, (cs, ru, kz) in enumerate(zip(cs_list, ru_list, kz_list), 1):
            row = [f"{dom}-{i}"]
            for n in sent_names:
                cs_c = toks[n][0](cs)
                ru_c = toks[n][0](ru)
                kz_c = toks[n][0](kz)
                row.extend([cs_c, ru_c, kz_c])
                prem_vs_ru[n].append((cs_c - ru_c) / ru_c * 100 if ru_c else 0)
                prem_vs_kz[n].append((cs_c - kz_c) / kz_c * 100 if kz_c else 0)
            rows.append(row)
        h = ["#"]
        for n in sent_names:
            h.extend([f"{n} CS", f"{n} RU", f"{n} KZ"])
        print(fmt_table(h, rows))

        print(f"\n  {dom} code-switching premium (mean ± SD):")
        for n in sent_names:
            m_ru, s_ru = mean_std(prem_vs_ru[n])
            m_kz, s_kz = mean_std(prem_vs_kz[n])
            print(f"    {n:<10} vs RU: {m_ru}% ± {s_ru}%   vs KZ: {m_kz}% ± {s_kz}%")
            domain_summary.append([dom, n, f"{m_ru}% ± {s_ru}%", f"{m_kz}% ± {s_kz}%"])

    # ═══════════════════════════════════════════════════════════
    # Cross-domain summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print("PART D — Cross-domain code-switching premium summary")
    print("=" * 76)
    print(fmt_table(["Domain", "Tokenizer", "CS premium vs RU", "CS premium vs KZ"], domain_summary))

    # ═══════════════════════════════════════════════════════════
    # CSV output
    # ═══════════════════════════════════════════════════════════
    if args.csv:
        print("\n" + "=" * 76)
        print("Writing CSV files")
        print("=" * 76)
        # Pair-level
        csv_write("casbench_word_pairs.csv", headers, pair_rows)
        csv_write("casbench_category_summary.csv", cat_headers, cat_rows)
        csv_write("casbench_pure_kz.csv", pkz_headers, pkz_rows)
        csv_write("casbench_domain_premium.csv",
                  ["Domain", "Tokenizer", "CS vs RU", "CS vs KZ"], domain_summary)

    # ═══════════════════════════════════════════════════════════
    # Paste-ready summary for the paper
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print("PASTE-READY SUMMARY (for paper §4)")
    print("=" * 76)
    print("""
Headline findings (fill into §4.1 of the paper):

1. Overall morphological-blending premium (mean ratio across all 50 pairs):
""")
    for n in names:
        vals = ratios_by_tok_cat[n]["ALL"]
        m, s = mean_std(vals)
        print(f"   {n:<10} {m} ± {s}× the RU baseline")
    print("\n2. Highest-cost category (per tokenizer):")
    for n in names:
        worst_cat, worst_val = None, 0
        for cat in ["ACC", "ABL", "LOC", "POSS", "PL"]:
            vals = ratios_by_tok_cat[n][cat]
            if vals and statistics.mean(vals) > worst_val:
                worst_val = statistics.mean(vals)
                worst_cat = cat
        print(f"   {n:<10} {worst_cat} ({worst_val:.2f}×)")
    print("\n3. KZ-tuned advantage on pure Kazakh words:")
    if "KZ-tuned" in names:
        kz_mean = statistics.mean(pkz_counts_by_tok["KZ-tuned"])
        for n in names:
            if n == "KZ-tuned":
                continue
            other_mean = statistics.mean(pkz_counts_by_tok[n])
            print(f"   {n:<10} {other_mean / kz_mean:.2f}× more tokens than KZ-tuned")

    print("\n4. Sentence-level code-switching premium (averaged across all domains and items):")
    for n in sent_names:
        all_vs_ru = []
        all_vs_kz = []
        for dom in ["FIN", "ADM", "EDU"]:
            cs_list = by_dom_var[dom]["CS"]
            ru_list = by_dom_var[dom]["RU"]
            kz_list = by_dom_var[dom]["KZ"]
            for cs, ru, kz in zip(cs_list, ru_list, kz_list):
                cs_c = toks[n][0](cs)
                ru_c = toks[n][0](ru)
                kz_c = toks[n][0](kz)
                all_vs_ru.append((cs_c - ru_c) / ru_c * 100 if ru_c else 0)
                all_vs_kz.append((cs_c - kz_c) / kz_c * 100 if kz_c else 0)
        m_ru, s_ru = mean_std(all_vs_ru)
        m_kz, s_kz = mean_std(all_vs_kz)
        print(f"   {n:<10} CS vs RU: {m_ru}% ± {s_ru}%   CS vs KZ: {m_kz}% ± {s_kz}%")

    print("\nPaste these into the paper as the new headline numbers.\n")


if __name__ == "__main__":
    main()
