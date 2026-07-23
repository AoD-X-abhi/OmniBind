"""
Data Preprocessing Script for Punjabi-English Neural Machine Translation
Extracts, cleans, deduplicates, and samples 100,000 sentence pairs from the Anuvaad corpus.
Uses Python standard library (csv, json, random, re, unicodedata, html) for maximum portability.
"""

import os
import re
import csv
import html
import unicodedata
import random


def clean_text(text: str) -> str:
    """Clean and normalize raw text line."""
    if not text:
        return ""
    
    # 1. Unescape HTML / XML entities (&amp;, &quot;, etc.)
    text = html.unescape(text)
    
    # 2. Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 3. Unicode NFC Normalization (crucial for Gurmukhi script consistency)
    text = unicodedata.normalize("NFC", text)
    
    # 4. Collapse multiple spaces & strip whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def is_valid_pair(en: str, pa: str, min_len: int = 5, max_len: int = 500, max_ratio: float = 3.0) -> bool:
    """Validate character lengths and length ratios between source and target."""
    len_en = len(en)
    len_pa = len(pa)
    
    if len_en < min_len or len_pa < min_len:
        return False
    if len_en > max_len or len_pa > max_len:
        return False
    
    ratio = max(len_en, len_pa) / max(min(len_en, len_pa), 1)
    if ratio > max_ratio:
        return False
        
    return True


def preprocess_data(
    en_path: str = "Anuvaad.en-pa.en",
    pa_path: str = "Anuvaad.en-pa.pa",
    output_csv: str = "punjabi_english_100k.csv",
    sample_size: int = 100000,
    seed: int = 42
):
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("  Punjabi-English Corpus Preprocessing & Subsetting")
    print("=" * 60)
    print(f"Input English Corpus: {en_path}")
    print(f"Input Punjabi Corpus: {pa_path}")
    print(f"Target Output CSV:    {output_csv}")
    print(f"Target Sample Size:   {sample_size:,}")
    print("-" * 60)

    if not os.path.exists(en_path) or not os.path.exists(pa_path):
        raise FileNotFoundError(f"Corpus files not found! Ensure {en_path} and {pa_path} are in working directory.")

    valid_pairs = []
    seen_pairs = set()
    total_processed = 0

    with open(en_path, "r", encoding="utf-8") as f_en, open(pa_path, "r", encoding="utf-8") as f_pa:
        for line_en, line_pa in zip(f_en, f_pa):
            total_processed += 1
            if total_processed % 250000 == 0:
                print(f"  Processed {total_processed:,} raw pairs... (Valid collected: {len(valid_pairs):,})")

            clean_en = clean_text(line_en)
            clean_pa = clean_text(line_pa)

            if is_valid_pair(clean_en, clean_pa):
                pair_key = (clean_en, clean_pa)
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    valid_pairs.append((clean_en, clean_pa))

    total_valid = len(valid_pairs)
    print("-" * 60)
    print(f"Extraction complete!")
    print(f"  Total Raw Pairs Evaluated:  {total_processed:,}")
    print(f"  Clean & Unique Pairs Found: {total_valid:,}")

    # Random Sampling
    random.seed(seed)
    if total_valid > sample_size:
        sampled_pairs = random.sample(valid_pairs, sample_size)
        print(f"  Randomly Sampled:           {sample_size:,} pairs")
    else:
        sampled_pairs = valid_pairs
        print(f"  Using all available pairs: {total_valid:,} pairs")

    # Shuffle
    random.shuffle(sampled_pairs)

    # Calculate train / val / test split counts
    n = len(sampled_pairs)
    n_train = int(n * 0.85)
    n_val = int(n * 0.075)
    n_test = n - n_train - n_val

    print("-" * 60)
    print("Dataset Split Allocation:")
    print(f"  Train Set:      {n_train:,} pairs (85%)")
    print(f"  Validation Set: {n_val:,} pairs (7.5%)")
    print(f"  Test Set:       {n_test:,} pairs (7.5%)")

    # Write to CSV
    with open(output_csv, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["english", "punjabi", "split"])

        for idx, (en, pa) in enumerate(sampled_pairs):
            if idx < n_train:
                split = "train"
            elif idx < n_train + n_val:
                split = "val"
            else:
                split = "test"
            writer.writerow([en, pa, split])

    file_size_mb = os.path.getsize(output_csv) / (1024 * 1024)
    print("=" * 60)
    print(f"SUCCESS! Dataset written to '{output_csv}' ({file_size_mb:.2f} MB)")
    print("=" * 60)

    # Display sample preview safely
    print("\nSample Preview (First 3 rows):")
    for idx, (en, pa) in enumerate(sampled_pairs[:3]):
        split_label = "TRAIN" if idx < n_train else ("VAL" if idx < n_train + n_val else "TEST")
        try:
            print(f"  [{split_label}]")
            print(f"    EN: {en}")
            print(f"    PA: {pa}\n")
        except UnicodeEncodeError:
            print(f"  [{split_label}]")
            print(f"    EN: {en}")
            print(f"    PA: [Gurmukhi Script Text - {len(pa)} chars]\n")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    preprocess_data()
