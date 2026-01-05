import os
import fnmatch
import glob
import re
import json
import argparse
import csv
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from decimal import Decimal, InvalidOperation


def parse_counts_field(field: str, max_token_digits: int = 8) -> List[int]:
    """
    Extract plausible integer tokens from the last column field.
    - Reject pathological contiguous digit runs (likely not structured counts)
    - Prefer structured formats (JSON list, key:value pairs, comma/space separated)
    - Fall back to small-length integer tokens only (<= max_token_digits)
    """
    # Pathological: extremely long contiguous digits -> reject
    if re.search(r"\d{50,}", field):
        return []

    # Try JSON array
    try:
        obj = json.loads(field)
        if isinstance(obj, (list, tuple)):
            vals = []
            for v in obj:
                if isinstance(v, int) and len(str(abs(v))) <= max_token_digits:
                    vals.append(v)
            if vals:
                return vals
    except Exception:
        pass

    # Key:value pairs (e.g. A:3;C:1)
    if ":" in field:
        vals = []
        for token in re.split(r"[;,\s]+", field.strip()):
            if not token or ":" not in token:
                continue
            _, val = token.split(":", 1)
            if re.fullmatch(r"-?\d{1," + str(max_token_digits) + r"}", val.strip()):
                try:
                    vals.append(int(val))
                except Exception:
                    continue
        if vals:
            return vals

    # Comma/space separated integers
    if re.search(r"[ ,;]", field):
        vals = []
        for token in re.split(r"[ ,;]+", field.strip()):
            if re.fullmatch(r"-?\d{1," + str(max_token_digits) + r"}", token):
                try:
                    vals.append(int(token))
                except Exception:
                    continue
        if vals:
            return vals

    # Fallback: capture only small integer tokens
    small_tokens = re.findall(r"-?\d{1," + str(max_token_digits) + r"}", field)
    try:
        return [int(t) for t in small_tokens]
    except Exception:
        return []


def extract_alt_events(reads: str, ref_base: str) -> List[str]:
    """
    Extract the sequence of ALT base observations from an mpileup reads string,
    aligned to observations (skip markers, handle indels), returning bases (A/C/G/T) only.
    """
    events: List[str] = []
    i = 0
    ref_upper = ref_base.upper()
    while i < len(reads):
        c = reads[i]
        if c == '^':  # start of read marker, skip next quality char
            i += 2
            continue
        if c == '$':  # end of read marker
            i += 1
            continue
        if c in '+-':  # indel: consume length then sequence
            i += 1
            num = ''
            while i < len(reads) and reads[i].isdigit():
                num += reads[i]
                i += 1
            try:
                n = int(num) if num else 0
            except Exception:
                n = 0
            i += n
            continue
        if c == '*':  # deletion placeholder
            i += 1
            continue
        # actual base observation
        base = c.upper()
        if base in 'ACGT':
            if base != ref_upper:
                events.append(base)
        # '.' or ',' are ref matches; other characters ignored
        i += 1
    return events


def extract_event_flags(reads: str, ref_base: str) -> List[bool]:
    """
    Extract a boolean sequence aligned to observations in an mpileup reads string.
    Returns a list where each element is True if the observation is ALT (!= ref),
    and False if the observation matches the REF. Indels and markers are skipped.
    """
    flags: List[bool] = []
    i = 0
    ref_upper = ref_base.upper()
    while i < len(reads):
        c = reads[i]
        if c == '^':  # start of read marker, skip next quality char
            i += 2
            continue
        if c == '$':  # end of read marker
            i += 1
            continue
        if c in '+-':  # indel: consume length then sequence
            i += 1
            num = ''
            while i < len(reads) and reads[i].isdigit():
                num += reads[i]
                i += 1
            try:
                n = int(num) if num else 0
            except Exception:
                n = 0
            i += n
            continue
        if c == '*':  # deletion placeholder
            i += 1
            continue
        # actual base observation
        if c in '.,':
            flags.append(False)
            i += 1
            continue
        base = c.upper()
        if base in 'ACGT':
            flags.append(base != ref_upper)
        # other characters ignored
        i += 1
    return flags


def align_counts_to_events(counts_field: str, num_events: int, max_width: int = 4) -> Tuple[List[int], str]:
    """
    Align a digit string from counts_field to the number of ALT events.
    - Keep only digits from counts_field
    - If len == num_events * w for some w in [1..max_width], chunk by that width (prefer smallest w)
    - If longer but not divisible, truncate to first num_events digits (w=1) as a fallback
    - If shorter than num_events, return empty (unalignable)
    Returns: (counts, mode) where mode is one of {'w1','w2','w3','w4','truncate','unalignable'}
    """
    digits = ''.join(ch for ch in counts_field if ch.isdigit())
    if num_events <= 0:
        return [], 'unalignable'
    for w in range(1, max_width + 1):
        if len(digits) == num_events * w:
            return [int(digits[j:j+w]) for j in range(0, len(digits), w)], f'w{w}'
    # Fallbacks
    if len(digits) > num_events:
        return [int(d) for d in digits[:num_events]], 'truncate'
    return [], 'unalignable'


def analyze_mpileup(
    mpileup_path: str,
    output_dir: str,
    sample_every: int = 1,
    max_points: int = 1000000,
    random_seed: int = 42,
    collect_alt_bases: bool = False,
) -> Tuple[dict, List[int], List[float], List[int]]:
    """
    Parse mpileup and compute correlation between read depth and the per-site
    average of integers in the last column. Also return values for histograms.

    Args:
        mpileup_path: Path to mpileup file.
        output_dir: Directory to write outputs.
        sample_every: Subsample factor for massively large files (use every Nth line).
        max_points: Cap the number of scatter points to keep memory and plot size reasonable.
        random_seed: Seed for reproducibility when subsampling scatter points.

    Returns:
        summary (dict): Summary statistics including correlation metrics.
        all_count_values (List[int]): Flattened list of all integer values from last column across sites.
        site_average_values (List[float]): List of per-site average of last-column integers.
        site_depths (List[int]): List of read depths corresponding to per-site averages.
    """
    rng = np.random.default_rng(random_seed)

    total_lines = 0
    parsed_sites = 0

    depths: List[int] = []
    # Legacy (ALT-focused) aggregates are preserved for backward compatibility
    site_avgs: List[float] = []  # ALT per-site averages
    all_values: List[int] = []   # ALT flattened values
    # New REF/ALT-split aggregates
    site_avgs_alt: List[float] = []
    site_avgs_ref: List[float] = []
    all_values_alt: List[int] = []
    all_values_ref: List[int] = []

    # Optional: per-ALT base flattened values (A/C/G/T)
    alt_base_to_all_values: dict = {"A": [], "C": [], "G": [], "T": []}

    # Skip accounting
    skipped_huge_digit_run = 0
    skipped_no_ints = 0
    skipped_unrepresentable = 0
    skipped_no_alt_events = 0
    skipped_unalignable = 0
    truncate_used = 0
    width_used = {"w1": 0, "w2": 0, "w3": 0, "w4": 0}

    # Format accounting
    lines_altcounts = 0
    lines_mpileup = 0
    depth_proxy_used = 0
    alignment_group_counts = {"all_events": 0, "alt_events": 0}

    sample_basename = os.path.basename(mpileup_path)
    enable_per_base = collect_alt_bases and (
        fnmatch.fnmatch(sample_basename, "*_EMS-1*") or
        fnmatch.fnmatch(sample_basename, "*_EMS-2*") or
        fnmatch.fnmatch(sample_basename, "*NT-12*")
    )

    with open(mpileup_path, "r") as f:
        for i, line in enumerate(f):
            total_lines += 1
            if sample_every > 1 and (i % sample_every) != 0:
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue

            # Auto-detect altcounts format: col4 = ACGT string, col5 = digits
            is_altcounts = bool(re.fullmatch(r"[ACGTacgt]+", parts[3])) and bool(re.fullmatch(r"\d+", parts[4]))

            if is_altcounts:
                lines_altcounts += 1
                alt_string = parts[3]
                counts_field = parts[4]
                alt_events = [b for b in alt_string.upper() if b in 'ACGT']

                counts, mode = align_counts_to_events(counts_field, len(alt_events), max_width=4)
                if not counts:
                    skipped_unalignable += 1
                    continue
                if mode == 'truncate':
                    truncate_used += 1
                elif mode in width_used:
                    width_used[mode] += 1

                # Determine depth: try next column if numeric, else use proxy (sum of counts)
                depth_val: Optional[int] = None
                if len(parts) > 5 and re.fullmatch(r"\d{1,9}", parts[5]):
                    try:
                        depth_val = int(parts[5])
                    except Exception:
                        depth_val = None
                if depth_val is None:
                    depth_val = int(sum(counts))
                    depth_proxy_used += 1

                try:
                    avg_alt_value = float(Decimal(sum(counts)) / Decimal(len(counts)))
                except (OverflowError, InvalidOperation):
                    skipped_unrepresentable += 1
                    continue

                # ALT-only alignment available in altcounts mode; no REF component
                alignment_group_counts["alt_events"] += 1

                depths.append(depth_val)
                site_avgs.append(avg_alt_value)  # legacy
                site_avgs_alt.append(avg_alt_value)
                all_values.extend(counts)  # legacy
                all_values_alt.extend(counts)
                if collect_alt_bases and len(counts) == len(alt_events):
                    for base, value in zip(alt_events, counts):
                        if base in alt_base_to_all_values:
                            alt_base_to_all_values[base].append(int(value))
                parsed_sites += 1
                continue

            # Fallback to mpileup-like format: depth in col4, reads in col5, counts in last col
            if len(parts) < 6:
                continue
            lines_mpileup += 1
            try:
                depth = int(parts[3])
            except ValueError:
                continue

            ref = parts[2]
            reads = parts[4]
            last_field = parts[-1]

            if enable_per_base:
                alt_bases_seq: List[str] = extract_alt_events(reads, ref)
                if not alt_bases_seq:
                    skipped_no_alt_events += 1
                    continue
                if re.search(r"\d{50,}", last_field) and not last_field.strip().isdigit():
                    skipped_huge_digit_run += 1
                    continue
                counts_alt, mode_alt = align_counts_to_events(last_field, len(alt_bases_seq), max_width=4)
                if not counts_alt:
                    skipped_unalignable += 1
                    continue
                if mode_alt == 'truncate':
                    truncate_used += 1
                elif mode_alt in width_used:
                    width_used[mode_alt] += 1
                alignment_group_counts["alt_events"] += 1
                try:
                    avg_alt_value = float(Decimal(sum(counts_alt)) / Decimal(len(counts_alt)))
                except (OverflowError, InvalidOperation):
                    skipped_unrepresentable += 1
                    continue
                depths.append(depth)
                site_avgs.append(avg_alt_value)
                all_values.extend(counts_alt)
                site_avgs_alt.append(avg_alt_value)
                all_values_alt.extend(counts_alt)
                if len(alt_bases_seq) == len(counts_alt):
                    for base, value in zip(alt_bases_seq, counts_alt):
                        if base in alt_base_to_all_values:
                            alt_base_to_all_values[base].append(int(value))
                parsed_sites += 1
            else:
                # Extract event flags (True for ALT, False for REF)
                event_flags = extract_event_flags(reads, ref)
                if not event_flags:
                    # No observable events
                    skipped_no_alt_events += 1
                    continue

                if re.search(r"\d{50,}", last_field) and not last_field.strip().isdigit():
                    skipped_huge_digit_run += 1
                    continue

                # Try aligning counts to ALL events first; fallback to ALT-only
                num_events_all = len(event_flags)
                num_events_alt = int(sum(1 for f in event_flags if f))

                counts_all, mode_all = align_counts_to_events(last_field, num_events_all, max_width=4)
                counts_alt, mode_alt = ([], 'unalignable') if num_events_alt == 0 else align_counts_to_events(last_field, num_events_alt, max_width=4)

                use_all = bool(counts_all)
                use_alt_only = bool(counts_alt) and not use_all
                if not use_all and not use_alt_only:
                    skipped_unalignable += 1
                    continue

                if use_all:
                    if mode_all == 'truncate':
                        truncate_used += 1
                    elif mode_all in width_used:
                        width_used[mode_all] += 1
                    alignment_group_counts["all_events"] += 1

                    # Split into REF vs ALT using flags
                    ref_counts = [c for c, is_alt in zip(counts_all, event_flags) if not is_alt]
                    alt_counts = [c for c, is_alt in zip(counts_all, event_flags) if is_alt]

                    alt_bases_seq: List[str] = extract_alt_events(reads, ref)

                    try:
                        avg_alt_value = float(Decimal(sum(alt_counts)) / Decimal(len(alt_counts))) if alt_counts else None
                        avg_ref_value = float(Decimal(sum(ref_counts)) / Decimal(len(ref_counts))) if ref_counts else None
                    except (OverflowError, InvalidOperation):
                        skipped_unrepresentable += 1
                        continue

                    # Only append paired depth when ALT average exists
                    if avg_alt_value is not None:
                        depths.append(depth)
                        # Legacy (ALT) aggregates
                        site_avgs.append(avg_alt_value)
                        all_values.extend(alt_counts)
                    # New REF/ALT aggregates
                    if avg_alt_value is not None:
                        site_avgs_alt.append(avg_alt_value)
                        all_values_alt.extend(alt_counts)
                    if avg_ref_value is not None:
                        site_avgs_ref.append(avg_ref_value)
                        all_values_ref.extend(ref_counts)
                    if collect_alt_bases and alt_counts and len(alt_bases_seq) == len(alt_counts):
                        for base, value in zip(alt_bases_seq, alt_counts):
                            if base in alt_base_to_all_values:
                                alt_base_to_all_values[base].append(int(value))
                    parsed_sites += 1
                elif use_alt_only:
                    # ALT-only
                    if mode_alt == 'truncate':
                        truncate_used += 1
                    elif mode_alt in width_used:
                        width_used[mode_alt] += 1
                    alignment_group_counts["alt_events"] += 1

                    try:
                        avg_alt_value = float(Decimal(sum(counts_alt)) / Decimal(len(counts_alt)))
                    except (OverflowError, InvalidOperation):
                        skipped_unrepresentable += 1
                        continue

                    depths.append(depth)
                    site_avgs.append(avg_alt_value)  # legacy
                    all_values.extend(counts_alt)     # legacy
                    site_avgs_alt.append(avg_alt_value)
                    all_values_alt.extend(counts_alt)
                    if collect_alt_bases:
                        alt_bases_seq: List[str] = extract_alt_events(reads, ref)
                        if len(alt_bases_seq) == len(counts_alt):
                            for base, value in zip(alt_bases_seq, counts_alt):
                                if base in alt_base_to_all_values:
                                    alt_base_to_all_values[base].append(int(value))
                    parsed_sites += 1

    # Subsample for scatter if too many points
    n_points = len(depths)
    if n_points > max_points:
        keep_idx = rng.choice(n_points, size=max_points, replace=False)
        keep_mask = np.zeros(n_points, dtype=bool)
        keep_mask[keep_idx] = True
        depths_plot = list(np.array(depths)[keep_mask])
        site_avgs_plot = list(np.array(site_avgs)[keep_mask])
    else:
        depths_plot = depths
        site_avgs_plot = site_avgs

    # Compute correlations
    pearson_r: Optional[float] = None
    pearson_p: Optional[float] = None
    spearman_r: Optional[float] = None
    spearman_p: Optional[float] = None

    if len(depths) >= 2 and len(site_avgs) >= 2:
        pearson_r, pearson_p = stats.pearsonr(depths, site_avgs)
        spearman_r, spearman_p = stats.spearmanr(depths, site_avgs)

    # Safe mean for potentially large integer values
    def safe_int_mean(v: List[int]) -> Optional[float]:
        if not v:
            return None
        try:
            return float(Decimal(sum(v)) / Decimal(len(v)))
        except (OverflowError, InvalidOperation):
            return None

    def safe_float_mean(v: List[float]) -> Optional[float]:
        if not v:
            return None
        try:
            return float(Decimal(sum(v)) / Decimal(len(v)))
        except (OverflowError, InvalidOperation):
            return None

    summary = {
        "mpileup": os.path.basename(mpileup_path),
        "total_lines": total_lines,
        "parsed_sites": parsed_sites,
        "all_values_count": len(all_values),
        "site_average_count": len(site_avgs),
        # New REF/ALT split counts
        "all_values_count_alt": len(all_values_alt),
        "all_values_count_ref": len(all_values_ref),
        "site_average_count_alt": len(site_avgs_alt),
        "site_average_count_ref": len(site_avgs_ref),
        "pearson_r": float(pearson_r) if pearson_r is not None else None,
        "pearson_p": float(pearson_p) if pearson_p is not None else None,
        "spearman_r": float(spearman_r) if spearman_r is not None else None,
        "spearman_p": float(spearman_p) if spearman_p is not None else None,
        "depth_summary": {
            "min": int(np.min(depths)) if depths else None,
            "max": int(np.max(depths)) if depths else None,
            "mean": float(np.mean(depths)) if depths else None,
            "median": float(np.median(depths)) if depths else None,
        },
        "site_avg_summary": {
            "min": float(np.min(site_avgs)) if site_avgs else None,
            "max": float(np.max(site_avgs)) if site_avgs else None,
            "mean": float(np.mean(site_avgs)) if site_avgs else None,
            "median": float(np.median(site_avgs)) if site_avgs else None,
        },
        "site_avg_summary_alt": {
            "min": float(np.min(site_avgs_alt)) if site_avgs_alt else None,
            "max": float(np.max(site_avgs_alt)) if site_avgs_alt else None,
            "mean": float(np.mean(site_avgs_alt)) if site_avgs_alt else None,
            "median": float(np.median(site_avgs_alt)) if site_avgs_alt else None,
        },
        "site_avg_summary_ref": {
            "min": float(np.min(site_avgs_ref)) if site_avgs_ref else None,
            "max": float(np.max(site_avgs_ref)) if site_avgs_ref else None,
            "mean": float(np.mean(site_avgs_ref)) if site_avgs_ref else None,
            "median": float(np.median(site_avgs_ref)) if site_avgs_ref else None,
        },
        "all_values_summary": {
            "min": int(np.min(all_values)) if all_values else None,
            "max": int(np.max(all_values)) if all_values else None,
            "mean": safe_int_mean(all_values),
            "median": float(np.median(all_values)) if all_values else None,
        },
        "all_values_summary_alt": {
            "min": int(np.min(all_values_alt)) if all_values_alt else None,
            "max": int(np.max(all_values_alt)) if all_values_alt else None,
            "mean": safe_int_mean(all_values_alt),
            "median": float(np.median(all_values_alt)) if all_values_alt else None,
        },
        "all_values_summary_ref": {
            "min": int(np.min(all_values_ref)) if all_values_ref else None,
            "max": int(np.max(all_values_ref)) if all_values_ref else None,
            "mean": safe_int_mean(all_values_ref),
            "median": float(np.median(all_values_ref)) if all_values_ref else None,
        },
        "skips": {
            "huge_digit_run": skipped_huge_digit_run,
            "no_ints_found": skipped_no_ints,
            "unrepresentable_avg": skipped_unrepresentable,
            "no_alt_events": skipped_no_alt_events,
            "unalignable_counts": skipped_unalignable,
        },
        "alignment_modes": {
            **width_used,
            "truncate": truncate_used,
        },
        "format_stats": {
            "altcounts_lines": lines_altcounts,
            "mpileup_lines": lines_mpileup,
            "depth_proxy_used": depth_proxy_used,
            "alignment_group_counts": alignment_group_counts,
        },
        "scatter_points_used": len(depths_plot),
        # Convenience top-level ref/alt means of per-site averages
        "mean_site_avg_alt": safe_float_mean(site_avgs_alt),
        "mean_site_avg_ref": safe_float_mean(site_avgs_ref),
    }

    # Append per-ALT-base summaries if requested
    if collect_alt_bases:
        def base_summary(values: List[int]) -> dict:
            if not values:
                return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
            return {
                "count": len(values),
                "min": int(np.min(values)),
                "max": int(np.max(values)),
                "mean": safe_int_mean(values),
                "median": float(np.median(values)),
            }

        summary["alt_bases_observed"] = [b for b in ["A", "C", "G", "T"] if alt_base_to_all_values[b]]
        summary["all_values_count_by_alt_base"] = {b: len(alt_base_to_all_values[b]) for b in ["A", "C", "G", "T"]}
        summary["all_values_summary_by_alt_base"] = {b: base_summary(alt_base_to_all_values[b]) for b in ["A", "C", "G", "T"]}

    # Save summary
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, f"{os.path.basename(mpileup_path)}.rca_summary.json")
    with open(summary_path, "w") as f_sum:
        json.dump(summary, f_sum, indent=2)

    # Visualizations
    sns.set(context="talk", style="whitegrid")

    # 1) Histogram of all integer values across genome (legacy ALT)
    if all_values:
        plt.figure(figsize=(12, 8))
        sns.histplot(all_values, bins=100, kde=False)
        plt.xlabel("RCA count values (flattened across sites)")
        plt.ylabel("Count")
        plt.title("Distribution of RCA count values across genome")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{os.path.basename(mpileup_path)}.rca_all_values_hist.png"), dpi=200)
        plt.close()

    # New: histograms split by ALT and REF
    if all_values_alt:
        plt.figure(figsize=(12, 8))
        sns.histplot(all_values_alt, bins=100, kde=False)
        plt.xlabel("ALT RCA count values (flattened across sites)")
        plt.ylabel("Count")
        plt.title("Distribution of ALT RCA count values across genome")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{os.path.basename(mpileup_path)}.rca_all_values_hist_alt.png"), dpi=200)
        plt.close()
    if all_values_ref:
        plt.figure(figsize=(12, 8))
        sns.histplot(all_values_ref, bins=100, kde=False)
        plt.xlabel("REF RCA count values (flattened across sites)")
        plt.ylabel("Count")
        plt.title("Distribution of REF RCA count values across genome")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{os.path.basename(mpileup_path)}.rca_all_values_hist_ref.png"), dpi=200)
        plt.close()

    # Optional: ALT per-base overlay histogram
    if collect_alt_bases and any(len(v) > 0 for v in alt_base_to_all_values.values()):
        plt.figure(figsize=(12, 8))
        # Compute a shared range for comparable bins
        combined_values = []
        for b in ["A", "C", "G", "T"]:
            combined_values.extend(alt_base_to_all_values[b])
        if combined_values:
            value_min = int(np.min(combined_values))
            value_max = int(np.max(combined_values))
            # Avoid degenerate range
            if value_max == value_min:
                value_max = value_min + 1
            for base, color in zip(["A", "C", "G", "T"], ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
                values = alt_base_to_all_values[base]
                if values:
                    sns.histplot(values, bins=100, binrange=(value_min, value_max), stat="density", kde=False, element="step", fill=False, label=base, color=color)
            plt.xlabel("ALT RCA count values by base")
            plt.ylabel("Density")
            plt.title("Distribution of ALT RCA count values by base (overlay)")
            plt.legend(title="ALT base")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{os.path.basename(mpileup_path)}.rca_all_values_hist_alt_by_base.png"), dpi=200)
        plt.close()

    # 2) Histogram of per-site average of integer values (legacy ALT)
    if site_avgs:
        plt.figure(figsize=(12, 8))
        sns.histplot(site_avgs, bins=100, kde=False)
        plt.xlabel("Per-site average RCA count")
        plt.ylabel("Count")
        plt.title("Distribution of per-site average RCA counts")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{os.path.basename(mpileup_path)}.rca_site_avg_hist.png"), dpi=200)
        plt.close()

    # New: per-site averages split
    if site_avgs_alt:
        plt.figure(figsize=(12, 8))
        sns.histplot(site_avgs_alt, bins=100, kde=False)
        plt.xlabel("Per-site average ALT RCA count")
        plt.ylabel("Count")
        plt.title("Distribution of per-site average ALT RCA counts")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{os.path.basename(mpileup_path)}.rca_site_avg_hist_alt.png"), dpi=200)
        plt.close()
    if site_avgs_ref:
        plt.figure(figsize=(12, 8))
        sns.histplot(site_avgs_ref, bins=100, kde=False)
        plt.xlabel("Per-site average REF RCA count")
        plt.ylabel("Count")
        plt.title("Distribution of per-site average REF RCA counts")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{os.path.basename(mpileup_path)}.rca_site_avg_hist_ref.png"), dpi=200)
        plt.close()

    # 3) Depth vs per-site average scatter with hexbin overlay for density (legacy ALT)
    if depths_plot and site_avgs_plot:
        plt.figure(figsize=(12, 9))
        plt.hexbin(depths_plot, site_avgs_plot, gridsize=60, cmap="viridis", mincnt=1)
        plt.colorbar(label="Number of sites")
        plt.xlabel("Total read depth")
        plt.ylabel("Per-site average RCA count")
        plt.title("Depth vs per-site average RCA count (hexbin density)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{os.path.basename(mpileup_path)}.rca_depth_vs_avg_hexbin.png"), dpi=200)
        plt.close()

        # Also provide a regression plot on a subsample (already subsampled if needed)
        plt.figure(figsize=(12, 9))
        sns.regplot(x=depths_plot, y=site_avgs_plot, scatter_kws={"alpha": 0.2, "s": 10}, line_kws={"color": "red"})
        plt.xlabel("Total read depth")
        plt.ylabel("Per-site average RCA count")
        plt.title("Depth vs per-site average RCA count (regression)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{os.path.basename(mpileup_path)}.rca_depth_vs_avg_reg.png"), dpi=200)
        plt.close()

        # New: regression for REF averages if available
        if site_avgs_ref:
            plt.figure(figsize=(12, 9))
            n_ref = min(len(depths_plot), len(site_avgs_ref))
            sns.regplot(x=depths_plot[:n_ref], y=site_avgs_ref[:n_ref], scatter_kws={"alpha": 0.2, "s": 10}, line_kws={"color": "red"})
            plt.xlabel("Total read depth")
            plt.ylabel("Per-site average REF RCA count")
            plt.title("Depth vs per-site average REF RCA count (regression)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{os.path.basename(mpileup_path)}.rca_depth_vs_avg_reg_ref.png"), dpi=200)
            plt.close()

    return summary, all_values, site_avgs, depths


def main():
    parser = argparse.ArgumentParser(description="Analyze RCA count integers in the last column of an mpileup file or all files in a directory.")
    parser.add_argument("mpileup", help="Path to mpileup file OR directory of mpileup files")
    parser.add_argument("output_dir", help="Directory to write outputs")
    parser.add_argument("--pattern", default="*.txt", help="Glob pattern to match files in directory mode (default: *.txt)")
    parser.add_argument("--sample-every", type=int, default=1, help="Use every Nth line to reduce memory/CPU for huge files")
    parser.add_argument("--max-points", type=int, default=1000000, help="Maximum points for scatter/hexbin plots")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    parser.add_argument("--collect-alt-bases", action="store_true", help="Collect and report RCA count distributions split by ALT base (A/C/G/T)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    summaries = []
    if os.path.isdir(args.mpileup):
        file_paths = sorted(glob.glob(os.path.join(args.mpileup, args.pattern)))
        if not file_paths:
            print(json.dumps({"error": "No files matched", "dir": args.mpileup, "pattern": args.pattern}, indent=2))
            return
        for fp in file_paths:
            summary, _, _, _ = analyze_mpileup(
                mpileup_path=fp,
                output_dir=args.output_dir,
                sample_every=args.sample_every,
                max_points=args.max_points,
                random_seed=args.seed,
                collect_alt_bases=args.collect_alt_bases,
            )
            summaries.append(summary)
        agg_path = os.path.join(args.output_dir, "rca_summaries.json")
        with open(agg_path, "w") as f:
            json.dump(summaries, f, indent=2)
        # Emit a CSV comparing ALT vs REF per-site average means per sample
        csv_path = os.path.join(args.output_dir, "rca_ref_alt_summary.csv")
        with open(csv_path, "w", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["sample", "mean_site_avg_alt", "mean_site_avg_ref", "site_count_alt", "site_count_ref"]) 
            for s in summaries:
                writer.writerow([
                    s.get("mpileup"),
                    s.get("mean_site_avg_alt"),
                    s.get("mean_site_avg_ref"),
                    s.get("site_average_count_alt"),
                    s.get("site_average_count_ref"),
                ])
        print(json.dumps({"processed_files": len(file_paths), "aggregate_summary": agg_path}, indent=2))
    else:
        summary, _, _, _ = analyze_mpileup(
            mpileup_path=args.mpileup,
            output_dir=args.output_dir,
            sample_every=args.sample_every,
            max_points=args.max_points,
            random_seed=args.seed,
            collect_alt_bases=args.collect_alt_bases,
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main() 