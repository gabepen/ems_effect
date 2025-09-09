import os
import sys
import argparse
from typing import Tuple, List, Dict, Optional, Set

import gzip

# Ensure we can import from ems_effect/src/modules
SCRIPT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from modules.parse import SeqContext, check_mutations  # type: ignore


def open_maybe_gzip(path: str):
    if path.endswith('.gz'):
        return gzip.open(path, 'rt')
    return open(path, 'r')


def compute_counts(
    mpileup_path: str,
    seq_context: SeqContext,
    min_depth: int,
    ems_only: bool,
    read_level: bool,
    min_alt: int,
) -> Tuple[int, int]:
    """Compute aggregate x and n from an mpileup.

    If read_level is False (trial mode):
        - x = number of mutated sites (requiring sum alt-reads >= min_alt)
        - n = number of callable sites
    If read_level is True:
        - x = total alt-read counts across callable sites
        - n = total read depth across callable sites
    """
    x_total = 0
    n_total = 0

    genome_seq_str = str(seq_context.genome)

    with open_maybe_gzip(mpileup_path) as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            chrom = parts[0]
            pos_1based = int(parts[1])
            ref_base = parts[2]
            depth = int(parts[3])

            pos_0based = pos_1based - 1

            # Skip overlapping regions
            if pos_0based < 0 or pos_0based >= len(seq_context.overlap_mask):
                continue
            if not seq_context.overlap_mask[pos_0based]:
                continue

            # Depth filter for callability
            if depth < min_depth:
                continue

            # EMS-only callable restriction
            if ems_only and ref_base not in ('G', 'C'):
                continue

            # Site is callable
            if read_level:
                n_total += depth
            else:
                n_total += 1

            # Detect mutations at this site
            entry = parts
            muts, _depth, _ = check_mutations(
                entry=entry,
                genestart=0,
                sample=chrom,
                ems_only=ems_only,
                base_counts={},
                genome_seq=genome_seq_str,
                context_counts={},
                context_size=7,
                gene_feature=None,
            )

            if read_level:
                # Sum all alt-read counts registered by check_mutations
                if muts:
                    x_total += sum(muts.values())
            else:
                # Trial mode: require at least min_alt alt reads
                if muts and sum(muts.values()) >= min_alt:
                    x_total += 1

    return x_total, n_total


def write_counts(output_path: str, x: int, n: int) -> None:
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as out:
        out.write('x\tn\n')
        out.write(f'{x}\t{n}\n')


def write_per_site_counts(output_path: str, mpileup_path: str, seq_context: SeqContext, min_depth: int, ems_only: bool, read_level: bool, min_alt: int) -> None:
    """Write one row per callable site.

    If read_level is False: n=1, x in {0,1} with x=1 only if alt-read count >= min_alt
    If read_level is True:  n=depth, x=alt-read count at the site
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    genome_seq_str = str(seq_context.genome)

    with open(output_path, 'w') as out, open_maybe_gzip(mpileup_path) as handle:
        out.write('site\tx\tn\n')
        for line in handle:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            chrom = parts[0]
            pos_1based = int(parts[1])
            ref_base = parts[2]
            depth = int(parts[3])
            pos_0based = pos_1based - 1

            # Skip overlapping regions
            if pos_0based < 0 or pos_0based >= len(seq_context.overlap_mask):
                continue
            if not seq_context.overlap_mask[pos_0based]:
                continue

            # Depth filter for callability
            if depth < min_depth:
                continue

            # EMS-only callable restriction
            if ems_only and ref_base not in ('G', 'C'):
                continue

            # Decide mutation presence / counts using check_mutations
            entry = parts
            muts, _depth, _ = check_mutations(
                entry=entry,
                genestart=0,
                sample=chrom,
                ems_only=ems_only,
                base_counts={},
                genome_seq=genome_seq_str,
                context_counts={},
                context_size=7,
                gene_feature=None,
            )

            if read_level:
                x_val = int(sum(muts.values())) if muts else 0
                n_val = int(depth)
            else:
                alt_reads = int(sum(muts.values())) if muts else 0
                x_val = 1 if alt_reads >= min_alt else 0
                n_val = 1

            out.write(f'{chrom}:{pos_1based}\t{x_val}\t{n_val}\n')


def derive_output_paths(base_output: str) -> Tuple[str, str]:
    base, ext = os.path.splitext(base_output)
    if not ext:
        ext = '.tsv'
    trial_path = f"{base}.trial{ext}"
    read_path = f"{base}.read{ext}"
    return trial_path, read_path


def find_sample_pairs(mpileup_dir: str) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """Find per-sample sites mpileup and mutations mpileup.

    Returns a list of (sample, sites_path, mutations_path).
    sites_path is the unfiltered sites mpileup; mutations_path is <sample>_variants_mutations.txt(.gz).
    """
    files = os.listdir(mpileup_dir)
    samples: Dict[str, Dict[str, Optional[str]]] = {}

    # First record mutations files
    for fname in files:
        if fname.endswith('_variants_mutations.txt') or fname.endswith('_variants_mutations.txt.gz'):
            sample = fname
            if sample.endswith('.gz'):
                sample = sample[:-3]
            sample = sample[:-len('_variants_mutations.txt')]
            samples.setdefault(sample, {})['mutations'] = os.path.join(mpileup_dir, fname)

    # For sites mpileup, prefer explicit 'sites' file; otherwise any mpileup for sample that is not the mutations file
    for fname in files:
        if not (fname.endswith('.txt') or fname.endswith('.txt.gz')):
            continue
        # Identify sample name prefix before first underscore
        # Accept forms starting with <sample>_ and not containing 'mutations'
        for sample in list(samples.keys()):
            if fname.startswith(sample + '_') and ('mutations' not in fname):
                # Prefer sites keywords
                if ('sites' in fname) or ('unfiltered' in fname):
                    samples[sample]['sites'] = os.path.join(mpileup_dir, fname)
                # Fallback if none set yet
                elif 'sites' not in samples[sample]:
                    samples[sample]['sites'] = os.path.join(mpileup_dir, fname)

    # Build result list
    out: List[Tuple[str, Optional[str], Optional[str]]] = []
    for sample, paths in samples.items():
        out.append((sample, paths.get('sites'), paths.get('mutations')))
    return sorted(out, key=lambda t: t[0])


def get_mutated_positions(mutations_mpile: str, seq_context: SeqContext, ems_only: bool) -> Set[int]:
    """Parse the mutations mpileup and return 1-based genomic positions that show mutations (respecting ems_only)."""
    genome_seq_str = str(seq_context.genome)
    mutant_pos: Set[int] = set()
    with open_maybe_gzip(mutations_mpile) as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            chrom = parts[0]
            pos_1based = int(parts[1])
            # Use check_mutations to confirm mutation and filter by EMS-only
            muts, _depth, _ = check_mutations(
                entry=parts,
                genestart=0,
                sample=chrom,
                ems_only=ems_only,
                base_counts={},
                genome_seq=genome_seq_str,
                context_counts={},
                context_size=7,
                gene_feature=None,
            )
            if muts:
                mutant_pos.add(pos_1based)
    return mutant_pos


def compute_trial_from_sites_and_mutations(
    sites_mpile: str,
    mutations_mpile: Optional[str],
    seq_context: SeqContext,
    min_depth: int,
    ems_only: bool,
) -> Tuple[int, int]:
    """Compute trial-level counts using sites for n and mutations list for x.

    If mutations_mpile is None, fall back to detecting mutations from sites mpileup directly.
    """
    mutant_positions: Set[int] = set()
    if mutations_mpile:
        mutant_positions = get_mutated_positions(mutations_mpile, seq_context, ems_only)

    x_total = 0
    n_total = 0
    genome_seq_str = str(seq_context.genome)

    with open_maybe_gzip(sites_mpile) as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            chrom = parts[0]
            pos_1based = int(parts[1])
            ref_base = parts[2]
            depth = int(parts[3])
            pos_0based = pos_1based - 1

            # Skip overlapping regions
            if pos_0based < 0 or pos_0based >= len(seq_context.overlap_mask):
                continue
            if not seq_context.overlap_mask[pos_0based]:
                continue
            # Depth filter
            if depth < min_depth:
                continue
            # EMS-only restriction for callable set
            if ems_only and ref_base not in ('G', 'C'):
                continue

            n_total += 1

            is_mut = False
            if mutant_positions:
                is_mut = pos_1based in mutant_positions
            else:
                # Fallback: detect from sites mpileup
                muts, _depth, _ = check_mutations(
                    entry=parts,
                    genestart=0,
                    sample=chrom,
                    ems_only=ems_only,
                    base_counts={},
                    genome_seq=genome_seq_str,
                    context_counts={},
                    context_size=7,
                    gene_feature=None,
                )
                is_mut = bool(muts)

            if is_mut:
                x_total += 1

    return x_total, n_total


def write_per_site_trial_from_sites_and_mutations(
    output_path: str,
    sites_mpile: str,
    mutations_mpile: Optional[str],
    seq_context: SeqContext,
    min_depth: int,
    ems_only: bool,
) -> None:
    """Write per-site trial rows using sites mpileup for n=1 and mutations list for x in {0,1}."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    mutant_positions: Set[int] = set()
    if mutations_mpile:
        mutant_positions = get_mutated_positions(mutations_mpile, seq_context, ems_only)

    with open(output_path, 'w') as out, open_maybe_gzip(sites_mpile) as handle:
        out.write('site\tx\tn\n')
        for line in handle:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            chrom = parts[0]
            pos_1based = int(parts[1])
            ref_base = parts[2]
            depth = int(parts[3])
            pos_0based = pos_1based - 1

            # Skip overlapping regions
            if pos_0based < 0 or pos_0based >= len(seq_context.overlap_mask):
                continue
            if not seq_context.overlap_mask[pos_0based]:
                continue
            # Depth filter
            if depth < min_depth:
                continue
            # EMS-only callable restriction
            if ems_only and ref_base not in ('G', 'C'):
                continue

            x_val = 1 if (mutations_mpile and (pos_1based in mutant_positions)) else 0
            out.write(f'{chrom}:{pos_1based}\t{x_val}\t1\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='Build ems_counts for ems_mutation_rate.py from mpileup(s) using parse module.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--mpileup', help='Path to a single mpileup file (optionally gzipped).')
    group.add_argument('--mpileup-dir', help='Directory containing per-sample mpileups. Expect <sample>_variants_mutations.txt and an unfiltered sites mpileup.')
    parser.add_argument('--fna', required=True, help='Path to reference FASTA (optionally gzipped).')
    parser.add_argument('--gff', required=True, help='Path to GFF annotation file.')
    parser.add_argument('--min-depth', type=int, default=10, help='Minimum depth to consider a site callable (default: 10).')
    parser.add_argument('--ems-only', action='store_true', help='Restrict to EMS transitions (C>T and G>A) and count only ref G/C sites for n.')
    parser.add_argument('--min-alt', type=int, default=3, help='Minimum alt-read count to call a site mutated in trial mode (default: 3).')
    parser.add_argument('--per-site', action='store_true', help='Write one row per callable site instead of aggregate counts.')
    parser.add_argument('--read-level', action='store_true', help='(Deprecated) Previously selected read-level output; now both outputs are always written.')
    parser.add_argument('--output', default='ems_counts.tsv', help='Output base path for single-sample mode (default: ems_counts.tsv).')
    parser.add_argument('--output-dir', help='Output directory for multi-sample mode. Creates <sample>.trial.tsv and <sample>.read.tsv per sample.')

    args = parser.parse_args()

    seq_ctx = SeqContext(fna=args.fna, annot=args.gff)

    if args.mpileup_dir:
        out_dir = args.output_dir or '.'
        os.makedirs(out_dir, exist_ok=True)
        pairs = find_sample_pairs(args.mpileup_dir)
        if not pairs:
            raise FileNotFoundError(f"No sample pairs found in {args.mpileup_dir} (need <sample>_variants_mutations.txt and a sites mpileup)")
        for sample, sites_mpile, mutations_mpile in pairs:
            if sites_mpile is None:
                print(f"Skipping {sample}: no sites mpileup found")
                continue
            base_out = os.path.join(out_dir, sample + '.tsv')
            trial_path, read_path = derive_output_paths(base_out)
            if args.per_site:
                # Trial per-site from sites+mutations
                write_per_site_trial_from_sites_and_mutations(
                    output_path=trial_path,
                    sites_mpile=sites_mpile,
                    mutations_mpile=mutations_mpile,
                    seq_context=seq_ctx,
                    min_depth=args.min_depth,
                    ems_only=args.ems_only,
                )
                # Read per-site from sites mpileup
                write_per_site_counts(
                    output_path=read_path,
                    mpileup_path=sites_mpile,
                    seq_context=seq_ctx,
                    min_depth=args.min_depth,
                    ems_only=args.ems_only,
                    read_level=True,
                    min_alt=args.min_alt,
                )
            else:
                # Trial aggregate from sites+mutations
                x_trial, n_trial = compute_trial_from_sites_and_mutations(
                    sites_mpile=sites_mpile,
                    mutations_mpile=mutations_mpile,
                    seq_context=seq_ctx,
                    min_depth=args.min_depth,
                    ems_only=args.ems_only,
                )
                write_counts(trial_path, x=x_trial, n=n_trial)

                # Read aggregate from sites mpileup
                x_read, n_read = compute_counts(
                    mpileup_path=sites_mpile,
                    seq_context=seq_ctx,
                    min_depth=args.min_depth,
                    ems_only=args.ems_only,
                    read_level=True,
                    min_alt=args.min_alt,
                )
                write_counts(read_path, x=x_read, n=n_read)
    else:
        # Single-sample mode (fallback to original behavior; single file is treated as sites mpileup)
        trial_path, read_path = derive_output_paths(args.output)
        if args.per_site:
            write_per_site_counts(
                output_path=trial_path,
                mpileup_path=args.mpileup,
                seq_context=seq_ctx,
                min_depth=args.min_depth,
                ems_only=args.ems_only,
                read_level=False,
                min_alt=args.min_alt,
            )
            write_per_site_counts(
                output_path=read_path,
                mpileup_path=args.mpileup,
                seq_context=seq_ctx,
                min_depth=args.min_depth,
                ems_only=args.ems_only,
                read_level=True,
                min_alt=args.min_alt,
            )
        else:
            x_trial, n_trial = compute_counts(
                mpileup_path=args.mpileup,
                seq_context=seq_ctx,
                min_depth=args.min_depth,
                ems_only=args.ems_only,
                read_level=False,
                min_alt=args.min_alt,
            )
            write_counts(trial_path, x=x_trial, n=n_trial)

            x_read, n_read = compute_counts(
                mpileup_path=args.mpileup,
                seq_context=seq_ctx,
                min_depth=args.min_depth,
                ems_only=args.ems_only,
                read_level=True,
                min_alt=args.min_alt,
            )
            write_counts(read_path, x=x_read, n=n_read)


if __name__ == '__main__':
    main() 