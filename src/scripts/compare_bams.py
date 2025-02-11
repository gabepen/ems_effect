
import pysam
import argparse
from typing import Dict, Tuple, List
from collections import Counter
import sys
from pathlib import Path

def get_alignment_stats(bam_path: str) -> Tuple[Dict, Counter, int]:
    """Get alignment statistics from BAM file.
    
    Args:
        bam_path (str): Path to sorted BAM file
        
    Returns:
        Tuple containing:
            - Dict: Mapping of read names to positions
            - Counter: Flag statistics
            - int: Total reads
    """
    alignments = {}
    flags = Counter()
    total_reads = 0
    
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam:
            total_reads += 1
            flags[read.flag] += 1
            
            if not read.is_unmapped:
                alignments[read.query_name] = {
                    'pos': (read.reference_name, read.reference_start),
                    'flag': read.flag,
                    'mapq': read.mapping_quality,
                    'cigar': read.cigarstring
                }
    
    return alignments, flags, total_reads

def compare_bams(bam1: str, bam2: str, output_dir: str) -> None:
    """Compare two sorted BAM files and output statistics.
    
    Args:
        bam1 (str): Path to first BAM file
        bam2 (str): Path to second BAM file
        output_dir (str): Directory for output files
    """
    # Get stats from both BAMs
    print("Processing first BAM file...")
    alns1, flags1, total1 = get_alignment_stats(bam1)
    print("Processing second BAM file...")
    alns2, flags2, total2 = get_alignment_stats(bam2)
    
    # Compare read sets
    reads1 = set(alns1.keys())
    reads2 = set(alns2.keys())
    shared_reads = reads1 & reads2
    
    # Compare alignments for shared reads
    position_diffs = []
    mapq_diffs = []
    flag_diffs = []
    cigar_diffs = []
    
    for read in shared_reads:
        aln1 = alns1[read]
        aln2 = alns2[read]
        
        if aln1['pos'] != aln2['pos']:
            position_diffs.append(read)
        if aln1['mapq'] != aln2['mapq']:
            mapq_diffs.append(read)
        if aln1['flag'] != aln2['flag']:
            flag_diffs.append(read)
        if aln1['cigar'] != aln2['cigar']:
            cigar_diffs.append(read)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Write summary report
    with open(Path(output_dir) / "bam_comparison_summary.txt", 'w') as f:
        f.write(f"BAM Comparison Summary\n")
        f.write(f"====================\n\n")
        
        f.write(f"BAM 1: {bam1}\n")
        f.write(f"BAM 2: {bam2}\n\n")
        
        f.write(f"Read Statistics:\n")
        f.write(f"---------------\n")
        f.write(f"Total reads in BAM 1: {total1:,}\n")
        f.write(f"Total reads in BAM 2: {total2:,}\n")
        f.write(f"Mapped reads in BAM 1: {len(reads1):,}\n")
        f.write(f"Mapped reads in BAM 2: {len(reads2):,}\n")
        f.write(f"Shared reads: {len(shared_reads):,}\n\n")
        
        f.write(f"Differences in Shared Reads:\n")
        f.write(f"--------------------------\n")
        f.write(f"Different positions: {len(position_diffs):,}\n")
        f.write(f"Different mapping qualities: {len(mapq_diffs):,}\n")
        f.write(f"Different flags: {len(flag_diffs):,}\n")
        f.write(f"Different CIGAR strings: {len(cigar_diffs):,}\n\n")
        
        f.write(f"Flag Statistics:\n")
        f.write(f"---------------\n")
        f.write(f"{'Flag':<8} {'BAM 1':<12} {'BAM 2':<12} {'Description':<40}\n")
        all_flags = sorted(set(flags1.keys()) | set(flags2.keys()))
        for flag in all_flags:
            desc = get_flag_description(flag)
            f.write(f"{flag:<8} {flags1[flag]:<12} {flags2[flag]:<12} {desc:<40}\n")
    
    # Write detailed differences if they exist
    if position_diffs:
        with open(Path(output_dir) / "position_differences.txt", 'w') as f:
            f.write(f"{'Read Name':<40} {'BAM 1 Position':<30} {'BAM 2 Position':<30}\n")
            for read in position_diffs:
                pos1 = f"{alns1[read]['pos'][0]}:{alns1[read]['pos'][1]}"
                pos2 = f"{alns2[read]['pos'][0]}:{alns2[read]['pos'][1]}"
                f.write(f"{read:<40} {pos1:<30} {pos2:<30}\n")

def get_flag_description(flag: int) -> str:
    """Get human-readable description of SAM flag."""
    descriptions = {
        0x1: "Paired",
        0x2: "Proper pair",
        0x4: "Unmapped",
        0x8: "Mate unmapped",
        0x10: "Reverse strand",
        0x20: "Mate reverse strand",
        0x40: "First in pair",
        0x80: "Second in pair",
        0x100: "Secondary alignment",
        0x200: "Failed QC",
        0x400: "PCR/Optical duplicate",
        0x800: "Supplementary alignment"
    }
    
    active_flags = []
    for bit, desc in descriptions.items():
        if flag & bit:
            active_flags.append(desc)
    
    return "; ".join(active_flags) if active_flags else "None"

def main():
    parser = argparse.ArgumentParser(description='Compare two sorted BAM files')
    parser.add_argument('-1', '--bam1', required=True,
                      help='Path to first BAM file')
    parser.add_argument('-2', '--bam2', required=True,
                      help='Path to second BAM file')
    parser.add_argument('-o', '--output', required=True,
                      help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.bam1).exists():
        sys.exit(f"Error: BAM file not found: {args.bam1}")
    if not Path(args.bam2).exists():
        sys.exit(f"Error: BAM file not found: {args.bam2}")
    
    compare_bams(args.bam1, args.bam2, args.output)

if __name__ == '__main__':
    main() 