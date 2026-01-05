# Full Repository Documentation: ems_effect

## Overview

The `ems_effect` repository contains a comprehensive suite of tools for analyzing EMS (ethyl methanesulfonate) mutagenesis data, including mutation rate estimation, sequence bias modeling, PROVEAN effect score calculation, and various downstream analyses. The repository is designed to process mpileup files from sequencing data and perform statistical modeling of mutation patterns.

## Repository Structure

```
ems_effect/
├── config/
│   └── reference_paths.yaml          # Configuration file with paths to reference data
├── src/
│   ├── rate_modeling/                 # Mutation rate estimation and sequence bias modeling
│   ├── modules/                        # Core utility modules
│   ├── scripts/                        # Additional analysis scripts
│   ├── utils/                          # Utility functions
│   ├── analyze_provean.py             # PROVEAN score analysis
│   ├── provean_effectscore.py         # PROVEAN effect score calculation
│   ├── preprocess_mpileup.py          # Mpileup file preprocessing
│   ├── permutation_sim.py             # Permutation simulations
│   ├── plot_filter_comparison.py      # Filter comparison plotting
│   ├── plotting.py                    # General plotting utilities
│   └── random_mutation_analysis.py    # Random mutation analysis
├── environment.yml                     # Conda environment specification
└── README.md                           # Repository readme
```

## Dependencies

The repository uses a conda environment defined in `environment.yml`. Key dependencies include:

- **Python 3.12** with scientific computing stack:
  - `numpy`, `pandas`, `scipy`, `statsmodels` (statistical modeling)
  - `matplotlib`, `seaborn` (visualization)
  - `biopython`, `pysam` (genomic data processing)
  - `scikit-learn` (machine learning utilities)
  
- **R 4.3.3** with tidyverse and specialized packages:
  - `ggplot2`, `dplyr`, `tidyr` (data manipulation and visualization)
  - `circlize` (circular genome visualization)
  - Various other R packages for GO enrichment and pathway analysis

- **Additional tools**:
  - `bcbio-gff` (GFF parsing)
  - `bx-python` (genomic interval operations)
  - `loguru` (logging)

## Core Modules (`src/modules/`)

### `parse.py`
Provides the `SeqContext` class for handling genomic sequence context and annotations:
- Loads genome FASTA and GFF annotation files
- Creates overlap masks for genomic features
- Provides sequence context extraction utilities
- Used throughout the codebase for genomic coordinate operations

### `annotate.py`
Annotation utilities for genomic features and mutations.

### `translate.py`
Translation utilities for converting DNA sequences to protein sequences.

### `provean_db.py`
Database interface for PROVEAN scores, providing efficient lookup and storage of precomputed effect scores.

## Main Analysis Scripts

### Mutation Rate Estimation (`src/rate_modeling/`)

See `manuscript_scope_documentation.md` for detailed documentation of the rate modeling pipeline.

### PROVEAN Analysis

#### `provean_effectscore.py`
Main script for calculating PROVEAN effect scores for mutations:
- Processes mpileup files to identify mutations
- Calculates PROVEAN scores using external PROVEAN executable
- Stores results in SQLite database and JSON format
- Supports parallel processing
- Handles EMS-specific mutations (C>T and G>A)

**Key features:**
- Configurable via YAML configuration file
- Supports exclusion of specific mutation types
- Caches results to avoid redundant calculations
- Integrates with `modules/parse.py` for genomic context

#### `analyze_provean.py`
Analysis and visualization of PROVEAN scores:
- Generates summary statistics
- Creates visualizations comparing mutation effects
- Performs statistical tests on effect score distributions
- Supports upset plots for mutation set intersections

### Data Preprocessing

#### `preprocess_mpileup.py`
Preprocesses mpileup files with quality filtering:
- Parses mpileup format to extract read information
- Applies depth-based filtering (percentile-based)
- Filters sites based on majority reference allele
- Analyzes read position bias using Kolmogorov-Smirnov tests
- Removes sites appearing in multiple control files
- Outputs filtered mutation counts

**Key filtering steps:**
1. Depth percentile filtering (removes top/bottom 10%)
2. Majority-reference filtering
3. Control-based site exclusion
4. Read position bias detection

### Additional Analysis Scripts

#### `permutation_sim.py`
Permutation-based statistical testing for mutation patterns.

#### `plot_filter_comparison.py`
Compares different filtering strategies and their effects on results.

#### `plotting.py`
General plotting utilities used across multiple scripts.

#### `random_mutation_analysis.py`
Analysis of random mutation patterns and null models.

## Utility Scripts (`src/scripts/`)

The `scripts/` directory contains various specialized analysis and data processing scripts:

- **Data processing:**
  - `merge_fastq_replicates.py` - Merges FASTQ replicate files
  - `rename_fastq_files.py` - Batch renaming of FASTQ files
  - `pull_raw_reads.py` - Extracts raw reads from BAM files

- **Mpileup analysis:**
  - `analyze_filtered_sites.py` - Analyzes sites filtered by preprocessing
  - `audit_mpileup_filters.py` - Audits filtering decisions
  - `analyze_ct_ga_discrepancy.py` - Analyzes C>T vs G>A mutation discrepancies
  - `analyze_rca_counts.py` - Analyzes read count artifacts

- **PROVEAN database:**
  - `check_provean_db_schema.py` - Validates PROVEAN database schema
  - `clean_provean_db.py` - Cleans and maintains PROVEAN database
  - `gene_to_protein_converter.py` - Converts gene IDs to protein sequences

- **Genomic analysis:**
  - `get_reference_kmer_counts.py` - Counts k-mers in reference genome
  - `extract_wd_ids.py` - Extracts Wolbachia gene IDs
  - `ns_immune_codons.py` - Analyzes non-synonymous immune codons

- **Pathway and GO analysis (R scripts):**
  - `kegg_analysis.r` / `kegg_analysis_uniprot.r` - KEGG pathway analysis
  - `wolbachia_go_enrichment.r` - GO enrichment analysis
  - `build_wolbachia_go_db.r` - Builds GO annotation database
  - `find_wolbachia_go.r` - Finds GO terms for Wolbachia genes
  - `circlized_genome_viz.r` - Circular genome visualization

- **Data integration:**
  - `json_gid_to_uniprot.py` - Maps gene IDs to UniProt IDs
  - `collect_supplemental_tables.py` - Collects data for supplemental tables
  - `diff_nuc_muts_dirs.py` - Compares mutation directories

## Utility Functions (`src/utils/`)

- `analyze_gff_overlaps.py` - Analyzes GFF feature overlaps
- `benchmark_provean.py` - Benchmarks PROVEAN performance
- `compare_bams.py` - Compares BAM files
- `compare_mutations.py` - Compares mutation sets
- `convert_score_table.py` - Converts score tables between formats
- `validate_json.py` - Validates JSON files

## Configuration

### `config/reference_paths.yaml`

Central configuration file specifying paths to:
- Reference genome FASTA file
- GFF annotation file
- Codon table (JSON format)
- PROVEAN database paths (SQLite and JSON)
- Gene information cache
- Transcriptomic data (module assignments, expression data)
- PROVEAN executable and data directories

## Data Flow

### Typical Workflow

1. **Data Input**: Mpileup files from sequencing data
2. **Preprocessing**: `preprocess_mpileup.py` filters and processes mpileup files
3. **Mutation Rate Estimation**: `rate_modeling/estimate_rates.py` calculates mutation rates
4. **Sequence Bias Modeling**: `rate_modeling/sequence_bias_modeling_sitelevel.py` models sequence context effects
5. **PROVEAN Scoring**: `provean_effectscore.py` calculates effect scores
6. **Analysis**: Various analysis scripts generate figures and statistics
7. **Visualization**: Plotting scripts create publication-ready figures

### Data Formats

- **Mpileup files**: Standard SAMtools mpileup format
- **Count files**: TSV format with columns for chromosome, position, depth, mutation counts
- **Rate files**: TSV format with per-sample or per-site mutation rates
- **Model files**: Pickle format for saved GLM models
- **JSON files**: Configuration and summary data

## Key Features

1. **EMS-specific analysis**: Focused on C>T and G>A mutations at G/C sites
2. **Statistical rigor**: Multiple estimation methods (simple rates, alpha correction, GLM)
3. **Sequence context modeling**: K-mer based models (3mer, 5mer, positional)
4. **Quality control**: Comprehensive filtering and bias detection
5. **Parallel processing**: Multiprocessing support for computationally intensive tasks
6. **Reproducibility**: Conda environment ensures consistent dependencies

## Usage Examples

### Setting up the environment

```bash
conda env create -f environment.yml
conda activate ems_effect
```

### Preprocessing mpileup files

```bash
python src/preprocess_mpileup.py \
    --mpileups /path/to/mpileups/*.mpileup \
    --output /path/to/output \
    --genome-fasta /path/to/genome.fna \
    --gff /path/to/annotation.gff
```

### Estimating mutation rates

```bash
python src/rate_modeling/estimate_rates.py \
    --counts-dir /path/to/counts \
    --output-dir /path/to/output \
    --genome-fasta /path/to/genome.fna \
    --gff /path/to/annotation.gff
```

### Calculating PROVEAN scores

```bash
python src/provean_effectscore.py \
    -m /path/to/mpileups \
    -o /path/to/output \
    -c config/reference_paths.yaml
```

## Notes

- The repository is designed for analysis of Wolbachia (wMel) genome data
- Most scripts support both control (NT) and treated (EMS) samples
- Time point analysis (3d, 7d) is supported where applicable
- The codebase uses 1-based genomic coordinates (GFF standard)
- EMS mutations are canonicalized: G>A mutations are reverse-complemented to C>T for analysis

