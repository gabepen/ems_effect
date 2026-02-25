# Manuscript Scope Documentation: Rate Modeling Analysis

## Overview

This documentation covers the analysis pipeline used in the manuscript, which focuses on mutation rate estimation and sequence bias modeling for EMS-mutagenized Wolbachia (wMel). The analysis is contained within the `src/rate_modeling/` directory.

## Directory Structure

```
src/
├── rate_modeling/
│   ├── collect_mutation_counts.py          # Parses mpileup files into .counts, generates exclusion mask
│   ├── collect_5mer_contexts.py            # Collects 5mer context counts
│   ├── estimate_rates.py                   # Main mutation rate estimation script
│   ├── sequence_bias_modeling_sitelevel.py # Sequence bias GLM modeling
│   ├── plot_5mer_mutation_rates.py         # Visualization of 5mer mutation rates
│   ├── plot_ems_spectra.py                 # Mutation spectra visualization
│   ├── regenerate_rate_plots.py            # Publication-ready rate plots
│   ├── correlate_rates_with_expression.py  # Expression-mutation rate correlation
│   ├── combine_model_evaluation_metrics.py # Combines evaluation metrics across models
│   ├── prediction_accuracy_metrics.py      # Model prediction accuracy evaluation
│   ├── cross_validation_evaluation.py      # Cross-validation framework
│   ├── residual_analysis.py                # Model residual diagnostics
│   ├── load_saved_models.py                # Utility to load saved GLM models
│   └── plot_sequence_bias_existing.py      # Sequence bias visualization
├── scripts/
│   └── collect_supplemental_tables.py      # Generates per-sample and per-gene supplemental tables
└── modules/
    └── parse.py                            # Genomic sequence context and GFF parsing utilities
```

## Analysis Pipeline

### Step 1: Data Collection and Filtering

#### `collect_mutation_counts.py`
**Purpose**: Parses raw mpileup variant files into per-site mutation count files, and optionally generates an exclusion mask from control samples.

**Inputs**:
- `--input-dir`: Directory containing `*_variants.txt` mpileup files
- `--output-dir`: Output directory for `.counts` files

**Outputs**:
- Per-sample `.counts` files (TSV format)
  - Columns: `chrom`, `pos`, `ref`, `ref_count`, `A_count`, `C_count`, `G_count`, `T_count`, `depth`
  - Includes all C/G sites passing filters (not just sites with mutations)
- Exclusion mask file (optional, via `--generate-exclusion-mask`)

**Filters applied during `.counts` generation**:
- Majority-ref: sites where non-reference reads exceed 50% of depth are removed
- Depth percentile: sites outside the 10th-90th percentile of per-sample depth are removed
- Read position bias (optional, `--position-bias`): KS test on alt allele read positions

**Exclusion mask generation** (`--generate-exclusion-mask`):
- Runs as a post-processing step after `.counts` files are written
- Identifies C/G sites with EMS-signature variants (any alt count) in >1 control sample
- Outputs `exclusion_mask.tsv` (`chrom\tpos` format)
- This mask is consumed by downstream scripts (`estimate_rates.py`, `collect_supplemental_tables.py`), not applied during `.counts` generation itself

**Usage**:
```bash
python src/rate_modeling/collect_mutation_counts.py \
    --input-dir /path/to/variants \
    --output-dir /path/to/counts \
    --generate-exclusion-mask \
    [--position-bias]
```

#### `collect_5mer_contexts.py`
**Purpose**: Collects 5mer context counts from processed count files.

**Inputs**:
- Directory of `.counts` files (from `collect_mutation_counts.py`)
- Reference genome FASTA
- GFF annotation file
- Optional exclusion mask

**Outputs**:
- Per-sample 5mer context counts (JSON format)
  - Total counts, gene counts, intergenic counts
  - Canonicalized to C-centered 5mers (G>A mutations reverse-complemented)

**Usage**:
```bash
python src/rate_modeling/collect_5mer_contexts.py \
    --counts-dir /path/to/counts \
    --genome-fasta /path/to/genome.fna \
    --gff /path/to/annotation.gff \
    --output-dir /path/to/output \
    [--exclusion-mask /path/to/mask.tsv]
```

### Step 2: Mutation Rate Estimation

#### `estimate_rates.py`
**Purpose**: Estimates mutation rates using multiple statistical methods.

**Inputs**:
- Directory of `.counts` files
- Reference genome FASTA
- GFF annotation file
- Optional exclusion mask
- Optional 5mer model file (for rate predictions)

**Outputs**:
- Per-sample mutation rates (TSV)
- Site-level GLM rates per sample (TSV)
- GLM-based rate estimates with confidence intervals
- Category-level rate comparisons (intergenic, synonymous, non-synonymous)
- Gene-based window analysis
- Publication-ready plots

**Estimation Methods**:

1. **Simple Rates**:
   - Low estimate: mutated positions / total depth
   - High estimate: total alt alleles / total depth

2. **Alpha Correction**:
   - Estimates background false positive rate from controls
   - Adjusts treated sample rates accordingly

3. **GLM Analysis**:
   - Negative Binomial regression
   - Models mutation counts with log(depth) offset
   - Includes treatment covariate (0 for controls, 1 for treated)
   - Category dummy variables (intergenic as reference, synonymous, non-synonymous)
   - Treatment x category interactions
   - Confidence intervals via asymptotic normal approximation

4. **Coverage-Dependent Analysis**:
   - Rate estimation across coverage bins
   - Identifies coverage-dependent biases

5. **5mer Model Predictions** (if model provided):
   - Predicts expected rates based on sequence context
   - Compares observed vs expected rates

**Usage**:
```bash
python src/rate_modeling/estimate_rates.py \
    --counts-dir /path/to/counts \
    --output-dir /path/to/output \
    --genome-fasta /path/to/genome.fna \
    --gff /path/to/annotation.gff \
    [--exclusion-mask /path/to/mask.tsv] \
    [--kmer5-model-path /path/to/5mer_model.pkl]
```

### Step 3: Sequence Bias Modeling

#### `sequence_bias_modeling_sitelevel.py`
**Purpose**: Models sequence context effects on mutation rates using GLMs.

**Inputs**:
- Directory of `.counts` files
- Reference genome FASTA
- Optional exclusion mask

**Outputs**:
- Fitted GLM models (pickle format)
- Model comparison statistics (AIC, BIC)
- Model summary reports
- Feature importance analysis

**Model Types**:

1. **Positional Model** (15 features): one-hot encoding of position within 5mer context
2. **3mer Model** (64 features): one-hot encoding of all 3mer contexts
3. **5mer Model** (1024 features): one-hot encoding of all 5mer contexts (canonicalized, C-centered)
4. **Positional-3mer Model**: combines positional and 3mer features

**Model Specification**:
- Response: EMS mutation counts
- Offset: log(depth)
- Covariates: Treatment (0/1) + sequence features
- Family: Poisson or Negative Binomial
- Link: Log

**Usage**:
```bash
python src/rate_modeling/sequence_bias_modeling_sitelevel.py \
    --counts-dir /path/to/counts \
    --genome-fasta /path/to/genome.fna \
    --output-dir /path/to/output \
    [--exclusion-mask /path/to/mask.tsv] \
    [--glm-family poisson|negative_binomial]
```

### Step 4: Model Evaluation

#### `cross_validation_evaluation.py`
**Purpose**: Performs k-fold cross-validation to assess model generalization.

**Metrics**: Deviance, Pseudo-R^2, MSE, MAE, Pearson/Spearman correlation.

**Usage**:
```bash
python src/rate_modeling/cross_validation_evaluation.py \
    --counts-dir /path/to/counts \
    --genome-fasta /path/to/genome.fna \
    --output-dir /path/to/output \
    --model-type 5mer \
    --n-folds 5
```

#### `prediction_accuracy_metrics.py`
**Purpose**: Computes prediction accuracy metrics including MSE, RMSE, MAE, pseudo-R^2 variants (McFadden's, Cox-Snell, Nagelkerke), correlation coefficients, and Poisson-specific metrics.

#### `residual_analysis.py`
**Purpose**: Diagnoses model fit through residual analysis (raw, Pearson, deviance, standardized residuals), overdispersion tests, and outlier detection.

#### `combine_model_evaluation_metrics.py`
**Purpose**: Combines evaluation metrics across multiple model runs for comparison.

### Step 5: Visualization

#### `plot_5mer_mutation_rates.py`
Heatmaps of 5mer mutation rates, sequence context effect plots, model prediction vs observed comparisons.

#### `plot_ems_spectra.py`
Mutation spectra bar plots (all substitution types) with statistical comparisons (Mann-Whitney U tests), grouped by control vs treated and time points.

#### `regenerate_rate_plots.py`
Publication-ready multi-panel figures: GLM rates per sample, mutation category significances, rates per treatment time group.

#### `plot_sequence_bias_existing.py`
Generates plots from existing sequence bias model summaries.

### Step 6: Expression Correlation Analysis

#### `correlate_rates_with_expression.py`
**Purpose**: Tests correlation between gene mutation rates and expression levels (transcription-coupled mutagenesis hypothesis).

**Usage**:
```bash
python src/rate_modeling/correlate_rates_with_expression.py \
    --mutation-rates /path/to/gene_rates.tsv \
    --expression /path/to/expression.tsv \
    --output-dir /path/to/output \
    [--module-assignments /path/to/modules.tsv]
```

### Step 7: Supplemental Tables

#### `collect_supplemental_tables.py` (in `src/scripts/`)
**Purpose**: Generates per-sample and per-gene supplemental mutation tables.

**Inputs**:
- Directory of `.counts` files
- GFF annotation file
- Reference genome FASTA
- Optional exclusion mask
- Optional codon table (for non-standard genetic codes)
- Optional output directory from `estimate_rates.py` (to incorporate GLM-estimated rates)

**Outputs**:
- `per_sample_mutation_table.tsv`: total, synonymous, and non-synonymous mutation counts and rates per sample
- `per_gene_mutation_table.tsv`: same breakdown per protein-coding gene, identified by NCBI GeneID

**Key behavior**:
- Restricted to protein-coding genes (tRNA, rRNA genes excluded)
- Sites classified as intergenic, synonymous, or non-synonymous based on CDS annotation and codon context
- Exclusion mask applied during site loading (counts are post-filtering)

**Usage**:
```bash
python src/scripts/collect_supplemental_tables.py \
    --counts-dir /path/to/counts \
    --gff-file /path/to/annotation.gff \
    --genome-fasta /path/to/genome.fna \
    --output-dir /path/to/output \
    [--exclusion-mask /path/to/mask.tsv] \
    [--codon-table /path/to/codon_table.json] \
    [--estimate-rates-output-dir /path/to/rates_output]
```

### Utility Scripts

#### `load_saved_models.py`
Loads and inspects saved GLM model files (pickle format).

## Typical Workflow

1. **Collect mutation counts and generate exclusion mask**:
   ```bash
   python src/rate_modeling/collect_mutation_counts.py \
       --input-dir /path/to/variants \
       --output-dir /path/to/counts \
       --generate-exclusion-mask
   ```

2. **Estimate mutation rates**:
   ```bash
   python src/rate_modeling/estimate_rates.py \
       --counts-dir /path/to/counts \
       --output-dir /path/to/rates \
       --genome-fasta /path/to/genome.fna \
       --gff /path/to/annotation.gff \
       --exclusion-mask /path/to/counts/exclusion_mask.tsv
   ```

3. **Model sequence bias**:
   ```bash
   python src/rate_modeling/sequence_bias_modeling_sitelevel.py \
       --counts-dir /path/to/counts \
       --genome-fasta /path/to/genome.fna \
       --output-dir /path/to/models \
       --exclusion-mask /path/to/counts/exclusion_mask.tsv
   ```

4. **Evaluate models**:
   ```bash
   python src/rate_modeling/cross_validation_evaluation.py \
       --counts-dir /path/to/counts \
       --genome-fasta /path/to/genome.fna \
       --output-dir /path/to/eval \
       --model-type 5mer --n-folds 5
   ```

5. **Generate visualizations**:
   ```bash
   python src/rate_modeling/regenerate_rate_plots.py \
       --output-dir /path/to/rates \
       --figure-output /path/to/figure.png
   ```

6. **Generate supplemental tables**:
   ```bash
   python src/scripts/collect_supplemental_tables.py \
       --counts-dir /path/to/counts \
       --gff-file /path/to/annotation.gff \
       --genome-fasta /path/to/genome.fna \
       --exclusion-mask /path/to/counts/exclusion_mask.tsv \
       --output-dir /path/to/supplemental \
       --estimate-rates-output-dir /path/to/rates
   ```

7. **Correlate with expression** (optional):
   ```bash
   python src/rate_modeling/correlate_rates_with_expression.py \
       --mutation-rates /path/to/gene_rates.tsv \
       --expression /path/to/expression.tsv \
       --output-dir /path/to/output
   ```

## Data Dependencies

### Required Input Files

1. **Variant files**: `*_variants.txt` mpileup-format files from variant calling pipeline
2. **Reference genome FASTA**: Genome sequence file (supports gzipped)
3. **GFF annotation file**: NCBI-format gene annotations (used for genic/intergenic and syn/non-syn classification)
4. **Expression data** (for correlation analysis): TSV with gene IDs and TPM values

### Intermediate Data Files

1. **`.counts` files**: Per-site allele counts (generated by `collect_mutation_counts.py`)
   - Format: `chrom  pos  ref  ref_count  A_count  C_count  G_count  T_count  depth`
2. **Exclusion mask**: Sites to exclude, generated from control samples (`chrom  pos`)
3. **5mer context JSON files**: Per-sample 5mer context counts
4. **Rate TSV files**: Per-sample and per-gene mutation rates
5. **Model pickle files**: Fitted GLM models

## Configuration

Paths to reference files can be specified via:
1. Command-line arguments (takes precedence)
2. `config/reference_paths.yaml` file

Key configuration paths:
- `references.genomic_fna`: Reference genome FASTA
- `references.annotation`: GFF annotation file
- `references.expression_data`: Expression data file
- `references.module_assignments`: Gene module assignments

## Key Assumptions and Design Decisions

1. **EMS-specific mutations**: Only C>T and G>A transitions at C/G sites
2. **Strand symmetry**: C>T and G>A treated equivalently via canonicalization to C-centered 5mers
3. **Depth offset**: Mutation rates modeled per unit sequencing depth
4. **Control-based exclusion**: Sites with EMS-signature variants in >1 control sample are excluded
5. **Depth percentile filtering**: Top and bottom 10% of per-sample depth distribution removed
6. **Majority-ref filtering**: Sites where >50% of reads are non-reference are removed
7. **1-based coordinates**: GFF standard throughout
8. **Protein-coding genes only**: Per-gene supplemental table restricted to protein-coding genes; tRNA/rRNA sites classified as intergenic in rate analyses
