# Manuscript Scope Documentation: Rate Modeling Analysis

## Overview

This documentation covers the analysis pipeline used in the manuscript, which focuses on mutation rate estimation and sequence bias modeling. The analysis is contained within the `src/rate_modeling/` directory and its dependencies.

## Scope

The manuscript analysis includes:
- **Primary scripts**: All scripts in `src/rate_modeling/`
- **Dependencies**: Scripts and modules used by rate modeling scripts
- **Data preprocessing**: Scripts that prepare data for rate modeling analysis

## Directory Structure

```
src/rate_modeling/
├── collect_mutation_counts.py          # Collects mutation counts from mpileup files
├── collect_5mer_contexts.py             # Collects 5mer context counts
├── estimate_rates.py                   # Main mutation rate estimation script
├── sequence_bias_modeling_sitelevel.py # Sequence bias GLM modeling
├── plot_5mer_mutation_rates.py         # Visualization of 5mer mutation rates
├── plot_ems_spectra.py                 # Mutation spectra visualization
├── regenerate_rate_plots.py            # Publication-ready rate plots
├── correlate_rates_with_expression.py  # Expression-mutation rate correlation
├── prediction_accuracy_metrics.py     # Model prediction accuracy evaluation
├── cross_validation_evaluation.py     # Cross-validation framework
├── residual_analysis.py               # Model residual diagnostics
├── load_saved_models.py               # Utility to load saved GLM models
└── plot_sequence_bias_existing.py     # Sequence bias visualization
```

## Dependencies

### Required Modules

- **`src/modules/parse.py`**: Used by `preprocess_mpileup.py` (via `SeqContext` class)
  - Provides genomic sequence context extraction
  - Handles GFF annotation parsing
  - Creates overlap masks for genomic features

### Data Preprocessing

- **`src/preprocess_mpileup.py`**: Preprocesses mpileup files before rate modeling
  - Filters sites based on depth percentiles
  - Applies majority-reference filtering
  - Removes sites appearing in control files
  - Analyzes read position bias
  - Outputs `.counts` files used by rate modeling scripts

## Analysis Pipeline

### Step 1: Data Collection

#### `collect_mutation_counts.py`
**Purpose**: Collects mutation counts from mpileup files with quality filtering.

**Inputs**:
- Directory of mpileup files
- Reference genome FASTA
- Optional exclusion mask file

**Outputs**:
- Per-sample `.counts` files (TSV format)
  - Columns: `chrom`, `pos`, `ref`, `depth`, `alt_count`, `alt_base`
  - Only includes EMS mutations (C>T at C sites, G>A at G sites)

**Key features**:
- Parses mpileup format, handling read markers (^, $, +, -)
- Filters sites based on:
  - Depth percentile (removes top/bottom 10%)
  - Majority reference allele requirement
  - Control file exclusion (sites in >1 control are removed)
- Read position bias analysis using Kolmogorov-Smirnov test
- Multiprocessing support for parallel processing

**Usage**:
```bash
python src/rate_modeling/collect_mutation_counts.py \
    --mpileup-dir /path/to/mpileups \
    --genome-fasta /path/to/genome.fna \
    --output-dir /path/to/output \
    [--exclusion-mask /path/to/mask.tsv]
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

**Key features**:
- Strand-collapsed analysis (C>T and G>A treated equivalently)
- Separates gene vs intergenic contexts
- Efficient processing using pre-computed count files

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
**Purpose**: Main script for estimating mutation rates using multiple statistical methods.

**Inputs**:
- Directory of `.counts` files
- Reference genome FASTA
- GFF annotation file
- Optional exclusion mask
- Optional 5mer model file (for rate predictions)

**Outputs**:
- Per-sample mutation rates (TSV)
- Site-level mutation rates (TSV)
- GLM-based rate estimates with confidence intervals
- Coverage-dependent rate analysis
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
   - Poisson or Negative Binomial regression
   - Models mutation counts with log(depth) offset
   - Includes treatment covariate (0 for controls, 1 for treated)
   - Provides uncertainty quantification via confidence intervals

4. **Coverage-Dependent Analysis**:
   - Rate estimation across coverage bins
   - Identifies coverage-dependent biases

5. **5mer Model Predictions** (if model provided):
   - Predicts expected rates based on sequence context
   - Compares observed vs expected rates

**Key features**:
- Sample name simplification (extracts EMS/NT group numbers and time points)
- Treatment day extraction (3d, 7d, control)
- Gene-based windowing for genic vs intergenic analysis
- Synonymous vs non-synonymous mutation classification
- Comprehensive ranking and comparison plots

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

1. **Positional Model** (15 features):
   - One-hot encoding of position within 5mer context
   - Captures position-specific effects

2. **3mer Model** (64 features):
   - One-hot encoding of all 3mer contexts
   - Captures local sequence context

3. **5mer Model** (1024 features):
   - One-hot encoding of all 5mer contexts
   - Most detailed sequence context model
   - Uses canonicalization (C-centered)

4. **Positional-3mer Model**:
   - Combines positional and 3mer features
   - Captures both position and local context effects

**Model Specification**:
- Response: EMS mutation counts (y)
- Offset: log(depth)
- Covariates: Treatment (0/1) + sequence features
- Family: Poisson or Negative Binomial
- Link: Log

**Key features**:
- Canonicalization: G-centered sites reverse-complemented to C-centered
- Strict EMS counting: at C sites count T only; at G sites count A only
- Model selection via AIC/BIC
- Thread control for parallel processing

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

**Inputs**:
- Directory of `.counts` files
- Reference genome FASTA
- Model type specification

**Outputs**:
- Cross-validation metrics (mean ± std across folds)
- Out-of-sample prediction accuracy
- Per-fold model performance

**Metrics**:
- Deviance (Poisson or Negative Binomial)
- Pseudo-R²
- Mean squared error (MSE)
- Mean absolute error (MAE)
- Correlation (Pearson, Spearman)

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
**Purpose**: Computes comprehensive prediction accuracy metrics on test data.

**Inputs**:
- Observed mutation counts
- Predicted mutation counts (from fitted model)
- Optional: model object for additional metrics

**Outputs**:
- MSE, RMSE, MAE
- Pseudo-R² (McFadden's, Cox-Snell, Nagelkerke)
- Correlation coefficients (Pearson, Spearman)
- Poisson-specific metrics (deviance, dispersion)
- Visualization plots

**Usage**:
```bash
python src/rate_modeling/prediction_accuracy_metrics.py \
    --observed /path/to/observed.tsv \
    --predicted /path/to/predicted.tsv \
    --output-dir /path/to/output
```

#### `residual_analysis.py`
**Purpose**: Diagnoses model fit issues through residual analysis.

**Inputs**:
- Fitted model object
- Design matrix (features)
- Observed counts
- Offset values

**Outputs**:
- Residual plots (raw, Pearson, deviance, standardized)
- Overdispersion tests
- Outlier detection
- Diagnostic summary

**Residual Types**:
- Raw residuals: observed - predicted
- Pearson residuals: (observed - predicted) / sqrt(variance)
- Deviance residuals: signed sqrt of deviance contribution
- Standardized residuals: z-scores

**Usage**:
```bash
python src/rate_modeling/residual_analysis.py \
    --model /path/to/model.pkl \
    --data /path/to/data.tsv \
    --output-dir /path/to/output
```

### Step 5: Visualization

#### `plot_5mer_mutation_rates.py`
**Purpose**: Visualizes 5mer-specific mutation rates and sequence bias patterns.

**Inputs**:
- Directory of `.counts` files
- Optional: fitted 5mer model
- Reference genome FASTA

**Outputs**:
- Heatmaps of 5mer mutation rates
- Sequence context effect plots
- Model prediction vs observed comparisons
- Clustering analysis of 5mer contexts

**Key features**:
- Canonicalized 5mer representation
- Sample comparison (control vs treated)
- Publication-ready figure generation

**Usage**:
```bash
python src/rate_modeling/plot_5mer_mutation_rates.py \
    --counts-dir /path/to/counts \
    --genome-fasta /path/to/genome.fna \
    --output-dir /path/to/output \
    [--model-dir /path/to/models]
```

#### `plot_ems_spectra.py`
**Purpose**: Plots mutation spectra (all substitution types) for visualization.

**Inputs**:
- Directory of `.counts` files or TSV rate files
- Minimum alt count threshold
- Minimum depth threshold

**Outputs**:
- Mutation spectra bar plots
- Statistical comparisons (Mann-Whitney U tests)
- Sample grouping (control vs treated, time points)

**Usage**:
```bash
python src/rate_modeling/plot_ems_spectra.py \
    /path/to/counts_dir \
    --output output.png \
    --min-alt 3 \
    --min-depth 10
```

#### `regenerate_rate_plots.py`
**Purpose**: Generates publication-ready multi-panel figures from rate estimation results.

**Inputs**:
- Output directory from `estimate_rates.py` (contains TSV files)

**Outputs**:
- Multi-panel publication figures:
  1. GLM rates per sample (bar plot with error bars)
  2. Mutation category significances (intergenic vs genic [syn/nonsyn])
  3. Rates per treatment time group

**Key features**:
- Publication-quality styling
- Statistical significance annotations
- Color schemes matching other figures
- Flexible figure layout

**Usage**:
```bash
python src/rate_modeling/regenerate_rate_plots.py \
    --output-dir /path/to/estimate_rates_output \
    --figure-output /path/to/figure.png
```

#### `plot_sequence_bias_existing.py`
**Purpose**: Generates plots from existing sequence bias model summaries.

**Inputs**:
- Model summary files or model directory

**Outputs**:
- Sequence bias visualization plots

**Usage**:
```bash
python src/rate_modeling/plot_sequence_bias_existing.py \
    --model-dir /path/to/models \
    --output-dir /path/to/output
```

### Step 6: Expression Correlation Analysis

#### `correlate_rates_with_expression.py`
**Purpose**: Tests correlation between gene mutation rates and expression levels.

**Inputs**:
- Gene-level mutation rates (from `estimate_rates.py`)
- Expression data (TPM values, TSV format)
- Optional: module assignments

**Outputs**:
- Correlation statistics (Pearson, Spearman)
- Linear regression results
- Scatter plots (rates vs expression)
- Outlier analysis
- Publication-ready figures

**Analysis**:
- Tests hypothesis: higher expression → higher mutation rates (transcription-coupled)
- Handles log transformation of expression data
- Performs outlier detection and influence analysis
- Module-specific analysis (if module assignments provided)

**Usage**:
```bash
python src/rate_modeling/correlate_rates_with_expression.py \
    --mutation-rates /path/to/gene_rates.tsv \
    --expression /path/to/expression.tsv \
    --output-dir /path/to/output \
    [--module-assignments /path/to/modules.tsv]
```

### Step 7: Model Utilities

#### `load_saved_models.py`
**Purpose**: Utility script to load and inspect saved GLM models.

**Inputs**:
- Model directory or specific model file

**Outputs**:
- Model summary information
- Feature coefficients
- Model statistics

**Usage**:
```bash
python src/rate_modeling/load_saved_models.py \
    --model-dir /path/to/models \
    --output-dir /path/to/output
```

## Data Dependencies

### Required Input Files

1. **Mpileup files**: Standard SAMtools mpileup format
   - Generated from BAM files using: `samtools mpileup`
   - Contains read information at each genomic position

2. **Reference genome FASTA**: Genome sequence file
   - Path specified in `config/reference_paths.yaml` or command line
   - Used for sequence context extraction

3. **GFF annotation file**: Gene annotations
   - Path specified in `config/reference_paths.yaml` or command line
   - Used for genic vs intergenic classification
   - Used for synonymous vs non-synonymous classification

4. **Expression data** (for correlation analysis):
   - TSV format with gene IDs and TPM values
   - Path specified in `config/reference_paths.yaml`

5. **Exclusion mask** (optional):
   - TSV format: `chrom\tpos`
   - Sites to exclude from analysis

### Intermediate Data Files

1. **`.counts` files**: Mutation counts per site
   - Generated by `collect_mutation_counts.py`
   - Format: TSV with columns: `chrom`, `pos`, `ref`, `depth`, `alt_count`, `alt_base`

2. **5mer context JSON files**: 5mer context counts
   - Generated by `collect_5mer_contexts.py`
   - JSON format with nested dictionaries

3. **Rate TSV files**: Mutation rates
   - Generated by `estimate_rates.py`
   - Multiple files: per-sample, per-site, per-gene

4. **Model pickle files**: Saved GLM models
   - Generated by `sequence_bias_modeling_sitelevel.py`
   - Can be loaded for predictions

## Statistical Methods

### Mutation Rate Estimation

1. **Simple Rate Estimates**:
   - Low: `P(mutated) = # mutated positions / total positions`
   - High: `P(mutation) = # alt alleles / total depth`

2. **Alpha Correction**:
   - Estimates false positive rate from controls: `α = rate_control`
   - Adjusts treated rates: `rate_adjusted = rate_treated - α`

3. **GLM Rate Estimation**:
   - Model: `log(E[y]) = β₀ + β₁×treatment + log(depth)`
   - Where `y` is mutation count, `depth` is sequencing depth
   - Rate = `exp(β₀ + β₁×treatment)` per unit depth
   - Confidence intervals via asymptotic normal approximation

### Sequence Bias Modeling

1. **GLM Framework**:
   - Response: Mutation counts (Poisson or Negative Binomial)
   - Offset: log(depth)
   - Features: Sequence context (positional, 3mer, 5mer, or combinations)
   - Covariate: Treatment (0/1)

2. **Model Selection**:
   - AIC (Akaike Information Criterion)
   - BIC (Bayesian Information Criterion)
   - Lower values indicate better fit (with penalty for complexity)

3. **Canonicalization**:
   - All mutations represented as C>T in C-centered 5mers
   - G>A mutations reverse-complemented: `G>A → C>T` in `RC(5mer)`

### Cross-Validation

- K-fold cross-validation (default k=5)
- Stratified by treatment group (if applicable)
- Out-of-sample metrics computed on held-out folds
- Mean ± standard deviation across folds

## Key Assumptions and Design Decisions

1. **EMS-specific mutations**: Only C>T and G>A at G/C sites
2. **Strand symmetry**: C>T and G>A treated equivalently via canonicalization
3. **Depth offset**: Mutation rates modeled per unit sequencing depth
4. **Control filtering**: Sites appearing in >1 control file are excluded
5. **Quality filtering**: Depth percentile filtering removes extreme coverage sites
6. **1-based coordinates**: GFF standard (Python 0-based internally converted)

## Output File Formats

### Count Files (`.counts`)
```
chrom	pos	ref	depth	alt_count	alt_base
NC_002978.6	100	C	50	3	T
NC_002978.6	101	G	48	2	A
```

### Rate Files (`.tsv`)
```
sample	rate_low	rate_high	rate_glm	glm_lower	glm_upper
EMS1_3d	1.2e-5	1.5e-5	1.3e-5	1.1e-5	1.5e-5
```

### Model Files (`.pkl`)
- Pickle format containing:
  - Fitted GLM model object
  - Feature names
  - Model metadata (family, link, etc.)

## Typical Workflow

1. **Preprocess mpileup files** (if not already done):
   ```bash
   python src/preprocess_mpileup.py --mpileups ... --output ...
   ```

2. **Collect mutation counts**:
   ```bash
   python src/rate_modeling/collect_mutation_counts.py --mpileup-dir ... --output-dir ...
   ```

3. **Estimate mutation rates**:
   ```bash
   python src/rate_modeling/estimate_rates.py --counts-dir ... --output-dir ...
   ```

4. **Model sequence bias**:
   ```bash
   python src/rate_modeling/sequence_bias_modeling_sitelevel.py --counts-dir ... --output-dir ...
   ```

5. **Evaluate models**:
   ```bash
   python src/rate_modeling/cross_validation_evaluation.py --counts-dir ... --output-dir ...
   ```

6. **Generate visualizations**:
   ```bash
   python src/rate_modeling/regenerate_rate_plots.py --output-dir ... --figure-output ...
   python src/rate_modeling/plot_ems_spectra.py ... --output ...
   ```

7. **Correlate with expression** (optional):
   ```bash
   python src/rate_modeling/correlate_rates_with_expression.py --mutation-rates ... --expression ...
   ```

## Configuration

Paths to reference files can be specified via:
1. Command-line arguments (takes precedence)
2. `config/reference_paths.yaml` file
3. Environment variables (where supported)

Key configuration paths:
- `references.genomic_fna`: Reference genome FASTA
- `references.annotation`: GFF annotation file
- `references.expression_data`: Expression data file
- `references.module_assignments`: Gene module assignments

## Notes

- All scripts support multiprocessing for parallel execution
- Thread limits can be controlled via environment variables (OMP_NUM_THREADS, etc.)
- Sample names are automatically simplified (extracts EMS/NT group numbers and time points)
- Control samples identified by "NT" in filename; treated samples by "EMS"
- Time points extracted from sample names (3d, 7d patterns)
- The analysis pipeline is designed for Wolbachia (wMel) genome but can be adapted

