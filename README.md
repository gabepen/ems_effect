# EMS Effect: Mutation Rate Estimation and Sequence Bias Modeling

This repository contains the analysis pipeline for mutation rate estimation and sequence bias modeling of EMS (ethyl methanesulfonate) mutagenesis data, as described in the manuscript.

## Documentation

For complete documentation of the analysis pipeline, see [manuscript_scope_documentation.md](manuscript_scope_documentation.md).

## Quick Start

The analysis pipeline is contained within the `src/rate_modeling/` directory. Key scripts include:

- **Data collection**: `collect_mutation_counts.py`, `collect_5mer_contexts.py`
- **Rate estimation**: `estimate_rates.py`
- **Sequence bias modeling**: `sequence_bias_modeling_sitelevel.py`
- **Visualization**: `plot_5mer_mutation_rates.py`, `plot_ems_spectra.py`, `regenerate_rate_plots.py`

## Dependencies

See `environment.yml` for the complete conda environment specification.

## Configuration

Reference file paths can be specified in `config/reference_paths.yaml` or via command-line arguments.

