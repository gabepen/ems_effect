#!/usr/bin/env Rscript

library(circlize)
library(tidyverse)
library(argparse)

# Set up argument parser
parser <- ArgumentParser(description='Create circular genome visualization of mutation density')
parser$add_argument('--variants', required=TRUE,
                   help='Path to variants.txt file from mpileup')
parser$add_argument('--gff', required=TRUE,
                   help='Path to reference GFF file')
parser$add_argument('--genome-length', type='integer', required=TRUE,
                   help='Length of genome in base pairs')
parser$add_argument('--window-size', type='integer', default=20000,
                   help='Size of windows for mutation density calculation (default: 20kb)')
parser$add_argument('--output', required=TRUE,
                   help='Output PNG file path')

args <- parser$parse_args()

# Process variants file into windows
variants <- read_tsv(args$variants, 
                    col_names=c("seqname", "pos", "ref", "depth", "bases", "qual")) %>%
  mutate(mut_count = as.numeric(str_count(bases, "[ACGT]"))) %>%
  mutate(window = floor(pos / args$window_size)) %>%
  group_by(window) %>%
  summarize(
    seqname = first(seqname),
    start = min(pos),
    end = max(pos),
    count = sum(mut_count),
    p = ppois(count, mean(count), lower.tail=FALSE)
  ) %>%
  mutate(
    color = ifelse(p <= 0.05, "red", "black"),
    seqname = "NC_002978.6",
    count = as.numeric(count)
  ) %>%
  select(seqname, start, end, count, color)

# Print data frame structure for debugging
print("Variants data frame structure:")
str(variants)

# Import gene annotations directly from GFF
genes <- read_tsv(args$gff, 
                 comment="#",
                 col_names=c("seqname", "source", "type", "start", "end", 
                           "score", "strand", "phase", "attributes")) %>%
  filter(type == "gene") %>%
  mutate(
    seqname = "NC_002978.6",
    start = as.numeric(start),
    end = as.numeric(end),
    color = ifelse(1:n() %% 2 == 0, "#4292C6", "#08519C")
  ) %>%
  select(seqname, start, end, color)

# Print data frame structure for debugging
print("Genes data frame structure:")
str(genes)

# Create genome info
genome_df <- data.frame(
  name = "NC_002978.6",
  start = 0,
  end = args$genome_length
)

# Initialize plot
png(args$output, height=37*350, width=40*350, res=600)
circos.clear()
circos.par(gap.after=10)

# Initialize genomic coordinates
circos.genomicInitialize(genome_df, major.by=2e5, plotType = NULL)

# Plot gene track
circos.genomicTrackPlotRegion(
  genes,
  track.height=0.1,
  ylim = c(0, 1),
  panel.fun=function(region, value, ...) {
    circos.genomicRect(region, value,
                      col=value$color,
                      ytop=1,
                      ybottom=0,
                      border=NA, ...)
  }
)

# Plot mutation density track with explicit numeric data and scaling
variants_plot <- variants %>%
  mutate(
    start = as.numeric(start),
    end = as.numeric(end),
    count = as.numeric(count)
  )

# Calculate reasonable y-limit using quantiles
max_count <- max(variants_plot$count, na.rm=TRUE)
q95 <- quantile(variants_plot$count, 0.95, na.rm=TRUE)  # 95th percentile
ylimit <- min(max_count, q95 * 2)  # Use either max or 2x the 95th percentile

print("Y-axis statistics:")
print(paste("Max count:", max_count))
print(paste("95th percentile:", q95))
print(paste("Using y-limit:", ylimit))

# Plot mutation density track
circos.genomicTrackPlotRegion(
  variants_plot,
  track.height=0.2,
  ylim = c(0, ylimit),  # Use calculated ylimit
  panel.fun=function(region, value, ...) {
    circos.genomicRect(region, value,
                      col=value$color,
                      ytop=pmin(as.numeric(value$count), ylimit),  # Cap values at ylimit
                      ybottom=0,
                      border=NA, ...)
    cell.xlim = get.cell.meta.data("cell.xlim")
    circos.lines(cell.xlim, c(0, 0), lty=1, col="#000000")
  }
)

# Add y-axis labels with note if values were capped
y_axis_labels <- c(0, ylimit)
if(max_count > ylimit) {
  y_axis_labels <- c(0, ylimit, max_count)
}

circos.yaxis(
  side="left",
  at=y_axis_labels,
  labels=y_axis_labels,
  labels.cex=0.6*par("cex")
)

dev.off()