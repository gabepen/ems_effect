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
parser$add_argument('--bam', required=TRUE,
                   help='Path to sorted BAM file for coverage')
parser$add_argument('--genome-length', type='integer', required=TRUE,
                   help='Length of genome in base pairs')
parser$add_argument('--window-size', type='integer', default=1000,
                   help='Size of windows for mutation density calculation (default: 1kb)')
parser$add_argument('--output', required=TRUE,
                   help='Output PNG file path')

args <- parser$parse_args()

# Process variants file into windows
variants_raw <- read_tsv(args$variants, 
                    col_names=c("seqname", "pos", "ref", "depth", "bases", "qual"))

# Filter for canonical EMS mutations (C>T and G>A)
ems_variants <- variants_raw %>%
  filter((ref == "C" & str_detect(bases, "T")) | (ref == "G" & str_detect(bases, "A")))

# Count EMS mutations per window
variants <- ems_variants %>%
  mutate(mut_count = as.numeric(str_count(bases, "[ACGT]"))) %>%
  mutate(window = floor(pos / args$window_size)) %>%
  group_by(window) %>%
  summarize(
    seqname = dplyr::first(seqname),
    start = min(pos),
    end = max(pos),
    count = sum(mut_count)
  ) %>%
  mutate(
    seqname = "NC_002978.6",
    count = as.numeric(count)
  ) %>%
  select(seqname, start, end, count)

# Print data frame structure for debugging
print("EMS Variants data frame structure:")
str(variants)

# Import gene annotations and calculate mutation rates per gene
genes <- read_tsv(args$gff, 
                 comment="#",
                 col_names=c("seqname", "source", "type", "start", "end", 
                           "score", "strand", "phase", "attributes")) %>%
  filter(type == "gene") %>%
  mutate(
    seqname = "NC_002978.6",
    start = as.numeric(start),
    end = as.numeric(end),
    # Ensure start < end by swapping if needed
    tmp_start = pmin(start, end),
    tmp_end = pmax(start, end),
    start = tmp_start,
    end = tmp_end,
    color = ifelse(1:n() %% 2 == 0, "#4292C6", "#08519C"),
    # Extract gene ID from attributes
    gene_id = str_extract(attributes, "ID=([^;]+)") %>% str_remove("ID="),
    # Calculate gene length using corrected coordinates
    length = end - start
  ) %>%
  select(-tmp_start, -tmp_end)  # Remove temporary columns

# Print some debugging info
print("Gene coordinate ranges:")
print(summary(genes$start))
print(summary(genes$end))
print("Number of genes where start > end:")
print(sum(genes$start > genes$end))

# Calculate coverage from BAM file
print("Reading coverage from BAM file...")
# Create windows for coverage calculation
window_starts <- seq(1, args$genome_length, by=args$window_size)
window_ends <- c(window_starts[-1] - 1, args$genome_length)

coverage_data <- data.frame(
  seqname = rep("NC_002978.6", length(window_starts)),
  start = window_starts,
  end = window_ends
)

print(paste("Number of windows:", nrow(coverage_data)))

# Use system call to samtools to calculate coverage
print("Calculating coverage...")
coverage_temp <- tempfile()
system(paste("samtools depth -a", args$bam, ">", coverage_temp))

# Read coverage data
coverage_raw <- read.table(coverage_temp, col.names=c("seqname", "pos", "depth"))
unlink(coverage_temp)  # Clean up temp file

# Calculate mean coverage per window
coverage_data$coverage <- sapply(1:nrow(coverage_data), function(i) {
  window_coverage <- coverage_raw$depth[coverage_raw$pos >= coverage_data$start[i] & 
                                      coverage_raw$pos <= coverage_data$end[i]]
  mean(window_coverage)
})

# Handle NA values and ensure valid coverage data
coverage_data$coverage[is.na(coverage_data$coverage)] <- 0

print("Coverage statistics:")
print(summary(coverage_data$coverage))

# Initialize plot
# Use a more standard figure size for publication
# (8x8 inches at 300 dpi)
png(args$output, height=8*300, width=8*300, res=300)
circos.clear()
circos.par(gap.after=10)

# Set base text size for the plot
par(cex=1)  # Slightly smaller base text size

# Create genome info
genome_df <- data.frame(
  name = "NC_002978.6",
  start = 0,
  end = args$genome_length
)

# Initialize genomic coordinates
circos.genomicInitialize(genome_df, major.by=2e5, plotType = NULL)

# Prepare gene data for plotting
genes_plot <- genes %>%
  select(seqname, start, end, color, gene_id) %>%
  as.data.frame()  # Convert to regular data frame

# Plotting coverage track
print("Plotting coverage track...")
max_coverage <- max(coverage_data$coverage, na.rm=TRUE)
if(max_coverage == 0) max_coverage <- 1  # Prevent zero y-limit
avg_coverage <- mean(coverage_data$coverage, na.rm=TRUE)

circos.genomicTrackPlotRegion(
  coverage_data,
  track.height=0.15,
  ylim=c(0, max_coverage),
  panel.fun=function(region, value, ...) {
    circos.genomicRect(region, value,
                      ytop=value$coverage,
                      ybottom=0,
                      col="#E6AB02",  # Golden yellow
                      border=NA)
    # Add average coverage line
    circos.lines(c(region$start[1], region$end[length(region$end)]),
                 c(avg_coverage, avg_coverage),
                 col="red", lwd=2, lty=2)
    # Add coverage axis with fixed ticks
    circos.yaxis(side="left",
                 at=c(0, 25000, 50000),
                 labels.cex=0.8)  # Slightly smaller axis labels
  }
)
# Add annotation for average coverage
text(0, -0.1 * max_coverage, paste0("Avg coverage: ", round(avg_coverage, 1)), col="red", cex=0.9, xpd=NA)

# Plot gene track (no low-mutation gene labels)
circos.genomicTrackPlotRegion(
  genes_plot,
  track.height=0.1,
  ylim = c(0, 1),
  panel.fun=function(region, value, ...) {
    circos.genomicRect(
      region,
      value,
      ytop=1,
      ybottom=0,
      col=value$color,
      border=NA
    )
  }
)

# Plot mutation density track (EMS only)
variants_plot <- variants %>%
  mutate(
    start = as.numeric(start),
    end = as.numeric(end),
    count = as.numeric(count)
  )

max_count <- max(variants_plot$count, na.rm=TRUE)
q95 <- quantile(variants_plot$count, 0.95, na.rm=TRUE)
ylimit <- min(max_count, q95 * 2)

print("Y-axis statistics:")
print(paste("Max count:", max_count))
print(paste("95th percentile:", q95))
print(paste("Using y-limit:", ylimit))

circos.genomicTrackPlotRegion(
  variants_plot,
  track.height=0.2,
  ylim = c(0, ylimit),
  panel.fun=function(region, value, ...) {
    circos.genomicRect(region, value,
                      col="darkorange",
                      ytop=pmin(value$count, ylimit),
                      ybottom=0,
                      border=NA, ...)
    # Add a line at the bottom of the track
    cell.xlim = get.cell.meta.data("cell.xlim")
    circos.lines(cell.xlim, c(0, 0), lty=1, col="#000000")
  }
)

# Add y-axis labels
y_axis_labels <- c(0, ylimit)
if(max_count > ylimit) {
  y_axis_labels <- c(0, ylimit, max_count)
}

circos.yaxis(
  side="left",
  at=y_axis_labels,
  labels=y_axis_labels,
  labels.cex=0.8  # Slightly smaller y-axis labels
)

# Add a legend, move to bottomright to avoid overlap
legend("bottomright", legend=c("Coverage", "Gene", "EMS mutation density", "Avg coverage"), 
       fill=c("#E6AB02", "#4292C6", "darkorange", NA), border=NA, lty=c(NA, NA, NA, 2), col=c(NA, NA, NA, "red"),
       cex=0.9, bty="n", text.col="black", text.font=1, x.intersp=0.7, y.intersp=0.9, pt.cex=0.8, inset=0.01)

dev.off()