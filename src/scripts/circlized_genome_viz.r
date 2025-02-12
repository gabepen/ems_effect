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
variants <- read_tsv(args$variants, 
                    col_names=c("seqname", "pos", "ref", "depth", "bases", "qual")) %>%
  mutate(mut_count = as.numeric(str_count(bases, "[ACGT]"))) %>%
  mutate(window = floor(pos / args$window_size)) %>%
  group_by(window) %>%
  summarize(
    seqname = dplyr::first(seqname),
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

# First fix the gene mutations calculation
gene_mutations <- variants %>%
  # Ensure numeric coordinates
  mutate(
    start = as.numeric(start),
    end = as.numeric(end)
  ) %>%
  # For each variant window, find overlapping genes
  rowwise() %>%
  mutate(
    gene_id = list(genes$gene_id[genes$start <= end & genes$end >= start])
  ) %>%
  unnest(gene_id) %>%
  # Group by gene and calculate totals
  group_by(gene_id) %>%
  summarize(
    total_mutations = sum(count),
    gene_start = min(genes$start[genes$gene_id == dplyr::first(gene_id)]),
    gene_end = max(genes$end[genes$gene_id == dplyr::first(gene_id)])
  ) %>%
  # Join with gene info
  left_join(genes, by="gene_id") %>%
  # Calculate mutation rate per base
  mutate(
    mutation_rate = total_mutations / length,
    start = gene_start,
    end = gene_end
  ) %>%
  # Identify low mutation genes (bottom 5%)
  mutate(is_low = mutation_rate < quantile(mutation_rate, 0.05, na.rm=TRUE))

# Print debugging info
print("Gene mutation statistics:")
print("Number of genes with mutations:")
print(nrow(gene_mutations))
print("Coordinate ranges in gene_mutations:")
print(summary(gene_mutations$start))
print(summary(gene_mutations$end))
print("Number of invalid coordinates:")
print(sum(gene_mutations$start > gene_mutations$end))

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
png(args$output, height=37*350, width=40*350, res=600)
circos.clear()
circos.par(gap.after=10)

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

# Prepare low mutation genes data
low_mut_genes <- gene_mutations %>%
  filter(is_low) %>%
  select(seqname, start, end, gene_id) %>%
  as.data.frame()

print("Genes plot data structure:")
str(genes_plot)
print("Sample of genes_plot data:")
print(head(genes_plot))

# After initializing plot and before gene track, add coverage track
print("Plotting coverage track...")
# Ensure valid y-limits
max_coverage <- max(coverage_data$coverage, na.rm=TRUE)
if(max_coverage == 0) max_coverage <- 1  # Prevent zero y-limit

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
    
    # Add coverage axis
    circos.yaxis(side="left",
                 at=pretty(c(0, max_coverage)),
                 labels.cex=0.4)
  }
)

# Plot gene track
circos.genomicTrackPlotRegion(
  genes_plot,
  track.height=0.1,
  ylim = c(0, 1),
  panel.fun=function(region, value, ...) {
    # Plot gene rectangles
    circos.genomicRect(
      region,
      value,
      ytop=1,
      ybottom=0,
      col=value$color,
      border=NA
    )
    
    # Add labels for low mutation genes if they overlap with current region
    if(nrow(low_mut_genes) > 0) {
      # Get genes that overlap with current region
      overlapping_genes <- low_mut_genes[
        low_mut_genes$start >= region$start[1] & 
        low_mut_genes$end <= region$end[length(region$end)],
      ]
      
      # Add labels for overlapping genes
      if(nrow(overlapping_genes) > 0) {
        for(i in 1:nrow(overlapping_genes)) {
          gene <- overlapping_genes[i,]
          mid_point <- (gene$start + gene$end)/2
          circos.text(
            mid_point,
            1.2,
            gene$gene_id,
            cex=0.4,
            col="darkred",
            facing="clockwise",  # Make text follow the circle
            niceFacing=TRUE,     # Adjust text angle for readability
            adj=c(0, 0.5)        # Center text vertically relative to point
          )
        }
      }
    }
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
q95 <- quantile(variants_plot$count, 0.95, na.rm=TRUE)
ylimit <- min(max_count, q95 * 2)

print("Y-axis statistics:")
print(paste("Max count:", max_count))
print(paste("95th percentile:", q95))
print(paste("Using y-limit:", ylimit))

# Plot mutation density track
circos.genomicTrackPlotRegion(
  variants_plot,
  track.height=0.2,
  ylim = c(0, ylimit),
  panel.fun=function(region, value, ...) {
    circos.genomicRect(region, value,
                      col=value$color,
                      ytop=pmin(as.numeric(value$count), ylimit),
                      ybottom=0,
                      border=NA, ...)
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
  labels.cex=0.6*par("cex")
)

dev.off()