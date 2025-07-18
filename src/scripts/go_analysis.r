# GO Enrichment Analysis Script
# Alternative to KEGG when REST API is blocked

# Set CRAN mirror to avoid installation errors
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Install BiocManager if not already installed
if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cloud.r-project.org")
}

# Install Bioconductor packages if not already installed
if (!requireNamespace("clusterProfiler", quietly = TRUE)) {
    BiocManager::install("clusterProfiler", update = FALSE, ask = FALSE)
}

if (!requireNamespace("org.Hs.eg.db", quietly = TRUE)) {
    BiocManager::install("org.Hs.eg.db", update = FALSE, ask = FALSE)
}

# Install and load argparse for command line argument parsing
if (!requireNamespace("argparse", quietly = TRUE)) {
    install.packages("argparse", repos = "https://cloud.r-project.org")
}

# Load libraries
library(clusterProfiler)
library(argparse)

# Set up argument parser
parser <- ArgumentParser(description = "GO enrichment analysis")
parser$add_argument("--gene_list", type = "character", 
                   help = "Comma-separated list of NCBI Gene IDs")
parser$add_argument("--gene_file", type = "character",
                   help = "File containing gene IDs (one per line)")
parser$add_argument("--pvalue_cutoff", type = "double", default = 0.05,
                   help = "P-value cutoff for enrichment analysis (default: 0.05)")
parser$add_argument("--output_prefix", type = "character", default = "go_enrichment",
                   help = "Prefix for output files (default: go_enrichment)")
parser$add_argument("--organism", type = "character", default = "org.Hs.eg.db",
                   help = "Organism annotation database (default: org.Hs.eg.db)")

# Parse arguments
args <- parser$parse_args()

# Check that either gene_list or gene_file is provided
if (is.null(args$gene_list) && is.null(args$gene_file)) {
    cat("ERROR: Either --gene_list or --gene_file must be provided\n")
    cat("Use --help for usage information\n")
    quit(status = 1)
}

# Get gene list from either argument
if (!is.null(args$gene_file)) {
    # Read genes from file
    cat("Reading genes from file:", args$gene_file, "\n")
    tryCatch({
        genes <- readLines(args$gene_file)
        # Remove empty lines and whitespace
        genes <- genes[nchar(trimws(genes)) > 0]
        genes <- trimws(genes)
        
        cat("Found", length(genes), "genes in file\n")
        if (length(genes) == 0) {
            cat("ERROR: No genes found in file\n")
            quit(status = 1)
        }
        
        # Convert to comma-separated string for processing
        args$gene_list <- paste(genes, collapse = ",")
        
    }, error = function(e) {
        cat("ERROR reading gene file:", e$message, "\n")
        quit(status = 1)
    })
}

# Convert gene_list string to vector
gene_list <- unlist(strsplit(args$gene_list, ","))
gene_list <- trimws(gene_list)  # Remove any whitespace

cat("Using organism database:", args$organism, "\n")
cat("Number of genes to analyze:", length(gene_list), "\n")
cat("First few gene IDs:", head(gene_list), "\n")

# Load organism database
tryCatch({
    if (args$organism == "org.Hs.eg.db") {
        library(org.Hs.eg.db)
        org_db <- org.Hs.eg.db
    } else {
        # Try to load the specified organism database
        library(args$organism, character.only = TRUE)
        org_db <- get(args$organism)
    }
    cat("✓ Organism database loaded successfully\n")
}, error = function(e) {
    cat("✗ Error loading organism database:", e$message, "\n")
    cat("Falling back to human database...\n")
    library(org.Hs.eg.db)
    org_db <- org.Hs.eg.db
})

# Perform GO enrichment analysis
cat("Performing GO enrichment analysis...\n")

# Biological Process
cat("Analyzing Biological Process (BP)...\n")
tryCatch({
    go_bp <- enrichGO(gene = gene_list,
                      OrgDb = org_db,
                      keyType = "ENTREZID",
                      ont = "BP",
                      pAdjustMethod = "BH",
                      pvalueCutoff = args$pvalue_cutoff)
    
    if (nrow(go_bp) > 0) {
        cat("✓ Found", nrow(go_bp), "enriched BP terms\n")
        
        # Save results
        bp_file <- paste0(args$output_prefix, "_BP_results.txt")
        write.table(as.data.frame(go_bp), file = bp_file, 
                    sep = "\t", quote = FALSE, row.names = FALSE)
        cat("BP results saved to:", bp_file, "\n")
        
        # Create dotplot
        pdf(paste0(args$output_prefix, "_BP_dotplot.pdf"))
        dotplot(go_bp, showCategory = 20)
        dev.off()
        cat("BP dotplot saved to:", paste0(args$output_prefix, "_BP_dotplot.pdf"), "\n")
    } else {
        cat("✗ No enriched BP terms found\n")
    }
}, error = function(e) {
    cat("✗ Error in BP analysis:", e$message, "\n")
})

# Molecular Function
cat("Analyzing Molecular Function (MF)...\n")
tryCatch({
    go_mf <- enrichGO(gene = gene_list,
                      OrgDb = org_db,
                      keyType = "ENTREZID",
                      ont = "MF",
                      pAdjustMethod = "BH",
                      pvalueCutoff = args$pvalue_cutoff)
    
    if (nrow(go_mf) > 0) {
        cat("✓ Found", nrow(go_mf), "enriched MF terms\n")
        
        # Save results
        mf_file <- paste0(args$output_prefix, "_MF_results.txt")
        write.table(as.data.frame(go_mf), file = mf_file, 
                    sep = "\t", quote = FALSE, row.names = FALSE)
        cat("MF results saved to:", mf_file, "\n")
        
        # Create dotplot
        pdf(paste0(args$output_prefix, "_MF_dotplot.pdf"))
        dotplot(go_mf, showCategory = 20)
        dev.off()
        cat("MF dotplot saved to:", paste0(args$output_prefix, "_MF_dotplot.pdf"), "\n")
    } else {
        cat("✗ No enriched MF terms found\n")
    }
}, error = function(e) {
    cat("✗ Error in MF analysis:", e$message, "\n")
})

# Cellular Component
cat("Analyzing Cellular Component (CC)...\n")
tryCatch({
    go_cc <- enrichGO(gene = gene_list,
                      OrgDb = org_db,
                      keyType = "ENTREZID",
                      ont = "CC",
                      pAdjustMethod = "BH",
                      pvalueCutoff = args$pvalue_cutoff)
    
    if (nrow(go_cc) > 0) {
        cat("✓ Found", nrow(go_cc), "enriched CC terms\n")
        
        # Save results
        cc_file <- paste0(args$output_prefix, "_CC_results.txt")
        write.table(as.data.frame(go_cc), file = cc_file, 
                    sep = "\t", quote = FALSE, row.names = FALSE)
        cat("CC results saved to:", cc_file, "\n")
        
        # Create dotplot
        pdf(paste0(args$output_prefix, "_CC_dotplot.pdf"))
        dotplot(go_cc, showCategory = 20)
        dev.off()
        cat("CC dotplot saved to:", paste0(args$output_prefix, "_CC_dotplot.pdf"), "\n")
    } else {
        cat("✗ No enriched CC terms found\n")
    }
}, error = function(e) {
    cat("✗ Error in CC analysis:", e$message, "\n")
})

cat("\n=== GO Analysis Complete ===\n")
cat("Note: This analysis uses human GO annotations.\n")
cat("For Wolbachia-specific analysis, you would need:\n")
cat("1. Wolbachia GO annotation database\n")
cat("2. Or use ortholog mapping to human genes\n")
cat("3. Or contact IT to unblock KEGG REST API\n") 