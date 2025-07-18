#!/usr/bin/env Rscript

# Wolbachia-specific GO Enrichment Analysis
# Uses custom Wolbachia GO database built from UniProt annotations

cat("Wolbachia GO Enrichment Analysis\n")
cat("===============================\n")
cat("This script performs GO enrichment analysis using UniProt IDs from Wolbachia wMel.\n")
cat("Expected input: UniProt IDs (e.g., A0A2H4V217, Q73FT6, etc.)\n\n")
cat("Usage examples:\n")
cat("  Rscript wolbachia_go_enrichment.r --gene_file my_uniprot_ids.txt\n")
cat("  Rscript wolbachia_go_enrichment.r --gene_list \"A0A2H4V217,Q73FT6,Q73FT7\"\n\n")

# Install required packages
if (!requireNamespace("argparse", quietly = TRUE)) {
    install.packages("argparse", repos = "https://cloud.r-project.org")
}

library(argparse)

# Set up argument parser
parser <- ArgumentParser(description = "Wolbachia GO enrichment analysis using UniProt IDs")
parser$add_argument("--gene_list", type = "character", 
                   help = "Comma-separated list of UniProt IDs (e.g., A0A2H4V217,Q73FT6)")
parser$add_argument("--gene_file", type = "character",
                   help = "File containing UniProt IDs (one per line)")
parser$add_argument("--pvalue_cutoff", type = "double", default = 0.05,
                   help = "P-value cutoff for enrichment analysis (default: 0.05)")
parser$add_argument("--output_prefix", type = "character", default = "wolbachia_go_enrichment",
                   help = "Prefix for output files (default: wolbachia_go_enrichment)")
parser$add_argument("--db_dir", type = "character", default = "wolbachia_go_db",
                   help = "Directory containing Wolbachia GO database (default: wolbachia_go_db)")

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

# Add _WOLPM suffix to UniProt IDs if not already present
gene_list_with_suffix <- sapply(gene_list, function(id) {
    if (!grepl("_WOLPM$", id)) {
        return(paste0(id, "_WOLPM"))
    } else {
        return(id)
    }
})

cat("Number of UniProt IDs to analyze:", length(gene_list), "\n")
cat("First few input UniProt IDs:", head(gene_list), "\n")
cat("First few IDs with suffix:", head(gene_list_with_suffix), "\n\n")

# Load Wolbachia GO database
cat("Loading Wolbachia GO database from:", args$db_dir, "\n")

gene_to_go_file <- file.path(args$db_dir, "gene_to_go.txt")
go_terms_file <- file.path(args$db_dir, "go_terms.txt")

if (!file.exists(gene_to_go_file) || !file.exists(go_terms_file)) {
    cat("ERROR: Wolbachia GO database files not found.\n")
    cat("Please run build_wolbachia_go_db.r first to create the database.\n")
    quit(status = 1)
}

# Load database files
tryCatch({
    gene_to_go <- read.table(gene_to_go_file, header = TRUE, sep = "\t", stringsAsFactors = FALSE)
    go_terms <- read.table(go_terms_file, header = TRUE, sep = "\t", stringsAsFactors = FALSE)
    
    cat("✓ Loaded Wolbachia GO database\n")
    cat("  - UniProt IDs with GO annotations:", length(unique(gene_to_go$GID)), "\n")
    cat("  - Unique GO terms:", length(unique(gene_to_go$GO)), "\n")
    cat("  - Total UniProt-GO associations:", nrow(gene_to_go), "\n\n")
    
}, error = function(e) {
    cat("ERROR loading Wolbachia GO database:", e$message, "\n")
    quit(status = 1)
})

# Function to perform GO enrichment analysis
perform_wolbachia_go_enrichment <- function(gene_list, gene_to_go, go_terms, pvalue_cutoff) {
    cat("Performing Wolbachia GO enrichment analysis...\n")
    
    # Create background gene set (all genes in database)
    background_genes <- unique(gene_to_go$GID)
    
    # Filter input genes to those in database
    available_genes <- gene_list[gene_list %in% background_genes]
    
    if (length(available_genes) == 0) {
        cat("✗ None of the input UniProt IDs are in the Wolbachia GO database\n")
        cat("Available UniProt IDs in database (first 10):", head(background_genes, 10), "\n")
        cat("Make sure your input contains valid UniProt IDs from Wolbachia wMel\n")
        return(NULL)
    }
    
    cat("✓ Found", length(available_genes), "UniProt IDs in Wolbachia GO database\n")
    
    # Perform enrichment analysis
    tryCatch({
        # Create enrichment result
        enrichment_result <- data.frame()
        
        # Get unique GO terms for input genes
        input_go_terms <- gene_to_go[gene_to_go$GID %in% available_genes, ]
        
        if (nrow(input_go_terms) == 0) {
            cat("✗ No GO terms found for input UniProt IDs\n")
            return(NULL)
        }
        
        # Calculate enrichment for each GO term
        unique_go_terms <- unique(input_go_terms$GO)
        
        for (go_term in unique_go_terms) {
            # Count genes with this GO term in input
            input_with_go <- sum(input_go_terms$GO == go_term)
            
            # Count genes with this GO term in background
            background_with_go <- sum(gene_to_go$GO == go_term)
            
            # Calculate enrichment
            total_input <- length(available_genes)
            total_background <- length(background_genes)
            
            # Fisher's exact test
            contingency_table <- matrix(c(
                input_with_go, total_input - input_with_go,
                background_with_go - input_with_go, 
                total_background - background_with_go - (total_input - input_with_go)
            ), nrow = 2)
            
            fisher_result <- fisher.test(contingency_table, alternative = "greater")
            
            # Get GO term description
            go_description <- go_terms$TERM[go_terms$GOID == go_term]
            if (length(go_description) == 0) go_description <- "Unknown"
            
            # Calculate enrichment ratio
            enrichment_ratio <- (input_with_go / total_input) / (background_with_go / total_background)
            
            enrichment_result <- rbind(enrichment_result, data.frame(
                ID = go_term,
                Description = go_description,
                GeneRatio = paste(input_with_go, total_input, sep = "/"),
                BgRatio = paste(background_with_go, total_background, sep = "/"),
                pvalue = fisher_result$p.value,
                p.adjust = fisher_result$p.value,  # No multiple testing correction for simplicity
                qvalue = fisher_result$p.value,
                geneID = paste(unique(input_go_terms$GID[input_go_terms$GO == go_term]), collapse = "/"),
                Count = input_with_go,
                EnrichmentRatio = enrichment_ratio,
                stringsAsFactors = FALSE
            ))
        }
        
        # Sort by p-value
        enrichment_result <- enrichment_result[order(enrichment_result$pvalue), ]
        
        # Filter by p-value cutoff
        significant_results <- enrichment_result[enrichment_result$pvalue <= pvalue_cutoff, ]
        
        cat("✓ Found", nrow(enrichment_result), "GO terms with annotations\n")
        cat("✓ Found", nrow(significant_results), "significantly enriched GO terms (p ≤", pvalue_cutoff, ")\n")
        
        return(list(all_results = enrichment_result, significant_results = significant_results))
        
    }, error = function(e) {
        cat("✗ Error in enrichment analysis:", e$message, "\n")
        return(NULL)
    })
}

# Perform enrichment analysis
enrichment_results <- perform_wolbachia_go_enrichment(gene_list_with_suffix, gene_to_go, go_terms, args$pvalue_cutoff)

if (is.null(enrichment_results)) {
    cat("No enrichment results to save.\n")
    quit(status = 1)
}

# Save results
cat("\nSaving results...\n")

# Save all results
all_results_file <- paste0(args$output_prefix, "_all_results.txt")
write.table(enrichment_results$all_results, all_results_file, 
            sep = "\t", quote = FALSE, row.names = FALSE)
cat("✓ All results saved to:", all_results_file, "\n")

# Save significant results
significant_results_file <- paste0(args$output_prefix, "_significant_results.txt")
write.table(enrichment_results$significant_results, significant_results_file, 
            sep = "\t", quote = FALSE, row.names = FALSE)
cat("✓ Significant results saved to:", significant_results_file, "\n")

# Create summary plots if significant results exist
if (nrow(enrichment_results$significant_results) > 0) {
    cat("\nCreating summary plots...\n")
    
    # Install ggplot2 if not available
    if (!requireNamespace("ggplot2", quietly = TRUE)) {
        install.packages("ggplot2", repos = "https://cloud.r-project.org")
    }
    library(ggplot2)
    
    # Create dotplot
    top_results <- head(enrichment_results$significant_results, 20)
    
    p <- ggplot(top_results, aes(x = EnrichmentRatio, y = reorder(Description, EnrichmentRatio))) +
        geom_point(aes(size = Count, color = pvalue)) +
        scale_color_gradient(low = "red", high = "blue") +
        labs(title = "Wolbachia GO Enrichment Analysis",
             x = "Enrichment Ratio",
             y = "GO Term",
             size = "Gene Count",
             color = "P-value") +
        theme_minimal() +
        theme(axis.text.y = element_text(size = 8))
    
    plot_file <- paste0(args$output_prefix, "_dotplot.pdf")
    ggsave(plot_file, p, width = 10, height = 8)
    cat("✓ Dotplot saved to:", plot_file, "\n")
    
    # Create barplot
    p2 <- ggplot(top_results, aes(x = reorder(Description, EnrichmentRatio), y = EnrichmentRatio)) +
        geom_bar(stat = "identity", fill = "steelblue") +
        coord_flip() +
        labs(title = "Wolbachia GO Enrichment Analysis",
             x = "GO Term",
             y = "Enrichment Ratio") +
        theme_minimal() +
        theme(axis.text.y = element_text(size = 8))
    
    barplot_file <- paste0(args$output_prefix, "_barplot.pdf")
    ggsave(barplot_file, p2, width = 10, height = 8)
    cat("✓ Barplot saved to:", barplot_file, "\n")
}

# Print summary
cat("\n=== Wolbachia GO Analysis Complete ===\n")
cat("Input UniProt IDs:", length(gene_list), "\n")
cat("UniProt IDs in database:", length(gene_list_with_suffix[gene_list_with_suffix %in% unique(gene_to_go$GID)]), "\n")
cat("GO terms tested:", nrow(enrichment_results$all_results), "\n")
cat("Significant terms:", nrow(enrichment_results$significant_results), "\n")

if (nrow(enrichment_results$significant_results) > 0) {
    cat("\nTop 5 enriched GO terms:\n")
    for (i in 1:min(5, nrow(enrichment_results$significant_results))) {
        result <- enrichment_results$significant_results[i, ]
        cat(i, ".", result$Description, "\n")
        cat("   GO ID:", result$ID, "\n")
        cat("   P-value:", result$pvalue, "\n")
        cat("   Enrichment ratio:", result$EnrichmentRatio, "\n")
        cat("   Gene count:", result$Count, "\n\n")
    }
} 