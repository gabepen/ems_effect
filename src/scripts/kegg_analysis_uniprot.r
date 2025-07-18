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

if (!requireNamespace("pathview", quietly = TRUE)) {
    BiocManager::install("pathview", update = FALSE, ask = FALSE)
}

# Install and load required packages
if (!requireNamespace("argparse", quietly = TRUE)) {
    install.packages("argparse", repos = "https://cloud.r-project.org")
}

if (!requireNamespace("httr", quietly = TRUE)) {
    install.packages("httr", repos = "https://cloud.r-project.org")
}

if (!requireNamespace("jsonlite", quietly = TRUE)) {
    install.packages("jsonlite", repos = "https://cloud.r-project.org")
}

# Load libraries
library(clusterProfiler)
library(pathview)
library(argparse)
library(httr)
library(jsonlite)

# Function to test KEGG REST API connectivity
test_kegg_api <- function() {
    cat("Testing KEGG REST API connectivity...\n")
    
    tryCatch({
        response <- GET("https://rest.kegg.jp/list/organism", timeout(10))
        if (status_code(response) == 200) {
            cat("✓ KEGG REST API is accessible\n")
            return(TRUE)
        } else {
            cat("✗ KEGG REST API returned status code:", status_code(response), "\n")
            return(FALSE)
        }
    }, error = function(e) {
        cat("✗ Error connecting to KEGG REST API:", e$message, "\n")
        return(FALSE)
    })
}

# Function to convert UniProt IDs to Wolbachia KEGG IDs
convert_uniprot_to_wolbachia_kegg <- function(uniprot_ids) {
    cat("Converting UniProt IDs to Wolbachia KEGG IDs...\n")
    
    matched_genes <- data.frame()
    
    for (uniprot_id in uniprot_ids) {
        tryCatch({
            # Clean the UniProt ID (remove 'up:' prefix if present)
            clean_id <- gsub("^up:", "", uniprot_id)
            
            # Try the correct KEGG REST API conversion format
            # The format should be: /conv/wol/uniprot:ID
            conversion_url <- paste0("https://rest.kegg.jp/conv/wol/uniprot:", clean_id)
            cat("Converting", clean_id, "...\n")
            
            response <- GET(conversion_url, timeout(10))
            if (status_code(response) == 200) {
                content_text <- content(response, "text")
                if (nchar(content_text) > 0) {
                    # Parse the conversion result
                    lines <- strsplit(content_text, "\n")[[1]]
                    for (line in lines) {
                        if (nchar(line) > 0) {
                            parts <- strsplit(line, "\t")[[1]]
                            if (length(parts) >= 2) {
                                # The format should be: wol:WD_XXXX    up:Q73IZ0
                                # So parts[1] should be the KEGG ID and parts[2] should be the UniProt ID
                                kegg_id <- parts[1]
                                uniprot_id_result <- parts[2]
                                
                                # Check if we got a Wolbachia KEGG ID
                                if (grepl("^wol:WD_", kegg_id)) {
                                    matched_genes <- rbind(matched_genes, data.frame(
                                        uniprot_id = clean_id,
                                        kegg_id = kegg_id
                                    ))
                                    cat("  Found match:", clean_id, "->", kegg_id, "\n")
                                } else {
                                    # If we got the wrong format, try the reverse
                                    if (grepl("^up:", kegg_id) && grepl("^wol:WD_", uniprot_id_result)) {
                                        matched_genes <- rbind(matched_genes, data.frame(
                                            uniprot_id = clean_id,
                                            kegg_id = uniprot_id_result
                                        ))
                                        cat("  Found match (reversed):", clean_id, "->", uniprot_id_result, "\n")
                                    } else {
                                        cat("  Unexpected format:", line, "\n")
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                cat("  No conversion found for", clean_id, "(status:", status_code(response), ")\n")
            }
        }, error = function(e) {
            cat("  Error converting", uniprot_id, ":", e$message, "\n")
        })
    }
    
    cat("Successfully matched", nrow(matched_genes), "out of", length(uniprot_ids), "UniProt IDs\n")
    return(matched_genes)
}

# Function to get background gene set for Wolbachia
get_wolbachia_background_genes <- function() {
    cat("Fetching Wolbachia background gene set...\n")
    
    tryCatch({
        response <- GET("https://rest.kegg.jp/list/wol", timeout(30))
        if (status_code(response) == 200) {
            content_text <- content(response, "text")
            lines <- strsplit(content_text, "\n")[[1]]
            
            background_genes <- c()
            for (line in lines) {
                if (nchar(line) > 0) {
                    parts <- strsplit(line, "\t")[[1]]
                    if (length(parts) >= 1) {
                        gene_id <- parts[1]
                        # Only include CDS genes (not tRNAs, etc.)
                        if (grepl("^wol:WD_\\d+$", gene_id)) {
                            background_genes <- c(background_genes, gene_id)
                        }
                    }
                }
            }
            cat("Found", length(background_genes), "CDS genes in Wolbachia background\n")
            return(background_genes)
        } else {
            cat("Failed to fetch background genes. Status code:", status_code(response), "\n")
            return(NULL)
        }
    }, error = function(e) {
        cat("Error fetching background genes:", e$message, "\n")
        return(NULL)
    })
}

# Set up argument parser
parser <- ArgumentParser(description = "KEGG pathway enrichment analysis for Wolbachia using UniProt IDs")
parser$add_argument("--uniprot_list", type = "character", 
                   help = "Comma-separated list of UniProt IDs")
parser$add_argument("--uniprot_file", type = "character",
                   help = "File containing UniProt IDs (one per line)")
parser$add_argument("--pvalue_cutoff", type = "double", default = 0.05,
                   help = "P-value cutoff for enrichment analysis (default: 0.05)")
parser$add_argument("--output_prefix", type = "character", default = "wolbachia_kegg_enrichment",
                   help = "Prefix for output files (default: wolbachia_kegg_enrichment)")
parser$add_argument("--organism", type = "character", default = "wol",
                   help = "KEGG organism code (default: wol for Wolbachia)")

# Parse arguments
args <- parser$parse_args()

# Check that either uniprot_list or uniprot_file is provided
if (is.null(args$uniprot_list) && is.null(args$uniprot_file)) {
    cat("ERROR: Either --uniprot_list or --uniprot_file must be provided\n")
    cat("Use --help for usage information\n")
    quit(status = 1)
}

# Get UniProt ID list from either argument
if (!is.null(args$uniprot_file)) {
    # Read UniProt IDs from file
    cat("Reading UniProt IDs from file:", args$uniprot_file, "\n")
    tryCatch({
        uniprot_ids <- readLines(args$uniprot_file)
        # Remove empty lines and whitespace
        uniprot_ids <- uniprot_ids[nchar(trimws(uniprot_ids)) > 0]
        uniprot_ids <- trimws(uniprot_ids)
        
        cat("Found", length(uniprot_ids), "UniProt IDs in file\n")
        if (length(uniprot_ids) == 0) {
            cat("ERROR: No UniProt IDs found in file\n")
            quit(status = 1)
        }
        
        # Convert to comma-separated string for processing
        args$uniprot_list <- paste(uniprot_ids, collapse = ",")
        
    }, error = function(e) {
        cat("ERROR reading UniProt ID file:", e$message, "\n")
        quit(status = 1)
    })
}

# Test API connectivity first
if (!test_kegg_api()) {
    cat("\nERROR: Cannot connect to KEGG REST API.\n")
    cat("This could be due to:\n")
    cat("1. Network connectivity issues\n")
    cat("2. Firewall blocking rest.kegg.jp\n")
    cat("3. KEGG server being down\n\n")
    cat("Please check your network connection and try again.\n")
    quit(status = 1)
}

# Convert uniprot_list string to vector
cat("Raw uniprot_list argument:", args$uniprot_list, "\n")
uniprot_list <- unlist(strsplit(args$uniprot_list, ","))
uniprot_list <- trimws(uniprot_list)  # Remove any whitespace

cat("Parsed UniProt ID list:", paste(uniprot_list, collapse=", "), "\n")
cat("Input UniProt ID list contains", length(uniprot_list), "IDs\n")

# Convert UniProt IDs to Wolbachia KEGG IDs
cat("Converting UniProt IDs to Wolbachia KEGG IDs...\n")
tryCatch({
    converted_ids <- convert_uniprot_to_wolbachia_kegg(uniprot_list)
    
    if (nrow(converted_ids) == 0) {
        cat("WARNING: No UniProt IDs were successfully converted to KEGG IDs.\n")
        cat("This could indicate:\n")
        cat("1. The UniProt IDs are not present in Wolbachia KEGG database\n")
        cat("2. The UniProt IDs are not in the expected format\n")
        cat("3. Network connectivity issues\n")
        cat("\nSuggestion: Verify your UniProt IDs are from Wolbachia proteins\n")
        quit(status = 1)
    }
    
    # Use converted KEGG IDs for analysis
    # Remove the 'wol:' prefix as clusterProfiler expects just the gene IDs
    gene_list_for_analysis <- gsub("^wol:", "", converted_ids$kegg_id)
    cat("Using", length(gene_list_for_analysis), "converted KEGG IDs for analysis\n")
    cat("Sample IDs:", paste(head(gene_list_for_analysis, 5), collapse=", "), "\n")
    
    # Get background gene set
    background_genes <- get_wolbachia_background_genes()
    if (is.null(background_genes)) {
        cat("WARNING: Could not fetch background gene set. Using default background.\n")
        background_genes <- NULL
    } else {
        # Remove the 'wol:' prefix from background genes too
        background_genes <- gsub("^wol:", "", background_genes)
        cat("Using", length(background_genes), "background genes (without wol: prefix)\n")
    }
    
    # Perform KEGG enrichment analysis
    cat("Performing KEGG enrichment analysis...\n")
    kegg_enrich <- enrichKEGG(gene = gene_list_for_analysis, 
                              organism = args$organism, 
                              pvalueCutoff = args$pvalue_cutoff,
                              universe = background_genes)
    
    # View results
    cat("KEGG Enrichment Results:\n")
    if (nrow(kegg_enrich) > 0) {
        print(head(kegg_enrich))
        
        # Save results to file
        output_file <- paste0(args$output_prefix, "_results.txt")
        write.table(as.data.frame(kegg_enrich), file = output_file, 
                    sep = "\t", quote = FALSE, row.names = FALSE)
        cat("Results saved to:", output_file, "\n")
        
        # Save conversion mapping
        conversion_file <- paste0(args$output_prefix, "_conversion_mapping.txt")
        write.table(converted_ids, file = conversion_file, 
                    sep = "\t", quote = FALSE, row.names = FALSE)
        cat("Conversion mapping saved to:", conversion_file, "\n")
        
        # Create and save dotplot
        pdf(paste0(args$output_prefix, "_dotplot.pdf"))
        dotplot(kegg_enrich)
        dev.off()
        cat("Dotplot saved to:", paste0(args$output_prefix, "_dotplot.pdf"), "\n")
        
        # Create and save barplot
        pdf(paste0(args$output_prefix, "_barplot.pdf"))
        barplot(kegg_enrich, showCategory = 20)
        dev.off()
        cat("Barplot saved to:", paste0(args$output_prefix, "_barplot.pdf"), "\n")
        
        # Create and save enrichment map
        pdf(paste0(args$output_prefix, "_enrichment_map.pdf"))
        emapplot(pairwise_termsim(kegg_enrich))
        dev.off()
        cat("Enrichment map saved to:", paste0(args$output_prefix, "_enrichment_map.pdf"), "\n")
        
    } else {
        cat("No significantly enriched pathways found.\n")
        cat("This could indicate:\n")
        cat("1. The gene list is too small\n")
        cat("2. The genes are not functionally related\n")
        cat("3. The p-value cutoff is too strict\n")
        cat("4. The genes are not present in KEGG pathways\n")
    }
    
}, error = function(e) {
    cat("ERROR during KEGG analysis:", e$message, "\n")
    cat("This could be due to:\n")
    cat("1. Network connectivity issues\n")
    cat("2. Invalid organism code\n")
    cat("3. KEGG server issues\n")
    cat("4. Invalid gene IDs\n")
    quit(status = 1)
})

cat("\nKEGG analysis completed successfully!\n")
cat("Summary:\n")
cat("- Input UniProt IDs:", length(uniprot_list), "\n")
cat("- Successfully converted to KEGG IDs:", nrow(converted_ids), "\n")
cat("- Enriched pathways found:", ifelse(nrow(kegg_enrich) > 0, nrow(kegg_enrich), 0), "\n") 