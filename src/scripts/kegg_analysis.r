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
    
    # Test the correct KEGG REST API URL
    test_url <- "https://rest.kegg.jp/list/organism"
    
    tryCatch({
        response <- GET(test_url, timeout(10))
        if (status_code(response) == 200) {
            cat("✓ KEGG REST API (rest.kegg.jp) is accessible\n")
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

# Function to get organism list from KEGG
get_kegg_organisms <- function() {
    cat("Fetching KEGG organism list...\n")
    
    tryCatch({
        response <- GET("https://rest.kegg.jp/list/organism", timeout(10))
        if (status_code(response) == 200) {
            content_text <- content(response, "text")
            # Parse the tab-separated response
            lines <- strsplit(content_text, "\n")[[1]]
            organisms <- data.frame()
            
            for (line in lines) {
                if (nchar(line) > 0) {
                    parts <- strsplit(line, "\t")[[1]]
                    if (length(parts) >= 3) {
                        organisms <- rbind(organisms, data.frame(
                            code = parts[1],
                            name = parts[2],
                            description = parts[3]
                        ))
                    }
                }
            }
            return(organisms)
        } else {
            cat("Failed to fetch organism list. Status code:", status_code(response), "\n")
            return(NULL)
        }
    }, error = function(e) {
        cat("Error fetching organism list:", e$message, "\n")
        return(NULL)
    })
}

# Function to check if Wolbachia is available in KEGG
check_wolbachia_availability <- function() {
    cat("Checking Wolbachia availability in KEGG...\n")
    
    tryCatch({
        # Check for Wolbachia specifically
        response <- GET("https://rest.kegg.jp/list/wol", timeout(10))
        if (status_code(response) == 200) {
            content_text <- content(response, "text")
            if (nchar(content_text) > 0 && !grepl("not found", content_text, ignore.case = TRUE)) {
                cat("✓ Wolbachia (wol) is available in KEGG\n")
                return(TRUE)
            } else {
                cat("✗ Wolbachia (wol) not found in KEGG\n")
                return(FALSE)
            }
        } else {
            cat("✗ Error checking Wolbachia availability. Status code:", status_code(response), "\n")
            return(FALSE)
        }
    }, error = function(e) {
        cat("✗ Error checking Wolbachia availability:", e$message, "\n")
        return(FALSE)
    })
}

# Set up argument parser
parser <- ArgumentParser(description = "KEGG pathway enrichment analysis")
parser$add_argument("--gene_list", type = "character", 
                   help = "Comma-separated list of NCBI Gene IDs")
parser$add_argument("--gene_file", type = "character",
                   help = "File containing gene IDs (one per line)")
parser$add_argument("--pvalue_cutoff", type = "double", default = 0.05,
                   help = "P-value cutoff for enrichment analysis (default: 0.05)")
parser$add_argument("--output_prefix", type = "character", default = "kegg_enrichment",
                   help = "Prefix for output files (default: kegg_enrichment)")
parser$add_argument("--organism", type = "character", default = "wol",
                   help = "KEGG organism code (default: wol for Wolbachia)")

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

# Test API connectivity first
if (!test_kegg_api()) {
    cat("\nERROR: Cannot connect to KEGG REST API.\n")
    cat("This could be due to:\n")
    cat("1. Network connectivity issues\n")
    cat("2. Firewall blocking rest.kegg.jp\n")
    cat("3. KEGG server being down\n\n")
    cat("Please check your network connection and try again.\n")
    cat("If the issue persists, consider using GO enrichment analysis as an alternative.\n")
    quit(status = 1)
}

# Check if the specified organism is available
if (!check_wolbachia_availability()) {
    cat("\nWARNING: Wolbachia (wol) may not be available in KEGG.\n")
    cat("Available organisms can be checked with: curl https://rest.kegg.jp/list/organism\n")
    cat("Continuing with analysis anyway...\n\n")
}

# Convert gene_list string to vector
cat("Raw gene_list argument:", args$gene_list, "\n")
cat("Length of raw gene_list:", nchar(args$gene_list), "\n")

gene_list <- unlist(strsplit(args$gene_list, ","))
gene_list <- trimws(gene_list)  # Remove any whitespace

cat("Parsed gene list:", paste(gene_list, collapse=", "), "\n")
cat("Input gene list contains", length(gene_list), "genes\n")

# Function to convert NCBI gene IDs to Wolbachia KEGG IDs using REST API
convert_ncbi_to_wolbachia_kegg <- function(ncbi_ids) {
    cat("Converting NCBI gene IDs to Wolbachia KEGG IDs...\n")
    
    # Use the correct KEGG REST API format: /conv/wol/ncbi-geneid
    cat("Using KEGG REST API conversion: /conv/wol/ncbi-geneid\n")
    
    # Try direct conversion using KEGG REST API
    matched_genes <- data.frame()
    
    for (ncbi_id in ncbi_ids) {
        tryCatch({
            # Use the correct URL format
            conversion_url <- paste0("https://rest.kegg.jp/conv/wol/ncbi-geneid:", ncbi_id)
            cat("Converting", ncbi_id, "...\n")
            
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
                                kegg_id <- parts[1]
                                ncbi_id_result <- parts[2]
                                
                                matched_genes <- rbind(matched_genes, data.frame(
                                    ncbi_geneid = ncbi_id,
                                    kegg = kegg_id
                                ))
                                cat("  Found match:", ncbi_id, "->", kegg_id, "\n")
                            }
                        }
                    }
                }
            } else {
                cat("  No conversion found for", ncbi_id, "(status:", status_code(response), ")\n")
            }
        }, error = function(e) {
            cat("  Error converting", ncbi_id, ":", e$message, "\n")
        })
    }
    
    cat("Successfully matched", nrow(matched_genes), "out of", length(ncbi_ids), "genes\n")
    return(matched_genes)
}

# Try to convert NCBI Gene IDs to KEGG IDs
cat("Converting NCBI Gene IDs to KEGG IDs...\n")
tryCatch({
    # For Wolbachia, we need to use a custom approach since clusterProfiler doesn't handle it well
    cat("Using custom Wolbachia gene ID conversion...\n")
    
    # Use our custom conversion function
    converted_ids <- convert_ncbi_to_wolbachia_kegg(gene_list)
    
    # If no matches found, try alternative approach
    if (nrow(converted_ids) == 0) {
        cat("No direct NCBI matches found. Trying alternative approach...\n")
        
        # For Wolbachia, we might need to use the genes directly
        # Let's check if the input genes are already in Wolbachia format
        cat("Checking if genes are already in Wolbachia format...\n")
        
        # Get Wolbachia gene list again
        wol_response <- GET("https://rest.kegg.jp/list/wol", timeout(10))
        if (status_code(wol_response) == 200) {
            wol_content <- content(wol_response, "text")
            wol_lines <- strsplit(wol_content, "\n")[[1]]
            
            # Extract Wolbachia gene IDs
            wol_gene_ids <- c()
            for (line in wol_lines) {
                if (nchar(line) > 0) {
                    parts <- strsplit(line, "\t")[[1]]
                    if (length(parts) >= 1) {
                        gene_id <- parts[1]
                        wol_gene_ids <- c(wol_gene_ids, gene_id)
                    }
                }
            }
            
            # Check if any input genes match Wolbachia gene IDs
            matched_wol_ids <- c()
            for (gene in gene_list) {
                # Try different formats
                possible_formats <- c(
                    gene,
                    paste0("wol:", gene),
                    paste0("WD_", gene)
                )
                
                for (format in possible_formats) {
                    if (format %in% wol_gene_ids) {
                        matched_wol_ids <- c(matched_wol_ids, format)
                        break
                    }
                }
            }
            
            if (length(matched_wol_ids) > 0) {
                cat("Found", length(matched_wol_ids), "genes in Wolbachia format\n")
                converted_ids <- data.frame(
                    ncbi_geneid = gene_list[1:length(matched_wol_ids)],
                    kegg = matched_wol_ids
                )
            }
        }
    }
    
    cat("Conversion results:\n")
    print(head(converted_ids))
    
    if (nrow(converted_ids) == 0) {
        cat("WARNING: No genes were successfully converted to KEGG IDs.\n")
        cat("This could indicate:\n")
        cat("1. The genes are not present in KEGG for Wolbachia\n")
        cat("2. The gene IDs are not in the expected format\n")
        cat("3. Network connectivity issues\n")
        cat("4. Need to use different gene identifiers\n")
        cat("\nSuggestion: Try using Wolbachia gene IDs (WD_XXXX format) instead of NCBI gene IDs\n")
        quit(status = 1)
    }
    
    # Use converted KEGG IDs for analysis
    gene_list_for_analysis <- converted_ids$kegg
    cat("Using", length(gene_list_for_analysis), "converted KEGG IDs for analysis\n")
    
    # Perform KEGG enrichment analysis
    cat("Performing KEGG enrichment analysis...\n")
    kegg_enrich <- enrichKEGG(gene = gene_list_for_analysis, 
                              organism = args$organism, 
                              pvalueCutoff = args$pvalue_cutoff)
    
    # View results
    cat("KEGG Enrichment Results:\n")
    if (nrow(kegg_enrich) > 0) {
        print(head(kegg_enrich))
        
        # Save results to file
        output_file <- paste0(args$output_prefix, "_results.txt")
        write.table(as.data.frame(kegg_enrich), file = output_file, 
                    sep = "\t", quote = FALSE, row.names = FALSE)
        cat("Results saved to:", output_file, "\n")
        
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
        
    } else {
        cat("No significantly enriched pathways found.\n")
        cat("This could indicate:\n")
        cat("1. The gene list is too small\n")
        cat("2. The genes are not functionally related\n")
        cat("3. The p-value cutoff is too strict\n")
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
