#!/usr/bin/env Rscript

# Script to find Wolbachia GO annotation databases
# This will help identify available Wolbachia-specific GO resources

cat("Searching for Wolbachia GO annotation databases...\n\n")

# Install required packages
if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cloud.r-project.org")
}

# Check for Wolbachia annotation packages
cat("Checking Bioconductor for Wolbachia annotation packages...\n")

# List of potential Wolbachia annotation packages
wolbachia_packages <- c(
    "org.Wolbachia.eg.db",
    "org.Wpip.eg.db",  # Wolbachia pipientis
    "org.Wmel.eg.db",  # Wolbachia melophagi
    "org.Wendosymbiont.eg.db",
    "org.Wolbachia.eg.db"
)

available_packages <- c()
for (pkg in wolbachia_packages) {
    tryCatch({
        if (requireNamespace(pkg, quietly = TRUE)) {
            cat("✓ Found package:", pkg, "\n")
            available_packages <- c(available_packages, pkg)
        } else {
            cat("✗ Package not available:", pkg, "\n")
        }
    }, error = function(e) {
        cat("✗ Error checking package", pkg, ":", e$message, "\n")
    })
}

if (length(available_packages) == 0) {
    cat("\nNo Wolbachia annotation packages found in Bioconductor.\n")
    cat("Let's check what's available...\n\n")
    
    # List all available organism packages
    cat("Available organism packages in Bioconductor:\n")
    tryCatch({
        # Get list of available packages
        bioc_packages <- BiocManager::available()
        org_packages <- bioc_packages[grepl("^org\\..*\\.eg\\.db$", bioc_packages)]
        
        cat("Found", length(org_packages), "organism annotation packages:\n")
        for (pkg in head(org_packages, 20)) {
            cat("  ", pkg, "\n")
        }
        if (length(org_packages) > 20) {
            cat("  ... and", length(org_packages) - 20, "more\n")
        }
        
        # Search for any packages containing "wolbachia" or "wol"
        wolbachia_related <- org_packages[grepl("wolbachia|wol", org_packages, ignore.case = TRUE)]
        if (length(wolbachia_related) > 0) {
            cat("\nWolbachia-related packages found:\n")
            for (pkg in wolbachia_related) {
                cat("  ", pkg, "\n")
            }
        } else {
            cat("\nNo Wolbachia-related packages found in Bioconductor.\n")
        }
        
    }, error = function(e) {
        cat("Error listing packages:", e$message, "\n")
    })
}

# Check for alternative sources
cat("\nChecking alternative sources for Wolbachia GO annotations...\n")

# 1. Check if we can create a custom database from UniProt
cat("1. Checking UniProt for Wolbachia GO annotations...\n")
tryCatch({
    # Install httr if not available
    if (!requireNamespace("httr", quietly = TRUE)) {
        install.packages("httr", repos = "https://cloud.r-project.org")
    }
    library(httr)
    
    # Search UniProt for Wolbachia proteins with GO annotations
    uniprot_url <- "https://rest.uniprot.org/uniprotkb/stream"
    query <- "organism_id:953 AND annotation:(type:go)"
    
    response <- GET(uniprot_url, query = list(query = query, format = "tsv", fields = "accession,go_id,go_term"))
    
    if (status_code(response) == 200) {
        content_text <- content(response, "text")
        lines <- strsplit(content_text, "\n")[[1]]
        
        if (length(lines) > 1) {  # More than just header
            cat("✓ Found Wolbachia GO annotations in UniProt\n")
            cat("  Number of entries:", length(lines) - 1, "\n")
            cat("  First few entries:\n")
            for (i in 2:min(6, length(lines))) {
                cat("    ", lines[i], "\n")
            }
        } else {
            cat("✗ No Wolbachia GO annotations found in UniProt\n")
        }
    } else {
        cat("✗ Error accessing UniProt (status:", status_code(response), ")\n")
    }
}, error = function(e) {
    cat("✗ Error checking UniProt:", e$message, "\n")
})

# 2. Check KEGG for Wolbachia GO annotations
cat("\n2. Checking KEGG for Wolbachia GO annotations...\n")
tryCatch({
    # Check if we can access KEGG
    kegg_response <- GET("https://rest.kegg.jp/list/wol", timeout(10))
    if (status_code(kegg_response) == 200) {
        cat("✓ KEGG Wolbachia data accessible\n")
        
        # Get a few Wolbachia genes and check for GO annotations
        wol_content <- content(kegg_response, "text")
        wol_lines <- strsplit(wol_content, "\n")[[1]]
        
        # Check first few genes for GO annotations
        for (i in 1:min(5, length(wol_lines))) {
            if (nchar(wol_lines[i]) > 0) {
                parts <- strsplit(wol_lines[i], "\t")[[1]]
                if (length(parts) >= 1) {
                    gene_id <- parts[1]
                    cat("  Checking gene:", gene_id, "\n")
                    
                    # Get gene details
                    gene_response <- GET(paste0("https://rest.kegg.jp/get/", gene_id), timeout(5))
                    if (status_code(gene_response) == 200) {
                        gene_content <- content(gene_response, "text")
                        if (grepl("GO:", gene_content)) {
                            cat("    ✓ Contains GO annotations\n")
                        } else {
                            cat("    ✗ No GO annotations found\n")
                        }
                    }
                }
            }
        }
    } else {
        cat("✗ Cannot access KEGG (status:", status_code(kegg_response), ")\n")
    }
}, error = function(e) {
    cat("✗ Error checking KEGG:", e$message, "\n")
})

# 3. Check for custom database creation options
cat("\n3. Checking options for creating custom Wolbachia GO database...\n")

cat("Options for creating Wolbachia GO database:\n")
cat("a) Use UniProt GO annotations for Wolbachia proteins\n")
cat("b) Use ortholog mapping to transfer GO annotations from model organisms\n")
cat("c) Use InterProScan to predict GO terms from protein sequences\n")
cat("d) Use existing Wolbachia genome annotations (if available)\n")

# Provide example code for creating custom database
cat("\nExample approach for creating custom Wolbachia GO database:\n")
cat("1. Download Wolbachia protein sequences from UniProt\n")
cat("2. Extract GO annotations for these proteins\n")
cat("3. Create a custom annotation database using AnnotationDbi\n")
cat("4. Use this database with clusterProfiler for GO enrichment\n")

cat("\nSearch completed!\n") 