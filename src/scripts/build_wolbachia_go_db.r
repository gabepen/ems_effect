#!/usr/bin/env Rscript

# Script to build a custom Wolbachia GO database from UniProt annotations
# This creates a usable GO database for Wolbachia gene enrichment analysis

cat("Building Wolbachia GO Database from UniProt\n")
cat("==========================================\n\n")

# Install required packages
if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cloud.r-project.org")
}

if (!requireNamespace("AnnotationDbi", quietly = TRUE)) {
    BiocManager::install("AnnotationDbi", update = FALSE, ask = FALSE)
}

if (!requireNamespace("GO.db", quietly = TRUE)) {
    BiocManager::install("GO.db", update = FALSE, ask = FALSE)
}

if (!requireNamespace("httr", quietly = TRUE)) {
    install.packages("httr", repos = "https://cloud.r-project.org")
}

library(AnnotationDbi)
library(GO.db)
library(httr)

# Function to download Wolbachia wMel GO annotations from UniProt
download_wolbachia_go_annotations <- function() {
    cat("Downloading Wolbachia wMel GO annotations from UniProt (new API)...\n")
    
    base_url <- "https://rest.uniprot.org/uniprotkb/search"
    query <- "organism_id:163164 AND go:*"
    fields <- "accession,id,gene_names,annotation_score,organism_name,go"
    format <- "tsv"
    size <- 500  # max per page
    
    all_lines <- NULL
    next_url <- base_url
    page <- 1
    
    repeat {
        cat(sprintf("  Fetching page %d...\n", page))
        response <- httr::GET(
            url = next_url,
            query = list(
                query = query,
                fields = fields,
                format = format,
                size = size
            ),
            httr::timeout(60)
        )
        
        if (httr::status_code(response) == 200) {
            content_text <- httr::content(response, "text")
            lines <- strsplit(content_text, "\n")[[1]]
            if (length(lines) > 1) {
                if (is.null(all_lines)) {
                    all_lines <- lines
                } else {
                    # skip header for subsequent pages
                    all_lines <- c(all_lines, lines[-1])
                }
            }
            # Check for pagination: look for 'Link' header with rel="next"
            link_header <- httr::headers(response)[["link"]]
            if (!is.null(link_header) && grepl('rel="next"', link_header)) {
                # Extract next URL
                next_url <- sub('^.*<([^>]+)>; rel="next".*$', '\\1', link_header)
                page <- page + 1
            } else {
                break
            }
        } else {
            cat("✗ Error downloading from UniProt (status:", httr::status_code(response), ")\n")
            cat("Response content:", httr::content(response, "text"), "\n")
            break
        }
    }
    
    if (!is.null(all_lines) && length(all_lines) > 1) {
        cat("✓ Downloaded", length(all_lines) - 1, "Wolbachia wMel protein entries with GO annotations\n")
        return(all_lines)
    } else {
        cat("✗ No Wolbachia wMel GO annotations found\n")
        return(NULL)
    }
}

# Function to parse UniProt GO annotations (wMel)
parse_uniprot_go_annotations <- function(lines) {
    cat("Parsing UniProt GO annotations...\n")
    
    if (is.null(lines) || length(lines) < 2) {
        return(NULL)
    }
    
    # Parse header
    header <- strsplit(lines[1], "\t")[[1]]
    cat("Header fields:", paste(header, collapse=", "), "\n")
    
    # Parse data
    annotations <- list()
    go_terms <- list()
    
    for (i in 2:length(lines)) {
        if (nchar(lines[i]) > 0) {
            parts <- strsplit(lines[i], "\t")[[1]]
            
            if (length(parts) >= 6) {
                accession <- parts[1]
                uniprot_id <- parts[2]
                gene_names <- parts[3]
                annotation_score <- parts[4]
                organism <- parts[5]
                go_annotations <- parts[6]
                
                # Parse GO annotations (format: "desc1 [GO:ID1]; desc2 [GO:ID2]; ...")
                if (nchar(go_annotations) > 0) {
                    go_items <- strsplit(go_annotations, "; ")[[1]]
                    go_id_list <- c()
                    go_term_list <- c()
                    for (item in go_items) {
                        m <- regexec("(.+) \\[GO:([0-9]+)\\]", item)
                        res <- regmatches(item, m)[[1]]
                        if (length(res) == 3) {
                            go_term_list <- c(go_term_list, res[2])
                            go_id <- paste0("GO:", res[3])
                            go_id_list <- c(go_id_list, go_id)
                            if (!(go_id %in% names(go_terms))) {
                                go_terms[[go_id]] <- res[2]
                            }
                        }
                    }
                    # Create gene ID (use UniProt ID as primary identifier)
                    gene_id <- uniprot_id
                    # Store annotations
                    annotations[[gene_id]] <- list(
                        accession = accession,
                        gene_names = gene_names,
                        organism = organism,
                        go_ids = go_id_list
                    )
                }
            }
        }
    }
    
    cat("✓ Parsed", length(annotations), "genes with GO annotations\n")
    cat("✓ Found", length(go_terms), "unique GO terms\n")
    
    return(list(annotations = annotations, go_terms = go_terms))
}

# Function to fetch GO term descriptions
fetch_go_descriptions <- function(go_terms) {
    cat("Fetching GO term descriptions...\n")
    
    # Since we already have descriptions from UniProt, use those
    # Only try to fetch from GO.db if we don't have a description
    go_descriptions <- list()
    
    for (go_id in names(go_terms)) {
        # Use the description we already parsed from UniProt
        if (go_terms[[go_id]] != "") {
            go_descriptions[[go_id]] <- go_terms[[go_id]]
        } else {
            # Fallback to GO.db if needed
            tryCatch({
                if (!requireNamespace("GO.db", quietly = TRUE)) {
                    BiocManager::install("GO.db", update = FALSE, ask = FALSE)
                }
                library(GO.db)
                term <- GO.db::GOTERM[[go_id]]
                if (!is.null(term)) {
                    go_descriptions[[go_id]] <- GO.db::Term(term)
                } else {
                    go_descriptions[[go_id]] <- "Unknown"
                }
            }, error = function(e) {
                go_descriptions[[go_id]] <- "Unknown"
            })
        }
    }
    
    cat("✓ Fetched descriptions for", length(go_descriptions), "GO terms\n")
    return(go_descriptions)
}

# Function to create custom GO database
create_custom_go_database <- function(parsed_data) {
    cat("Creating custom GO database...\n")
    
    if (is.null(parsed_data)) {
        cat("✗ No data to create database from\n")
        return(NULL)
    }
    
    annotations <- parsed_data$annotations
    go_terms <- parsed_data$go_terms
    
    # Fetch GO descriptions
    go_descriptions <- fetch_go_descriptions(go_terms)
    
    # Create data frames for database
    gene_to_go <- data.frame()
    go_info <- data.frame()
    
    # Build gene-to-GO mappings
    for (gene_id in names(annotations)) {
        go_ids <- annotations[[gene_id]]$go_ids
        
        for (go_id in go_ids) {
            gene_to_go <- rbind(gene_to_go, data.frame(
                GID = gene_id,
                GO = go_id,
                EVIDENCE = "IEA",  # Inferred from Electronic Annotation
                stringsAsFactors = FALSE
            ))
        }
    }
    
    # Build GO term information
    for (go_id in names(go_terms)) {
        description <- go_descriptions[[go_id]]
        if (is.null(description) || description == "") {
            description <- "Unknown"
        }
        go_info <- rbind(go_info, data.frame(
            GOID = go_id,
            TERM = description,
            ONTOLOGY = "BP",  # Default to Biological Process
            stringsAsFactors = FALSE
        ))
    }
    
    cat("✓ Created gene-to-GO mappings:", nrow(gene_to_go), "entries\n")
    cat("✓ Created GO term information:", nrow(go_info), "entries\n")
    
    return(list(gene_to_go = gene_to_go, go_info = go_info))
}

# Function to perform GO enrichment analysis with custom database
perform_custom_go_enrichment <- function(custom_db, gene_list) {
    cat("Performing GO enrichment analysis with custom database...\n")
    
    if (is.null(custom_db)) {
        cat("✗ No custom database available\n")
        return(NULL)
    }
    
    # Install clusterProfiler if not available
    if (!requireNamespace("clusterProfiler", quietly = TRUE)) {
        BiocManager::install("clusterProfiler", update = FALSE, ask = FALSE)
    }
    library(clusterProfiler)
    
    # Create background gene set (all genes in database)
    background_genes <- unique(custom_db$gene_to_go$GID)
    
    # Filter input genes to those in database
    available_genes <- gene_list[gene_list %in% background_genes]
    
    if (length(available_genes) == 0) {
        cat("✗ None of the input genes are in the Wolbachia GO database\n")
        return(NULL)
    }
    
    cat("✓ Found", length(available_genes), "genes in Wolbachia GO database\n")
    
    # Perform enrichment analysis
    tryCatch({
        # Create enrichment result manually
        enrichment_result <- data.frame()
        
        # Get unique GO terms for input genes
        input_go_terms <- custom_db$gene_to_go[custom_db$gene_to_go$GID %in% available_genes, ]
        
        if (nrow(input_go_terms) == 0) {
            cat("✗ No GO terms found for input genes\n")
            return(NULL)
        }
        
        # Calculate enrichment for each GO term
        unique_go_terms <- unique(input_go_terms$GO)
        
        for (go_term in unique_go_terms) {
            # Count genes with this GO term in input
            input_with_go <- sum(input_go_terms$GO == go_term)
            
            # Count genes with this GO term in background
            background_with_go <- sum(custom_db$gene_to_go$GO == go_term)
            
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
            go_description <- custom_db$go_info$TERM[custom_db$go_info$GOID == go_term]
            if (length(go_description) == 0) go_description <- "Unknown"
            
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
                stringsAsFactors = FALSE
            ))
        }
        
        # Sort by p-value
        enrichment_result <- enrichment_result[order(enrichment_result$pvalue), ]
        
        cat("✓ Found", nrow(enrichment_result), "enriched GO terms\n")
        return(enrichment_result)
        
    }, error = function(e) {
        cat("✗ Error in enrichment analysis:", e$message, "\n")
        return(NULL)
    })
}

# Main execution
main <- function() {
    # Download and parse Wolbachia GO annotations
    lines <- download_wolbachia_go_annotations()
    parsed_data <- parse_uniprot_go_annotations(lines)
    
    if (is.null(parsed_data)) {
        cat("Failed to obtain Wolbachia GO annotations. Exiting.\n")
        return()
    }
    
    # Create custom database
    custom_db <- create_custom_go_database(parsed_data)
    
    if (is.null(custom_db)) {
        cat("Failed to create custom database. Exiting.\n")
        return()
    }
    
    # Save database to files
    output_dir <- "/storage1/gabe/ems_effect_code/wolbachia_go_db"
    dir.create(output_dir, showWarnings = FALSE)
    
    write.table(custom_db$gene_to_go, 
                file.path(output_dir, "gene_to_go.txt"), 
                sep = "\t", quote = FALSE, row.names = FALSE)
    
    write.table(custom_db$go_info, 
                file.path(output_dir, "go_terms.txt"), 
                sep = "\t", quote = FALSE, row.names = FALSE)
    
    cat("✓ Custom Wolbachia GO database saved to:", output_dir, "\n")
    
    # Example usage
    cat("\nExample usage:\n")
    cat("1. Save your gene list to a file (e.g., my_genes.txt)\n")
    cat("2. Run: Rscript wolbachia_go_enrichment.r --gene_file my_genes.txt\n")
    
    cat("\nDatabase summary:\n")
    cat("- Genes with GO annotations:", length(unique(custom_db$gene_to_go$GID)), "\n")
    cat("- Unique GO terms:", length(unique(custom_db$gene_to_go$GO)), "\n")
    cat("- Total gene-GO associations:", nrow(custom_db$gene_to_go), "\n")
}

# Run main function
main() 