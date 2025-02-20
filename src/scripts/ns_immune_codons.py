import json
from typing import Dict, List, Set


def analyze_ems_mutations(codon_table: Dict[str, List[str]]) -> List[str]:
    '''Find codons that only produce synonymous mutations from EMS changes.
    
    Args:
        codon_table: Dictionary mapping amino acids to their codons
        
    Returns:
        List[str]: List of codons that can only produce synonymous mutations 
                  from C>T or G>A changes
    '''
    # Create reverse lookup (codon -> amino acid)
    codon_to_aa = {}
    for aa, codons in codon_table.items():
        if aa != 'starts':  # Skip start codons list
            for codon in codons:
                codon_to_aa[codon] = aa
    
    immune_codons = []
    
    for codon in codon_to_aa:
        original_aa = codon_to_aa[codon]
        is_immune = True
        has_ems_site = False
        
        # Check each position in the codon
        for i, base in enumerate(codon):
            if base == 'C':
                has_ems_site = True
                # Test C>T mutation
                mutated = codon[:i] + 'T' + codon[i+1:]
                if mutated in codon_to_aa and codon_to_aa[mutated] != original_aa:
                    is_immune = False
                    break
            elif base == 'G':
                has_ems_site = True
                # Test G>A mutation
                mutated = codon[:i] + 'A' + codon[i+1:]
                if mutated in codon_to_aa and codon_to_aa[mutated] != original_aa:
                    is_immune = False
                    break
        
        if is_immune and has_ems_site:
            immune_codons.append(codon)
            
    return sorted(immune_codons)

# Test with the codon table
with open('/storage1/gabe/ems_effect_code/data/references/11.json') as f:
    codon_table = json.load(f)

immune_codons = analyze_ems_mutations(codon_table)

print("\nCodons immune to non-synonymous EMS mutations:")
for codon in immune_codons:
    aa = [aa for aa, codons in codon_table.items() if codon in codons][0]
    print(f"{codon} ({aa})")

print(f"\nTotal immune codons: {len(immune_codons)}")
print("Total codons in codon table: ", len(codon_table))

