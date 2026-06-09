#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import os
import numpy as np
from typing import Dict, List, Tuple


def parse_result_file(file_path: str) -> Dict[float, Dict[str, float]]:
    results = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
                              
        ratio_line = lines[i].strip()
        if ratio_line.startswith('[') and ratio_line.endswith(']'):
            ratios = [int(x.strip()) for x in ratio_line[1:-1].split(',')]
            train_ratio = ratios[0] / 100.0                 
            
                   
            if i + 1 < len(lines):
                result_line = lines[i + 1].strip()
                metrics = {}
                
                        
                patterns = {
                    'MRR': r'MRR:\s*([\d.]+)',
                    'Hit1': r'Hit1:\s*([\d.]+)',
                    'Hit3': r'Hit3:\s*([\d.]+)',
                    'Hit10': r'Hit10:\s*([\d.]+)',
                    'Emerging_MRR': r'Emerging_MRR:\s*([\d.]+)',
                    'Emerging_Hit1': r'Emerging_Hit1:\s*([\d.]+)',
                    'Emerging_Hit3': r'Emerging_Hit3:\s*([\d.]+)',
                    'Emerging_Hit10': r'Emerging_Hit10:\s*([\d.]+)',
                    'Known_MRR': r'Known_MRR:\s*([\d.]+)',
                    'Known_Hit1': r'Known_Hit1:\s*([\d.]+)',
                    'Known_Hit3': r'Known_Hit3:\s*([\d.]+)',
                    'Known_Hit10': r'Known_Hit10:\s*([\d.]+)',
                }
                
                for key, pattern in patterns.items():
                    match = re.search(pattern, result_line)
                    if match:
                        metrics[key] = float(match.group(1))
                
                results[train_ratio] = metrics
                i += 2
            else:
                i += 1
        else:
            i += 1
    
    return results


def calculate_lpi(results: Dict[float, Dict[str, float]], metric_name: str) -> float:
              
    sorted_ratios = sorted(results.keys())
    
    if len(sorted_ratios) < 2:
        return 0.0
    
    lpi = 0.0
    
    for i in range(len(sorted_ratios) - 1):
        alpha_i = sorted_ratios[i]
        alpha_next = sorted_ratios[i + 1]
        
                    
        p_i = results[alpha_i].get(metric_name, 0.0)
        p_next = results[alpha_next].get(metric_name, 0.0)
        
                
        lpi += (p_i + p_next) / 2.0 * (alpha_next - alpha_i)
    
    return lpi


def calculate_mgi(results: Dict[float, Dict[str, float]]) -> float:
    n = len(results)
    if n == 0:
        return 0.0
    
    mgi_sum = 0.0
    
    for alpha, metrics in results.items():
        mrr_overall = metrics.get('MRR', 0.0)
        mrr_known = metrics.get('Known_MRR', 0.0)
        mrr_emerging = metrics.get('Emerging_MRR', 0.0)
        
        if mrr_overall > 0:
            diff = abs(mrr_known - mrr_emerging) / mrr_overall
            mgi_sum += diff
    
    mgi = mgi_sum / n
    return mgi


def calculate_hsr(results: Dict[float, Dict[str, float]], metric_name: str = 'MRR') -> float:
    if len(results) == 0:
        return 0.0
    
    sorted_ratios = sorted(results.keys())
    alpha_min = sorted_ratios[0]
    alpha_max = sorted_ratios[-1]
    
    mrr_min = results[alpha_min].get(metric_name, 0.0)
    mrr_max = results[alpha_max].get(metric_name, 0.0)
    
    if mrr_max == 0:
        return 0.0
    
    hsr = (mrr_max - mrr_min) / mrr_max * 100.0
    return hsr


def main():
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = 'DORADO_ICEWS14.txt'

    print('=' * 80)
    print('Generalized Evaluation Metrics')
    print('=' * 80)
    print(f'Result file: {file_path}\n')

    results = parse_result_file(file_path)

    if len(results) == 0:
        print('Error: failed to parse any result data.')
        return

    print('Parsed training ratios and corresponding metrics:')
    sorted_ratios = sorted(results.keys())
    for alpha in sorted_ratios:
        print(
            f"  Training ratio alpha = {alpha:.1f}: MRR = {results[alpha].get('MRR', 0.0):.4f}, "
            f"Known_MRR = {results[alpha].get('Known_MRR', 0.0):.4f}, "
            f"Emerging_MRR = {results[alpha].get('Emerging_MRR', 0.0):.4f}"
        )
    print()

    print('=' * 80)
    print('1. Lifelong Performance Index (LPI)')
    print('=' * 80)
    lpi_mrr = calculate_lpi(results, 'MRR')
    lpi_hit1 = calculate_lpi(results, 'Hit1')
    lpi_hit10 = calculate_lpi(results, 'Hit10')

    print(f'LPI_MRR    = {lpi_mrr:.6f}')
    print(f'LPI_Hit@1  = {lpi_hit1:.6f}')
    print(f'LPI_Hit@10 = {lpi_hit10:.6f}')
    print()

    print('=' * 80)
    print('2. Memorization-Generalization Imbalance (MGI)')
    print('=' * 80)
    mgi = calculate_mgi(results)
    print(f'MGI = {mgi:.6f}')
    print('(Lower MGI indicates a better balance between memorization and generalization.)')
    print()

    print('=' * 80)
    print('3. Horizon Sensitivity Rate (HSR)')
    print('=' * 80)
    hsr = calculate_hsr(results, 'MRR')
    alpha_min = min(results.keys())
    alpha_max = max(results.keys())
    mrr_min = results[alpha_min].get('MRR', 0.0)
    mrr_max = results[alpha_max].get('MRR', 0.0)

    print(f'Training ratio range: alpha_min = {alpha_min:.1f}, alpha_max = {alpha_max:.1f}')
    print(f'MRR(alpha_min) = {mrr_min:.4f}, MRR(alpha_max) = {mrr_max:.4f}')
    print(f'HSR = {hsr:.4f}%')
    print('(Lower HSR indicates weaker dependence on the historical interaction horizon.)')
    print()

    print('=' * 80)
    print('Metric Summary')
    print('=' * 80)
    print(f'LPI_MRR:    {lpi_mrr:.6f}')
    print(f'LPI_Hit@1:  {lpi_hit1:.6f}')
    print(f'LPI_Hit@10: {lpi_hit10:.6f}')
    print(f'MGI:        {mgi:.6f}')
    print(f'HSR:        {hsr:.4f}%')
    print('=' * 80)


if __name__ == '__main__':
    main()
