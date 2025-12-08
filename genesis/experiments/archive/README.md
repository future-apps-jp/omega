# Experiment Results Archive

This directory contains archived experiment results from the Artificial Physics project.

## Archive Date: 2025-12-08

### Archived Files

1. **experiment_results_archive_20251208.tar.gz** (440K)
   - Compressed archive of all historical experiment results
   - Contains JSON result files and text reports

2. **Individual archived files:**
   - `experiment2_separated_20251207_192640.json` (2.1M) - Experiment 2 separated conditions
   - `experiment2_dynamic_20251207_200500.json` (1.5M) - Experiment 2 with dynamic tasks
   - `extended_results_20251207_191557.json` (1.3M) - Extended experiments with N=100
   - `phase25_results_20251207_180610.json` (116K) - Phase 25 (Experiment 1) results
   - `phase26_results_20251207_181048.json` (14K) - Phase 26 (Experiment 2) results
   - `exp2a_n1000_test_20251207_213226.json` (371B) - N=1000 test results

### Current Active Results

The following file remains in `results/` directory as the current active result:
- `all_experiments_dynamic_20251207_200835.json` (1.2M) - Final consolidated results used in the paper

### Archive Purpose

These files were archived to reduce disk usage while preserving historical data for reproducibility. The compressed archive contains all original files and can be extracted if needed.

### Extraction

To extract the archive:
```bash
cd archive/
tar -xzf experiment_results_archive_20251208.tar.gz
```

