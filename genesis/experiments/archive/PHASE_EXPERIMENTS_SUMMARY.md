# Phase Experiments Data Summary

**Date:** 2025-12-08

## Overview

All phase experiments directories in `sk-quantum/` have been reviewed for data archiving.

## Findings

### Total Size
- **All phase experiments combined:** 360K
- **Individual phase sizes:** 4K - 40K per phase

### Phase-by-Phase Breakdown

| Phase | Size | Main Files | Status |
|-------|------|------------|--------|
| Phase 0 | 16K | 1 file | Keep (small) |
| Phase 1 | 36K | 3 files | Keep (small) |
| Phase 2 | 12K | 1 file | Keep (small) |
| Phase 4 | 24K | 2 files | Keep (small) |
| Phase 5 | 16K | 1 file | Keep (small) |
| Phase 6 | 20K | 2 files | Keep (small) |
| Phase 7 | 20K | 2 files | Keep (small) |
| Phase 8 | 24K | 2 files | Keep (small) |
| Phase 9 | 28K | 2 files | Keep (small) |
| Phase 10 | 24K | 2 files | Keep (small) |
| Phase 13 | 32K | 3 files | Keep (small) |
| Phase 14 | 4K | 0 files | Keep (small) |
| Phase 15 | 4K | 0 files | Keep (small) |
| Phase 19 | 12K | 1 file | Keep (small) |
| Phase 20 | 36K | 3 files | Keep (small) |
| Phase 21 | 40K | 4 files | Keep (small) |
| Phase 22 | 12K | 1 file | Keep (small) |

### File Types Found

1. **Python scripts** (`.py`): Experiment execution scripts
2. **Markdown results** (`RESULTS_*.md`): Analysis and findings documentation
3. **JSON data** (`.json`): Small result files (3-4K each)
   - `phase20/experiments/complexity_results.json` (3.1K)
   - `phase21/experiments/results_local_*.json` (3.5K)
   - `phase21/experiments/results_sv1_*.json` (1.3K)

## Recommendation

**No archiving needed** for phase experiments directories because:

1. **Total size is minimal** (360K combined)
2. **Files are essential** - Python scripts and RESULTS_*.md files are critical for reproducibility
3. **No large data files** - All JSON files are <4K
4. **Active research data** - These files are referenced in papers and should remain accessible

## Comparison with Genesis Experiments

| Location | Size | Status |
|----------|------|--------|
| `genesis/experiments/results/` | 1.2M | **Active** (paper data) |
| `genesis/experiments/archive/` | 5.3M | **Archived** (historical) |
| `sk-quantum/phase*/experiments/` | 360K | **Keep** (too small to archive) |

## Conclusion

Phase experiments data is already well-organized and minimal in size. No archiving action required.


