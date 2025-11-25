# Evaluation Results

This directory contains results from accuracy evaluation runs.

## Structure

```
results/
└── accuracy/
    ├── bpm/
    │   ├── 2025-11-24_143052_bpm_eval_commit-abc123.json
    │   ├── 2025-11-24_143052_bpm_eval_commit-abc123.md
    │   ├── 2025-11-24_143052_bpm_eval_commit-abc123.png
    │   ├── latest.json (symlink)
    │   └── latest.md (symlink)
    └── drops/
        └── ...
```

## File Formats

### Markdown (`.md`)

Human-readable summary with:
- Evaluation metadata (timestamp, git commit, sample size)
- Summary metrics (MAE, RMSE, accuracy)
- Worst outliers table
- Error distribution

**Usage:** Quick review of results

### JSON (`.json`)

Machine-readable full results with:
- Complete metadata
- Individual file results
- Detailed metrics
- Error distributions
- Outlier analysis

**Usage:** Programmatic analysis, comparison across runs

### PNG (`.png`)

Error distribution histogram (optional, requires matplotlib)

**Usage:** Visual analysis of error patterns

## Symlinks

`latest.json` and `latest.md` always point to the most recent evaluation results for easy access.

## JSON Schema

```json
{
  "metadata": {
    "analysis_type": "bpm",
    "timestamp": "2025-11-24T14:30:52",
    "git_commit": "abc123",
    "git_branch": "main",
    "sample_size": 100,
    "sampling_strategy": "random",
    "sampling_seed": 42,
    "reference_source": "spotify",
    "tolerance": 2.5
  },
  "summary": {
    "total_files": 100,
    "successful": 97,
    "failed": 3,
    "mean_absolute_error": 1.84,
    "root_mean_square_error": 2.91,
    "accuracy_within_tolerance": 87.6,
    "error_distribution": {...}
  },
  "outliers": [...],
  "results": [
    {
      "file": "/path/to/track.mp3",
      "reference": 128.0,
      "computed": 127.8,
      "error": -0.2,
      "success": true,
      "computation_time": 0.843,
      "error_message": null
    }
  ]
}
```

## AI Assistant Integration

This format is optimized for AI assistants like Claude:

**Quick queries:**
- "What's the current BPM accuracy?" → Read `bpm/latest.md`
- "Compare accuracy across commits" → Compare JSON files by git_commit
- "Show worst outliers" → Read outliers section
- "Has accuracy improved?" → Compare MAE/RMSE across evaluations

**Example:**
```
User: "Did my algorithm changes improve BPM accuracy?"
Claude: *Reads latest.md and previous evaluation*
"Yes! MAE improved from 2.14 BPM to 1.84 BPM (14% better).
Accuracy within ±2.5 BPM increased from 82.3% to 87.6%."
```

## Version Control

Results can be committed to git for historical tracking:

```bash
# Track results
git add benchmarks/results/accuracy/
git commit -m "BPM evaluation: MAE 1.84, accuracy 87.6%"

# Compare across commits
git diff HEAD~1 benchmarks/results/accuracy/bpm/latest.md
```

Or add to `.gitignore` if results are considered ephemeral.
