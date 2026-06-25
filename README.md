# Xact PMF Analysis

Research code and notebooks for Xact metals quality control, event screening, PMF input preparation, PMF output diagnostics, and source contribution time analysis.

## Current focus

- Generate PMF-ready concentration, uncertainty, sample-key, and species-mapping files.
- Compare PMF factor-number solutions.
- Merge EPA PMF factor contribution outputs back to timestamps.
- Explore time-aware PMF factor behavior for the full-period 6-factor solution.

The active notebook is:

- `pmf_full_6f_time_analysis_notebook.ipynb`

## Data policy

Raw measurement files, PMF input/output tables, plots, and generated result folders are intentionally ignored by Git. This keeps the GitHub repository focused on code and reproducible workflow notes.

If a small derived table or figure should be versioned, add it intentionally with `git add -f path/to/file`.

## Getting Started

Create a Python environment and install the common analysis packages:

```bash
pip install -r requirements.txt
```

Most scripts currently expect the project root to be:

```text
D:\Documents\PhD-Research\Xact python code
```

As the project gets cleaned up, path handling can be moved into a shared configuration file.
