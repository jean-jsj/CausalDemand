"""Reference baseline estimators: the 2x2 (instruments x text) reference grid.

For each demand family, four variants cross instrument use (off/on) with
product-text use (off/on). Every estimator consumes ONLY a cell's public
files and emits the submission CSVs defined in docs/SUBMISSION_FORMAT.md,
so each variant directory is a complete, scoreable submission.
"""
