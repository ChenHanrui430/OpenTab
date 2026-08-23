# Released test data

This directory contains only the processed closed-world test splits used by
the inference artifact:

- `tmwf_closed_world_test.arrow`: 3,750 synthetic 1--5-tab TMWF test traces
  over 50 monitored classes.
- `mtwf_closed_world_test.arrow`: 3,750 synthetic 1--5-tab MTWF test traces
  over 100 monitored classes.

Each Arrow file contains the processed traffic features (`data` and `time`),
the ground-truth label set, and label text. Training and validation splits,
raw captures, open-world splits, and data-generation code are intentionally
not included in this anonymous artifact.

The data are redistributed solely for artifact evaluation. The code license
in the repository does not supersede the terms of the original TMWF and MTWF
datasets; users remain responsible for complying with those terms.

