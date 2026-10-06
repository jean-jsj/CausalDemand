import sys
from pathlib import Path

FOUNDATION = Path(__file__).resolve().parent
MODELS = FOUNDATION.parent
for p in (str(MODELS), str(FOUNDATION)):
    if p not in sys.path:
        sys.path.insert(0, p)

import fm_runner

fm_runner.HARNESS = MODELS


def main(argv: list) -> None:
    a = argv[1:]
    if len(a) < 4 or a[3] not in fm_runner.SIZES:
        sys.exit("usage: python models/foundation/run_fm.py DATASET_DIR PANEL_DATASET_DIR OUT_DIR "
                 "{tabpfn,tabfm,chronos2} [SAMPLER_SEED [STORE_WEEKS]] [check]")
    rest = [x for x in a[4:] if x != "check"]
    extra = {"store_weeks": int(rest[1])} if len(rest) > 1 else {}
    fm_runner.run(Path(a[0]), Path(a[1]), Path(a[2]), a[3], int(rest[0]) if rest else 0,
                  check="check" in a[4:], **extra)


if __name__ == "__main__":
    main(sys.argv)
