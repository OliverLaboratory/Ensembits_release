# Code conventions

## File headers

Every `.py` file has a docstring with:
- one-line title
- what it does (inputs, steps, outputs)
- which part of the paper it implements (tex file, section, equation label)
- origin (where it was ported from, what changed)
- how to run (command line)

## Inline markers

| Marker | Meaning |
|---|---|
| `FIXED(source, date)` | A behaviour change applied here; the comment says what changed |
| `Open(source)` | A genuinely open item (usually paper text pending the text pass) |
| `Caveat` | A known design limitation (documented, not a bug) |

## Imports and paths

The codebase root (`ensembits_code/`) is on `sys.path`. Scripts add it with:

```python
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT
```

Data and outputs are resolved through `ensembits.paths` (REPO_ROOT, CKPT_DIR).
No absolute or machine-specific paths in the source.

## Testing

After editing: `python -m py_compile <file>`, then `python -m pytest tests/`.
