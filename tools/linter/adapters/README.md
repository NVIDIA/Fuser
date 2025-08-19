# Lintrunner Adapters

These files adapt our various linters to work with `lintrunner`.

## Adding a new linter
1. Add linter entry to `.lintrunner.toml`

```
[[linter]]
code = 'LINTER_NAME'
include_patterns = [
    '**/*.h',
    '**/*.cpp',
]
exclude_patterns = []
command = []
init_command = []
is_formatter = true
```

2. Create adapter for linter
It must accept two arguments.
* `{{DRYRUN}}` - A bool flag that prints an explanation of what the linter would do.

<!-- CI IGNORE -->
```python
parser.add_argument(
    "--dry-run", help="do not install anything, just print what would be done."
)
```

* `{{PATHSFILE}}` - a variadic argument for all files passed to the linter.

<!-- CI IGNORE -->
```python
parser.add_argument(
    "filenames",
    nargs="+",
    help="paths to lint",
)
```
