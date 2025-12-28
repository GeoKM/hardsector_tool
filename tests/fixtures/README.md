# SCP fixtures

Large SCP captures used by the slow tests are not checked into git. To exercise
those tests, place the files in the following locations:

- `tests/ACMS80217/ACMS80217-HS32.scp`
- `tests/ACMS80221/ACMS80221-HS32.scp`
- `tests/ACMS80227/ACMS80227-HS32.scp`

Additional ACMS fixture directories follow the same pattern under `tests/ACMS*/`.

By default the tests will skip scenarios that require these captures when the
files are absent. To run them explicitly, supply the slow marker:

```bash
pytest -m slow
```
