# Changelog — test_vision_analysis_mock.py

> **Surface**: Local sample-response workflow and its shell exit status.
>
> **Append rule**: Prepend dated entries; preserve prior entries.

## 2026-10-15 — Return failure status after a caught mock workflow error
Keep the existing TEST FAILED message and traceback, then exit 1 so developers and automation can distinguish a failed workflow from a successful run. The normal path, fixture parsing, scoring, and cache format remain unchanged.
**Commit**: uncommitted candidate. **Author**: Codex.
**Touches**: [test_mock_runner_exit.md](../tests/test_mock_runner_exit.md), [README.md](../docs/README.md).

---

## Entry template

```md
## YYYY-MM-DD — Short imperative title
What changed and why; state the effect for the developer or user.
**Commit**: <verified commit or uncommitted candidate>. **Author**: <name>.
**Touches**: <related lane links>.
```
