# Changelog — test_mock_runner_exit.py

> **Surface**: Offline subprocess regression for the mock runner.
>
> **Append rule**: Prepend dated entries; preserve prior entries.

## 2026-10-15 — Check native success and injected failure exit codes
Exercise the actual main entrypoint with native success and explicitly injected exceptions at all four workflow steps. Bound concurrent callers to two children, use separate temporary working directories, and repeat failure followed by success; child processes receive no parent credentials and guarded Python socket/process events are blocked.
**Commit**: uncommitted candidate. **Author**: Codex.
**Touches**: [test_vision_analysis_mock.md](../scripts/test_vision_analysis_mock.md), [README.md](../docs/README.md).

---

## Entry template

```md
## YYYY-MM-DD — Short imperative title
What changed and why; state the effect for the developer or user.
**Commit**: <verified commit or uncommitted candidate>. **Author**: <name>.
**Touches**: <related lane links>.
```
