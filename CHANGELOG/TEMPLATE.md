# Surface history entry format

Create one lane for each touched script, test, or documentation surface. Describe who benefits, the observed problem, and the resulting behavior. Record a verified commit when known; use `uncommitted candidate` before publication. Never invent old entries or rewrite earlier history.

```md
# Changelog — <repository path>

> **Surface**: <user or developer job served by this file>.
>
> **Append rule**: Prepend dated entries; preserve prior entries.

## YYYY-MM-DD — Short imperative title
What changed and why; state the effect for the developer or user.
**Commit**: <verified commit or uncommitted candidate>. **Author**: <name>.
**Touches**: <related lane links>.
```
