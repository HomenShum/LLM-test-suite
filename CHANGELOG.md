# Changelog

All notable changes to this project will be documented in this file.

## v0.1.1 - 2025-10-03
- Add live app URL to README and footer:
  - https://llm-test-suite-cafecorner.streamlit.app/
- Fix Streamlit duplicate element key by prefixing judge buttons with context key
- Cleanup: remove non-essential Markdown docs; ignore future non-essential docs by default
- Cleanup: remove backup/corrupted Streamlit files; add .gitignore rules

## v0.1.0 - 2025-10-03
- Test6: Overall Model Leaderboard
  - Ground-truth-based ranking (avg absolute error) when available
  - Consensus proxy ranking when ground truth is absent
- Compact LLM Judge panel alongside the leaderboard
- Session persistence for evaluation context

