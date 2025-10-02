# cost_tracker/tracker.py
from __future__ import annotations
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import time

_COST_DIR = Path("model_costs")
_COST_DIR.mkdir(exist_ok=True)

def _usd(x: float) -> float:
    """Round to 4 decimal places for USD cents precision."""
    return float(Decimal(x).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))

def _save_cost_params(provider: str, model: str, params: dict) -> None:
    """Save cost parameters to disk cache."""
    p = _COST_DIR / f"{model.replace('/','_')}_{provider.replace('/','_')}_cost_parameters.md"
    p.write_text(
        f"# Cost Parameters for {model} ({provider})\n\n"
        f"input_per_mtok_usd: {params['input_per_mtok_usd']}\n"
        f"output_per_mtok_usd: {params['output_per_mtok_usd']}\n"
        f"\nNotes: auto-resolved via web search + parsing.\n",
        encoding="utf-8"
    )

def _load_cost_params(provider: str, model: str) -> dict | None:
    """Load cost parameters from disk cache."""
    p = _COST_DIR / f"{model.replace('/','_')}_{provider.replace('/','_')}_cost_parameters.md"
    if not p.exists():
        return None
    vals = {}
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" in line:
            k, v = [x.strip() for x in line.split(":", 1)]
            try:
                vals[k.lower()] = float(v.split()[0])
            except:
                pass
    if "input_per_mtok_usd" in vals and "output_per_mtok_usd" in vals:
        return {"input_per_mtok_usd": vals["input_per_mtok_usd"], "output_per_mtok_usd": vals["output_per_mtok_usd"]}
    return None

# ---------- Provider registry ----------
# A usage extractor is: (provider:str, model:str, raw_obj, raw_json) -> dict(prompt_tokens, completion_tokens, total_tokens)
_USAGE_EXTRACTORS: dict[str, callable] = {}

def register_usage_extractor(provider_name: str, fn: callable) -> None:
    """Register/override how to pull token counts for a provider."""
    _USAGE_EXTRACTORS[provider_name.lower()] = fn

def _extract_usage(provider: str, model: str, raw_obj=None, raw_json=None) -> dict:
    """Extract usage information from API response using registered extractors or fallback."""
    fn = _USAGE_EXTRACTORS.get(provider.lower())
    if fn:
        return fn(provider, model, raw_obj, raw_json)

    # Fallback extractor (tries common shapes):
    #  1) OpenAI-style: usage.prompt_tokens / completion_tokens / total_tokens
    u = None
    if isinstance(raw_json, dict) and "usage" in raw_json:
        u = raw_json.get("usage") or {}
        return {
            "prompt_tokens": int(u.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(u.get("completion_tokens", 0) or 0),
            "total_tokens": int(u.get("total_tokens", 0) or 0) or int(u.get("total", 0) or 0),
        }
    #  2) Anthropic-style objects: usage.{input_tokens, output_tokens}
    u = getattr(raw_obj, "usage", None)
    if u is not None:
        pt = int(getattr(u, "input_tokens", 0) or getattr(u, "prompt_tokens", 0) or 0)
        ct = int(getattr(u, "output_tokens", 0) or getattr(u, "completion_tokens", 0) or 0)
        return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}
    #  3) Gemini objects: usage_metadata.promptTokenCount / candidatesTokenCount / totalTokenCount
    um = getattr(raw_obj, "usage_metadata", None)
    if um is not None:
        pt = int(getattr(um, "promptTokenCount", 0) or 0)
        ct = int(getattr(um, "candidatesTokenCount", 0) or 0)
        tt = int(getattr(um, "totalTokenCount", 0) or (pt + ct))
        return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}
    #  4) Cohere stream/meta: billed_units.input_tokens/output_tokens (if present in JSON)
    if isinstance(raw_json, dict):
        meta = raw_json.get("meta") or {}
        billed = (meta.get("billed_units") or meta.get("billedUnits") or {})
        if billed:
            pt = int(billed.get("input_tokens", 0) or 0)
            ct = int(billed.get("output_tokens", 0) or 0)
            return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}
    #  5) last resort
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

class CostTracker:
    """Provider-agnostic token & cost tracker with pluggable usage extractors and web-searched pricing."""
    def __init__(self) -> None:
        self.by_call: list[dict] = []
        self.totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "input_cost_usd": 0.0,
            "output_cost_usd": 0.0,
            "total_cost_usd": 0.0,
        }
        self.pricing_cache: dict[tuple[str,str], dict] = {}

    def _get_price(self, provider: str, model: str, resolver) -> dict:
        """Get pricing for a model, using cache or resolver."""
        key = (provider, model)
        if key in self.pricing_cache:
            return self.pricing_cache[key]
        # disk cache
        disk = _load_cost_params(provider, model)
        if disk:
            self.pricing_cache[key] = disk
            return disk
        # web-resolve on demand
        params = resolver(provider, model) if resolver else None
        if not params:
            params = {"input_per_mtok_usd": 0.0, "output_per_mtok_usd": 0.0}
        self.pricing_cache[key] = params
        _save_cost_params(provider, model, params)
        return params

    def update(self, *, provider: str, model: str, api: str,
               raw_response_obj=None, raw_response_json=None,
               pricing_resolver=None, request_context: dict | None = None) -> dict:
        """
        Update tracker with a new API call.
        
        Args:
            provider: Provider name (e.g., "OpenAI", "Anthropic", "OpenRouter")
            model: Model identifier
            api: API endpoint name (e.g., "chat.completions", "messages.create")
            raw_response_obj: Raw response object from SDK
            raw_response_json: Raw JSON response
            pricing_resolver: Function to resolve pricing (provider, model) -> dict
            request_context: Optional context about the request
            
        Returns:
            dict: Record of this call with tokens and costs
        """
        usage = _extract_usage(provider, model, raw_response_obj, raw_response_json)
        pt = int(usage.get("prompt_tokens", 0) or 0)
        ct = int(usage.get("completion_tokens", 0) or 0)
        tt = int(usage.get("total_tokens", 0) or (pt + ct))

        rates = self._get_price(provider, model, pricing_resolver)
        in_rate = float(rates.get("input_per_mtok_usd", 0.0))
        out_rate = float(rates.get("output_per_mtok_usd", 0.0))
        in_cost = _usd((pt / 1_000_000.0) * in_rate)
        out_cost = _usd((ct / 1_000_000.0) * out_rate)

        rec = {
            "ts": time.time(),
            "provider": provider,
            "model": model,
            "api": api,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": tt,
            "input_cost_usd": in_cost,
            "output_cost_usd": out_cost,
            "total_cost_usd": _usd(in_cost + out_cost),
        }
        if request_context:
            rec["context"] = request_context
        self.by_call.append(rec)

        t = self.totals
        t["prompt_tokens"] += pt
        t["completion_tokens"] += ct
        t["total_tokens"] += tt
        t["input_cost_usd"] = _usd(t["input_cost_usd"] + in_cost)
        t["output_cost_usd"] = _usd(t["output_cost_usd"] + out_cost)
        t["total_cost_usd"] = _usd(t["total_cost_usd"] + in_cost + out_cost)

        return rec

    def reset(self) -> None:
        """Reset all tracking data."""
        self.by_call.clear()
        self.totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "input_cost_usd": 0.0,
            "output_cost_usd": 0.0,
            "total_cost_usd": 0.0,
        }

    def get_summary(self) -> dict:
        """Get a summary of costs by provider and model."""
        summary = {}
        for call in self.by_call:
            key = (call["provider"], call["model"])
            if key not in summary:
                summary[key] = {
                    "provider": call["provider"],
                    "model": call["model"],
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "input_cost_usd": 0.0,
                    "output_cost_usd": 0.0,
                    "total_cost_usd": 0.0,
                }
            s = summary[key]
            s["calls"] += 1
            s["prompt_tokens"] += call["prompt_tokens"]
            s["completion_tokens"] += call["completion_tokens"]
            s["total_tokens"] += call["total_tokens"]
            s["input_cost_usd"] = _usd(s["input_cost_usd"] + call["input_cost_usd"])
            s["output_cost_usd"] = _usd(s["output_cost_usd"] + call["output_cost_usd"])
            s["total_cost_usd"] = _usd(s["total_cost_usd"] + call["total_cost_usd"])
        return summary

