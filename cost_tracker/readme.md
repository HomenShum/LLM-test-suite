# cost_tracker (provider-agnostic)

Drop-in token & cost tracker for ANY LLM provider.

- Pluggable "usage extractors" per provider (one small function).
- Autonomous pricing discovery via Linkup Search API (no hardcoding).
- Caches per-model rates to: `model_costs/{model}_{provider}_cost_parameters.md`
- Fallback to hardcoded pricing for common models when Linkup is unavailable.

## Features

- **Provider-agnostic**: Works with OpenAI, Anthropic, Cohere, Groq, Mistral, Together, Perplexity, OpenRouter, and more
- **Automatic pricing**: Uses Linkup Search API to find current pricing from official sources
- **Smart caching**: Caches pricing to disk to avoid repeated lookups
- **Pluggable extractors**: Register custom usage extractors for any provider
- **Fallback pricing**: Built-in pricing database for common models
- **Detailed tracking**: Per-call and aggregate statistics

## Install

Copy `cost_tracker/` into your project. Optionally set `LINKUP_API_KEY` for autonomous pricing discovery.

## Quick Start

```python
from cost_tracker import CostTracker, combined_price_lookup

# Create tracker (typically once per session)
ct = CostTracker()

# After each API call, update the tracker
ct.update(
    provider="OpenAI",
    model="gpt-4o-mini",
    api="chat.completions",
    raw_response_obj=response_obj,      # SDK response object
    raw_response_json=response_json,    # or raw JSON dict
    pricing_resolver=combined_price_lookup
)

# View totals
print(ct.totals)
# {'prompt_tokens': 1234, 'completion_tokens': 567, 'total_tokens': 1801,
#  'input_cost_usd': 0.1851, 'output_cost_usd': 0.3402, 'total_cost_usd': 0.5253}

# View per-call details
for call in ct.by_call:
    print(f"{call['provider']}/{call['model']}: ${call['total_cost_usd']}")

# Get summary by provider/model
summary = ct.get_summary()
for (provider, model), stats in summary.items():
    print(f"{provider}/{model}: {stats['calls']} calls, ${stats['total_cost_usd']}")
```

## Usage Extractors

The tracker includes a fallback extractor that handles most OpenAI-compatible APIs. For providers with different response formats, register a custom extractor:

```python
from cost_tracker import register_usage_extractor

def anthropic_extractor(provider, model, raw_obj, raw_json):
    """Extract usage from Anthropic Messages API response."""
    u = getattr(raw_obj, "usage", None) or {}
    return {
        "prompt_tokens": int(getattr(u, "input_tokens", 0) or 0),
        "completion_tokens": int(getattr(u, "output_tokens", 0) or 0),
        "total_tokens": int((getattr(u, "input_tokens", 0) or 0) + 
                           (getattr(u, "output_tokens", 0) or 0))
    }

register_usage_extractor("Anthropic", anthropic_extractor)
```

### Built-in Fallback Extractor

The fallback extractor automatically handles:

1. **OpenAI-style**: `usage.prompt_tokens`, `usage.completion_tokens`, `usage.total_tokens`
2. **Anthropic-style**: `usage.input_tokens`, `usage.output_tokens`
3. **Gemini-style**: `usage_metadata.promptTokenCount`, `usage_metadata.candidatesTokenCount`
4. **Cohere-style**: `meta.billed_units.input_tokens`, `meta.billed_units.output_tokens`

## Pricing Resolution

The tracker supports multiple pricing resolution strategies:

### 1. Linkup Search API (Autonomous)

```python
from cost_tracker import linkup_price_lookup

ct.update(
    provider="Anthropic",
    model="claude-3-5-sonnet-20241022",
    api="messages.create",
    raw_response_obj=resp,
    pricing_resolver=linkup_price_lookup  # Searches web for pricing
)
```

Requires `LINKUP_API_KEY` environment variable.

### 2. Fallback Pricing (Hardcoded)

```python
from cost_tracker import fallback_price_lookup

ct.update(
    provider="OpenAI",
    model="gpt-4o-mini",
    api="chat.completions",
    raw_response_obj=resp,
    pricing_resolver=fallback_price_lookup  # Uses built-in pricing DB
)
```

### 3. Combined (Recommended)

```python
from cost_tracker import combined_price_lookup

ct.update(
    provider="OpenAI",
    model="gpt-4o-mini",
    api="chat.completions",
    raw_response_obj=resp,
    pricing_resolver=combined_price_lookup  # Tries Linkup, falls back to hardcoded
)
```

## Pricing Cache

Pricing is cached to `model_costs/{model}_{provider}_cost_parameters.md`:

```markdown
# Cost Parameters for gpt-4o-mini (OpenAI)

input_per_mtok_usd: 0.15
output_per_mtok_usd: 0.6

Notes: auto-resolved via web search + parsing.
```

You can manually edit these files to override pricing.

## Provider Examples

### OpenAI

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=OPENAI_API_KEY)
resp = await client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)

ct.update(
    provider="OpenAI",
    model="gpt-4o-mini",
    api="chat.completions",
    raw_response_obj=resp,
    pricing_resolver=combined_price_lookup
)
```

### OpenRouter

```python
import httpx

headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
body = {"model": "mistralai/mistral-small-3.2-24b-instruct", "messages": messages}
async with httpx.AsyncClient() as client:
    r = await client.post("https://openrouter.ai/api/v1/chat/completions", 
                          headers=headers, json=body)
    data = r.json()

ct.update(
    provider="OpenRouter",
    model="mistralai/mistral-small-3.2-24b-instruct",
    api="chat.completions",
    raw_response_json=data,
    pricing_resolver=combined_price_lookup
)
```

### Anthropic

```python
from anthropic import AsyncAnthropic

client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
resp = await client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello"}]
)

ct.update(
    provider="Anthropic",
    model="claude-3-5-sonnet-20241022",
    api="messages.create",
    raw_response_obj=resp,
    pricing_resolver=combined_price_lookup
)
```

### Google Gemini

```python
from google import genai

client = genai.Client(api_key=GEMINI_API_KEY)
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hello"
)

ct.update(
    provider="Google",
    model="gemini-2.5-flash",
    api="generate_content",
    raw_response_obj=resp,
    pricing_resolver=combined_price_lookup
)
```

## API Reference

### CostTracker

#### `__init__()`
Create a new cost tracker instance.

#### `update(*, provider, model, api, raw_response_obj=None, raw_response_json=None, pricing_resolver=None, request_context=None)`
Update tracker with a new API call.

**Parameters:**
- `provider` (str): Provider name (e.g., "OpenAI", "Anthropic")
- `model` (str): Model identifier
- `api` (str): API endpoint name
- `raw_response_obj`: Raw response object from SDK
- `raw_response_json`: Raw JSON response dict
- `pricing_resolver`: Function `(provider, model) -> dict` to resolve pricing
- `request_context` (dict): Optional context to attach to this call

**Returns:** dict with call details

#### `reset()`
Reset all tracking data.

#### `get_summary()`
Get summary of costs grouped by provider and model.

**Returns:** dict mapping `(provider, model)` to aggregated stats

### Functions

#### `register_usage_extractor(provider_name, fn)`
Register a custom usage extractor for a provider.

**Parameters:**
- `provider_name` (str): Provider name (case-insensitive)
- `fn` (callable): Function `(provider, model, raw_obj, raw_json) -> dict`

#### `linkup_price_lookup(provider, model)`
Look up pricing using Linkup Search API.

#### `fallback_price_lookup(provider, model)`
Look up pricing from built-in database.

#### `combined_price_lookup(provider, model)`
Try Linkup first, fall back to built-in database.

## Notes

* OpenAI/Groq/Mistral often return OpenAI-style `usage.*`.
* Anthropic exposes `usage.input_tokens/output_tokens` on Messages API.
* Cohere streams expose `meta.billed_units.input_tokens/output_tokens`.
* If a provider returns nothing, the fallback returns zeros (you can also add a local tokenizer estimator).
* Pricing is per 1M tokens in USD.

## License

This module is provided as-is for use in your projects.

