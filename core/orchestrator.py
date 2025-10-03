"""
Unified Orchestrator and related classes for agent coordination.
Extracted from streamlit_test_v5.py to reduce main file size.

This module contains:
- Budget management
- Task and verification classes
- Knowledge indexing
- Agent coordination patterns
- Gemini LLM client integration
- UnifiedOrchestrator class
"""

import asyncio
import time
import hashlib
import streamlit as st
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Set
from google import genai
from google.genai import types

# Import from leaf agent scaffold
from leaf_agent_scaffold import (
    SupervisorAgent,
    AgentType,
    WebResearchAgent,
    CodeExecutorAgent,
    ContentGeneratorAgent,
    ValidatorAgent,
    TaskPlanner,
    ResultSynthesizer,
    PolicyUpdater
)

# Import stateful components
from utils.stateful_components import (
    MemoryManager,
    SecurityAuditAgent,
    SelfCorrectionManager
)


class Budget:
    """Budget manager with dual modes: turn-based or cost-based tracking"""
    def __init__(self,
                 mode: str = "turns",  # "turns" or "cost"
                 max_turns: int = 10,
                 max_cost_usd: float = 5.0,
                 max_tokens: int = 1_000_000):
        self.mode = mode
        self.max_turns = max_turns
        self.max_cost = max_cost_usd
        self.max_tokens = max_tokens

        self.current_turn = 0
        self.spent_cost = 0.0
        self.spent_tokens = 0

    def left(self) -> float:
        """Returns remaining budget as fraction (0.0 to 1.0)"""
        if self.mode == "turns":
            return (self.max_turns - self.current_turn) / self.max_turns if self.max_turns > 0 else 0.0
        else:  # "cost" mode
            cost_left = (self.max_cost - self.spent_cost) / self.max_cost if self.max_cost > 0 else 0.0
            token_left = (self.max_tokens - self.spent_tokens) / self.max_tokens if self.max_tokens > 0 else 0.0
            return min(cost_left, token_left)

    def advance_turn(self):
        """Increment turn counter"""
        self.current_turn += 1

    def consume(self, cost: float, tokens: int):
        """Track actual spending (useful for reporting in both modes)"""
        self.spent_cost += cost
        self.spent_tokens += tokens

    def exhausted(self) -> bool:
        """Check if budget is exhausted"""
        if self.mode == "turns":
            return self.current_turn >= self.max_turns
        else:
            return (self.spent_cost >= self.max_cost or
                    self.spent_tokens >= self.max_tokens)


@dataclass
class TurnMetrics:
    """Metrics tracked per turn for progress visualization"""
    turn: int
    tasks_attempted: int
    tasks_verified: int
    best_accuracy: float
    improvement: float  # delta from previous turn
    cost_spent: float
    tokens_used: int
    timestamp: float


@dataclass
class OrchestratorResult:
    """Unified result format for all orchestrator modes and patterns"""
    mode: str  # "inference", "analysis", "research"
    coordination_pattern: str  # "solo", "subagent", "multi_agent"
    final_score: float  # Accuracy, success rate, or quality score
    total_turns: int
    best_turn: int
    total_cost: float
    total_tokens: int
    solution: Any  # Code, consensus, or synthesis
    history: List[TurnMetrics]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """Task representation for orchestrator"""
    id: str
    goal: str
    parent_ids: Set[str] = field(default_factory=set)
    inputs: Dict[str, Any] = field(default_factory=dict)
    priority: float = 0.0
    novelty_score: float = 0.0
    estimated_roi: float = 0.0
    code: Optional[str] = None

    def signature(self) -> str:
        """Hash for deduplication"""
        content = f"{self.goal}|{self.code}|{sorted(self.inputs.items())}"
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class VerificationResult:
    """Verification result with claims and evidence"""
    task_id: str
    claims: List[str]
    evidence: Dict[str, Any]
    verdict: str  # "verified", "failed", "partial"
    confidence: float
    outputs: Any


class TaskCache:
    """Cache for task results with deduplication"""
    def __init__(self):
        self.cache: Dict[str, VerificationResult] = {}
        self.verified_tasks: Set[str] = set()

    def has(self, signature: str) -> bool:
        return signature in self.cache

    def get(self, signature: str) -> Optional[VerificationResult]:
        return self.cache.get(signature)

    def store(self, signature: str, result: VerificationResult):
        self.cache[signature] = result
        if result.verdict == "verified":
            self.verified_tasks.add(result.task_id)


class KnowledgeIndex:
    """Knowledge index for storing verified outputs"""
    def __init__(self):
        self.entries: List[Dict[str, Any]] = []
        self.embeddings = None  # Would use actual embeddings in production

    async def store(self, outputs: Any, verified: VerificationResult):
        """Flatten, embed, and tag verified outputs"""
        entry = {
            "task_id": verified.task_id,
            "outputs": outputs,
            "verdict": verified.verdict,
            "confidence": verified.confidence,
            "claims": verified.claims,
            "timestamp": time.time()
        }
        self.entries.append(entry)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant verified knowledge"""
        # Simplified: would use actual semantic search
        return self.entries[-top_k:]


class FineTuneDatasetCollector:
    """Collects examples for fine-tuning datasets"""
    def __init__(self):
        self.examples: List[Dict[str, Any]] = []

    def add_example(self, input_data: Any, output_data: Any, metadata: Optional[Dict] = None):
        """Add a training example"""
        self.examples.append({
            "input": input_data,
            "output": output_data,
            "metadata": metadata or {}
        })

    def export(self) -> List[Dict]:
        """Export collected examples"""
        return self.examples


# ============================================
# Agent Coordination Patterns
# ============================================

class AgentCoordinationPattern(Enum):
    """
    Coordination patterns define how agents work together.

    SOLO: Single agent executes the task independently
    SUBAGENT: Hierarchical delegation with specialized subagents (decomposer, generator, evaluator, etc.)
    MULTI_AGENT: Peer collaboration with independent proposals, cross-review, and consensus
    LEAF_SCAFFOLD: Hierarchical multi-agent scaffold with supervisor and specialized leaf agents
    """
    SOLO = "solo"
    SUBAGENT = "subagent"
    MULTI_AGENT = "multi_agent"
    LEAF_SCAFFOLD = "leaf_scaffold"


# ============================================
# Leaf Agent Scaffold Integration
# ============================================

class GeminiLLMClient:
    """Wrapper for Gemini API to use with leaf agents."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)

    async def generate(self, prompt: str) -> str:
        """Generate text response."""
        # Import here to avoid circular dependency
        from core.pricing import custom_gemini_price_lookup
        
        response = await asyncio.to_thread(
            lambda: self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
        )

        # Track cost
        if st.session_state.cost_tracker:
            st.session_state.cost_tracker.update(
                provider="Google",
                model="gemini-2.5-flash",
                api="generate_content",
                raw_response_obj=response,
                pricing_resolver=custom_gemini_price_lookup
            )

        return response.text
    async def generate_with_code_exec(self, prompt: str) -> str:
        """Generate with code execution."""
        # Import here to avoid circular dependency
        from core.pricing import custom_gemini_price_lookup
        
        try:
            response = await asyncio.to_thread(
                lambda: self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(code_execution={})]
                    )
                )
            )

            # Track cost
            if st.session_state.cost_tracker:
                st.session_state.cost_tracker.update(
                    provider="Google",
                    model="gemini-2.5-flash",
                    api="generate_content",
                    raw_response_obj=response,
                    pricing_resolver=custom_gemini_price_lookup
                )

            # Extract code execution results
            output_parts = []
            if response and hasattr(response, 'candidates') and response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        output_parts.append(part.text)
                    elif hasattr(part, 'code_execution_result'):
                        if hasattr(part.code_execution_result, 'output') and part.code_execution_result.output:
                            output_parts.append(part.code_execution_result.output)

            result = "\n".join(output_parts) if output_parts else "Code execution completed but no output was generated."
            return result

        except Exception as e:
            # Return error message instead of raising to allow graceful handling
            return f"Code execution failed: {str(e)}"
class GeminiTaskPlanner(TaskPlanner):
    """Task planner that delegates sub-tasks using Gemini."""

    def __init__(self, available_agents: List[AgentType], llm_client: GeminiLLMClient):
        super().__init__(available_agents)
        self.llm_client = llm_client

    async def _call_llm_for_planning(self, prompt: str) -> str:
        """Invoke Gemini to produce a structured task plan."""
        from core.pricing import custom_gemini_price_lookup  # Local import to avoid cycles

        response = await asyncio.to_thread(
            lambda: self.llm_client.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
        )

        tracker = getattr(st.session_state, "cost_tracker", None)
        if tracker:
            tracker.update(
                provider="Google",
                model="gemini-2.5-flash",
                api="generate_content",
                raw_response_obj=response,
                pricing_resolver=custom_gemini_price_lookup
            )

        return response.text


class GeminiResultSynthesizer(ResultSynthesizer):
    """Synthesize multi-agent outputs with Gemini."""

    def __init__(self, llm_client: GeminiLLMClient):
        self.llm_client = llm_client

    async def _call_llm_for_synthesis(self, prompt: str) -> str:
        """Invoke Gemini to summarize and refine results."""
        return await self.llm_client.generate(prompt)



