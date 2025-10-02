"""
Stateful Components for Advanced Agent Features

This module implements:
- MemoryManager: Core and archival memory with RAG
- SecurityAuditAgent: Security auditing for agent operations
- SelfCorrectionManager: Policy modification and rethink mechanisms
"""

import re
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


# ============================================
# Memory Manager
# ============================================

@dataclass
class CoreMemoryBlock:
    """A core memory block (Persona, Purpose, Tool Guidelines)."""
    name: str
    content: str
    version: int = 1
    last_modified_turn: int = 0
    modification_count: int = 0


@dataclass
class ArchivalEntry:
    """An entry in archival memory."""
    id: str
    content: str
    tags: List[str]
    source_agent: str
    timestamp: str
    embedding: Optional[List[float]] = None  # For future RAG implementation


class MemoryManager:
    """
    Manages core and archival memory for agents.
    
    Core Memory:
        - Persona: Agent identity and role
        - Purpose: Current goal and objectives
        - Tool Guidelines: Rules for tool usage
    
    Archival Memory:
        - Long-term storage of observations and learnings
        - RAG-based retrieval
    """
    
    def __init__(self, logger=None):
        """
        Initialize memory manager.
        
        Args:
            logger: Optional DashboardLogger instance for logging memory operations
        """
        self.logger = logger
        
        # Core memory blocks
        self.core_blocks: Dict[str, CoreMemoryBlock] = {
            "Persona": CoreMemoryBlock(
                name="Persona",
                content="I am an AI agent designed to solve complex tasks through iterative refinement."
            ),
            "Purpose": CoreMemoryBlock(
                name="Purpose",
                content="My current purpose will be set based on the user's goal."
            ),
            "Tool Guidelines": CoreMemoryBlock(
                name="Tool Guidelines",
                content="I should use tools judiciously and verify outputs before proceeding."
            )
        }
        
        # Archival memory
        self.archival_entries: List[ArchivalEntry] = []
        
        # Statistics
        self.stats = {
            "total_retrievals": 0,
            "total_insertions": 0,
            "cache_hits": 0,
            "avg_retrieval_latency_ms": 0.0
        }
    
    def update_block(self, block_name: str, new_content: str, turn: int, 
                    trigger: str = "manual", change_summary: str = "") -> bool:
        """
        Update a core memory block.
        
        Args:
            block_name: Name of the block to update
            new_content: New content for the block
            turn: Current turn number
            trigger: What triggered the update
            change_summary: Summary of changes
            
        Returns:
            True if update was successful
        """
        if block_name not in self.core_blocks:
            return False
        
        block = self.core_blocks[block_name]
        old_content = block.content
        
        # Update block
        block.content = new_content
        block.version += 1
        block.last_modified_turn = turn
        block.modification_count += 1
        
        # Log rethink event if logger is available
        if self.logger:
            self.logger.log_rethink_event(
                turn=turn,
                block_name=block_name,
                trigger=trigger,
                change_summary=change_summary or f"Updated {block_name}",
                old_content=old_content,
                new_content=new_content
            )
        
        return True
    
    def insert_archival(self, content: str, tags: List[str], source_agent: str, turn: int):
        """
        Insert an entry into archival memory.
        
        Args:
            content: Content to store
            tags: Tags for categorization
            source_agent: Agent that created this entry
            turn: Current turn number
        """
        entry_id = hashlib.md5(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        entry = ArchivalEntry(
            id=entry_id,
            content=content,
            tags=tags,
            source_agent=source_agent,
            timestamp=datetime.now().isoformat()
        )
        
        self.archival_entries.append(entry)
        self.stats["total_insertions"] += 1
        
        # Log memory write event
        if self.logger:
            self.logger.log_execution_event(
                turn=turn,
                agent=source_agent,
                event_type="MEMORY_WRITE",
                message=f"Inserted archival entry: {content[:50]}...",
                severity="INFO",
                entry_id=entry_id,
                tags=tags
            )
    
    def search_archival(self, query: str, tags: Optional[List[str]] = None, 
                       top_k: int = 5) -> List[ArchivalEntry]:
        """
        Search archival memory (simple keyword-based for now).
        
        Args:
            query: Search query
            tags: Optional tag filters
            top_k: Number of results to return
            
        Returns:
            List of matching archival entries
        """
        import time
        start_time = time.time()
        
        self.stats["total_retrievals"] += 1
        
        # Simple keyword matching (can be enhanced with embeddings later)
        query_lower = query.lower()
        results = []
        
        for entry in self.archival_entries:
            # Tag filter
            if tags and not any(tag in entry.tags for tag in tags):
                continue
            
            # Keyword matching
            if query_lower in entry.content.lower():
                results.append(entry)
        
        # Update latency stats
        latency_ms = (time.time() - start_time) * 1000
        self.stats["avg_retrieval_latency_ms"] = (
            (self.stats["avg_retrieval_latency_ms"] * (self.stats["total_retrievals"] - 1) + latency_ms) 
            / self.stats["total_retrievals"]
        )
        
        return results[:top_k]

    def enforce_constraints_in_prompt(self, base_prompt: str) -> str:
        """
        Prepend memory constraints to a prompt.

        Args:
            base_prompt: The original prompt

        Returns:
            Enhanced prompt with constraints prepended
        """
        tool_guidelines = self.core_blocks.get("Tool Guidelines")
        if not tool_guidelines:
            return base_prompt

        constraints = tool_guidelines.content

        enhanced_prompt = f"""CRITICAL CONSTRAINTS FROM MEMORY:
{constraints}

You MUST respect these constraints in your output.

---

{base_prompt}"""

        return enhanced_prompt

    def get_snapshot(self, turn: int) -> Dict[str, Any]:
        """
        Get a snapshot of current memory state.
        
        Args:
            turn: Current turn number
            
        Returns:
            Dictionary containing memory state
        """
        snapshot = {
            "core_blocks": {
                name: {
                    "content": block.content,
                    "version": block.version,
                    "last_modified_turn": block.last_modified_turn,
                    "modification_count": block.modification_count
                }
                for name, block in self.core_blocks.items()
            },
            "archival_entries": [
                {
                    "id": entry.id,
                    "content": entry.content[:100] + "..." if len(entry.content) > 100 else entry.content,
                    "tags": entry.tags,
                    "source_agent": entry.source_agent,
                    "timestamp": entry.timestamp
                }
                for entry in self.archival_entries
            ],
            "statistics": self.stats.copy()
        }
        
        # Log snapshot if logger is available
        if self.logger:
            self.logger.log_memory_snapshot(
                turn=turn,
                core_blocks=snapshot["core_blocks"],
                archival_entries=snapshot["archival_entries"],
                statistics=snapshot["statistics"]
            )
        
        return snapshot


# ============================================
# Security Audit Agent
# ============================================

class SecurityAuditAgent:
    """
    Performs security audits on agent operations.
    
    Audit Types:
        - Prompt Sanitization: Check for injection attacks
        - Code Sandbox Verification: Validate code execution safety
        - Output Scanning: Detect PII and sensitive data
    """
    
    def __init__(self, logger=None):
        """
        Initialize security audit agent.
        
        Args:
            logger: Optional DashboardLogger instance for logging audits
        """
        self.logger = logger
        
        # Blocked patterns for prompt sanitization
        self.blocked_patterns = [
            r"ignore\s+previous\s+instructions",
            r"system\s*:\s*you\s+are",
            r"<\s*script",
            r"eval\s*\(",
            r"exec\s*\("
        ]
        
        # Dangerous imports for code sandbox
        self.dangerous_imports = [
            "os", "sys", "subprocess", "shutil", "socket",
            "requests", "urllib", "http", "ftplib"
        ]
        
        # PII patterns
        self.pii_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
        }
    
    def audit_prompt(self, prompt: str, agent: str, turn: int) -> Tuple[bool, str, Optional[str]]:
        """
        Audit a prompt for security issues.
        
        Args:
            prompt: The prompt to audit
            agent: Agent making the request
            turn: Current turn number
            
        Returns:
            Tuple of (is_safe, status, sanitized_prompt)
        """
        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                if self.logger:
                    self.logger.log_security_audit(
                        turn=turn,
                        agent=agent,
                        operation_type="Prompt Injection",
                        risk_level="HIGH",
                        status="Blocked",
                        matched_pattern=pattern,
                        original_prompt=prompt[:100]
                    )
                return False, "Blocked", None
        
        # Passed all checks
        if self.logger:
            self.logger.log_security_audit(
                turn=turn,
                agent=agent,
                operation_type="Prompt Sanitization",
                risk_level="LOW",
                status="Pass"
            )
        
        return True, "Pass", prompt

    def audit_code(self, code: str, agent: str, turn: int) -> Tuple[bool, str, List[str]]:
        """
        Audit code for dangerous operations.

        Args:
            code: Code to audit
            agent: Agent executing the code
            turn: Current turn number

        Returns:
            Tuple of (is_safe, status, warnings)
        """
        warnings = []

        # Check for dangerous imports
        for dangerous_import in self.dangerous_imports:
            if re.search(rf"\bimport\s+{dangerous_import}\b", code) or \
               re.search(rf"\bfrom\s+{dangerous_import}\s+import\b", code):
                warnings.append(f"Dangerous import detected: {dangerous_import}")

        # Determine risk level
        if warnings:
            risk_level = "HIGH" if len(warnings) > 2 else "MEDIUM"
            status = "Warning"
        else:
            risk_level = "LOW"
            status = "Pass"

        # Log audit
        if self.logger:
            self.logger.log_security_audit(
                turn=turn,
                agent=agent,
                operation_type="Code Execution",
                risk_level=risk_level,
                status=status,
                warnings=warnings,
                code_snippet=code[:200]
            )

        return len(warnings) == 0, status, warnings

    def scan_output(self, output: str, agent: str, turn: int) -> Tuple[bool, List[str]]:
        """
        Scan output for PII and sensitive data.

        Args:
            output: Output to scan
            agent: Agent that produced the output
            turn: Current turn number

        Returns:
            Tuple of (is_clean, detected_pii_types)
        """
        detected_pii = []

        for pii_type, pattern in self.pii_patterns.items():
            if re.search(pattern, output):
                detected_pii.append(pii_type)

        # Log scan
        if self.logger:
            risk_level = "HIGH" if detected_pii else "LOW"
            status = "Warning" if detected_pii else "Pass"

            self.logger.log_security_audit(
                turn=turn,
                agent=agent,
                operation_type="Output Scan",
                risk_level=risk_level,
                status=status,
                detected_pii=detected_pii
            )

        return len(detected_pii) == 0, detected_pii


# ============================================
# Self-Correction Manager
# ============================================

class SelfCorrectionManager:
    """
    Manages self-correction and policy modification.

    Features:
        - Track rethink events
        - Maintain revert points
        - Analyze correction effectiveness
    """

    def __init__(self, memory_manager: MemoryManager, logger=None):
        """
        Initialize self-correction manager.

        Args:
            memory_manager: MemoryManager instance to modify
            logger: Optional DashboardLogger instance
        """
        self.memory_manager = memory_manager
        self.logger = logger

        # Rethink history
        self.rethink_history: List[Dict[str, Any]] = []

        # Revert points (snapshots of memory states)
        self.revert_points: Dict[str, Dict[str, Any]] = {}

    def trigger_rethink(self, turn: int, block_name: str, validator_feedback: str,
                       new_content: str) -> bool:
        """
        Trigger a rethink event based on validator feedback.

        Args:
            turn: Current turn number
            block_name: Memory block to modify
            validator_feedback: Feedback that triggered the rethink
            new_content: New content for the block

        Returns:
            True if rethink was successful
        """
        # Create revert point before modification
        revert_point_id = f"revert_{turn}_{block_name}_{len(self.revert_points)}"
        self.revert_points[revert_point_id] = {
            "turn": turn,
            "block_name": block_name,
            "content": self.memory_manager.core_blocks[block_name].content,
            "timestamp": datetime.now().isoformat()
        }

        # Perform update
        success = self.memory_manager.update_block(
            block_name=block_name,
            new_content=new_content,
            turn=turn,
            trigger=f"Validator feedback: {validator_feedback[:50]}",
            change_summary=f"Modified {block_name} based on validator feedback"
        )

        if success:
            # Record rethink event
            self.rethink_history.append({
                "turn": turn,
                "block_name": block_name,
                "trigger": validator_feedback,
                "revert_point_id": revert_point_id,
                "timestamp": datetime.now().isoformat()
            })

        return success

    def revert_to_point(self, revert_point_id: str, turn: int) -> bool:
        """
        Revert memory to a previous state.

        Args:
            revert_point_id: ID of the revert point
            turn: Current turn number

        Returns:
            True if revert was successful
        """
        if revert_point_id not in self.revert_points:
            return False

        revert_point = self.revert_points[revert_point_id]

        # Restore block content
        success = self.memory_manager.update_block(
            block_name=revert_point["block_name"],
            new_content=revert_point["content"],
            turn=turn,
            trigger="Manual revert",
            change_summary=f"Reverted to state from turn {revert_point['turn']}"
        )

        return success

    def analyze_correction_effectiveness(self, performance_history: List[float]) -> Dict[str, Any]:
        """
        Analyze the effectiveness of rethink events.

        Args:
            performance_history: List of performance scores by turn

        Returns:
            Analysis results
        """
        if not self.rethink_history or len(performance_history) < 2:
            return {
                "total_rethinks": 0,
                "successful_rethinks": 0,
                "success_rate": 0.0,
                "avg_improvement": 0.0
            }

        successful_rethinks = 0
        improvements = []

        for rethink in self.rethink_history:
            turn = rethink["turn"]

            # Check if performance improved in the next turn
            if turn < len(performance_history) - 1:
                improvement = performance_history[turn + 1] - performance_history[turn]
                if improvement > 0:
                    successful_rethinks += 1
                improvements.append(improvement)

        return {
            "total_rethinks": len(self.rethink_history),
            "successful_rethinks": successful_rethinks,
            "success_rate": successful_rethinks / len(self.rethink_history) if self.rethink_history else 0.0,
            "avg_improvement": sum(improvements) / len(improvements) if improvements else 0.0,
            "most_modified_blocks": self._get_most_modified_blocks()
        }

    def _get_most_modified_blocks(self) -> List[Tuple[str, int]]:
        """Get blocks that were modified most frequently."""
        from collections import Counter

        block_counts = Counter([r["block_name"] for r in self.rethink_history])
        return block_counts.most_common(3)

