"""
Persistent Dashboard Logger for Agent Execution Tracking

This module provides comprehensive logging infrastructure for the evaluation suite,
enabling historical review of test executions without re-running them.

Directory Structure:
    agent_dashboard_logs/
    ├── test_runs/
    │   ├── {test_id}_{timestamp}/
    │   │   ├── metadata.json
    │   │   ├── execution_log.jsonl
    │   │   ├── memory_snapshots.jsonl
    │   │   ├── security_audits.jsonl
    │   │   ├── rethink_history.jsonl
    │   │   └── gantt_timeline.json
    └── index.json
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict


# ============================================
# Configuration
# ============================================

LOGS_BASE_DIR = Path("agent_dashboard_logs")
TEST_RUNS_DIR = LOGS_BASE_DIR / "test_runs"
INDEX_FILE = LOGS_BASE_DIR / "index.json"


# ============================================
# Data Models
# ============================================

@dataclass
class ExecutionLogEntry:
    """Single execution event log entry."""
    timestamp: str  # ISO-8601
    turn: int
    agent: str
    event_type: str  # TOOL_RULE_ENFORCED | MEMORY_WRITE | SECURITY_AUDIT | RETHINK_TRIGGERED
    severity: str  # INFO | WARNING | ERROR | SECURITY_ALERT
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemorySnapshot:
    """Memory state snapshot at a specific turn."""
    timestamp: str
    turn: int
    core_blocks: Dict[str, Any]  # Persona, Purpose, Tool Guidelines
    archival_entries: List[Dict[str, Any]]
    statistics: Dict[str, Any]


@dataclass
class SecurityAudit:
    """Security audit event."""
    timestamp: str
    turn: int
    agent: str
    operation_type: str  # Code Exec, Web Search, File Access
    risk_level: str  # LOW | MEDIUM | HIGH | CRITICAL
    status: str  # Pass | Blocked | Warning
    details: Dict[str, Any]


@dataclass
class RethinkEvent:
    """Self-correction/rethink event."""
    timestamp: str
    turn: int
    block_name: str
    trigger: str  # Validator feedback
    change_summary: str
    old_content: str
    new_content: str
    revert_point_id: str


@dataclass
class GanttTask:
    """Gantt timeline task entry."""
    task_id: str
    parent_id: Optional[str]
    agent: str
    agent_type: str
    start_time: float
    end_time: float
    status: str
    metadata: Dict[str, Any]


@dataclass
class RunMetadata:
    """Metadata for a test run."""
    run_id: str
    test_type: str
    test_name: str
    timestamp: str
    model: str
    dataset_info: Dict[str, Any]
    configuration: Dict[str, Any]
    summary_metrics: Dict[str, Any] = field(default_factory=dict)


# ============================================
# Dashboard Logger
# ============================================

class DashboardLogger:
    """Centralized logging utility for dashboard visualizations."""
    
    def __init__(self, test_id: str, test_type: str, test_name: str = "Test 5", 
                 model: str = "gemini-2.5-flash", dataset_info: Optional[Dict] = None,
                 configuration: Optional[Dict] = None):
        """
        Initialize logger and create run directory.
        
        Args:
            test_id: Unique identifier for this test run
            test_type: Type of test (e.g., "orchestrator", "classification")
            test_name: Human-readable test name
            model: Model being used
            dataset_info: Information about the dataset
            configuration: Test configuration parameters
        """
        self.test_id = test_id
        self.test_type = test_type
        self.test_name = test_name
        
        # Generate unique run ID with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{test_id}_{timestamp}"
        
        # Create run directory
        self.run_dir = TEST_RUNS_DIR / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.metadata_file = self.run_dir / "metadata.json"
        self.execution_log_file = self.run_dir / "execution_log.jsonl"
        self.memory_snapshots_file = self.run_dir / "memory_snapshots.jsonl"
        self.security_audits_file = self.run_dir / "security_audits.jsonl"
        self.rethink_history_file = self.run_dir / "rethink_history.jsonl"
        self.gantt_timeline_file = self.run_dir / "gantt_timeline.json"
        
        # Initialize metadata
        self.metadata = RunMetadata(
            run_id=self.run_id,
            test_type=test_type,
            test_name=test_name,
            timestamp=datetime.now().isoformat(),
            model=model,
            dataset_info=dataset_info or {},
            configuration=configuration or {}
        )
        
        # Write initial metadata
        self._write_metadata()
        
        # Gantt tasks accumulator
        self.gantt_tasks: List[GanttTask] = []
        
        # Update index
        self._update_index()
    
    def log_execution_event(self, turn: int, agent: str, event_type: str, 
                           message: str, severity: str = "INFO", **metadata):
        """
        Log an execution event.
        
        Args:
            turn: Current turn number
            agent: Agent identifier
            event_type: Type of event (TOOL_RULE_ENFORCED, MEMORY_WRITE, etc.)
            message: Human-readable message
            severity: Event severity (INFO, WARNING, ERROR, SECURITY_ALERT)
            **metadata: Additional event-specific data
        """
        entry = ExecutionLogEntry(
            timestamp=datetime.now().isoformat(),
            turn=turn,
            agent=agent,
            event_type=event_type,
            severity=severity,
            message=message,
            metadata=metadata
        )
        
        self._append_jsonl(self.execution_log_file, asdict(entry))
    
    def log_memory_snapshot(self, turn: int, core_blocks: Dict[str, Any], 
                           archival_entries: List[Dict[str, Any]], 
                           statistics: Optional[Dict[str, Any]] = None):
        """
        Log a memory state snapshot.
        
        Args:
            turn: Current turn number
            core_blocks: Core memory blocks (Persona, Purpose, Tool Guidelines)
            archival_entries: List of archival memory entries
            statistics: Memory statistics (size, retrieval latency, etc.)
        """
        snapshot = MemorySnapshot(
            timestamp=datetime.now().isoformat(),
            turn=turn,
            core_blocks=core_blocks,
            archival_entries=archival_entries,
            statistics=statistics or {}
        )
        
        self._append_jsonl(self.memory_snapshots_file, asdict(snapshot))
    
    def log_security_audit(self, turn: int, agent: str, operation_type: str,
                          risk_level: str, status: str, **details):
        """
        Log a security audit event.
        
        Args:
            turn: Current turn number
            agent: Agent performing the operation
            operation_type: Type of operation (Code Exec, Web Search, File Access)
            risk_level: Risk level (LOW, MEDIUM, HIGH, CRITICAL)
            status: Audit result (Pass, Blocked, Warning)
            **details: Additional audit details
        """
        audit = SecurityAudit(
            timestamp=datetime.now().isoformat(),
            turn=turn,
            agent=agent,
            operation_type=operation_type,
            risk_level=risk_level,
            status=status,
            details=details
        )
        
        self._append_jsonl(self.security_audits_file, asdict(audit))
    
    def log_rethink_event(self, turn: int, block_name: str, trigger: str,
                         change_summary: str, old_content: str, new_content: str):
        """
        Log a self-correction/rethink event.
        
        Args:
            turn: Current turn number
            block_name: Name of memory block being modified
            trigger: What triggered the rethink (e.g., validator feedback)
            change_summary: Brief summary of changes
            old_content: Previous block content
            new_content: New block content
        """
        revert_point_id = f"revert_{turn}_{block_name}_{int(time.time())}"
        
        event = RethinkEvent(
            timestamp=datetime.now().isoformat(),
            turn=turn,
            block_name=block_name,
            trigger=trigger,
            change_summary=change_summary,
            old_content=old_content,
            new_content=new_content,
            revert_point_id=revert_point_id
        )
        
        self._append_jsonl(self.rethink_history_file, asdict(event))
    
    def log_gantt_task(self, task_id: str, agent: str, agent_type: str,
                      start_time: float, end_time: float, status: str,
                      parent_id: Optional[str] = None, **metadata):
        """
        Log a Gantt timeline task.
        
        Args:
            task_id: Unique task identifier
            agent: Agent name
            agent_type: Type of agent (orchestrator, main_agent, sub_agent, etc.)
            start_time: Task start timestamp
            end_time: Task end timestamp
            status: Task status (complete, running, error)
            parent_id: Parent task ID (for hierarchical tasks)
            **metadata: Additional task metadata
        """
        task = GanttTask(
            task_id=task_id,
            parent_id=parent_id,
            agent=agent,
            agent_type=agent_type,
            start_time=start_time,
            end_time=end_time,
            status=status,
            metadata=metadata
        )
        
        self.gantt_tasks.append(task)
    
    def finalize_run(self, summary_metrics: Dict[str, Any]):
        """
        Finalize the run by writing summary metrics and Gantt timeline.
        
        Args:
            summary_metrics: Final metrics for the run (accuracy, cost, etc.)
        """
        # Update metadata with summary
        self.metadata.summary_metrics = summary_metrics
        self._write_metadata()
        
        # Write Gantt timeline
        gantt_data = {
            "run_id": self.run_id,
            "tasks": [asdict(task) for task in self.gantt_tasks]
        }
        
        with open(self.gantt_timeline_file, 'w') as f:
            json.dump(gantt_data, f, indent=2)
        
        # Update index with final status
        self._update_index(status="complete")

    # ============================================
    # Static Methods for Loading Historical Runs
    # ============================================

    @staticmethod
    def load_run(run_id: str) -> Dict[str, Any]:
        """
        Load all logs for a specific run.

        Args:
            run_id: Run identifier

        Returns:
            Dictionary containing all log data
        """
        run_dir = TEST_RUNS_DIR / run_id

        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        # Load metadata
        metadata_file = run_dir / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Load JSONL files
        execution_log = DashboardLogger._load_jsonl(run_dir / "execution_log.jsonl")
        memory_snapshots = DashboardLogger._load_jsonl(run_dir / "memory_snapshots.jsonl")
        security_audits = DashboardLogger._load_jsonl(run_dir / "security_audits.jsonl")
        rethink_history = DashboardLogger._load_jsonl(run_dir / "rethink_history.jsonl")

        # Load Gantt timeline
        gantt_file = run_dir / "gantt_timeline.json"
        gantt_timeline = {}
        if gantt_file.exists():
            with open(gantt_file, 'r') as f:
                gantt_timeline = json.load(f)

        return {
            "metadata": metadata,
            "execution_log": execution_log,
            "memory_snapshots": memory_snapshots,
            "security_audits": security_audits,
            "rethink_history": rethink_history,
            "gantt_timeline": gantt_timeline
        }

    @staticmethod
    def load_index() -> Dict[str, Any]:
        """
        Load the index file containing all run metadata.

        Returns:
            Dictionary with 'runs' key containing list of run metadata
        """
        if not INDEX_FILE.exists():
            return {'runs': []}

        with open(INDEX_FILE, 'r') as f:
            index = json.load(f)

        return index

    @staticmethod
    def list_all_runs() -> List[Dict[str, Any]]:
        """
        List all available test runs.

        Returns:
            List of run metadata dictionaries
        """
        index = DashboardLogger.load_index()
        return index.get("runs", [])

    # ============================================
    # Private Helper Methods
    # ============================================

    def _write_metadata(self):
        """Write metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(asdict(self.metadata), f, indent=2)

    def _append_jsonl(self, filepath: Path, data: Dict):
        """Append a JSON line to a JSONL file."""
        with open(filepath, 'a') as f:
            f.write(json.dumps(data) + '\n')

    @staticmethod
    def _load_jsonl(filepath: Path) -> List[Dict]:
        """Load all entries from a JSONL file."""
        if not filepath.exists():
            return []

        entries = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        return entries

    def _update_index(self, status: str = "running"):
        """Update the master index with this run."""
        # Ensure index file exists
        if not INDEX_FILE.exists():
            INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(INDEX_FILE, 'w') as f:
                json.dump({"runs": []}, f)

        # Load existing index
        with open(INDEX_FILE, 'r') as f:
            index = json.load(f)

        # Update or add this run
        runs = index.get("runs", [])

        # Remove existing entry for this run_id (if updating)
        runs = [r for r in runs if r.get("run_id") != self.run_id]

        # Add updated entry
        runs.append({
            "run_id": self.run_id,
            "test_type": self.test_type,
            "test_name": self.test_name,
            "timestamp": self.metadata.timestamp,
            "model": self.metadata.model,
            "status": status,
            "summary_metrics": self.metadata.summary_metrics
        })

        # Sort by timestamp (newest first)
        runs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        # Write back
        index["runs"] = runs
        with open(INDEX_FILE, 'w') as f:
            json.dump(index, f, indent=2)

