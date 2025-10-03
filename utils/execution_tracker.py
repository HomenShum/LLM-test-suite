"""
Execution tracking for test runs.
Extracted from streamlit_test_v5.py to reduce main file size.

This module contains:
- ExecutionEvent class
- ExecutionTracker class
- Event logging and timeline export
"""

import time
import streamlit as st
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ExecutionEvent:
    """Universal event for tracking all test executions."""
    test_name: str  # "Test 1", "Test 2", etc.
    event_type: str  # "start", "progress", "complete", "error"
    agent_id: str  # Unique identifier for this agent/task
    agent_name: str  # Human-readable name
    agent_type: str  # "orchestrator", "main_agent", "sub_agent", "batch"
    parent_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    duration: Optional[float] = None
    progress: float = 0.0  # 0-100
    status: str = "pending"  # "pending", "running", "complete", "error"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'test_name': self.test_name,
            'event_type': self.event_type,
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'agent_type': self.agent_type,
            'parent_id': self.parent_id,
            'timestamp': self.timestamp,
            'duration': self.duration,
            'progress': self.progress,
            'status': self.status,
            'metadata': self.metadata
        }


class ExecutionTracker:
    """Centralized execution tracking for all tests."""

    def __init__(self):
        self.events: List[ExecutionEvent] = []
        self.active_agents: Dict[str, ExecutionEvent] = {}
        self.start_time = time.time()

    def emit(self, test_name: str, event_type: str, agent_id: str, agent_name: str,
             agent_type: str, parent_id: Optional[str] = None, **metadata):
        """Emit a new event."""

        # Calculate relative timestamp
        rel_timestamp = time.time() - self.start_time

        event = ExecutionEvent(
            test_name=test_name,
            event_type=event_type,
            agent_id=agent_id,
            agent_name=agent_name,
            agent_type=agent_type,
            parent_id=parent_id,
            timestamp=rel_timestamp,
            metadata=metadata
        )

        # Update status
        if event_type == "start":
            event.status = "running"
            event.progress = 0
            self.active_agents[agent_id] = event
        elif event_type == "progress":
            if agent_id in self.active_agents:
                self.active_agents[agent_id].progress = metadata.get('progress', 0)
                event.progress = metadata.get('progress', 0)
                event.status = "running"
        elif event_type == "complete":
            event.status = "complete"
            event.progress = 100
            if agent_id in self.active_agents:
                start_event = self.active_agents[agent_id]
                event.duration = rel_timestamp - start_event.timestamp
                del self.active_agents[agent_id]
        elif event_type == "error":
            event.status = "error"
            if agent_id in self.active_agents:
                start_event = self.active_agents[agent_id]
                event.duration = rel_timestamp - start_event.timestamp
                del self.active_agents[agent_id]

        self.events.append(event)

        # Store in session state
        if 'execution_events' not in st.session_state:
            st.session_state['execution_events'] = []
        st.session_state['execution_events'].append(event.to_dict())

    def get_test_events(self, test_name: str) -> List[ExecutionEvent]:
        """Get all events for a specific test."""
        return [e for e in self.events if e.test_name == test_name]

    def export_timeline(self, test_name: Optional[str] = None) -> pd.DataFrame:
        """Export timeline as DataFrame."""
        events_to_export = self.get_test_events(test_name) if test_name else self.events
        return pd.DataFrame([e.to_dict() for e in events_to_export])

    def reset(self):
        """Reset tracker for new run."""
        self.events = []
        self.active_agents = {}
        self.start_time = time.time()

    def get_active_count(self) -> int:
        """Get count of currently active agents."""
        return len(self.active_agents)

    def get_completed_count(self, test_name: Optional[str] = None) -> int:
        """Get count of completed events."""
        events = self.get_test_events(test_name) if test_name else self.events
        return sum(1 for e in events if e.status == "complete")

    def get_error_count(self, test_name: Optional[str] = None) -> int:
        """Get count of error events."""
        events = self.get_test_events(test_name) if test_name else self.events
        return sum(1 for e in events if e.status == "error")

    def get_total_duration(self, test_name: Optional[str] = None) -> float:
        """Get total duration of all completed events."""
        events = self.get_test_events(test_name) if test_name else self.events
        return sum(e.duration for e in events if e.duration is not None)

    def get_average_duration(self, test_name: Optional[str] = None) -> float:
        """Get average duration of completed events."""
        events = self.get_test_events(test_name) if test_name else self.events
        completed = [e for e in events if e.duration is not None]
        if not completed:
            return 0.0
        return sum(e.duration for e in completed) / len(completed)

    def get_status_summary(self, test_name: Optional[str] = None) -> Dict[str, int]:
        """Get summary of event statuses."""
        events = self.get_test_events(test_name) if test_name else self.events
        summary = {
            "pending": 0,
            "running": 0,
            "complete": 0,
            "error": 0
        }
        for event in events:
            if event.status in summary:
                summary[event.status] += 1
        return summary

    def get_latest_event(self, test_name: Optional[str] = None) -> Optional[ExecutionEvent]:
        """Get the most recent event."""
        events = self.get_test_events(test_name) if test_name else self.events
        return events[-1] if events else None

    def clear_test_events(self, test_name: str):
        """Clear all events for a specific test."""
        self.events = [e for e in self.events if e.test_name != test_name]
        # Also remove from active agents
        to_remove = [aid for aid, e in self.active_agents.items() if e.test_name == test_name]
        for aid in to_remove:
            del self.active_agents[aid]

