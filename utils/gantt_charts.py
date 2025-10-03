"""
Gantt chart visualization functions for execution timelines.
Extracted from streamlit_test_v5.py to reduce main file size.
"""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any

# Plotly configuration
PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True
}

# Enhanced color scheme for different agent types
AGENT_COLORS = {
    'orchestrator': '#6366F1',      # Indigo
    'main_agent': '#16A34A',        # Green
    'web_researcher': '#0EA5E9',    # Sky Blue
    'code_executor': '#F59E0B',     # Amber
    'knowledge_retriever': '#8B5CF6', # Purple
    'content_generator': '#EC4899',  # Pink
    'validator': '#14B8A6',         # Teal
    'editor': '#EF4444',            # Red
    'sub_agent': '#64748B',         # Slate
    'task': '#94A3B8'               # Light Slate
}


def render_agent_gantt_chart(test_name: str = "classification"):
    """Renders an interactive Gantt chart for agent/batch execution timeline."""
    if 'last_progress_metadata' not in st.session_state:
        st.info("No execution data yet. Run a test to see the timeline.")
        return

    metadata = st.session_state.last_progress_metadata.get(test_name)
    if not metadata or not metadata.get('batch_timestamps'):
        st.info(f"No timeline data for {test_name}. Run the test first.")
        return

    # Prepare Gantt chart data
    gantt_data = []

    # Add batches as tasks
    for i, (end_time, duration, size, success_rate) in enumerate(zip(
        metadata.get('batch_timestamps', []),
        metadata.get('batch_latencies', []),
        metadata.get('batch_sizes', []),
        metadata.get('success_rates', [])
    )):
        start_time = end_time - duration
        gantt_data.append({
            'Task': f'Batch {i+1}',
            'Start': start_time,
            'Finish': end_time,
            'Resource': 'Async Worker',
            'Description': f'{size} items, {success_rate*100:.0f}% success',
            'Progress': success_rate * 100
        })

    if not gantt_data:
        return

    # Create Gantt chart using Plotly
    df_gantt = pd.DataFrame(gantt_data)

    fig = go.Figure()

    # Color scale based on success rate
    colors = ['#EF4444' if p < 50 else '#F59E0B' if p < 80 else '#10B981'
              for p in df_gantt['Progress']]

    for idx, row in df_gantt.iterrows():
        fig.add_trace(go.Bar(
            y=[row['Task']],
            x=[row['Finish'] - row['Start']],
            base=row['Start'],
            orientation='h',
            name=row['Task'],
            marker=dict(color=colors[idx]),
            text=row['Description'],
            textposition='inside',
            hovertemplate=(
                f"<b>{row['Task']}</b><br>" +
                f"Start: {row['Start']:.1f}s<br>" +
                f"Duration: {row['Finish']-row['Start']:.2f}s<br>" +
                f"{row['Description']}<br>" +
                "<extra></extra>"
            ),
            showlegend=False
        ))

    fig.update_layout(
        title="Batch Execution Timeline (Gantt Chart)",
        xaxis_title="Time (seconds)",
        yaxis_title="Batch",
        height=400,
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='#F7F7FB',
        paper_bgcolor='white'
    )

    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def render_test5_gantt_chart():
    """Renders Gantt chart for Test 5 orchestrator execution with enhanced colors."""
    if 'orchestrator_events' not in st.session_state or not st.session_state['orchestrator_events']:
        st.info("No execution data yet. Run Test 5 to see the timeline.")
        return

    events = st.session_state['orchestrator_events']

    # Build timeline data from events
    agents = {}  # agent_id -> {start, end, name, type, parent, status, data}

    for event in events:
        agent_id = event['agent_id']

        if agent_id not in agents:
            agents[agent_id] = {
                "id": agent_id,
                "name": event['data'].get('name', agent_id),
                "type": event['data'].get('type', 'task'),
                "parent": event['data'].get('parent'),
                "start": event['timestamp'],
                "end": None,
                "status": "running",
                "data": event['data'].copy()
            }

        if event['type'] == "complete":
            agents[agent_id]['end'] = event['timestamp']
            agents[agent_id]['status'] = "complete"
            agents[agent_id]['data'].update(event['data'])
        elif event['type'] == "error":
            agents[agent_id]['end'] = event['timestamp']
            agents[agent_id]['status'] = "error"
            agents[agent_id]['data'].update(event['data'])

    # Create Gantt chart data
    gantt_data = []

    if not agents:
        st.info("No agent data available.")
        return

    max_time = max([a.get('end', a['start']) for a in agents.values()])

    # Add orchestrator (overall execution)
    gantt_data.append({
        'Task': 'Orchestrator',
        'Start': 0,
        'Finish': max_time,
        'Resource': 'Test 5',
        'Description': f"{len([a for a in agents.values() if a['type'] == 'main_agent'])} turns executed",
        'Progress': 100 if all(a['status'] == 'complete' for a in agents.values() if a['type'] == 'main_agent') else 75,
        'Status': 'complete'
    })

    # Add turns (main agents) and their sub-tasks
    for agent in sorted(agents.values(), key=lambda x: x['start']):
        if agent['type'] == 'main_agent':
            duration = (agent.get('end', max_time) - agent['start'])
            accuracy = agent['data'].get('accuracy', 0) * 100

            gantt_data.append({
                'Task': agent['name'],
                'Start': agent['start'],
                'Finish': agent.get('end', max_time),
                'Resource': 'Turn',
                'Description': f"Accuracy: {accuracy:.1f}%",
                'Progress': accuracy,
                'Status': agent['status']
            })

            # Add sub-agents (code gen, evaluation)
            sub_agents = [a for a in agents.values() if a.get('parent') == agent['id']]
            for sub in sorted(sub_agents, key=lambda x: x['start']):
                sub_duration = (sub.get('end', max_time) - sub['start'])
                gantt_data.append({
                    'Task': f"  └─ {sub['name']}",  # Indent for hierarchy
                    'Start': sub['start'],
                    'Finish': sub.get('end', max_time),
                    'Resource': 'Sub-task',
                    'Description': f"{sub_duration:.2f}s",
                    'Progress': 100 if sub['status'] == 'complete' else 50,
                    'Status': sub['status']
                })

    if not gantt_data:
        st.info("No timeline data available.")
        return

    # Render using Plotly with enhanced colors
    df_gantt = pd.DataFrame(gantt_data)

    # Determine colors based on agent type and status
    def get_agent_color(task_name, status):
        """Get color based on agent type and status."""
        task_lower = task_name.lower()

        # Determine base color from agent type
        if 'orchestrator' in task_lower:
            base_color = AGENT_COLORS['orchestrator']
        elif 'web' in task_lower or 'research' in task_lower:
            base_color = AGENT_COLORS['web_researcher']
        elif 'code' in task_lower or 'executor' in task_lower:
            base_color = AGENT_COLORS['code_executor']
        elif 'knowledge' in task_lower or 'retriev' in task_lower:
            base_color = AGENT_COLORS['knowledge_retriever']
        elif 'content' in task_lower or 'generat' in task_lower:
            base_color = AGENT_COLORS['content_generator']
        elif 'validat' in task_lower:
            base_color = AGENT_COLORS['validator']
        elif 'editor' in task_lower or 'edit' in task_lower:
            base_color = AGENT_COLORS['editor']
        elif 'turn' in task_lower or 'main' in task_lower:
            base_color = AGENT_COLORS['main_agent']
        else:
            base_color = AGENT_COLORS['task']

        # Adjust opacity based on status
        if status == 'error':
            return '#EF4444'  # Red for errors
        elif status == 'running':
            return base_color + '99'  # Semi-transparent for running
        else:
            return base_color  # Full color for complete

    colors = [get_agent_color(row['Task'], row['Status']) for _, row in df_gantt.iterrows()]

    fig = go.Figure()

    for idx, row in df_gantt.iterrows():
        fig.add_trace(go.Bar(
            y=[row['Task']],
            x=[row['Finish'] - row['Start']],
            base=row['Start'],
            orientation='h',
            name=row['Task'],
            marker=dict(
                color=colors[idx],
                line=dict(color='rgba(0,0,0,0.1)', width=1)
            ),
            text=row['Description'],
            textposition='inside',
            textfont=dict(color='white', size=10),
            hovertemplate=(
                f"<b>{row['Task']}</b><br>" +
                f"Start: {row['Start']:.1f}s<br>" +
                f"Duration: {row['Finish']-row['Start']:.2f}s<br>" +
                f"Status: {row['Status']}<br>" +
                f"{row['Description']}<br>" +
                "<extra></extra>"
            ),
            showlegend=False
        ))

    fig.update_layout(
        title="Test 5: Orchestrator Execution Timeline",
        xaxis_title="Time (seconds)",
        yaxis_title="Agent / Task",
        height=max(400, len(gantt_data) * 40),
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='#F7F7FB',
        paper_bgcolor='white'
    )

    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_turns = len([a for a in agents.values() if a['type'] == 'main_agent'])
        st.metric("Total Turns", total_turns)

    with col2:
        completed = len([a for a in agents.values() if a['status'] == 'complete'])
        st.metric("Completed Tasks", completed)

    with col3:
        total_time = max_time if agents else 0
        st.metric("Total Time", f"{total_time:.1f}s")

    with col4:
        best_acc = max([a['data'].get('accuracy', 0) for a in agents.values() if 'accuracy' in a['data']], default=0) * 100
        st.metric("Best Accuracy", f"{best_acc:.1f}%")

