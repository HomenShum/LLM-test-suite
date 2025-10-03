"""Agent dashboard tab extracted from the main app."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from utils.execution_tracker import ExecutionEvent, ExecutionTracker

def _event_to_dict(event: Any) -> Dict[str, Any]:
    """Normalize ExecutionEvent or dict into a dictionary."""
    if isinstance(event, dict):
        return event
    if hasattr(event, "to_dict"):
        data = event.to_dict()
        if 'metadata' not in data:
            data['metadata'] = getattr(event, 'metadata', {}) or {}
        return data
    return {}

def generate_gantt_data(test_name: str, tracker: ExecutionTracker) -> pd.DataFrame:
    """Generate Gantt chart data from events."""

    events = tracker.get_test_events(test_name)

    if not events:
        return pd.DataFrame()

    # Group events by agent_id
    agent_timeline = {}

    for event in events:
        if event.agent_id not in agent_timeline:
            agent_timeline[event.agent_id] = {
                'name': event.agent_name,
                'type': event.agent_type,
                'parent': event.parent_id,
                'start': event.timestamp,
                'end': event.timestamp,
                'status': event.status,
                'progress': event.progress,
                'metadata': event.metadata.copy()
            }
        else:
            # Update end time and status
            agent_timeline[event.agent_id]['end'] = max(
                agent_timeline[event.agent_id]['end'],
                event.timestamp
            )
            agent_timeline[event.agent_id]['status'] = event.status
            agent_timeline[event.agent_id]['progress'] = max(
                agent_timeline[event.agent_id]['progress'],
                event.progress
            )
            # Merge metadata
            agent_timeline[event.agent_id]['metadata'].update(event.metadata)

    # Build DataFrame
    rows = []
    for agent_id, data in agent_timeline.items():
        duration = data['end'] - data['start']

        # Indent based on hierarchy
        indent = "  " if data['parent'] else ""

        rows.append({
            'Task': f"{indent}{data['name']}",
            'Start': data['start'],
            'Duration': duration,
            'Status': data['status'],
            'Progress': data['progress'],
            'Type': data['type'],
            'Metadata': data['metadata'],
            'Code': data['metadata'].get('code', '')  # For hover display
        })

    return pd.DataFrame(rows)













# ---------- Rigorous Reporting with Scikit-learn and LLM Explanation ----------

# --- PATCH 3: Gemini Code Execution Helper ---


from utils.plotly_config import PLOTLY_CONFIG

_CONFIGURED = False

def configure(context: Dict[str, Any]) -> None:
    global _CONFIGURED
    for key, value in context.items():
        if key.startswith("__") or key in {"configure"}:
            continue
        globals()[key] = value
    _CONFIGURED = True

def render_agent_dashboard(tab) -> None:
    if not _CONFIGURED:
        st.error("Agent dashboard module not configured. Call configure() first.")
        return
    with tab:
        st.header("🎯 Agent Execution Dashboard")
        st.caption("Real-time monitoring and visualization of all test executions")
        
        # Enhanced dashboard with historical data loading and mock data
        def load_historical_data():
            """Load historical execution data from dashboard logs."""
            try:
                from utils.dashboard_logger import DashboardLogger
        
                # Load index
                index_data = DashboardLogger.load_index()
                historical_runs = []
        
                for run_info in index_data.get('runs', []):
                    try:
                        run_data = DashboardLogger.load_run(run_info['run_id'])
                        historical_runs.append({
                            'run_info': run_info,
                            'data': run_data
                        })
                    except Exception as e:
                        st.warning(f"Could not load run {run_info['run_id']}: {e}")
        
                return historical_runs
            except Exception as e:
                st.warning(f"Could not load historical data: {e}")
                return []
        
        def generate_mock_data():
            """Generate mock execution data for demonstration with RELATIVE timestamps."""
            mock_events = []
            base_time = 0  # Start at 0 for relative timing
        
            # Mock Test 1: Classification Test
            for i in range(3):
                mock_events.append({
                    'test_name': 'Classification Test',
                    'agent_id': f'classifier_{i+1}',
                    'event_type': 'start',
                    'timestamp': base_time + i * 60,  # Start at 0s, 60s, 120s
                    'data': {
                        'model': ['gpt-4o-mini', 'claude-3-5-sonnet', 'gemini-2.5-flash'][i],
                        'batch_size': 50,
                        'task': f'Classification Batch {i+1}'
                    }
                })
                mock_events.append({
                    'test_name': 'Classification Test',
                    'agent_id': f'classifier_{i+1}',
                    'event_type': 'complete',
                    'timestamp': base_time + i * 60 + 45,  # Duration: 45s each
                    'data': {
                        'accuracy': [0.87, 0.91, 0.89][i],
                        'f1_score': [0.85, 0.90, 0.88][i],
                        'cost_usd': [0.12, 0.18, 0.08][i],
                        'tokens': [2400, 3200, 1800][i]
                    }
                })
        
            # Mock Test 5: Orchestrator
            orchestrator_start = 200  # Start at 200s
            mock_events.append({
                'test_name': 'Test 5',
                'agent_id': 'orchestrator',
                'event_type': 'start',
                'timestamp': orchestrator_start,
                'data': {
                    'task': 'Research: TechCorp Fundraising',
                    'mode': 'leaf_scaffold',
                    'target_person': 'John Smith, CEO'
                }
            })
        
            # Sub-agents for orchestrator (parallel execution)
            sub_agents = [
                ('web_researcher', 'LinkedIn Profile Analysis', 25, 0.92),
                ('knowledge_retriever', 'Company Background Research', 35, 0.88),
                ('content_generator', 'Executive Summary Generation', 20, 0.95),
                ('validator', 'Fact Verification', 15, 0.85)
            ]
        
            for i, (agent_type, task, duration, score) in enumerate(sub_agents):
                start_time = orchestrator_start + 5 + i * 10  # Staggered start
                mock_events.extend([
                    {
                        'test_name': 'Test 5',
                        'agent_id': f'{agent_type}_{i+1}',
                        'event_type': 'start',
                        'timestamp': start_time,
                        'data': {
                            'agent_type': agent_type,
                            'task': task,
                            'parent': 'orchestrator'
                        }
                    },
                    {
                        'test_name': 'Test 5',
                        'agent_id': f'{agent_type}_{i+1}',
                        'event_type': 'complete',
                        'timestamp': start_time + duration,
                        'data': {
                            'score': score,
                            'output_size': f'{1.2 + i * 0.3:.1f} KB',
                            'confidence': score * 0.9
                        }
                    }
                ])
        
            mock_events.append({
                'test_name': 'Test 5',
                'agent_id': 'orchestrator',
                'event_type': 'complete',
                'timestamp': orchestrator_start + 80,  # Total duration: 80s
                'data': {
                    'final_score': 0.89,
                    'total_cost': 0.45,
                    'research_quality': 'High',
                    'hallucination_risk': 0.12
                }
            })
        
            return mock_events
        
        # Get data sources
        tracker = st.session_state.get('execution_tracker')
        historical_data = load_historical_data()
        
        # Data source selector
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            data_source = st.selectbox(
                "📊 Data Source",
                options=['current_session', 'historical', 'mock_demo'],
                format_func=lambda x: {
                    'current_session': '🔄 Current Session Data',
                    'historical': '📚 Historical Runs',
                    'mock_demo': '🎭 Demo Data (Mock)'
                }[x],
                index=2 if not tracker or not tracker.events else 0
            )
        
        with col2:
            if st.button("🔄 Refresh", width='content'):
                st.rerun()
        
        with col3:
            auto_refresh = st.checkbox("Auto-refresh", value=False)
        
        # Load appropriate data based on selection
        if data_source == 'current_session' and tracker and tracker.events:
            events_data = tracker.events
            st.success(f"✅ Loaded {len(events_data)} events from current session")
        elif data_source == 'historical' and historical_data:
            # Combine all historical events
            events_data = []
            for run in historical_data:
                for event in run['data'].get('execution_log', []):
                    events_data.append(event)
            st.success(f"✅ Loaded {len(events_data)} events from {len(historical_data)} historical runs")
        elif data_source == 'mock_demo':
            events_data = generate_mock_data()
            st.info(f"🎭 Generated {len(events_data)} mock events for demonstration")
        else:
            events_data = []
            if data_source == 'current_session':
                st.info("No execution data in current session. Run any test to see live data.")
            elif data_source == 'historical':
                st.info("No historical data found. Historical runs will appear here after tests complete.")
            else:
                st.info("Select a data source to view the dashboard.")
        
        if events_data:
            event_dicts = [_event_to_dict(e) for e in events_data]
            # Enhanced Gantt chart renderer for mock/historical/current data
            def render_enhanced_gantt_from_events(events, title="Agent Execution Timeline"):
                """Render enhanced Gantt chart from event data with vibrant, high-contrast colors."""
                # Vibrant, high-contrast color scheme for maximum visual distinction
                AGENT_COLORS_VIBRANT = {
                    'orchestrator': '#6366F1',      # Vibrant Indigo
                    'classifier': '#10B981',        # Emerald Green
                    'web_researcher': '#0EA5E9',    # Sky Blue
                    'code_executor': '#F59E0B',     # Amber Orange
                    'knowledge_retriever': '#8B5CF6', # Purple
                    'content_generator': '#EC4899',  # Hot Pink
                    'validator': '#14B8A6',         # Teal
                    'editor': '#EF4444',            # Red
                    'main_agent': '#22C55E',        # Bright Green
                    'sub_agent': '#64748B'          # Slate Gray
                }
        
                # Build agent timeline
                agents = {}
                min_time = float('inf')

                for event in events:
                    # Handle both dict and ExecutionEvent objects
                    if isinstance(event, dict):
                        agent_id = event.get('agent_id', 'unknown')
                        timestamp = event.get('timestamp', 0)
                        event_type = event.get('event_type', 'unknown')
                        agent_name = event.get('agent_name', agent_id)
                        agent_type = event.get('agent_type', 'task')
                        metadata = event.get('metadata', {})
                    else:
                        # ExecutionEvent object
                        agent_id = getattr(event, 'agent_id', 'unknown')
                        timestamp = getattr(event, 'timestamp', 0)
                        event_type = getattr(event, 'event_type', 'unknown')
                        agent_name = getattr(event, 'agent_name', agent_id)
                        agent_type = getattr(event, 'agent_type', 'task')
                        metadata = getattr(event, 'metadata', {})

                    # Track minimum timestamp for normalization
                    min_time = min(min_time, timestamp)

                    if agent_id not in agents:
                        agents[agent_id] = {
                            'id': agent_id,
                            'name': agent_name,
                            'type': agent_type,
                            'start': timestamp,
                            'end': None,
                            'status': 'running',
                            'data': metadata
                        }

                    if event_type == 'complete':
                        agents[agent_id]['end'] = timestamp
                        agents[agent_id]['status'] = 'complete'
                        agents[agent_id]['data'].update(metadata)
        
                if not agents:
                    st.info("No agent data to visualize.")
                    return
        
                # Normalize timestamps to start at 0
                if min_time == float('inf'):
                    min_time = 0
        
                # Create Gantt data with normalized timestamps
                gantt_data = []
                for agent in sorted(agents.values(), key=lambda x: x['start']):
                    normalized_start = agent['start'] - min_time
                    end_time = agent.get('end') or (agent['start'] + 30)
                    normalized_end = end_time - min_time
                    duration = normalized_end - normalized_start
        
                    # Determine color (vibrant, high-contrast scheme)
                    agent_type = agent.get('type', 'task')
                    color = AGENT_COLORS_VIBRANT.get(agent_type, '#94A3B8')

                    if agent['status'] == 'error':
                        color = '#DC2626'  # Bright Red for errors
                    elif agent['status'] == 'running':
                        # Convert hex to rgba with transparency for running agents
                        # Extract RGB from hex color
                        hex_color = color.lstrip('#')
                        r = int(hex_color[0:2], 16)
                        g = int(hex_color[2:4], 16)
                        b = int(hex_color[4:6], 16)
                        color = f'rgba({r},{g},{b},0.8)'  # 80% opacity for running
        
                    gantt_data.append({
                        'Task': agent['name'],
                        'Start': normalized_start,
                        'Finish': normalized_end,
                        'Duration': duration,
                        'Type': agent_type,
                        'Status': agent['status'],
                        'Color': color
                    })
        
                # Create Plotly figure with enhanced styling
                fig = go.Figure()
        
                for row in gantt_data:
                    fig.add_trace(go.Bar(
                        y=[row['Task']],
                        x=[row['Duration']],
                        base=row['Start'],
                        orientation='h',
                        name=row['Task'],
                        marker=dict(
                            color=row['Color'],
                            line=dict(color='rgba(255,255,255,0.3)', width=2),  # White border for contrast
                            pattern=dict(
                                shape="" if row['Status'] == 'complete' else "/" if row['Status'] == 'running' else "x"
                            )
                        ),
                        text=f"{row['Duration']:.1f}s",
                        textposition='inside',
                        textfont=dict(color='white', size=12, family='Arial Black', weight='bold'),
                        hovertemplate=(
                            f"<b style='font-size:14px'>{row['Task']}</b><br>" +
                            f"<b>Type:</b> {row['Type']}<br>" +
                            f"<b>Start:</b> {row['Start']:.1f}s<br>" +
                            f"<b>Duration:</b> {row['Duration']:.1f}s<br>" +
                            f"<b>Status:</b> {row['Status']}<br>" +
                            "<extra></extra>"
                        ),
                        showlegend=False
                    ))
        
                fig.update_layout(
                    title=dict(
                        text=title,
                        font=dict(size=20, weight='bold', color='#1F2937')
                    ),
                    xaxis=dict(
                        title=dict(text="Time (seconds)", font=dict(size=14, weight='bold')),
                        gridcolor='#D1D5DB',
                        showgrid=True,
                        zeroline=True,
                        zerolinecolor='#6B7280',
                        zerolinewidth=2
                    ),
                    yaxis=dict(
                        title=dict(text="Agent / Task", font=dict(size=14, weight='bold')),
                        gridcolor='#E5E7EB',
                        tickfont=dict(size=11, weight='bold')
                    ),
                    height=max(450, len(gantt_data) * 50),
                    showlegend=False,
                    hovermode='closest',
                    plot_bgcolor='#FFFFFF',
                    paper_bgcolor='#F9FAFB',
                    margin=dict(l=220, r=60, t=90, b=70),
                    font=dict(family='Arial, sans-serif')
                )
        
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        
                # Color Legend
                st.markdown("### 🎨 Agent Type Color Legend")
                legend_cols = st.columns(5)
        
                legend_items = [
                    ("Orchestrator", AGENT_COLORS_VIBRANT['orchestrator']),
                    ("Web Researcher", AGENT_COLORS_VIBRANT['web_researcher']),
                    ("Code Executor", AGENT_COLORS_VIBRANT['code_executor']),
                    ("Knowledge Retriever", AGENT_COLORS_VIBRANT['knowledge_retriever']),
                    ("Content Generator", AGENT_COLORS_VIBRANT['content_generator']),
                    ("Validator", AGENT_COLORS_VIBRANT['validator']),
                    ("Classifier", AGENT_COLORS_VIBRANT['classifier']),
                    ("Main Agent", AGENT_COLORS_VIBRANT['main_agent']),
                    ("Editor", AGENT_COLORS_VIBRANT['editor']),
                    ("Sub Agent", AGENT_COLORS_VIBRANT['sub_agent'])
                ]
        
                for idx, (name, color) in enumerate(legend_items[:5]):
                    with legend_cols[idx]:
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; margin-bottom: 8px;">
                            <div style="width: 20px; height: 20px; background-color: {color};
                                        border: 2px solid white; border-radius: 4px; margin-right: 8px;
                                        box-shadow: 0 2px 4px rgba(0,0,0,0.2);"></div>
                            <span style="font-size: 11px; font-weight: 600;">{name}</span>
                        </div>
                        """, unsafe_allow_html=True)
        
                if len(legend_items) > 5:
                    legend_cols2 = st.columns(5)
                    for idx, (name, color) in enumerate(legend_items[5:]):
                        with legend_cols2[idx]:
                            st.markdown(f"""
                            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                                <div style="width: 20px; height: 20px; background-color: {color};
                                            border: 2px solid white; border-radius: 4px; margin-right: 8px;
                                            box-shadow: 0 2px 4px rgba(0,0,0,0.2);"></div>
                                <span style="font-size: 11px; font-weight: 600;">{name}</span>
                            </div>
                            """, unsafe_allow_html=True)
        
                st.divider()
        
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Agents", len(agents))
                with col2:
                    completed = len([a for a in agents.values() if a['status'] == 'complete'])
                    st.metric("Completed", completed)
                with col3:
                    total_time = max([(a['end'] or a['start']) - min_time for a in agents.values()]) if agents else 0
                    st.metric("Total Time", f"{total_time:.1f}s")
                with col4:
                    avg_duration = sum([((a['end'] or a['start']) - a['start']) for a in agents.values()]) / len(agents) if agents else 0
                    st.metric("Avg Duration", f"{avg_duration:.1f}s")
        
            # Render the enhanced Gantt chart
            st.subheader("📊 Execution Timeline")
            render_enhanced_gantt_from_events(events_data, title=f"{data_source.replace('_', ' ').title()} - Agent Execution Timeline")
        
            st.divider()
        
            # Test selector with available tests
            test_name_map = {
                "classification": "Classification Test",
                "pruning": "Pruning Test",
                "test5": "Test 5",
                "smoke_test": "Smoke Test Run"
            }
        
            col1, col2, col3 = st.columns([2, 1, 1])
        
            with col1:
                # Build options dynamically based on available data
                if data_source == 'mock_demo':
                    test_options = ["classification", "test5"]
                else:
                    # Extract from events
                    available_tests = set()
                    for event in event_dicts:
                        test_name = event.get('test_name', '')
                        if 'Classification' in test_name:
                            available_tests.add('classification')
                        elif 'Test 5' in test_name or 'Orchestrator' in test_name:
                            available_tests.add('test5')
                        elif 'Pruning' in test_name:
                            available_tests.add('pruning')
        
                    test_options = sorted(list(available_tests)) if available_tests else ["test5"]
        
                # Use old selector for backward compatibility
                test_selector = st.selectbox(
                    "Select Test to Visualize",
                    options=test_options,
                    format_func=lambda x: {
                        "classification": "Classification Tests (1-3)",
                        "pruning": "Context Pruning (Test 4)",
                        "test5": "Agent Self-Refinement (Test 5)",
                        "smoke_test": "🔍 Smoke Test Run"
                    }.get(x, x),
                    index=test_options.index(st.session_state.get('selected_test_in_dashboard_selector', test_options[0])) if st.session_state.get('selected_test_in_dashboard_selector') in test_options else 0,
                    key='dashboard_test_selector'
                )
        
                # Map to actual test name
                selected_test = test_name_map.get(test_selector, "Classification Test")
        
                # Store selection for persistence
                st.session_state['selected_test_in_dashboard_selector'] = test_selector
        
            with col2:
                if st.button("🔄 Refresh", key='dashboard_refresh_btn_2', width='content'):
                    st.rerun()
        
            with col3:
                auto_refresh = st.checkbox("Auto-refresh", key='dashboard_auto_refresh_2', value=False)
        
            st.divider()
        
            # Add historical run selector for Test 5
            if test_selector == "test5":
                st.markdown("### 📂 Load Past Run (Historical Replay)")
        
                all_runs = DashboardLogger.list_all_runs()
                test5_runs = [r for r in all_runs if r.get("test_type") == "orchestrator"]
        
                if test5_runs:
                    run_options = ["Live Execution"] + [
                        f"{r['run_id']} - {r.get('model', 'unknown')} - {r.get('timestamp', '')[:19]} ({r.get('summary_metrics', {}).get('final_score', 0.0):.2%} accuracy)"
                        for r in test5_runs
                    ]
        
                    selected_run = st.selectbox(
                        "Select Run to View",
                        options=run_options,
                        help="View historical test runs without re-execution"
                    )
        
                    # Load historical data if selected
                    if selected_run != "Live Execution":
                        run_id = selected_run.split(" - ")[0]
                        try:
                            historical_data = DashboardLogger.load_run(run_id)
                            st.session_state['historical_run_data'] = historical_data
                            st.session_state['view_mode'] = 'historical'
                            st.success(f"✅ Loaded historical run: {run_id}")
                        except Exception as e:
                            st.error(f"Failed to load run: {e}")
                            st.session_state['view_mode'] = 'live'
                    else:
                        st.session_state['view_mode'] = 'live'
                else:
                    st.info("No historical Test 5 runs available yet. Run Test 5 to create logs.")
                    st.session_state['view_mode'] = 'live'
        
                st.divider()
        
            # Create enhanced dashboard tabs
            dashboard_tabs = st.tabs([
                "📊 Overview",
                "📈 Gantt Timeline",
                "📋 Task Cards",
                "📜 Event Log",
                "📥 Export",
                "🧠 Memory Inspector",
                "🔒 Security Audit Log",
                "📜 Rethink History",
                "🔍 Interactive Tools"
            ])
        
            # --- TAB 1: OVERVIEW ---
            with dashboard_tabs[0]:
                st.subheader("Execution Overview")
        
                test_events = tracker.get_test_events(selected_test)
        
                if not test_events:
                    st.info(f"No execution data for {selected_test}. Run the test first.")
                else:
                    # KPIs
                    col1, col2, col3, col4 = st.columns(4)
        
                    with col1:
                        total_agents = len(set([e.agent_id for e in test_events]))
                        st.metric("Total Agents", total_agents)
        
                    with col2:
                        completed = len([e for e in test_events if e.status == "complete"])
                        st.metric("Completed", completed)
        
                    with col3:
                        running = len(tracker.active_agents)
                        st.metric("Running", running)
        
                    with col4:
                        errors = len([e for e in test_events if e.status == "error"])
                        st.metric("Errors", errors)
        
                    st.divider()
        
                    # Timeline summary
                    st.subheader("Timeline Summary")
        
                    start_time = min([e.timestamp for e in test_events])
                    end_time = max([e.timestamp for e in test_events])
                    total_duration = end_time - start_time
        
                    col1, col2 = st.columns(2)
        
                    with col1:
                        st.metric("Total Duration", f"{total_duration:.1f}s")
        
                    with col2:
                        durations = [e.duration for e in test_events if e.duration]
                        avg_agent_duration = sum(durations) / max(len(durations), 1) if durations else 0
                        st.metric("Avg Agent Duration", f"{avg_agent_duration:.2f}s")
        
            # --- TAB 2: GANTT TIMELINE ---
            with dashboard_tabs[1]:
                st.subheader("Execution Timeline (Gantt Chart)")
        
                test_events = tracker.get_test_events(selected_test)
        
                if not test_events:
                    st.info("No events recorded for this test.")
                else:
                    # Build Gantt data
                    gantt_df = generate_gantt_data(selected_test, tracker)
        
                    if gantt_df.empty:
                        st.info("No timeline data available.")
                    else:
                        # Color scale based on status
                        color_map = {
                            'complete': '#10B981',
                            'running': '#F59E0B',
                            'error': '#EF4444',
                            'pending': '#94A3B8'
                        }
        
                        colors = [color_map.get(status, '#94A3B8') for status in gantt_df['Status']]
        
                        fig = go.Figure()
        
                        for idx, row in gantt_df.iterrows():
                            # Add hover text with metadata
                            hover_text = f"<b>{row['Task']}</b><br>"
                            hover_text += f"Start: {row['Start']:.1f}s<br>"
                            hover_text += f"Duration: {row['Duration']:.2f}s<br>"
                            hover_text += f"Status: {row['Status']}<br>"
        
                            if 'Metadata' in row and row['Metadata']:
                                metadata = row['Metadata']
                                if isinstance(metadata, dict):
                                    for key, value in list(metadata.items())[:3]:  # Show first 3 metadata items
                                        if key != 'code':  # Skip code in hover
                                            hover_text += f"{key}: {value}<br>"
        
                            fig.add_trace(go.Bar(
                                y=[row['Task']],
                                x=[row['Duration']],
                                base=row['Start'],
                                orientation='h',
                                name=row['Task'],
                                marker=dict(color=colors[idx]),
                                text=f"{row['Progress']:.0f}%",
                                textposition='inside',
                                hovertemplate=hover_text + "<extra></extra>",
                                showlegend=False
                            ))
        
                        fig.update_layout(
                            title=f"{selected_test}: Execution Timeline",
                            xaxis_title="Time (seconds)",
                            yaxis_title="Agent / Task",
                            height=max(400, len(gantt_df) * 40),
                            showlegend=False,
                            hovermode='closest',
                            plot_bgcolor='#F7F7FB',
                            paper_bgcolor='white'
                        )
        
                        st.plotly_chart(fig, width='content', config=PLOTLY_CONFIG)
        
                        # Summary metrics below chart
                        col1, col2, col3, col4 = st.columns(4)
        
                        with col1:
                            st.metric("Total Tasks", len(gantt_df))
        
                        with col2:
                            completed = len(gantt_df[gantt_df['Status'] == 'complete'])
                            st.metric("Completed", completed)
        
                        with col3:
                            total_duration = gantt_df['Start'].max() + gantt_df.loc[gantt_df['Start'].idxmax(), 'Duration']
                            st.metric("Total Time", f"{total_duration:.1f}s")
        
                        with col4:
                            avg_progress = gantt_df['Progress'].mean()
                            st.metric("Avg Progress", f"{avg_progress:.0f}%")
        
            # --- TAB 3: TASK CARDS ---
            with dashboard_tabs[2]:
                render_task_cards(selected_test, tracker)
        
            # --- TAB 4: EVENT LOG ---
            with dashboard_tabs[3]:
                st.subheader("Event Log")
        
                # Check if we have historical Test 5 data with enhanced logging
                view_mode = st.session_state.get('view_mode', 'live')
        
                if view_mode == 'historical' and 'historical_run_data' in st.session_state and test_selector == "test5":
                    # Enhanced Event Log for Test 5 with structured entries
                    run_data = st.session_state['historical_run_data']
                    execution_log = run_data.get('execution_log', [])
        
                    if not execution_log:
                        st.info("No execution log data available for this run.")
                    else:
                        st.markdown("### Structured Execution Log")
        
                        # Enhanced filter controls
                        col1, col2, col3, col4 = st.columns(4)
        
                        with col1:
                            all_event_types = set(entry.get('event_type', 'UNKNOWN') for entry in execution_log)
                            event_type_filter = st.multiselect(
                                "Event Type",
                                options=sorted(all_event_types),
                                default=list(all_event_types),
                                help="Filter by event type: TOOL_RULE_ENFORCED, MEMORY_WRITE, SECURITY_AUDIT, RETHINK_TRIGGERED"
                            )
        
                        with col2:
                            all_severities = set(entry.get('severity', 'INFO') for entry in execution_log)
                            severity_filter = st.multiselect(
                                "Severity",
                                options=sorted(all_severities),
                                default=list(all_severities),
                                help="Filter by severity: INFO, WARNING, ERROR, SECURITY_ALERT"
                            )
        
                        with col3:
                            all_agents = set(entry.get('agent', 'Unknown') for entry in execution_log)
                            agent_filter = st.multiselect(
                                "Agent",
                                options=sorted(all_agents),
                                default=list(all_agents)
                            )
        
                        with col4:
                            # Turn range filter
                            max_turn = max((entry.get('turn', 0) for entry in execution_log), default=0)
                            turn_range = st.slider(
                                "Turn Range",
                                min_value=0,
                                max_value=max_turn,
                                value=(0, max_turn),
                                help="Filter events by turn number"
                            )
        
                        # Apply filters
                        filtered_log = [
                            entry for entry in execution_log
                            if entry.get('event_type') in event_type_filter
                            and entry.get('severity') in severity_filter
                            and entry.get('agent') in agent_filter
                            and turn_range[0] <= entry.get('turn', 0) <= turn_range[1]
                        ]
        
                        # Display with visual indicators
                        if filtered_log:
                            st.caption(f"Showing {len(filtered_log)} of {len(execution_log)} events")
        
                            # Event type icons and colors
                            event_icons = {
                                'TOOL_RULE_ENFORCED': '⚙️',
                                'MEMORY_WRITE': '💾',
                                'SECURITY_AUDIT': '🔒',
                                'RETHINK_TRIGGERED': '🔄',
                                'CODE_GENERATED': '📝',
                                'VALIDATION': '✅'
                            }
        
                            severity_colors = {
                                'INFO': '🔵',
                                'WARNING': '🟡',
                                'ERROR': '🔴',
                                'SECURITY_ALERT': '🔴'
                            }
        
                            # Create enhanced table
                            log_data = []
                            for entry in filtered_log:
                                event_type = entry.get('event_type', 'UNKNOWN')
                                severity = entry.get('severity', 'INFO')
        
                                icon = event_icons.get(event_type, '📋')
                                color = severity_colors.get(severity, '⚪')
        
                                log_data.append({
                                    'Turn': entry.get('turn', 0),
                                    'Event': f"{icon} {event_type}",
                                    'Severity': f"{color} {severity}",
                                    'Agent': entry.get('agent', 'Unknown'),
                                    'Message': entry.get('message', '')[:80] + "..." if len(entry.get('message', '')) > 80 else entry.get('message', ''),
                                    'Timestamp': entry.get('timestamp', '')[:19]
                                })
        
                            df_log = pd.DataFrame(log_data)
                            st.dataframe(df_log, use_container_width=True, height=500)
        
                            # Event type distribution
                            st.divider()
                            st.markdown("### Event Distribution")
        
                            col1, col2 = st.columns(2)
        
                            with col1:
                                # Event type counts
                                from collections import Counter
                                event_counts = Counter([entry.get('event_type', 'UNKNOWN') for entry in filtered_log])
        
                                st.markdown("**Event Types:**")
                                for event_type, count in event_counts.most_common():
                                    icon = event_icons.get(event_type, '📋')
                                    st.text(f"{icon} {event_type}: {count}")
        
                            with col2:
                                # Severity counts
                                severity_counts = Counter([entry.get('severity', 'INFO') for entry in filtered_log])
        
                                st.markdown("**Severity Levels:**")
                                for severity, count in severity_counts.most_common():
                                    color = severity_colors.get(severity, '⚪')
                                    st.text(f"{color} {severity}: {count}")
                        else:
                            st.info("No events match the selected filters.")
                else:
                    # Standard Event Log for other tests
                    test_events = tracker.get_test_events(selected_test)
        
                    if not test_events:
                        st.info("No events to display.")
                    else:
                        # Filter controls
                        col1, col2 = st.columns([1, 1])
        
                        with col1:
                            event_type_filter = st.multiselect(
                                "Event Type",
                                options=["start", "progress", "complete", "error"],
                                default=["start", "complete", "error"]
                            )
        
                        with col2:
                            agent_type_filter = st.multiselect(
                                "Agent Type",
                                options=["orchestrator", "main_agent", "sub_agent", "batch"],
                                default=["orchestrator", "main_agent", "sub_agent", "batch"]
                            )
        
                        # Filter events
                        filtered_events = [
                            e for e in test_events
                            if e.event_type in event_type_filter and e.agent_type in agent_type_filter
                        ]
        
                        # Display as table
                        if filtered_events:
                            df_events = pd.DataFrame([
                                {
                                    'Timestamp': f"{e.timestamp:.2f}s",
                                    'Event': e.event_type,
                                    'Agent': e.agent_name,
                                    'Type': e.agent_type,
                                    'Status': e.status,
                                    'Progress': f"{e.progress:.0f}%",
                                    'Duration': f"{e.duration:.2f}s" if e.duration else "—",
                                    'Metadata': str(e.metadata)[:50] + "..." if e.metadata else ""
                                }
                                for e in filtered_events
                            ])
        
                            st.dataframe(df_events, width='content', height=400)
                        else:
                            st.info("No events match the filters.")
        
            # --- TAB 5: EXPORT ---
            with dashboard_tabs[4]:
                st.subheader("Export Execution Data")
        
                test_events = tracker.get_test_events(selected_test)
        
                if not test_events:
                    st.info("No data to export.")
                else:
                    # Export options
                    col1, col2 = st.columns(2)
        
                    with col1:
                        st.markdown("**Export Timeline Data**")
        
                        df_export = tracker.export_timeline(selected_test)
        
                        csv_bytes = df_export.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download CSV",
                            data=csv_bytes,
                            file_name=f"{selected_test.lower().replace(' ', '_')}_timeline.csv",
                            mime="text/csv",
                            width='content'
                        )
        
                        json_str = df_export.to_json(orient='records', indent=2)
                        st.download_button(
                            "📥 Download JSON",
                            data=json_str,
                            file_name=f"{selected_test.lower().replace(' ', '_')}_timeline.json",
                            mime="application/json",
                            width='content'
                        )
        
                    with col2:
                        st.markdown("**Export Gantt Chart**")
        
                        # Generate Gantt data
                        gantt_df = generate_gantt_data(selected_test, tracker)
        
                        if not gantt_df.empty:
                            csv_gantt = gantt_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📥 Download Gantt CSV",
                                data=csv_gantt,
                                file_name=f"{selected_test.lower().replace(' ', '_')}_gantt.csv",
                                mime="text/csv",
                                width='content'
                            )
        
                    # Preview
                    st.markdown("**Data Preview**")
                    st.dataframe(df_export.head(20), width='content')
        
            # --- TAB 6: MEMORY INSPECTOR ---
            with dashboard_tabs[5]:
                st.subheader("🧠 Memory Inspector")
        
                # Get historical data if in historical mode
                view_mode = st.session_state.get('view_mode', 'live')
        
                if view_mode == 'historical' and 'historical_run_data' in st.session_state:
                    run_data = st.session_state['historical_run_data']
                    memory_snapshots = run_data.get('memory_snapshots', [])
        
                    if not memory_snapshots:
                        st.info("No memory data available for this run.")
                    else:
                        # Get latest snapshot
                        latest_snapshot = memory_snapshots[-1]
        
                        st.markdown("### Current Memory Blocks")
                        core_blocks = latest_snapshot.get('core_blocks', {})
        
                        for block_name, block_data in core_blocks.items():
                            with st.expander(f"📝 {block_name}", expanded=False):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Version", block_data.get('version', 1))
                                with col2:
                                    st.metric("Last Modified Turn", block_data.get('last_modified_turn', 0))
                                with col3:
                                    st.metric("Modifications", block_data.get('modification_count', 0))
        
                                st.code(block_data.get('content', ''), language='text')
        
                        st.divider()
        
                        st.markdown("### Archival Memory Index")
                        archival_entries = latest_snapshot.get('archival_entries', [])
        
                        if archival_entries:
                            # Pagination
                            page_size = 50
                            total_pages = (len(archival_entries) + page_size - 1) // page_size
        
                            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
                            start_idx = (page - 1) * page_size
                            end_idx = min(start_idx + page_size, len(archival_entries))
        
                            # Filters
                            col1, col2 = st.columns(2)
                            with col1:
                                all_tags = set()
                                for entry in archival_entries:
                                    all_tags.update(entry.get('tags', []))
        
                                tag_filter = st.multiselect("Filter by Tags", options=sorted(all_tags))
        
                            with col2:
                                all_agents = set(entry.get('source_agent', 'Unknown') for entry in archival_entries)
                                agent_filter = st.multiselect("Filter by Source Agent", options=sorted(all_agents))
        
                            # Apply filters
                            filtered_entries = archival_entries
                            if tag_filter:
                                filtered_entries = [e for e in filtered_entries if any(tag in e.get('tags', []) for tag in tag_filter)]
                            if agent_filter:
                                filtered_entries = [e for e in filtered_entries if e.get('source_agent') in agent_filter]
        
                            # Display entries
                            st.caption(f"Showing {start_idx + 1}-{end_idx} of {len(filtered_entries)} entries")
        
                            for entry in filtered_entries[start_idx:end_idx]:
                                with st.expander(f"[{entry.get('source_agent', 'Unknown')}] {entry.get('content', '')[:100]}...", expanded=False):
                                    st.text(entry.get('content', ''))
                                    st.caption(f"Tags: {', '.join(entry.get('tags', []))} | {entry.get('timestamp', '')}")
                        else:
                            st.info("No archival memory entries.")
        
                        st.divider()
        
                        st.markdown("### Memory Statistics")
                        stats = latest_snapshot.get('statistics', {})
        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Entries", len(archival_entries))
                        with col2:
                            st.metric("Total Retrievals", stats.get('total_retrievals', 0))
                        with col3:
                            st.metric("Cache Hits", stats.get('cache_hits', 0))
                        with col4:
                            avg_latency = stats.get('avg_retrieval_latency_ms', 0.0)
                            st.metric("Avg Retrieval Latency", f"{avg_latency:.2f}ms")
                else:
                    st.info("Memory Inspector is only available for Test 5 historical runs. Run Test 5 and then load a past run to view memory data.")
        
            # --- TAB 7: SECURITY AUDIT LOG ---
            with dashboard_tabs[6]:
                st.subheader("🔒 Security Audit Log")
        
                view_mode = st.session_state.get('view_mode', 'live')
        
                if view_mode == 'historical' and 'historical_run_data' in st.session_state:
                    run_data = st.session_state['historical_run_data']
                    security_audits = run_data.get('security_audits', [])
        
                    if not security_audits:
                        st.info("No security audit data available for this run.")
                    else:
                        st.markdown("### Audit Events Table")
        
                        # Filters
                        col1, col2, col3 = st.columns(3)
        
                        with col1:
                            all_operations = set(audit.get('operation_type', 'Unknown') for audit in security_audits)
                            operation_filter = st.multiselect("Operation Type", options=sorted(all_operations), default=list(all_operations))
        
                        with col2:
                            all_risk_levels = set(audit.get('risk_level', 'UNKNOWN') for audit in security_audits)
                            risk_filter = st.multiselect("Risk Level", options=sorted(all_risk_levels), default=list(all_risk_levels))
        
                        with col3:
                            all_statuses = set(audit.get('status', 'Unknown') for audit in security_audits)
                            status_filter = st.multiselect("Status", options=sorted(all_statuses), default=list(all_statuses))
        
                        # Apply filters
                        filtered_audits = [
                            audit for audit in security_audits
                            if audit.get('operation_type') in operation_filter
                            and audit.get('risk_level') in risk_filter
                            and audit.get('status') in status_filter
                        ]
        
                        # Display table
                        audit_data = []
                        for audit in filtered_audits:
                            status_icon = {
                                'Pass': '🟢',
                                'Blocked': '🔴',
                                'Warning': '🟡'
                            }.get(audit.get('status', ''), '⚪')
        
                            audit_data.append({
                                'Turn': audit.get('turn', 0),
                                'Agent': audit.get('agent', 'Unknown'),
                                'Operation': audit.get('operation_type', 'Unknown'),
                                'Risk': audit.get('risk_level', 'UNKNOWN'),
                                'Status': f"{status_icon} {audit.get('status', 'Unknown')}",
                                'Details': str(audit.get('details', {}))[:50] + "..."
                            })
        
                        if audit_data:
                            st.dataframe(pd.DataFrame(audit_data), use_container_width=True, height=400)
        
                            # Security Metrics Dashboard
                            st.divider()
                            st.markdown("### Security Metrics")
        
                            col1, col2, col3, col4 = st.columns(4)
        
                            with col1:
                                st.metric("Total Audits", len(security_audits))
        
                            with col2:
                                blocked = len([a for a in security_audits if a.get('status') == 'Blocked'])
                                st.metric("Blocked Operations", blocked)
        
                            with col3:
                                warnings = len([a for a in security_audits if a.get('status') == 'Warning'])
                                st.metric("Warnings", warnings)
        
                            with col4:
                                passed = len([a for a in security_audits if a.get('status') == 'Pass'])
                                st.metric("Passed", passed)
        
                            # Pie chart: Distribution of operation types
                            operation_counts = {}
                            for audit in security_audits:
                                op_type = audit.get('operation_type', 'Unknown')
                                operation_counts[op_type] = operation_counts.get(op_type, 0) + 1
        
                            if operation_counts:
                                fig = go.Figure(data=[go.Pie(
                                    labels=list(operation_counts.keys()),
                                    values=list(operation_counts.values()),
                                    hole=0.3
                                )])
        
                                fig.update_layout(
                                    title="Distribution of Operation Types",
                                    height=300
                                )
        
                                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                        else:
                            st.info("No audits match the selected filters.")
                else:
                    st.info("Security Audit Log is only available for Test 5 historical runs. Run Test 5 and then load a past run to view security data.")
        
            # --- TAB 8: RETHINK HISTORY ---
            with dashboard_tabs[7]:
                st.subheader("📜 Rethink History")
        
                view_mode = st.session_state.get('view_mode', 'live')
        
                if view_mode == 'historical' and 'historical_run_data' in st.session_state:
                    run_data = st.session_state['historical_run_data']
                    rethink_history = run_data.get('rethink_history', [])
        
                    if not rethink_history:
                        st.info("No rethink events recorded for this run.")
                    else:
                        st.markdown("### Policy Modification Log")
        
                        # Display rethink events
                        for idx, event in enumerate(rethink_history, 1):
                            with st.expander(f"🔄 Rethink #{idx} - Turn {event.get('turn', 0)} - {event.get('block_name', 'Unknown')}", expanded=False):
                                col1, col2 = st.columns(2)
        
                                with col1:
                                    st.markdown("**Trigger:**")
                                    st.text(event.get('trigger', 'N/A'))
        
                                    st.markdown("**Change Summary:**")
                                    st.text(event.get('change_summary', 'N/A'))
        
                                with col2:
                                    st.markdown("**Revert Point ID:**")
                                    st.code(event.get('revert_point_id', 'N/A'))
        
                                    st.markdown("**Timestamp:**")
                                    st.text(event.get('timestamp', 'N/A'))
        
                                st.divider()
        
                                # Side-by-side diff
                                st.markdown("**Content Diff:**")
                                col_before, col_after = st.columns(2)
        
                                with col_before:
                                    st.markdown("*Before:*")
                                    st.code(event.get('old_content', ''), language='text')
        
                                with col_after:
                                    st.markdown("*After:*")
                                    st.code(event.get('new_content', ''), language='text')
        
                        st.divider()
        
                        # Self-Correction Analytics
                        st.markdown("### Self-Correction Analytics")
        
                        # Most frequently modified blocks
                        from collections import Counter
                        block_counts = Counter([event.get('block_name', 'Unknown') for event in rethink_history])
                        most_common = block_counts.most_common(3)
        
                        if most_common:
                            st.markdown("**Most Frequently Modified Blocks:**")
                            for block_name, count in most_common:
                                st.text(f"  • {block_name}: {count} modification(s)")
        
                        # Average turns between rethinks
                        if len(rethink_history) > 1:
                            turns = [event.get('turn', 0) for event in rethink_history]
                            avg_gap = sum(turns[i+1] - turns[i] for i in range(len(turns)-1)) / (len(turns) - 1)
                            st.metric("Average Turns Between Rethinks", f"{avg_gap:.1f}")
                else:
                    st.info("Rethink History is only available for Test 5 historical runs. Run Test 5 and then load a past run to view rethink data.")
        
            # --- TAB 9: INTERACTIVE TOOLS ---
            with dashboard_tabs[8]:
                st.subheader("🔍 Interactive Tools")
        
                view_mode = st.session_state.get('view_mode', 'live')
        
                if view_mode == 'historical' and 'historical_run_data' in st.session_state:
                    run_data = st.session_state['historical_run_data']
                    memory_snapshots = run_data.get('memory_snapshots', [])
        
                    if not memory_snapshots:
                        st.info("No memory data available for interactive tools.")
                    else:
                        latest_snapshot = memory_snapshots[-1]
                        archival_entries = latest_snapshot.get('archival_entries', [])
        
                        # Memory Search Simulator
                        st.markdown("### 🔍 Memory Search Simulator")
                        st.caption("Demonstrate RAG mechanism interactively")
        
                        col1, col2 = st.columns([3, 1])
        
                        with col1:
                            search_query = st.text_input("Search Query", placeholder="Enter keywords to search archival memory...")
        
                        with col2:
                            # Tag filters
                            all_tags = set()
                            for entry in archival_entries:
                                all_tags.update(entry.get('tags', []))
        
                            tag_filters = st.multiselect("Filter by Tags", options=sorted(all_tags))
        
                        if st.button("🔍 Search Archival Memory", type="primary"):
                            if not search_query:
                                st.warning("Please enter a search query.")
                            else:
                                # Simple keyword search
                                results = []
                                query_lower = search_query.lower()
        
                                for entry in archival_entries:
                                    # Tag filter
                                    if tag_filters and not any(tag in entry.get('tags', []) for tag in tag_filters):
                                        continue
        
                                    # Keyword matching
                                    content = entry.get('content', '').lower()
                                    if query_lower in content:
                                        # Calculate simple relevance score (keyword frequency)
                                        relevance = content.count(query_lower) / max(len(content.split()), 1)
                                        results.append((entry, relevance))
        
                                # Sort by relevance
                                results.sort(key=lambda x: x[1], reverse=True)
                                top_results = results[:5]
        
                                if top_results:
                                    st.success(f"Found {len(results)} matching entries. Showing top 5:")
        
                                    for idx, (entry, relevance) in enumerate(top_results, 1):
                                        with st.expander(f"Result #{idx} (Relevance: {relevance:.4f})", expanded=idx==1):
                                            st.text(entry.get('content', ''))
                                            st.caption(f"Source: {entry.get('source_agent', 'Unknown')} | Tags: {', '.join(entry.get('tags', []))}")
                                else:
                                    st.info("No matching entries found.")
        
                        st.divider()
        
                        # Tool Rule Tester
                        st.markdown("### ⚙️ Tool Rule Tester")
                        st.caption("Test tool usage rules from policy blocks")
        
                        core_blocks = latest_snapshot.get('core_blocks', {})
                        tool_guidelines = core_blocks.get('Tool Guidelines', {})
        
                        if tool_guidelines:
                            st.markdown("**Current Tool Guidelines:**")
                            st.code(tool_guidelines.get('content', 'No guidelines available'), language='text')
        
                            st.markdown("**Test a Task:**")
                            test_task = st.text_area("Sample Task Description", placeholder="Enter a task description to test against tool rules...")
        
                            if st.button("🧪 Test Rule Enforcement", type="secondary"):
                                if not test_task:
                                    st.warning("Please enter a task description.")
                                else:
                                    # Simple rule matching (can be enhanced)
                                    guidelines_text = tool_guidelines.get('content', '').lower()
                                    task_lower = test_task.lower()
        
                                    # Check for common rule keywords
                                    triggered_rules = []
        
                                    if 'verify' in guidelines_text and 'verify' not in task_lower:
                                        triggered_rules.append("⚠️ Guideline suggests verification, but task doesn't mention it")
        
                                    if 'test' in guidelines_text and 'test' not in task_lower:
                                        triggered_rules.append("⚠️ Guideline suggests testing, but task doesn't mention it")
        
                                    if 'cautious' in guidelines_text or 'careful' in guidelines_text:
                                        triggered_rules.append("ℹ️ Guidelines emphasize caution - ensure proper validation")
        
                                    if triggered_rules:
                                        st.warning("**Rule Enforcement Suggestions:**")
                                        for rule in triggered_rules:
                                            st.text(f"  • {rule}")
                                    else:
                                        st.success("✅ Task appears to align with tool guidelines")
                        else:
                            st.info("No Tool Guidelines available in this run.")
        
                # --- LIVE SMOKE TEST UTILITY (Always Available) ---
                st.divider()
                st.subheader("🔍 Live Smoke Test Utility")
                st.caption("Rapidly validate core planning, policy RAG, and execution across key scenarios.")
        
                smoke_scenario = st.selectbox(
                    "Select Scenario for Live Smoke Test",
                    options=list(SMOKE_TEST_SCENARIOS.keys()),
                    index=0,
                    key='smoke_test_select_final'
                )
        
                st.markdown(f"**Goal:** {SMOKE_TEST_SCENARIOS[smoke_scenario]['goal'][:150]}...")
        
                # Display memory policy if applicable
                policy = SMOKE_TEST_SCENARIOS[smoke_scenario]['policy']
                if policy:
                    with st.expander(f"Policy Loaded (RAG Source): {policy.splitlines()[0]}", expanded=False):
                        st.code(policy, language='text')
        
                if st.button("▶️ Run Live Smoke Test (1-Turn Execution)", type="primary", use_container_width=True):
                    if not GEMINI_API_KEY:
                        st.error("GEMINI_API_KEY is required to run the smoke test.")
                    else:
                        # Run test synchronously and update dashboard selection
                        with st.spinner(f"Running **{smoke_scenario}**... (1 turn execution)"):
                            test_result = asyncio.run(run_live_smoke_test(smoke_scenario))
        
                        # Immediate Display and Logging
                        st.markdown("---")
                        if test_result['success']:
                            st.success("✅ Smoke Test PASSED: Core Orchestration Path Verified")
                            st.markdown("### Final Answer")
                            st.markdown(test_result['final_answer'])
                            st.markdown("### Specialized Output")
                            st.code(test_result['code_output'], language='python')
        
                            st.info(f"📊 Log available in **Event Log** and **Gantt Timeline** tabs. Select 'Smoke Test Run' from the test selector above.")
                        else:
                            st.error("❌ Smoke Test FAILED: Orchestrator Encountered an Error")
                            st.text(f"Error: {test_result['error']}")
                            st.warning("Check API Keys and ensure GEMINI_API_KEY is set correctly.")
        
                        # Clean up temporary policy setting
                        if 'demo_memory_policy' in st.session_state:
                            del st.session_state['demo_memory_policy']
                        if 'demo_scenario' in st.session_state:
                            del st.session_state['demo_scenario']
        
                        # Force dashboard selector to point to the smoke test to show log immediately
                        st.session_state['selected_test_in_dashboard'] = 'Smoke Test Run'
                        st.session_state['selected_test_in_dashboard_selector'] = 'smoke_test'
                        st.rerun()
        
            # Auto-refresh
            if auto_refresh:
                time.sleep(2)
                st.rerun()
        
        



