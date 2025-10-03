"""
Script to remove extracted visualization functions from streamlit_test_v5.py
"""

import re

# Read the file
with open('streamlit_test_v5.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and mark lines to remove
functions_to_remove = [
    'render_model_comparison_chart',
    'render_organized_results',
    'render_progress_replay',
    'render_agent_gantt_chart',
    'render_test5_gantt_chart',
    'render_universal_gantt_chart',
    'render_task_cards',
    'render_single_task_card',
    'render_live_agent_status',
    'render_agent_task_cards'
]

# Track which lines to keep
keep_lines = []
i = 0
removed_count = 0

while i < len(lines):
    line = lines[i]
    
    # Check if this line starts a function we want to remove
    is_target_function = False
    for func_name in functions_to_remove:
        if re.match(rf'^def {func_name}\(', line):
            is_target_function = True
            print(f"Found {func_name} at line {i+1}")
            break
    
    if is_target_function:
        # Skip this function - find where it ends
        indent_level = len(line) - len(line.lstrip())
        i += 1  # Move past the def line
        
        # Skip until we find a line with same or less indentation that's not blank/comment
        while i < len(lines):
            current_line = lines[i]
            current_indent = len(current_line) - len(current_line.lstrip())
            
            # If we hit a non-empty, non-comment line with same or less indentation, we're done
            if current_line.strip() and not current_line.strip().startswith('#'):
                if current_indent <= indent_level:
                    # Check if it's another def or class
                    if current_line.strip().startswith('def ') or current_line.strip().startswith('class '):
                        break
            i += 1
        
        removed_count += 1
    else:
        keep_lines.append(line)
        i += 1

# Write the cleaned file
with open('streamlit_test_v5.py', 'w', encoding='utf-8') as f:
    f.writelines(keep_lines)

print(f"\nRemoved {removed_count} functions")
print(f"Original lines: {len(lines)}")
print(f"New lines: {len(keep_lines)}")
print(f"Lines removed: {len(lines) - len(keep_lines)}")

