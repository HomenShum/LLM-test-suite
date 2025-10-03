"""Quick fix to remove leftover code between Pydantic comment and run_live_smoke_test"""

# Read the file
with open('streamlit_test_v5.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the markers
pydantic_comment_idx = None
smoke_test_idx = None

for i, line in enumerate(lines):
    if "# Pydantic models moved to core/models.py" in line:
        pydantic_comment_idx = i
    if line.strip().startswith("async def run_live_smoke_test"):
        smoke_test_idx = i
        break

if pydantic_comment_idx and smoke_test_idx:
    print(f"Found Pydantic comment at line {pydantic_comment_idx + 1}")
    print(f"Found smoke test at line {smoke_test_idx + 1}")
    print(f"Removing lines {pydantic_comment_idx + 2} to {smoke_test_idx}")
    
    # Keep everything before the leftover code and after it
    new_lines = lines[:pydantic_comment_idx + 1]
    new_lines.append("\n")
    new_lines.append("# ============================================================\n")
    new_lines.append("# DEMONSTRATION SCENARIO CONFIGURATIONS\n")
    new_lines.append("# ============================================================\n")
    new_lines.append("# NOTE: Scenario configurations moved to config/scenarios.py\n")
    new_lines.append("# Imported above as: PI_AGENT_GOAL_PROMPT, FOLDING_POLICY_BLOCK, etc.\n")
    new_lines.append("\n")
    new_lines.extend(lines[smoke_test_idx:])
    
    # Write back
    with open('streamlit_test_v5.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"\n✅ Removed {smoke_test_idx - pydantic_comment_idx - 1} lines of leftover code")
    print(f"Original: {len(lines)} lines")
    print(f"New: {len(new_lines)} lines")
else:
    print("❌ Could not find markers")

