from pathlib import Path
import re

p = Path("app.py")
if not p.exists():
    raise SystemExit("ERROR: app.py not found. Put this file inside the bot folder, then run it again.")

s = p.read_text(encoding="utf-8")
original = s

# Streamlit Cloud is crashing because some pages are rendered inside other panels,
# and those pages also contain st.expander(). Streamlit does not allow nested expanders.
# This patch converts ALL expander blocks into bordered containers.
#
# Example:
#   with st.expander("Title", expanded=False):
# becomes:
#   with st.container(border=True):
#       st.markdown("### Title")
#
# The inside block remains valid because indentation stays the same.

def make_title(arg_text: str) -> str:
    arg_text = arg_text.strip()
    # Extract a clean title for normal string/f-string first argument.
    m = re.match(r'^[fFrRbBuU]*([\'"])(.*?)\1', arg_text)
    if m:
        title = m.group(2)
        # Remove braces from f-string titles so markdown does not execute code.
        title = title.replace("{", "").replace("}", "")
        title = title.replace('"', "'")
        return title[:120] if title else "Details"
    return "Details"

lines = s.splitlines()
out = []
changed = 0

for line in lines:
    m = re.match(r'^(\s*)with\s+st\.expander\((.*)\):\s*$', line)
    if m:
        indent = m.group(1)
        args = m.group(2)
        # Best-effort first argument only; do not try to evaluate code.
        first_arg = args.split(",", 1)[0]
        title = make_title(first_arg)
        out.append(f'{indent}with st.container(border=True):')
        out.append(f'{indent}    st.markdown("### {title}")')
        changed += 1
    else:
        out.append(line)

s = "\n".join(out) + ("\n" if original.endswith("\n") else "")

# Also fix common manually-introduced accordion indentation issue.
s = s.replace('''    if layout_mode == "Accordion panels":
    for idx, panel_name in enumerate(available_panels):
''', '''    if layout_mode == "Accordion panels":
        for idx, panel_name in enumerate(available_panels):
''')

p.write_text(s, encoding="utf-8")

print(f"OK: Converted {changed} st.expander blocks into safe bordered containers.")
print("Now run:")
print("python -m py_compile app.py")
print("git add app.py")
print('git commit -m "Disable nested expanders for Streamlit Cloud"')
print("git push")
