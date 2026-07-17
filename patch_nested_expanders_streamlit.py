from pathlib import Path
import re

p = Path("app.py")
if not p.exists():
    raise SystemExit("ERROR: app.py not found. Put this patch file inside the bot folder, then run it again.")

s = p.read_text(encoding="utf-8")
original = s

def replace_expander_with_container(text, title):
    # Replace any exact st.expander(title, expanded=True/False) line with a bordered container + heading.
    # Keeps the same indentation so the inner block remains valid.
    escaped = re.escape(title)
    pattern = re.compile(
        rf'^(?P<indent>[ \t]*)with st\.expander\("{escaped}"(?:,\s*expanded\s*=\s*(?:True|False))?\):\s*$',
        re.MULTILINE
    )
    def repl(m):
        indent = m.group("indent")
        return f'{indent}with st.container(border=True):\n{indent}    st.markdown("### {title}")'
    return pattern.sub(repl, text)

# Current Streamlit crash:
# Nested expander in Market & Technical Scanner -> multi_style_trading_desk_panel.
titles_to_convert = [
    "Risk, RR, and Position Controls",
    "Trading style settings",
    "Scenario Finder Settings",
    "Scanner Settings",
    "Pattern Settings",
    "Indicator Settings",
    "Chart Settings",
    "Advanced Settings",
]

for t in titles_to_convert:
    s = replace_expander_with_container(s, t)

# Also make the top-level workspace Accordion layout safe. No page should be wrapped
# inside st.expander because many pages already contain their own expanders.
old_block = '''    if layout_mode == "Accordion panels":
        for idx, panel_name in enumerate(available_panels):
            with st.expander(panel_name, expanded=(idx < 2)):
                PAGE_RENDERERS[panel_name]()
        return
'''
new_block = '''    if layout_mode == "Accordion panels":
        for idx, panel_name in enumerate(available_panels):
            with st.container(border=True):
                st.markdown(f"## {panel_name}")
                PAGE_RENDERERS[panel_name]()
        return
'''
s = s.replace(old_block, new_block)

# Fix a possible wrongly-indented replacement from manual edit.
s = s.replace('''    if layout_mode == "Accordion panels":
    for idx, panel_name in enumerate(available_panels):
''', '''    if layout_mode == "Accordion panels":
        for idx, panel_name in enumerate(available_panels):
''')

if s == original:
    print("WARNING: No matching expander lines were changed. Your file may already be patched or text is different.")
else:
    p.write_text(s, encoding="utf-8")
    print("OK: Nested expander crash patch applied.")
    print("Changed app.py. Now run: git add app.py && git commit -m \"Fix nested expander crash\" && git push")
