"""
Fix and patch the renderBatters and renderBowlers JS functions in index.html
to have correct template literal syntax for the RAE/RPAE stat cells.
"""

content = open('index.html', encoding='utf-8').read()

# Find the renderBatters function's broken RAE line
# The broken pattern produces:  ${b.hp_rae >= 0 ? '+' : ''}</td>${  <newline>  b.hp_rae...
# We need to find this broken cell and fix it

# Strategy: find the two broken JS lines and replace them
# In renderBatters:
OLD_BAT_STAT = (
    "      <td class=\"stat-cell ${b.hp_sr.toFixed(0)}\"></td>\n"
    "      <td class=\"stat-cell ${b.hp_rae >= 0 ? 'pos' : 'neg'}\">${b.hp_rae >= 0 ? '+' : ''}</td>${b.hp_rae.toFixed(1)}</td>\n"
)

# Instead, do a targeted search for the problematic pattern
# Look for the render function region between the two function names
rB_start = content.find('function renderBatters(list)')
rB_end = content.find('function renderBowlers(list)')
rBowl_start = rB_end
rBowl_end = content.find('/* ---- PLAYER MODAL ----')

bat_fn = content[rB_start:rB_end]
bowl_fn = content[rBowl_start:rBowl_end]

print("=== BAT FN STAT CELLS ===")
# Find the hp_sr and hp_rae lines
for line in bat_fn.split('\n'):
    if 'hp_sr' in line or 'hp_rae' in line or 'hp_rpae' in line:
        print(repr(line[:100]))

print("\n=== BOWL FN STAT CELLS ===")
for line in bowl_fn.split('\n'):
    if 'hp_econ' in line or 'hp_rpae' in line:
        print(repr(line[:100]))
