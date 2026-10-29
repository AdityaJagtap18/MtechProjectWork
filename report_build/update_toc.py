import re, subprocess, sys
src = open('report_build/build_report.js').read()
entries = re.findall(r'^\s*\["(.+?)", (\d+), ([12])\],$', src, flags=re.M)
text = subprocess.run(['pdftotext','-layout','QGNN_V4_Final_Research_Report.pdf','-'],capture_output=True,text=True).stdout
pages = text.split('\f')
def norm(t): return re.sub(r'\s+',' ',t.replace('\\u201c','\u201c').replace('\\u201d','\u201d').replace('\\u2014','\u2014')).strip()
changed = 0
new_src = src
for title, old, lvl in entries:
    t = norm(title)
    found = None
    for i, pg in enumerate(pages, start=1):
        if i <= 3: continue
        for line in pg.split('\n'):
            if norm(line) == t:
                found = i; break
        if found: break
    if not found:
        print('NOT FOUND', title); continue
    if int(old) != found:
        changed += 1
        new_src = new_src.replace(f'["{title}", {old}, {lvl}]', f'["{title}", {found}, {lvl}]', 1)
open('report_build/build_report.js','w').write(new_src)
print('changed', changed)
