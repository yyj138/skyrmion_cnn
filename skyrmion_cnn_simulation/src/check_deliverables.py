"""交付前校验: 训练图纯净性 + 标签一致性"""
from PIL import Image
import os, csv

# 1. 训练图全部为灰度 mode L（无伪彩污染）
pom_dir = r'e:\大创\outputs\dataset\pom_images'
bad = []
n = 0
for f in os.listdir(pom_dir):
    if f.endswith('.png'):
        n += 1
        img = Image.open(os.path.join(pom_dir, f))
        if img.mode not in ('L', '1', 'I;16'):
            bad.append((f, img.mode))
print(f'训练图共 {n} 张，非灰度图: {bad if bad else "无 (全部纯净)"}')

# 2. presentation 图全部带 _pres 后缀
pres_dir = r'e:\大创\outputs\visualizations\presentation'
pres_files = [f for f in os.listdir(pres_dir) if f.endswith('.png')]
ok_suf = all(f.endswith('_pres.png') for f in pres_files)
print(f'展示图共 {len(pres_files)} 张，全部带 _pres 后缀: {ok_suf}')

# 3. labels_demo.csv 无 NaN 且文件一一对应
with open(r'e:\大创\outputs\dataset\labels\labels_demo.csv', encoding='utf-8') as fh:
    rows = list(csv.DictReader(fh))
nan_free = all(r['free_energy'] not in ('', 'nan', 'NaN') for r in rows)
match = all(os.path.exists(os.path.join(pom_dir, r['filename'])) for r in rows)
print(f'demo 标签 {len(rows)} 行，无 NaN: {nan_free}，文件一一对应: {match}')
