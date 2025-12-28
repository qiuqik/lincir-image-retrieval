#!/usr/bin/env python3
import csv, json, os, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / 'cir_db.csv'
TEST_OOD_CSV = ROOT / 'test_ood.csv'
IMAGES_DIR = ROOT / 'data' / 'test_ood' / 'EUROPEANA' / 'images'
OUT_DIR = ROOT / 'test'
OUT_DIR.mkdir(exist_ok=True)

SPLIT_OUT = OUT_DIR / 'split.eufcc.json'
CAP_OUT = OUT_DIR / 'cap.eufcc.json'

# collect available image ids (filename without ext)
image_files = [p.name for p in IMAGES_DIR.iterdir() if p.is_file()]
image_ids = [os.path.splitext(n)[0] for n in image_files]

# build split map from test_ood.csv (preferred) or fallback to available images
split_map = {}
if TEST_OOD_CSV.exists():
    with TEST_OOD_CSV.open(newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for r in reader:
            if not r:
                continue
            src_id = r[0].strip()
            if not src_id:
                continue
            img_filename = f"{src_id}.jpg"
            rel_path = f"./data/test_ood/EUROPEANA/images/{img_filename}"
            split_map[src_id] = rel_path
else:
    for iid in image_ids:
        split_map[iid] = f"./data/test_ood/EUROPEANA/images/{iid}.jpg"

rows = []
with CSV_PATH.open(newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    headers = next(reader)
    for r in reader:
        if not r:
            continue
        rows.append(r)

cap_list = []
next_pairid = 1
set_id_counter = 1

for r in rows:
    # CSV expected: id1, id2, ... , query
    id1 = r[0].strip()
    id2_raw = r[1].strip() if len(r) > 1 else ''
    caption = r[-1].strip() if len(r) >= 1 else ''

    # build id2 list (comma separated)
    id2_list = [x.strip() for x in id2_raw.split(',') if x.strip()] if id2_raw else []
    # reference is id1
    reference = id1

    # target_hard is first id2 if present
    target_hard = id2_list[0] if id2_list else None

    # members: start with id2_list in order, ensure target_hard at index 0
    members = list(id2_list)
    if target_hard and members and members[0] != target_hard:
        if target_hard in members:
            members.remove(target_hard)
        members.insert(0, target_hard)

    # ensure reference included (append if missing)
    if reference not in members:
        members.append(reference)

    # ensure unique members, preserve order
    seen = set()
    uniq_members = []
    for m in members:
        if m not in seen:
            uniq_members.append(m)
            seen.add(m)
    members = uniq_members

    # pad to 6 members using image_ids pool
    if len(members) < 6:
        pool = [i for i in image_ids if i not in members]
        while len(members) < 6 and pool:
            pick = random.choice(pool)
            members.append(pick)
            pool.remove(pick)
    if len(members) > 6:
        members = members[:6]

    # compute ranks (0-based)
    reference_rank = members.index(reference) if reference in members else -1
    target_rank = members.index(target_hard) if target_hard and target_hard in members else -1

    # build target_soft dict: each id2 -> 1.0
    target_soft = {m: 1.0 for m in id2_list} if id2_list else {}

    # build cap entry matching user's requested schema
    cap_entry = {
        "pairid": next_pairid,
        "reference": reference,
        "target_hard": target_hard,
        "target_soft": target_soft,
        "target_rank": target_rank,
        "caption": caption,
        "img_set": {
            "id": set_id_counter,
            "members": members,
            "reference_rank": reference_rank
        }
    }
    cap_list.append(cap_entry)

    next_pairid += 1
    set_id_counter += 1

# write outputs
with SPLIT_OUT.open('w', encoding='utf-8') as f:
    json.dump(split_map, f, ensure_ascii=False, indent=2)

with CAP_OUT.open('w', encoding='utf-8') as f:
    json.dump(cap_list, f, ensure_ascii=False, indent=2)

print(f"Wrote {SPLIT_OUT} ({len(split_map)} entries) and {CAP_OUT} ({len(cap_list)} entries)")
