import json

with open('dataset/WLASL_v0.3.json') as f:
    wlasl = json.load(f)

print('Total glosses:', len(wlasl))

for i, gloss_entry in enumerate(wlasl[:5]):
    print(f'Gloss: {gloss_entry["gloss"]}')
    print(f'  Instances: {len(gloss_entry["instances"])}')
    for inst in gloss_entry["instances"][:2]:
        print(f'    video_id: {inst["video_id"]}, split: {inst["split"]}, fps: {inst.get("fps", "N/A")}')

# Check how many videos we have
import os
video_files = set(os.listdir('dataset/videos'))
print(f'\nTotal video files: {len(video_files)}')

# Check how many WLASL videos exist
found = 0
missing = 0
for gloss_entry in wlasl:
    for inst in gloss_entry["instances"]:
        vid = inst["video_id"] + ".mp4"
        if vid in video_files:
            found += 1
        else:
            missing += 1

print(f'WLASL videos found: {found}, missing: {missing}')

# Check class list
with open('dataset/wlasl_class_list.txt') as f:
    class_list = f.read().strip().split('\n')
print(f'\nClass list entries: {len(class_list)}')
for line in class_list[:10]:
    print(f'  {line}')