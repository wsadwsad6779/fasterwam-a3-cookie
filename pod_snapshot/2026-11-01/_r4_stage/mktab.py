import os
H = {}
for line in open('/workspace/a3_eval/disk_hashes.txt'):
    if line.startswith('DONE'):
        continue
    parts = line.split(None, 1)
    if len(parts) != 2:
        continue
    H[parts[1].strip()] = parts[0]
rows = []
for line in open('/workspace/a3_eval/disk_files.txt'):
    size, _, path = line.rstrip('\n').partition('\t')
    rows.append((int(size), path, H.get(path, 'MISSING')))
rows.sort(reverse=True)
tot = sum(r[0] for r in rows)
print('files=%d total_bytes=%d' % (len(rows), tot))
for size, path, h in rows[:34]:
    print('%d\t%s\t%s' % (size, path, h))
