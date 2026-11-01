OUT=/workspace/a3_eval
: > $OUT/disk_files.txt
for d in /workspace/a3_train_runs /workspace/a3_data /workspace/FasterWAM/checkpoints /workspace/data /workspace/a3_eval /workspace/fasterwam_a3_bridge/bridge/_next /workspace/finetune; do
  find "$d" -type f ! -name '*.mp4' ! -name 'disk_*' -printf '%s\t%p\n' >> $OUT/disk_files.txt 2>/dev/null
done
awk -F'\t' '{print $2}' $OUT/disk_files.txt | xargs -d '\n' -n 8 sha256sum > $OUT/disk_hashes.txt 2>$OUT/disk_hash_err.txt
echo DONE_HASH >> $OUT/disk_hashes.txt
