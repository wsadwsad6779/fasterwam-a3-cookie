set -u
D=/workspace/_r6_stage/out
echo "driver start $(date -u +%FT%TZ)" > $D/driver.mark
bash /root/hfstage/run_slice.sh P0small 300 4 --batch P0 --max-file-bytes 2000000000; echo "P0small rc=$? $(date -u +%FT%TZ)" >> $D/driver.mark
bash /root/hfstage/run_slice.sh P0big 300 12 --batch P0 --min-file-bytes 2000000001; echo "P0big rc=$? $(date -u +%FT%TZ)" >> $D/driver.mark
bash /root/hfstage/run_slice.sh P1small 300 4 --batch P1 --max-file-bytes 2000000000; echo "P1small rc=$? $(date -u +%FT%TZ)" >> $D/driver.mark
bash /root/hfstage/run_slice.sh P1big 300 12 --batch P1 --min-file-bytes 2000000001; echo "P1big rc=$? $(date -u +%FT%TZ)" >> $D/driver.mark
echo "ALLSLICES $(date -u +%FT%TZ)" >> $D/driver.mark
