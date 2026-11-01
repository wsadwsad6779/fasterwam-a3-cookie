cd /workspace || exit 1
{
echo "=== R2 head ==="
head -40 r2.log
echo "=== R2 300-340 ==="
sed -n "300,340p" r2.log
echo "=== R2 tail ==="
tail -40 r2.log
} > /workspace/_a1.txt 2>&1
echo CMDDONE >> /workspace/_a1.txt
