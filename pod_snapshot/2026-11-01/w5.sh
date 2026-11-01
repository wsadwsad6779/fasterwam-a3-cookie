F=/workspace/_r5_stage/single_lowforce_probe.py
sed -i '/^if __name__ == /d; /^    main()$/d' $F
grep -c main $F
python3 -m py_compile $F && echo COMPILE1_OK
printf 'if __name__ == "__main__":\n    main()\n' >> $F
python3 -m py_compile $F && echo COMPILE2_OK
tail -22 $F
echo DONE