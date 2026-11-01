import pathlib
R = pathlib.Path("/workspace/fasterwam-a3-cookie")
S = pathlib.Path("/workspace/fasterwam_a3_bridge/bridge/_next")
sec = pathlib.Path("/workspace/dagger_section.md").read_bytes().replace(b"\r\n", b"\n")
for p in [R / "bridge" / "_next" / "README.md", S / "README.md"]:
    b = p.read_bytes()
    p.write_bytes(b.rstrip() + b"\n\n" + sec.rstrip() + b"\n")
files = ["dagger_recover.py", "dagger_to_dataset.py", "check_a3_dagger_dataset.py", "resume_held_expert.py", "dagger_resume.py"]
for f in files:
    txt = (S / f).read_bytes().replace(b"\r\n", b"\n")
    (R / "bridge" / "_next" / f).write_bytes(txt)
print("README_BYTES", (R / "bridge" / "_next" / "README.md").stat().st_size, "SYNCED", files)
