import pathlib
R = pathlib.Path("/workspace/fasterwam-a3-cookie")
head = pathlib.Path("/workspace/readme_head.md").read_bytes()
sec = pathlib.Path("/workspace/dagger_section.md").read_bytes().replace(b"\r\n", b"\n")
new = head.rstrip() + b"\n\n" + sec.rstrip() + b"\n"
(R / "bridge/_next/README.md").write_bytes(new)
pathlib.Path("/workspace/fasterwam_a3_bridge/bridge/_next/README.md").write_bytes(new)
print("README", len(new))
