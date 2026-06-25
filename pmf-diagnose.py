"""
PMF File Diagnostic
===================
Run this after prepare_pmf_inputs.py to inspect any conc/unc pair at the
byte level. Paste the output here so we can identify the exact issue.
"""

import os
from pathlib import Path

# ── EDIT THESE to point at any conc/unc pair ────────────────────────────────
CONC_FILE = Path("pmf_ready/Xact_PMF_fireworks_2023_conc.csv")
UNC_FILE  = Path("pmf_ready/Xact_PMF_fireworks_2023_unc.csv")
# ────────────────────────────────────────────────────────────────────────────

SEP = "─" * 60


def read_raw(path):
    with open(path, "rb") as f:
        return f.read()


def first_n_lines_raw(data: bytes, n=5):
    lines = data.split(b"\n")
    return lines[:n]


def detect_line_ending(data: bytes):
    crlf = data.count(b"\r\n")
    lf   = data.count(b"\n") - crlf   # bare LF only
    cr   = data.count(b"\r") - crlf   # bare CR only
    return crlf, lf, cr


def detect_bom(data: bytes):
    if data[:3] == b"\xef\xbb\xbf":
        return "UTF-8 BOM"
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return "UTF-16 BOM"
    return "None"


def show_header_bytes(data: bytes):
    first_line = data.split(b"\n")[0]
    print(f"  Header bytes (hex): {first_line.hex()}")
    print(f"  Header decoded:     {first_line!r}")


def show_first_dates(data: bytes, n=5):
    lines = data.split(b"\n")
    print(f"  First {n} data rows (raw bytes):")
    for line in lines[1 : n + 1]:
        if line.strip():
            date_field = line.split(b",")[0]
            print(f"    {date_field!r}  hex={date_field.hex()}")


def compare_dates(conc_data: bytes, unc_data: bytes, n=10):
    conc_lines = [l for l in conc_data.split(b"\n") if l.strip()][1:]
    unc_lines  = [l for l in unc_data.split(b"\n")  if l.strip()][1:]

    print(f"\n  Comparing first {n} date fields side-by-side:")
    mismatches = 0
    for i, (cl, ul) in enumerate(zip(conc_lines[:n], unc_lines[:n])):
        cd = cl.split(b",")[0]
        ud = ul.split(b",")[0]
        match = "✓" if cd == ud else "✗ MISMATCH"
        print(f"    row {i+1:>3}  conc={cd!r}  unc={ud!r}  {match}")
        if cd != ud:
            mismatches += 1

    if mismatches == 0:
        print("  → All compared date fields are byte-identical")
    else:
        print(f"  → {mismatches} mismatches found in first {n} rows")

    # also check total row counts
    print(f"\n  Total data rows:  conc={len(conc_lines)}  unc={len(unc_lines)}")
    if len(conc_lines) != len(unc_lines):
        print("  ✗ ROW COUNT MISMATCH — this will always cause EPA PMF to reject the pair")
    else:
        print("  ✓ Row counts match")


# ── Run diagnostics ──────────────────────────────────────────────────────────
print(SEP)
print("PMF File Diagnostic")
print(SEP)

for label, path in [("CONC", CONC_FILE), ("UNC", UNC_FILE)]:
    print(f"\n[{label}] {path}")
    print(f"  File size: {os.path.getsize(path):,} bytes")

    data = read_raw(path)

    bom = detect_bom(data)
    print(f"  BOM:       {bom}")

    crlf, lf, cr = detect_line_ending(data)
    print(f"  Line endings:  CRLF={crlf}  bare-LF={lf}  bare-CR={cr}")
    if crlf > 0 and lf == 0 and cr == 0:
        print("  → Windows CRLF  ✓")
    elif lf > 0 and crlf == 0:
        print("  → Unix LF only  ← may cause EPA PMF issues on Windows")
    else:
        print("  → Mixed line endings  ← likely problematic")

    print()
    show_header_bytes(data)
    print()
    show_first_dates(data)

print(f"\n{SEP}")
conc_data = read_raw(CONC_FILE)
unc_data  = read_raw(UNC_FILE)
compare_dates(conc_data, unc_data)

print(f"\n{SEP}")
print("Suggested next fix based on findings:")
print("  If line endings show 'bare-LF':")
print("    → Remove lineterminator from to_csv() (already done)")
print("       and ensure you are running on Windows, not WSL")
print("  If BOM shows 'None':")
print("    → Try  encoding='utf-8-sig'  in to_csv()")
print("  If date bytes contain unexpected characters (\\r, spaces, etc.):")
print("    → Strip the Date column after strftime()")
print(SEP)