#!/usr/bin/env python3
"""Verify the immutable launch snapshot before executing GPU code."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-manifest',type=Path,required=True)
    args=parser.parse_args()
    record=json.loads(args.snapshot_manifest.read_text())
    root=args.snapshot_manifest.resolve().parent
    for relative,expected in record['files'].items():
        path=root/relative
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=expected:
            raise SystemExit(f'Snapshot hash mismatch: {relative}')
    print(f"Verified {len(record['files'])} snapshot files.",flush=True)


if __name__=='__main__':
    main()
