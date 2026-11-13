"""Upload a local Zarr store to the GRID4EARTH bucket.

Usage:
    python scripts/upload_s3.py LOCAL_STORE S3_URL [--profile grid4earth]
        [--endpoint https://s3.gra.io.cloud.ovh.net]

Run it from a machine with internet access (on Datarmor: a login node, the
compute nodes have none). Trials go under
``s3://grid4earth/public/trials/<github-id>-<date>-<name>/`` and are then
readable at ``https://data.grid4earth.eu/trials/...``.
"""
import argparse
import os
import time

import s3fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("local")
    ap.add_argument("url")
    ap.add_argument("--profile", default="grid4earth")
    ap.add_argument("--endpoint", default="https://s3.gra.io.cloud.ovh.net")
    a = ap.parse_args()
    if not a.url.startswith("s3://grid4earth/public/trials/"):
        raise SystemExit("refusing to write outside s3://grid4earth/public/trials/")
    fs = s3fs.S3FileSystem(profile=a.profile, client_kwargs={"endpoint_url": a.endpoint})
    files = [os.path.join(r, f) for r, _, fs_ in os.walk(a.local) for f in fs_]
    size = sum(os.path.getsize(f) for f in files)
    print(f"{len(files)} files, {size / 1e9:.2f} GB", flush=True)
    t0 = time.time()
    if fs.exists(a.url):
        fs.rm(a.url, recursive=True)
    fs.put(a.local.rstrip("/") + "/", a.url.rstrip("/") + "/", recursive=True, batch_size=64)
    print(f"uploaded in {time.time() - t0:.0f} s; {len(fs.find(a.url))} objects")


if __name__ == "__main__":
    main()
