import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import av
import pandas as pd
from tqdm import tqdm


def download(url: str, dest: Path, timeout_s: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "birds_hand download_xc_to_ogg/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        data = r.read()
    dest.write_bytes(data)


def transcode_to_ogg_vorbis(src: Path, dst: Path, sample_rate: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    inp = av.open(str(src), mode="r", options={"analyzeduration": "10M", "probesize": "10M"})
    try:
        astreams = [s for s in inp.streams if s.type == "audio" and s.codec]
        if not astreams:
            raise ValueError("no audio stream in file")
        a_in = astreams[0]
    except Exception:
        inp.close()
        raise

    out = av.open(str(dst), "w", format="ogg")
    a_out = out.add_stream("libvorbis", rate=sample_rate, layout="mono")

    resampler = av.audio.resampler.AudioResampler(
        format="fltp",
        layout="mono",
        rate=sample_rate,
    )

    try:
        for frame in inp.decode(a_in):
            for rframe in resampler.resample(frame):
                for packet in a_out.encode(rframe):
                    out.mux(packet)
        for packet in a_out.encode(None):
            out.mux(packet)
    finally:
        out.close()
        inp.close()


skip_existing = True
mapdf = pd.read_csv("data/class_counts.csv")
train = pd.read_csv("data/train.csv")
trfilenames = train["filename"].tolist()
# print(trfilenames[0])
# print(f'{len(trfilenames)=}')
# trfilenames = list(set(trfilenames))
# print(f'{len(trfilenames)=} after dedup')

path = "data/xc_parsed"
for datacsv in tqdm(os.listdir(path)):
    if datacsv == ".DS_Store":
        continue
    df = pd.read_csv(path + "/" + datacsv)
    # print('nan places: ', df['country'].isna().sum())
    df["country"] = df["country"].fillna("none")
    if len(df) > 500:
        new_df = df[df["country"].str.contains("brasil|brazil", case=False)]
        # print(f'{new_df.shape=}')
        dfh = df[~df["country"].str.contains("brasil|brazil", case=False)]
        # print(f'{dfh.shape=}')
        df = pd.concat([new_df, dfh.head(500 - len(new_df))])
        # print(f'{df.shape=}')
        # break
    # print(datacsv, len(df))

    # print(datacsv)
    prlabel = datacsv.split(".")[0]
    # prlabel = mapdf[mapdf['scientific_name'] == df.iloc[0]['scientific_name']].iloc[0]['primary_label']
    print("### ", f"{prlabel=}")

    print("before remove empty urls: ", len(df))
    df = df.fillna("")
    df = df[df["file_url"].str.startswith("http")]
    print("after remove empty urls: ", len(df))
    df = df.reset_index(drop=True)

    # df.to_csv(f'{prlabel}.csv', index=False)

    new = 0
    intrain = 0

    for i in range(len(df)):
        url = df.iloc[i]["file_url"]
        recid = str(df.iloc[i]["id"])
        fn = prlabel + "/" + "XC" + recid

        if fn + ".ogg" in trfilenames:
            print(f"[{i}]: {fn} already in train!!!")
            intrain += 1
            continue
        else:
            print(f"[{i}]: {fn} downloading...")
            new += 1

        outpath = Path("data/xc_downloaded")
        out_ogg = outpath / (fn + ".ogg")

        if skip_existing and out_ogg.is_file() and out_ogg.stat().st_size > 0:
            continue

        temp_dir = Path("tmp_downloads")
        temp_dir.mkdir(exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="dl_ogg_", dir=str(temp_dir)) as tmpd:
            tmp = Path(tmpd)
            suf = Path(urllib.parse.urlparse(url).path).suffix
            if not suf or len(suf) > 6:
                suf = ".bin"
            raw = tmp / f"in_{i}{suf}"
            try:
                timeout = 120
                download(url, raw, timeout)
            except (OSError, urllib.error.URLError, TimeoutError) as e:
                print(f"[{i}] fail download {fn}: {e}", file=sys.stderr)
                continue
            try:
                sample_rate = 32000
                transcode_to_ogg_vorbis(raw, out_ogg, sample_rate)
            except (OSError, ValueError, RuntimeError, av.FFmpegError) as e:
                print(f"[{i}] fail encode {fn}: {e}", file=sys.stderr)
                continue
            import time

            time.sleep(0.3)
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()
    print(f"{new=}, {intrain=}")
