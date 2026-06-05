import os
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import av
import pandas as pd
from redownload_corrupted import download, parse_label_basename, transcode_to_ogg_vorbis
from tqdm import tqdm

skip_existing = True
mapdf = pd.read_csv("data/inat_parsed/class_counts.csv")
train = pd.read_csv("data/inat_parsed/train.csv")
trfilenames = train["filename"].tolist()
# print(trfilenames[0])
# print(f'{len(trfilenames)=}')
# trfilenames = list(set(trfilenames))
# print(f'{len(trfilenames)=} after dedup')

path = "data/inat_parsed"
for datacsv in tqdm(os.listdir(path)):
    if datacsv == ".DS_Store":
        continue
    df = pd.read_csv(path + "/" + datacsv)
    # print('nan places: ', df['place_guess'].isna().sum())
    df["place_guess"] = df["place_guess"].fillna("none")
    if len(df) > 500:
        new_df = df[df["place_guess"].str.contains("brasil|brazil|pantanal", case=False)]
        # print(f'{new_df.shape=}')
        dfh = df[~df["place_guess"].str.contains("brasil|brazil|pantanal", case=False)]
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
    df = df[df["sound_url"].str.startswith("http")]
    print("after remove empty urls: ", len(df))
    df = df.reset_index(drop=True)

    # df.to_csv(f'{prlabel}.csv', index=False)

    intrain = 0
    new = 0

    for i in range(len(df)):
        # if i == 1:
        #     break

        urlrow = df.iloc[i]["sound_url"]
        urls = [u.strip() for u in urlrow.split(",") if u.strip()]

        for url in urls:
            fn = prlabel + "/" + "iNat" + str(url.split("sounds/")[-1].split(".")[0])

            if fn + ".ogg" in trfilenames:
                print(f"[{i}]: {fn} already in train!!!")
                intrain += 1
                continue
            else:
                print(f"[{i}]: {fn} downloading...")
                new += 1

            label, base = parse_label_basename(fn)
            if not base.lower().endswith(".ogg"):
                base = f"{Path(base).stem}.ogg"

            outpath = Path("data/inat_downloaded")
            out_ogg = outpath / label / base

            if skip_existing and out_ogg.is_file() and out_ogg.stat().st_size > 0:
                continue

            with tempfile.TemporaryDirectory(prefix="dl_ogg_") as tmpd:
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
    print(f"{new=}, {intrain=}")
