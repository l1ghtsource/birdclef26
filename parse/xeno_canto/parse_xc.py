import os
import time

import pandas as pd
import requests
from tqdm import tqdm

API_KEY = os.getenv("XENO_CANTO_API_KEY")
BASE_URL = "https://xeno-canto.org/api/3/recordings"
OUTPUT_DIR = "data/xc_parsed"

df = pd.read_csv("data/class_counts.csv")
# df = df[df['xc_share'] <= 0].reset_index(drop=True)
# df = df[df['xc_share'] > 0].reset_index(drop=True)
df = df.sort_values(by="count")

os.makedirs(OUTPUT_DIR, exist_ok=True)

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; XenoCantoCollector/1.0; +mailto:your@email.com)"})
retries = requests.adapters.Retry(
    total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"]
)
session.mount("https://", requests.adapters.HTTPAdapter(max_retries=retries))

for idx in tqdm(range(len(df)), desc="Обработка видов"):
    # if idx == 1:
    #     break
    scientific_name = df.loc[idx, "scientific_name"]
    primary_label = df.loc[idx, "primary_label"]
    outcsv = f"{OUTPUT_DIR}/{primary_label}.csv"

    allrows = []
    pg = 1
    while True:
        q = f'sp:"{scientific_name}"'
        params = {"query": q, "page": pg, "key": API_KEY}
        try:
            r = session.get(BASE_URL, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except requests.exceptions.HTTPError as e:
            print(f"{scientific_name}: {e}")
            break
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"{scientific_name}: {e}")
            break

        recs = data.get("recordings", [])
        nrecs = int(data.get("numRecordings", 0))
        npages = int(data.get("numPages", 1))

        if nrecs == 0:
            print(f"\n{scientific_name}: no records")
            break

        for rec in recs:
            row = {
                "id": rec.get("id"),
                "genus": rec.get("gen"),
                "specific_epithet": rec.get("sp"),
                "subspecies": rec.get("ssp"),
                "group": rec.get("grp"),
                "english_name": rec.get("en"),
                "recordist": rec.get("rec"),
                "country": rec.get("cnt"),
                "locality": rec.get("loc"),
                "latitude": rec.get("lat"),
                "longitude": rec.get("lng"),
                "altitude": rec.get("alt"),
                "type": rec.get("type"),
                "sex": rec.get("sex"),
                "stage": rec.get("stage"),
                "method": rec.get("method"),
                "url": rec.get("url"),
                "file_url": rec.get("file"),
                "file_name": rec.get("file-name"),
                "license": rec.get("lic"),
                "quality": rec.get("q"),
                "length": rec.get("length"),
                "time": rec.get("time"),
                "date": rec.get("date"),
                "uploaded": rec.get("uploaded"),
                "also": "; ".join(rec.get("also", [])) if rec.get("also") else "",
                "remarks": rec.get("rmk"),
                "animal_seen": rec.get("animal-seen"),
                "playback_used": rec.get("playback-used"),
                "temperature": rec.get("temp"),
                "recorder": rec.get("dvc"),
                "microphone": rec.get("mic"),
                "sample_rate": rec.get("smp"),
                "annotation_set": rec.get("annotation-set"),
            }
            allrows.append(row)

        if pg < npages:
            pg += 1
            time.sleep(0.3)
        else:
            break

    if allrows:
        outdf = pd.DataFrame(allrows)
        outdf.to_csv(outcsv, index=False, encoding="utf-8")
        print(f"{scientific_name}: saved {len(outdf)} -> {outcsv}")
    else:
        print(f"{scientific_name}: 0 recs")

    time.sleep(0.6)
