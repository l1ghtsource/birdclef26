import pandas as pd
from pyinaturalist import get_observations
from tqdm import tqdm

df = pd.read_csv("data/class_counts.csv")
df = df.sort_values(by="count")

for idx in tqdm(range(len(df))):
    name = df.loc[idx, "scientific_name"]
    outcsv = df.loc[idx, "primary_label"] + ".csv"

    observations = get_observations(taxon_name=name, sounds=True, page="all", per_page=100)
    results = observations["results"]
    total = observations["total_results"]

    print(f"{total=}, loaded: {len(results)}")

    if total == 0:
        print(" -> skip!!!")
        continue

    rows = []
    for obs in results:
        us = obs.get("user") or {}
        tax = obs.get("taxon") or {}
        soundset = set()
        for s in obs.get("observation_sounds") or []:
            sobj = s.get("sound") if "sound" in s else s
            if sobj and sobj.get("file_url"):
                soundset.add(sobj["file_url"])
        for s in obs.get("sounds") or []:
            if s.get("file_url"):
                soundset.add(s["file_url"])
        surl = ", ".join(sorted(soundset))
        loc = obs.get("location")
        lat = loc[0] if loc and len(loc) == 2 else None
        lon = loc[1] if loc and len(loc) == 2 else None
        row = {
            "id": obs.get("id"),
            "observed_on": obs.get("observed_on"),
            "user_login": us.get("login"),
            "user_name": us.get("name"),
            "created_at": obs.get("created_at"),
            "updated_at": obs.get("updated_at"),
            "quality_grade": obs.get("quality_grade"),
            "license": obs.get("license_code"),
            "url": obs.get("uri"),
            "sound_url": surl,
            "place_guess": obs.get("place_guess"),
            "latitude": lat,
            "longitude": lon,
            "scientific_name": tax.get("name"),
            "common_name": tax.get("preferred_common_name"),
            "iconic_taxon_name": tax.get("iconic_taxon_name"),
            "taxon_id": tax.get("id"),
        }
        rows.append(row)

    outdf = pd.DataFrame(rows)
    outdf.to_csv(f"data/inat_parsed/{outcsv}", index=False, encoding="utf-8")
    print(f"{outdf.shape=}")
