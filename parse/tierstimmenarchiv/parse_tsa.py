import os
import time
from urllib.parse import parse_qs, quote, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

INPUT_CSV = "data/class_counts.csv"
OUTPUT_DIR = "data/tsa_parsed"
DELAY = 1.0

BASE_URL = "https://suche.tierstimmenarchiv.de/search/query.html"
FIXED_PARAMS = {
    "from_year": "1900",
    "to_year": "2026",
    "show_not_downloadable": "false",
    "language": "english",
    "requested": "species,locality,duration,recording_date,description,sound_type,collection,usage_permission",
}
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; BioacousticsBot/1.0)"}


def build_url(species_name):
    params = FIXED_PARAMS.copy()
    params["species"] = species_name
    params["results_per_page"] = 500
    query = "&".join(f"{k}={quote(str(v))}" for k, v in params.items())
    return f"{BASE_URL}?{query}"


def parse_results(html):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="results_table")
    if not table:
        return [], 0

    rows = table.find("tbody").find_all("tr")
    data = []
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 3:
            continue

        audio_tag = row.find("audio")
        preview_url = None
        if audio_tag and audio_tag.get("src"):
            src = audio_tag["src"]
            preview_url = "https://suche.tierstimmenarchiv.de" + src

        unique_id = None
        details_link = row.find("a", href=lambda x: x and "showdetails.html" in x)
        if details_link:
            qs = parse_qs(urlparse(details_link["href"]).query)
            unique_id = qs.get("unique_identifier", [None])[0]
        if not unique_id:
            checkbox = row.find("input", {"type": "checkbox", "name": "basket_check"})
            if checkbox:
                unique_id = checkbox.get("value")

        species_cell = row.find("td", class_="species")
        species_name = species_cell.get_text(strip=True) if species_cell else ""

        def get_td_text(idx):
            if idx < len(cols):
                return cols[idx].get_text(separator=" ", strip=True)
            return ""

        locality = get_td_text(2)
        duration = get_td_text(3)
        recording_date = get_td_text(4)
        description = get_td_text(5)
        sound_type = get_td_text(6)
        collection = get_td_text(7)
        usage_permission = get_td_text(8)

        data.append(
            {
                "unique_identifier": unique_id,
                "scientific_name": species_name,
                "locality": locality,
                "duration": duration,
                "recording_date": recording_date,
                "description": description,
                "sound_type": sound_type,
                "collection": collection,
                "usage_permission": usage_permission,
                "preview_audio_url": preview_url,
            }
        )

    return data


def scrape_species(species_name):
    url = build_url(species_name)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"error | {species_name} | {e}")
        return []

    records = parse_results(resp.text)
    return records


os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(INPUT_CSV).sort_values("count")

for idx in tqdm(range(len(df))):
    name = df.loc[idx, "scientific_name"]
    label = df.loc[idx, "primary_label"]
    out_csv = os.path.join(OUTPUT_DIR, f"{label}.csv")

    print(f"### {name} ({label=})")
    records = scrape_species(name)
    print(f"{len(records)=}")

    if not records:
        print("-> skip")
        continue

    out_df = pd.DataFrame(records)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"saved {out_df.shape[0]} rown -> {out_csv}")

    time.sleep(DELAY)
