import xml.etree.ElementTree as ET
import requests
import pandas as pd

CL_BASE_URL = "https://raw.githubusercontent.com/acl-org/acl-anthology/refs/heads/master/data/xml/{year_conference}.xml"
CL_CONFERENCES = ["emnlp"]
YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
VOLUMES = {
    2014: ["1"],
    2015: ["1"],
    2016: ["1"],
    2017: ["1"],
    2018: ["1"],
    2019: ["1"],
    2020: ["main"],
    2021: ["main"],
    2022: ["main"],
    2023: ["main"],
    2024: ["main"]
}

def get_text(elem):
    return "".join(elem.itertext()).strip() if elem is not None else ""

def parse_cl_xml(conference,year):
    if year < 2020:
        year_conference = "D" + str(year)[2:]
    else:
        year_conference = str(year) + "." + str(conference)
    print(year_conference)

    xml_url = CL_BASE_URL.format(year_conference=year_conference)
    try:
        response = requests.get(xml_url)
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except requests.exceptions.RequestException as e:
        print(f"[WARNING] Failed to fetch {xml_url}: {e}")
        return None # Return empty DataFrame to be safely concatenated

    # Get volumes
    volume_list = []
    volume_index = -1

    for volume in root.findall(".//volume"):
        volume_id = volume.get("id")
        volume_list.append(volume_id)
        
    # Get papers
    papers = []
    for paper in root.findall(".//paper"):
        paper_id = paper.get("id")
        
        if paper_id == '1':
            volume_index += 1
        
        # Skip non-long or short paper
        if volume_list[volume_index] in VOLUMES[year]:
            title = get_text(paper.find("title"))
            abstract = get_text(paper.find("abstract"))
            url = get_text(paper.find("url"))
            doi = get_text(paper.find("doi"))
            bibkey = get_text(paper.find("bibkey"))
            award = get_text(paper.find("award"))  # may be empty

            if "Invited Talk:" in title:
                print("skip invited talk")
                continue

            paper_data = {
                "paper_id": f"{conference}-{year}-{volume_list[volume_index]}-{paper_id}",
                "title": title,
                "abstract": abstract,
                "conference": conference,
                "year": year,
                "volume": volume_list[volume_index],
                "url": url,
                "doi": doi,
                "bibkey": bibkey,
            }

            if award:
                paper_data["award"] = award

            papers.append(paper_data)

    return pd.DataFrame(papers)

list_dfs = []
for conference in CL_CONFERENCES:
    for year in YEARS:
        df = parse_cl_xml(conference, year)
        if df is not None and len(df) != 0:
            list_dfs.append(df)

combined_df = pd.concat(list_dfs, ignore_index=True)
combined_df.to_csv("../../data/csv/emnlp_conference.csv", index=False)