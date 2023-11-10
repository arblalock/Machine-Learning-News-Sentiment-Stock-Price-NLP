import os
import csv
from newsapi import NewsApiClient
import requests
import json
import pandas as pd

newsapi = NewsApiClient(api_key=os.environ.get("NEWS_KEY"))
source_save_path = "./data/news/"
START_DATE = "2020-08-18"
END_DATE = "2020-09-16"
START_TIME = "05:00:00"  # 9am EST (market open)
END_TIME = "16:00:00"  # Noon EST
SOURCES = "bloomberg,the-wall-street-journal,cnbc"

source_save_name = (
    START_DATE + "_" + END_DATE + "_T" + START_TIME[:2] + "-" + END_TIME[:2] + ".csv"
)
date_array = pd.bdate_range(start=START_DATE, end=END_DATE)
all_articles = []

for d in date_array:
    start_time = d.strftime("%Y-%m-%d") + "T" + START_TIME
    end_time = d.strftime("%Y-%m-%d") + "T" + END_TIME
    all_articles_req = newsapi.get_everything(
        sources=SOURCES,
        from_param=start_time,
        to=end_time,
        language="en",
        sort_by="publishedAt",
        page_size=100,
    )

    all_articles = all_articles + all_articles_req["articles"]

all_articles_df = pd.DataFrame(all_articles)
all_articles_df = pd.concat(
    [
        all_articles_df.drop(["source"], axis=1),
        all_articles_df["source"].apply(pd.Series),
    ],
    axis=1,
)
all_articles_df.to_csv(source_save_path + source_save_name)


with open(source_save_path + source_save_name, "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    for data in all_articles:
        writer.writerow(data)

with open(source_save_path + source_save_name, "w") as outfile:
    json.dump(all_articles, outfile)
