"""
Builds data/activity_hourly.csv (% of each hour spent in a vehicle / sitting still) from the original activity_recognition.csv (386 MB).

You do NOT need to run this: activity_hourly.csv is already in the data folder.
It is only here to show how that file was made. To run it, download
activity_recognition.csv from https://doi.org/10.5281/zenodo.10975463
and put it in this folder.
"""
from pathlib import Path
import pandas as pd

here = Path(__file__).parent

activity = pd.read_csv(here / "activity_recognition.csv", usecols=["participant", "timestamp", "label"])
activity = activity[activity["label"] != "TILTING"]          # tilting = phone moved, not the person

activity["hour"] = pd.to_datetime(activity["timestamp"], format="ISO8601").dt.floor("h")
activity["pct_in_vehicle"] = activity["label"] == "IN_VEHICLE"
activity["pct_still"] = activity["label"] == "STILL"

# % of the records in each hour that say IN_VEHICLE / STILL
hourly = activity.groupby(["participant", "hour"])[["pct_in_vehicle", "pct_still"]].mean() * 100
hourly = hourly.round(2).reset_index()

hourly.to_csv(here.parent / "data" / "activity_hourly.csv", index=False)
print("Saved", len(hourly), "rows")
