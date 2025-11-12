# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 21:07:41 2025

@author:Kwangho Baek baek0040@umn.edu; dptm22203@gmail.com
"""

import re
import pandas as pd

df=pd.read_csv('filteredSwissMetro.csv')

# --- 1) Identify columns ---
# Generic (duplicate) columns: no underscore (except CHOICE handled separately)
generic_cols = [c for c in df.columns if "_" not in c and c != "CHOICE"]

# Alternative-specific columns: split on first underscore -> (ALT, ATTR)
alt_cols = [c for c in df.columns if "_" in c]
pairs = [re.match(r"^([^_]+)_(.+)$", c).groups() for c in alt_cols]
alternatives = sorted(set(a for a,_ in pairs))
attrs = sorted(set(b for _,b in pairs))   # all attribute names across alts

# --- 2) Build long table by stacking each alternative ---
long_frames = []
for alt in alternatives:
    # columns of this alternative
    this_alt_cols = [c for c in alt_cols if c.startswith(f"{alt}_")]
    # rename "{ALT}_{ATTR}" -> "{ATTR}"
    renamed = {c: c.split("_", 1)[1] for c in this_alt_cols}
    tmp = pd.concat(
        [
            df[generic_cols].reset_index(drop=True),
            df[this_alt_cols].rename(columns=renamed).reset_index(drop=True),
        ],
        axis=1,
    )
    tmp["alt"] = alt
    long_frames.append(tmp)

long_df = pd.concat(long_frames, axis=0, ignore_index=True)

# Keep a tidy column order (generic, then attributes, then 'alt'/'choice')
ordered_cols = generic_cols + [a for a in attrs if a in long_df.columns] + ["alt"]
long_df = long_df[ordered_cols]

# --- 3) Add a per-row choice indicator ---
# Map numeric CHOICE code -> alternative name.
# (This matches the classic Swissmetro-style coding: 1=TRAIN, 2=SM, 3=CAR.)
choice_map = {1: "TRAIN", 2: "SM", 3: "CAR"}
chosen_alt = df["CHOICE"].map(choice_map)

# Repeat chosen_alt to match the 3x stacking (exactly once per alternative)
# We stacked in the order given by 'alternatives', three blocks of length len(df).
repeated_chosen = pd.concat([chosen_alt]*len(alternatives), ignore_index=True)

long_df["match"] = (long_df["alt"] == repeated_chosen).astype(int)


# --- 5) Some Hardcoding
long_df.loc[(long_df["alt"] != "CAR") & (pd.isna(long_df["HE"])), "HE"] = 1440 #1 day
long_df.loc[long_df.SEATS==0, "SEATS"] = 2 # convert non-airline seat (what is this?) to 2 and keep 0 for nans
long_df = long_df.fillna(0)
revert_map = {"TRAIN": 0, "SM": 1, "CAR": 2}
long_df["alt"] = long_df["alt"].map(revert_map)
long_df = long_df.sort_values(["chid", "alt"]).reset_index(drop=True)
long_df['CO']=long_df['CO']/100
long_df['HE']=long_df['HE']/100
long_df['TT']=long_df['TT']/100

# the below code will replace the pinned dfIn.csv for replication
if __name__=="__main__":
    0
    #long_df.to_csv('dfIn.csv',index=False)
