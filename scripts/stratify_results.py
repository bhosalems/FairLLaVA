import json
import pandas as pd
import numpy as np
import argparse


def _derive_race_major_from_raw_race(raw_race: pd.Series) -> pd.Series:
    """Map MIMIC admissions-style `race` strings into coarse `race_major` buckets."""
    race = raw_race.astype(str)
    conds = [
        race.str.contains(r'^WHITE', case=False, na=False),
        race.str.contains(r'^BLACK', case=False, na=False),
        race.str.contains(r'^ASIAN', case=False, na=False),
        race.str.contains(r'HISPANIC', case=False, na=False),
        race.str.contains(r'AMERICAN INDIAN', case=False, na=False),
        race.str.contains(r'HAWAIIAN|PACIFIC ISLANDER', case=False, na=False),
        race.str.contains(r'^OTHER$', case=False, na=False),
        race.str.contains(r'DECLINED|UNABLE TO OBTAIN', case=False, na=False),
        race.str.contains(r'^UNKNOWN$', case=False, na=False),
    ]
    labels = [
        'White',
        'Black or African American',
        'Asian',
        'Hispanic or Latino',
        'American Indian or Alaska Native',
        'Native Hawaiian or Pacific Islander',
        'Other',
        'Declined / Unable to obtain',
        'Unknown',
    ]
    return pd.Series(np.select(conds, labels, default='Other'), index=raw_race.index)

def mimic_cxr_process(records):
    df = pd.DataFrame.from_records(records)
    # "id" (this id is patient ID and Study ID) We need patient ID only for fairness analysis
    df['subject_id'] = df['id'].apply(lambda x: int(x.split("_")[0]))
    patients_data = pd.read_csv("/a2il/data/mbhosale/MrFair/physionet.org/mimc-cxr-jpeg/patients.csv")
    merged_df = df.merge(patients_data, on='subject_id', how='left', validate='m:1')

    # Pandas will suffix overlapping column names (e.g., gender_x / gender_y). Prefer patients.csv gender.
    if 'gender' not in merged_df.columns:
        if 'gender_y' in merged_df.columns:
            merged_df['gender'] = merged_df['gender_y']
        elif 'gender_x' in merged_df.columns:
            merged_df['gender'] = merged_df['gender_x']
        elif 'SexEncoded' in merged_df.columns:
            se = pd.to_numeric(merged_df['SexEncoded'], errors='coerce')
            merged_df['gender'] = np.where(se == 0, 'M', np.where(se == 1, 'F', np.nan))

    # Normalize gender to M/F when possible
    if 'gender' in merged_df.columns:
        g = merged_df['gender'].astype(str).str.strip().str.lower()
        merged_df.loc[g.isin(['m', 'male', '0']), 'gender'] = 'M'
        merged_df.loc[g.isin(['f', 'female', '1']), 'gender'] = 'F'

    # We do need age and gender for fairness analysis, so we will just drop the rows missing
    merged_df = merged_df.dropna(subset=['anchor_age', 'gender'])
    
    bins   = [0, 44, 65, np.inf]
    labels = ['0–44', '44–65', '65+']
    merged_df = merged_df.copy()

    merged_df.loc[:, 'age_group'] = pd.cut(
        merged_df['anchor_age'],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True
    )
    return merged_df

def padchest_process(records):
    preds = pd.DataFrame.from_records(records)
    dem_df = pd.read_json("/a2il/data/mbhosale/MrFair/Padchest/test_findings.json")
    preds['ImageID'] = preds['id']
    # print(f"Preds before merge: {len(preds)}")
    # print(f"Dem_df rows: {len(dem_df)}")
    merged_df = preds.merge(dem_df, on='ImageID', how='left', validate='m:1')
    # print(f"After merge: {len(merged_df)}")
    # print(f"Non-null Age: {merged_df['Age'].notna().sum()}")
    # print(f"Non-null gender: {merged_df['gender'].notna().sum()}")
    merged_df = merged_df.dropna(subset=['Age', 'gender'])
    # print(f"After dropna: {len(merged_df)}")
    bins   = [0, 44, 65, np.inf]
    labels = ['0–44', '44–65', '65+']
    merged_df = merged_df.copy()
    merged_df.loc[:, 'age_group'] = pd.cut(
        merged_df['Age'],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True
    )
    return merged_df


def ham10000_process(records):
    """Process HAM10000 prediction records.

    Assumes merged_preds.jsonl records may already include demographics copied from the query JSON:
      - gender: "male"/"female" (preferred) or "M"/"F"
      - SexEncoded: 0/1 (optional)
      - anchor_age_group: 0/1/2 (preferred) or Age (optional)

    Produces columns:
      - gender: "M" or "F"
      - age_group: one of {"0–44", "44–65", "65+"}
    """
    df = pd.DataFrame.from_records(records)
    df = df.copy()

    # Normalize gender
    if "gender" in df.columns:
        g = df["gender"].astype(str).str.strip().str.lower()
        df.loc[g.isin(["m", "male", "0"]) , "gender"] = "M"
        df.loc[g.isin(["f", "female", "1"]) , "gender"] = "F"
    elif "SexEncoded" in df.columns:
        se = pd.to_numeric(df["SexEncoded"], errors="coerce")
        df["gender"] = np.where(se == 0, "M", np.where(se == 1, "F", np.nan))
    else:
        df["gender"] = np.nan

    # Compute age_group
    if "anchor_age_group" in df.columns:
        ag = pd.to_numeric(df["anchor_age_group"], errors="coerce")
        # Match your bin names used elsewhere
        mapping = {0: "0–44", 1: "44–65", 2: "65+"}
        df["age_group"] = ag.map(mapping)
    elif "Age" in df.columns:
        age = pd.to_numeric(df["Age"], errors="coerce")
        bins = [0, 44, 65, np.inf]
        labels = ['0–44', '44–65', '65+']
        df.loc[:, 'age_group'] = pd.cut(
            age,
            bins=bins,
            labels=labels,
            right=True,
            include_lowest=True
        )
    else:
        df["age_group"] = np.nan

    # Keep only rows with demographics needed for stratification
    df = df.dropna(subset=["gender", "age_group"], how="any")
    return df

def main(pred_dir, dataset):
    records = []
    with open(pred_dir+"/merged_preds.jsonl", "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)      # urlsafe json parser, keeps strings intact
            records.append(rec)

    ds = dataset.lower()
    if ds == "mimic-cxr":
        merged_df = mimic_cxr_process(records)
    elif ds == "padchest":
        merged_df = padchest_process(records)
    elif ds in {"ham10000", "ham"}:
        merged_df = ham10000_process(records)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
            
    # Lets separate out the results according to demographic groups
    keys = ['id','query','reference','prediction']
    if 'greenscore' in merged_df.columns:
        keys.append('greenscore')
    # Write subgroup files; skip empty splits to avoid downstream empty-eval crashes.
    def write_if_nonempty(df_slice, out_name: str):
        if df_slice is None or len(df_slice) == 0:
            return
        df_slice[keys].to_json(pred_dir + '/' + out_name, orient='records', lines=True)

    write_if_nonempty(merged_df[merged_df['gender'] == 'M'], 'gender_m.jsonl')
    write_if_nonempty(merged_df[merged_df['gender'] == 'F'], 'gender_f.jsonl')
    write_if_nonempty(merged_df[merged_df['age_group'] == '0–44'], 'age_0_44.jsonl')
    write_if_nonempty(merged_df[merged_df['age_group'] == '44–65'], 'age_44_65.jsonl')
    write_if_nonempty(merged_df[merged_df['age_group'] == '65+'], 'age_65_inf.jsonl')

    if ds == 'mimic-cxr':
        admissions = pd.read_csv("/a2il/data/mbhosale/MrFair/physionet.org/mimc-cxr-jpeg/admissions.csv")

        # admissions.csv contains raw `race`, not `race_major`
        if 'race_major' not in admissions.columns:
            admissions['race_major'] = _derive_race_major_from_raw_race(admissions['race'])
        admissions = (admissions.drop_duplicates(subset=['subject_id'], keep='first')[['subject_id','insurance','race_major','marital_status']])
        merged_df = merged_df.merge(admissions, on='subject_id', how='left', validate='m:1')

        # Backward/forward compat for race columns:
        # - Older runs may already have `race_major` in merged_preds.jsonl.
        # - Newer merges may introduce raw `race`.
        # - Pandas may suffix overlapping names on merge (race_major_x/y).
        if 'race_major' not in merged_df.columns:
            if 'race_major_x' in merged_df.columns and 'race_major_y' in merged_df.columns:
                merged_df['race_major'] = merged_df['race_major_x'].combine_first(merged_df['race_major_y'])
            elif 'race_major_x' in merged_df.columns:
                merged_df['race_major'] = merged_df['race_major_x']
            elif 'race_major_y' in merged_df.columns:
                merged_df['race_major'] = merged_df['race_major_y']
            elif 'race' in merged_df.columns:
                merged_df['race_major'] = _derive_race_major_from_raw_race(merged_df['race'])

        if 'race_major' in merged_df.columns:
            write_if_nonempty(merged_df[merged_df['race_major'] == 'White'], 'race_white.jsonl')
            write_if_nonempty(merged_df[merged_df['race_major'] == 'Black or African American'], 'race_black.jsonl')
            write_if_nonempty(merged_df[merged_df['race_major'] == 'Asian'], 'race_asian.jsonl')
            write_if_nonempty(merged_df[merged_df['race_major'] == 'Other'], 'race_other.jsonl')
            write_if_nonempty(merged_df[merged_df['race_major'] == 'Hispanic or Latino'], 'race_hispanic.jsonl')

    merged_df.to_json(pred_dir+'/merged_demographics.jsonl', orient='records', lines=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Process demographic data")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory for prediction file")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name: mimic-cxr, padchest, ham10000")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.pred_dir, args.dataset)
