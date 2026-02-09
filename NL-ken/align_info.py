import os
import json

# import glob
import pandas as pd

transcript_text = "/large/datas/frederikus-h/lectrans-testset/25.11.transcript_translate/transcript_translate_annotated_by_sbj_all_text.json"
slide_text = "/large/datas/frederikus-h/lectrans-testset/25.11.slides_annot/slide_translation_annot_by_sbj_all_text.json"

subjects = {
    "anat-1": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/anat-1_-_LQ7IXyQSXBk_-_transcripts.xlsx",
        "video_id": "LQ7IXyQSXBk",
    },
    "anat-2": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/anat-2_-_IvK-UGOI5ZQ_-_transcripts.xlsx",
        "video_id": "IvK-UGOI5ZQ",
    },
    "bio-1": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/bio-1_-_pUH2_SGnm9g_-_transcripts.xlsx",
        "video_id": "pUH2_SGnm9g",
    },
    "bio-2": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/bio-2_-_dEh_Zk9d0sE_-_transcripts.xlsx",
        "video_id": "dEh_Zk9d0sE",
    },
    "bio-3": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/bio-3_-_u7AnKjGOEgI_-_transcripts.xlsx",
        "video_id": "u7AnKjGOEgI",
    },
    "bio-4": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/bio-4_-_gb0arNyiLCI_-_transcripts.xlsx",
        "video_id": "gb0arNyiLCI",
    },
    "dental": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/dental_-_uLz-Ur_Drk4_-_transcripts.xlsx",
        "video_id": "uLz-Ur_Drk4",
    },
    "intmd_microeconomics": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/intmd_microeconomics_-_rhOpABGewYA_-_transcripts.xlsx",
        "video_id": "rhOpABGewYA",
    },
    "intro_to_cs_programming_python": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/intro_to_cs_programming_python_-_tsMrUdynbQQ_-_transcripts.xlsx",
        "video_id": "tsMrUdynbQQ",
    },
    "intro_to_neural_computation": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/intro_to_neural_computation_-_K1pxJVdqlxw_-_transcripts.xlsx",
        "video_id": "K1pxJVdqlxw",
    },
    "intro_to_physics_20century": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/intro_to_physics_20century_-_Fce9xf_kjyc_-_transcripts.xlsx",
        "video_id": "Fce9xf_kjyc",
    },
    "ml-1": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/ml-1_-_L1TiP9P55-8_-_transcripts.xlsx",
        "video_id": "L1TiP9P55-8",
    },
    "psy-1": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/psy-1_-_j2-9yymHfeU_-_transcripts.xlsx",
        "video_id": "j2-9yymHfeU",
    },
    "psy-2": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/psy-2_-_uNrB2uG_9gs_-_transcripts.xlsx",
        "video_id": "uNrB2uG_9gs",
    },
    "speaking": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/speaking_-_KL_I9eE2aGk_-_transcripts.xlsx",
        "video_id": "KL_I9eE2aGk",
    },
    "theory_of_computation": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/theory_of_computation_-_aVv9WXwW95w_-_transcripts.xlsx",
        "video_id": "aVv9WXwW95w",
    },
    "urban_energy_systems_and_policy": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/urban_energy_systems_and_policy_-_3WXWESx6jWY_-_transcripts.xlsx",
        "video_id": "3WXWESx6jWY",
    },
}

slide_dir = "/large/datas/frederikus-h/lectrans-testset/25.11.slides_annot/slide_translation_251214"
image_timestamp = (
    "/large/datas/frederikus-h/lectrans-testset/25.11.slides_annot/annot_by_sbj"
)
# /large/datas/frederikus-h/lectrans-testset/25.11.slides_annot/annot_by_sbj/anat-1/LQ7IXyQSXBk_-_scene_id_to_timestamp.csv


def prepare_timestamp(timestamp):
    if pd.isna(timestamp):
        return None

    timestamp = str(timestamp)
    timestamp = timestamp.split(" -->")[0]

    if timestamp == "":
        return None

    parts = timestamp.split(":")

    try:
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)

        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)

        elif len(parts) == 1:
            return float(parts[0])

    except ValueError:
        return None

    return None


def align_info(subject_name, paths, output_dir):
    print(f"Processing: {subject_name}")
    image_dir = os.path.join(
        slide_dir, subject_name, subjects[subject_name]["video_id"]
    )

    scene_img_csv_path = os.path.join(
        image_timestamp,
        subject_name,
        f"{subjects[subject_name]['video_id']}_-_scene_id_to_timestamp.csv",
    )
    scene_df = pd.read_csv(scene_img_csv_path)
    # get timestamps from scene_timestamp
    scene_df["img_timestamp"] = (
        scene_df["scene_timestamp"]
        .str.split("-", expand=True)[0]
        .apply(prepare_timestamp)
    )

    # save the json path
    scene_df["json_name"] = scene_df["scene_id"]

    # prepare img_name (.jpg)
    scene_df["img_name"] = scene_df["scene_id"].str.replace(
        ".json", ".jpg", regex=False
    )

    transcript_df = pd.read_excel(paths["transcript_xlsx"])
    transcript_df["Start_sec"] = transcript_df["timestamp"].apply(prepare_timestamp)
    transcript_df["End_sec"] = transcript_df["Start_sec"].shift(-1)
    # print(transcript_df[["timestamp", "Start_sec", "End_sec"]].head(10))

    # assume that the last row ends 30 seconds after its start time
    transcript_df.loc[transcript_df.index[-1], "End_sec"] = (
        transcript_df.loc[transcript_df.index[-1], "Start_sec"] + 30.0
    )

    results = []

    # json
    video_key = f"{subject_name}/{subjects[subject_name]['video_id']}"
    results_json = {video_key: {}}

    for i, row in transcript_df.iterrows():
        # from transcript file (xlsx)
        start_sec = row["Start_sec"]
        end_sec = row["End_sec"]

        """matched_val = scene_df[
            (scene_df["img_timestamp"] >= start_sec) &
            (scene_df["img_timestamp"] < end_sec)
        ]"""

        # the slide image at start_sec
        start_slide = (
            scene_df[scene_df["img_timestamp"] <= start_sec]
            .sort_values("img_timestamp")
            .tail(1)
        )

        remaining_slides = scene_df[
            (scene_df["img_timestamp"] > start_sec)
            & (scene_df["img_timestamp"] < end_sec)
        ]

        matched_val = pd.concat([start_slide, remaining_slides]).drop_duplicates()

        # add image and json for the first segment
        if i == 0 and matched_val.empty:
            matched_val = scene_df.iloc[[0]]

        matched_images = matched_val["img_name"].tolist()
        matched_jsons = matched_val["json_name"].tolist()

        image_paths = [
            os.path.join(image_dir, image)
            for image in matched_images
            if os.path.exists(os.path.join(image_dir, image))
        ]
        json_paths = [
            os.path.join(image_dir, json)
            for json in matched_jsons
            if os.path.exists(os.path.join(image_dir, json))
        ]

        """results.append({
            "subject": subject_name,
            "start_sec": row["Start_sec"],
            "end_sec": row["End_sec"],
            "transcript": row.get("transcript", ""),
            "ja": row.get("ja", ""),
            "zh": row.get("zh", ""),
            "th": row.get("th", ""),
            "id": row.get("id", ""),
            "es": row.get("es", ""),
            "fr": row.get("fr", ""),
            "de": row.get("de", ""),
            "images": ", ".join(sorted(image_paths)),
            "ocrs": ", ".join(sorted(set(json_paths)))
        })"""

        # json
        img_timestamp = str(row["timestamp"])
        img_timestamp_key = img_timestamp.split(" -->")[0]
        results_json[video_key][img_timestamp_key] = {
            "images": sorted(image_paths),
            "ocrs": sorted(set(json_paths)),
            "start_sec": row["Start_sec"],
            "end_sec": row["End_sec"],
            "en": row.get("transcript", ""),
            "ja": row.get("ja", ""),
            "zh": row.get("zh", ""),
            "th": row.get("th", ""),
            "id": row.get("id", ""),
            "es": row.get("es", ""),
            "fr": row.get("fr", ""),
            "de": row.get("de", ""),
        }

    # result_df = pd.DataFrame(results)

    """# if images = NaN, assume that the slide displayed on the screen has not changed
    # -> use the slide images from the previous transcript segment
    result_df["images"] = result_df["images"].replace("", pd.NA)
    result_df["ocrs"] = result_df["ocrs"].replace("", pd.NA)"""

    """# keep only the last path
    for column in ["images", "ocrs"]:
        last_val = None
        new_vals = []
        for val in result_df[column]:
            if pd.isna(val):
                new_vals.append(last_val)
            else:
                last_val = get_last_paths(val)
                new_vals.append(last_val)
        result_df[column] = new_vals"""

    os.makedirs(output_dir, exist_ok=True)

    """output_path = os.path.join(
        output_dir, f"{subject_name}_transcript_slide.csv"
    )
    result_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")"""

    # json
    output_path = os.path.join(output_dir, f"{subject_name}_transcript_slide.json")
    with open(output_path, "w", encoding="utf-8") as output_json:
        json.dump(results_json, output_json, ensure_ascii=False, indent=2)
    print(f"Saved: {output_path}")


aligned_img_base_dir = "/large/datas/nina-n/NL-ken/transcript_img_combined_json"
for subject_name, paths in subjects.items():
    align_info(subject_name, paths, output_dir=aligned_img_base_dir)
