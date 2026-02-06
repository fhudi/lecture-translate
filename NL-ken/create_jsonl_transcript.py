import os
import json
import glob
import pandas as pd
from pathlib import Path
import base64

lang_src = "English"

target_languages = {
    "ja": "Japanese",
    "es": "Spanish",
    "fr": "French",
    "id": "Indonesian",
    "th": "Thai",
    "zh": "Chinese",
    "de": "German"
}

subjects = {
    "anat-1": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/anat-1_-_LQ7IXyQSXBk_-_transcripts.xlsx",
        "video_id": "LQ7IXyQSXBk"
        },
    "anat-2": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/anat-2_-_IvK-UGOI5ZQ_-_transcripts.xlsx",
        "video_id": "IvK-UGOI5ZQ"
        },
    "bio-1": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/bio-1_-_pUH2_SGnm9g_-_transcripts.xlsx",
        "video_id": "pUH2_SGnm9g"
        },
    "bio-2": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/bio-2_-_dEh_Zk9d0sE_-_transcripts.xlsx",
        "video_id": "dEh_Zk9d0sE"
        },
    "bio-3": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/bio-3_-_u7AnKjGOEgI_-_transcripts.xlsx",
        "video_id": "u7AnKjGOEgI"
        },
    "bio-4": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/bio-4_-_gb0arNyiLCI_-_transcripts.xlsx",
        "video_id": "gb0arNyiLCI"
        },
    "dental": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/dental_-_uLz-Ur_Drk4_-_transcripts.xlsx",
        "video_id": "uLz-Ur_Drk4"
        },
    "intmd_microeconomics": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/intmd_microeconomics_-_rhOpABGewYA_-_transcripts.xlsx",
        "video_id": "rhOpABGewYA"
        },
    "intro_to_cs_programming_python": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/intro_to_cs_programming_python_-_tsMrUdynbQQ_-_transcripts.xlsx",
        "video_id": "tsMrUdynbQQ"
        },
    "intro_to_neural_computation": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/intro_to_neural_computation_-_K1pxJVdqlxw_-_transcripts.xlsx",
        "video_id": "K1pxJVdqlxw"
        },
    "intro_to_physics_20century": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/intro_to_physics_20century_-_Fce9xf_kjyc_-_transcripts.xlsx",
        "video_id": "Fce9xf_kjyc"
        },
    "ml-1": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/ml-1_-_L1TiP9P55-8_-_transcripts.xlsx",
        "video_id": "L1TiP9P55-8"
        },
    "psy-1": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/psy-1_-_j2-9yymHfeU_-_transcripts.xlsx",
        "video_id": "j2-9yymHfeU"
        },
    "psy-2": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/psy-2_-_uNrB2uG_9gs_-_transcripts.xlsx",
        "video_id": "uNrB2uG_9gs"
        },
    "speaking": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/speaking_-_KL_I9eE2aGk_-_transcripts.xlsx",
        "video_id": "KL_I9eE2aGk"
        },
    "theory_of_computation": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/theory_of_computation_-_aVv9WXwW95w_-_transcripts.xlsx",
        "video_id": "aVv9WXwW95w"
        },
    "urban_energy_systems_and_policy": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/urban_energy_systems_and_policy_-_3WXWESx6jWY_-_transcripts.xlsx",
        "video_id": "3WXWESx6jWY"
        }
}

gpt_4x = [
    "gpt-4.1-2025-04-14",
    "o4-mini-2025-04-16"
]

gpt_5x = [
    "gpt-5-2025-08-07",
    "gpt-5-mini-2025-08-07",
    "gpt-5-nano-2025-08-07"
]

def nocontext(model):
    output_dir = "/large/datas/nina-n/NL-ken/input_jsonl_nocontext"
    output_model_dir = os.path.join(output_dir, model)
    os.makedirs(output_model_dir, exist_ok=True)

    # for subject, input_file in subjects.items():
    for subject, paths in subjects.items():
        output_file = f"{output_dir}/{model}/{subject}_nocontext_{model}.jsonl"
        print(f"Creating {output_file}")
        
        # testset = pd.read_excel(input_file, engine="openpyxl")
        testset = pd.read_excel(paths["transcript_xlsx"], engine="openpyxl")
        request_count = 0

        with open(output_file, "w", encoding="utf-8") as f:
            for transcript_i, row in testset.iterrows():
                transcript_en = row["transcript"]

                for language_code, language_name in target_languages.items():
                    '''prompt_user = f"""\
                    Translate the following from {lang_src} to {language_name}.
                    Please return only the translation.

                    {lang_src}:
                    "{transcript_en}"

                    {language_name}:
                    """'''

                    prompt_user = f"""\
                    You are a translation system for lecture video transcripts.
                    Translate the spoken transcript into the specified target language.

                    {lang_src} transcript:
                    "{transcript_en}"

                    {language_name}:
                    """
                    
                    content_list = [
                        {
                            "type": "text",
                            "text": prompt_user
                        }
                    ]

                    if model in gpt_4x:
                        request_json = {
                            "custom_id": f"{subject}_nocontext_{model}_{transcript_i}_{language_code}",
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": model,
                                "messages": [
                                    {"role": "user",
                                    "content": content_list} 
                                ],
                                "temperature": 0,
                                # "max_tokens": 2048
                            }
                        }
                    if model in gpt_5x:
                        request_json = {
                            "custom_id": f"{subject}_nocontext_{model}_{transcript_i}_{language_code}",
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": model,
                                "messages": [
                                    {"role": "user",
                                    "content": content_list} 
                                ],
                                # "max_completion_tokens": 2048
                            }
                        }

                    f.write(json.dumps(request_json) + "\n")
                    request_count += 1
        print("File created")

slide_dir = "/large/datas/frederikus-h/lectrans-testset/25.11.slides_annot/slide_translation_251214"
image_timestamp = "/large/datas/frederikus-h/lectrans-testset/25.11.slides_annot/annot_by_sbj"
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
    image_dir = os.path.join(slide_dir, subject_name, subjects[subject_name]["video_id"])

    # scene_df = scene json (extracted text), slide images
    scene_img_csv_path = os.path.join(image_timestamp, subject_name, f'{subjects[subject_name]["video_id"]}_-_scene_id_to_timestamp.csv')
    scene_df = pd.read_csv(scene_img_csv_path)
    # get timestamps from scene_timestamp
    scene_df["img_timestamp"] = (scene_df["scene_timestamp"].str.split("-", expand=True)[0].apply(prepare_timestamp))
    
    # save the json path
    scene_df["json_name"] = scene_df["scene_id"]
    # prepare img_name (.jpg)
    scene_df["img_name"] = scene_df["scene_id"].str.replace(".json", ".jpg", regex=False)

    transcript_df = pd.read_excel(paths["transcript_xlsx"])
    transcript_df["Start_sec"] = transcript_df["timestamp"].apply(prepare_timestamp)
    transcript_df["End_sec"] = transcript_df["Start_sec"].shift(-1)
    # print(transcript_df[["timestamp", "Start_sec", "End_sec"]].head(10))

    # assume that the last row ends 30 seconds after its start time
    transcript_df.loc[transcript_df.index[-1], "End_sec"] = transcript_df.loc[transcript_df.index[-1], "Start_sec"] + 30.0

    results = []

    for _, row in transcript_df.iterrows():
        # from transcript file (xlsx)
        start_sec = row["Start_sec"]
        end_sec = row["End_sec"]

        matched_val = scene_df[
            (scene_df["img_timestamp"] >= start_sec) &
            (scene_df["img_timestamp"] < end_sec)
        ]
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

        results.append({
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
        })

    result_df = pd.DataFrame(results)
    
    # if images = NaN, assume that the slide displayed on the screen has not changed
    # -> use the slide images from the previous transcript segment
    result_df["images"] = result_df["images"].replace("", pd.NA).ffill()
    result_df["ocrs"] = result_df["ocrs"].replace("", pd.NA).ffill()
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir, f"{subject_name}_transcript_slide.csv"
    )
    result_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

aligned_img_base_dir = "/large/datas/nina-n/NL-ken/transcript_img_combined"

for subject_name, paths in subjects.items():
    align_info(
        subject_name,
        paths,
        output_dir=aligned_img_base_dir
    )

def encode_image_to_base64(image_path: Path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def slideimage(model):
    output_dir = "/large/datas/nina-n/NL-ken/input_jsonl_slideimage"
    output_model_dir = os.path.join(output_dir, model)
    os.makedirs(output_model_dir, exist_ok=True)

    combined_csv_dir = "/large/datas/nina-n/NL-ken/transcript_img_combined"
    
    for subject, paths in subjects.items():
        # get slide paths 
        combined_csv_path = f'{combined_csv_dir}/{subject}_transcript_slide.csv'
        if not os.path.exists(combined_csv_path):
            print(f"Creating {combined_csv_path}")
            align_info(subject, paths, output_dir=combined_csv_dir)
        image_df = pd.read_csv(combined_csv_path)

        output_file = f"{output_dir}/{model}/{subject}_slideimage_{model}.jsonl"
        print(f"Creating {output_file}")
        
        testset = pd.read_excel(paths["transcript_xlsx"], engine="openpyxl")
        request_count = 0

        with open(output_file, "w", encoding="utf-8") as f:
            for transcript_i, row in testset.iterrows():
                transcript_en = row["transcript"]
                valid_image_count = 0
                    
                images_raw = []
                if transcript_i < len(image_df):
                    images_str = image_df.loc[transcript_i, "images"]
                    if isinstance(images_str, str) and images_str.strip():
                        images_raw = images_str.split(", ")               
                img_files = [Path(img.strip()) for img in images_raw if img.strip()]

                for language_code, language_name in target_languages.items():
                    prompt_user = f"""\
                    You are a translation system for lecture video transcripts.
                    Translate the spoken transcript into the specified target language.

                    In addition to the transcript, you are given auxiliary image context from the corresponding presentation slides. 
                    This context is provided to supplement semantic understanding and disambiguate technical terms, 
                    but it is not part of the content to be translated.

                    {lang_src} transcript:
                    "{transcript_en}"

                    {language_name}:
                    """
                    
                    content_list = [
                        {
                            "type": "text",
                            "text": prompt_user
                        }
                    ]

                    for img_path in img_files:
                        if img_path.exists():
                            image_base64 = encode_image_to_base64(img_path)
                            content_list.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            })
                            valid_image_count += 1
                        else:
                            print(f"Not found: {img_path}")

                    if model in gpt_4x:
                        request_json = {
                            "custom_id": f"{subject}_slideimage_{model}_{transcript_i}_{language_code}",
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": model,
                                "messages": [
                                    {"role": "user",
                                    "content": content_list} 
                                ],
                                "temperature": 0,
                                # "max_tokens": 2048
                            }
                        }
                    if model in gpt_5x:
                        request_json = {
                            "custom_id": f"{subject}_slideimage_{model}_{transcript_i}_{language_code}",
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": model,
                                "messages": [
                                    {"role": "user",
                                    "content": content_list} 
                                ],
                                # "max_completion_tokens": 2048
                            }
                        }

                    f.write(json.dumps(request_json) + "\n")
                    request_count += 1
        print("File created")

slideimage("gpt-5-2025-08-07")

def slidetext(model):
    output_dir = "/large/datas/nina-n/NL-ken/input_jsonl_slidetext"
    output_model_dir = os.path.join(output_dir, model)
    os.makedirs(output_model_dir, exist_ok=True)

    combined_csv_dir = "/large/datas/nina-n/NL-ken/transcript_img_combined"
    
    for subject, input_file in subjects.items():
        # get json paths 
        combined_csv_path = f'{combined_csv_dir}/{subject}_transcript_slide.csv'
        if not os.path.exists(combined_csv_path):
            print(f"Creating {combined_csv_path}")
            align_info(subject, paths, output_dir=combined_csv_dir)
        ocr_df = pd.read_csv(combined_csv_path)

        output_file = f"{output_dir}/{model}/{subject}_slidetext_{model}.jsonl"
        print(f"Creating {output_file}")
        
        testset = pd.read_excel(paths["transcript_xlsx"], engine="openpyxl")
        request_count = 0

        with open(output_file, "w", encoding="utf-8") as f:
            for transcript_i, row in testset.iterrows():
                transcript_en = row["transcript"]

                slide_text = ""
                if transcript_i < len(ocr_df):
                    json_file_str = ocr_df.loc[transcript_i, "ocrs"]
                    if isinstance(json_file_str, str):
                        json_files = [
                            Path(j.strip())
                            for j in json_file_str.split(", ") if j.strip()
                        ]

                        for json_file in json_files:
                            with open(json_file, "r", encoding="utf-8") as ocr_file:
                                data = json.load(ocr_file)
                            for item in data:
                                if "value" in item and "text" in item["value"]:
                                    for ocr_text in item["value"]["text"]:
                                        slide_text += ocr_text + "\n"

                for language_code, language_name in target_languages.items():
                    prompt_user = f"""\
                    You are a translation system for lecture video transcripts.
                    Translate the spoken transcript into the specified target language.

                    In addition to the transcript, you are given auxiliary textual context extracted from the corresponding presentation slides. 
                    This context is provided to supplement semantic understanding and disambiguate technical terms, 
                    but it is not part of the content to be translated.

                    The context is as follows:

                    {slides_text}

                    {lang_src} transcript:
                    "{transcript_en}"

                    {language_name}:
                    """
                    
                    content_list = [
                        {
                            "type": "text",
                            "text": prompt_user
                        }
                    ]

                    if model in gpt_4x:
                        request_json = {
                            "custom_id": f"{subject}_slidetext_{model}_{transcript_i}_{language_code}",
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": model,
                                "messages": [
                                    {"role": "user",
                                    "content": content_list} 
                                ],
                                "temperature": 0,
                                # "max_tokens": 2048
                            }
                        }
                    if model in gpt_5x:
                        request_json = {
                            "custom_id": f"{subject}_slidetext_{model}_{transcript_i}_{language_code}",
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": model,
                                "messages": [
                                    {"role": "user",
                                    "content": content_list} 
                                ],
                                # "max_completion_tokens": 2048
                            }
                        }

                    f.write(json.dumps(request_json) + "\n")
                    request_count += 1
        print("File created")

# nocontext("gpt-5-2025-08-07")
# slideimage("gpt-5-2025-08-07")
# slidetext("gpt-5-2025-08-07")

# model = "o4-mini-2025-04-16"
# model = "gpt-4.1-2025-04-14"

# model = "gpt-5-2025-08-07"
# model = "gpt-5-mini-2025-08-07"
# model = "gpt-5-nano-2025-08-07"