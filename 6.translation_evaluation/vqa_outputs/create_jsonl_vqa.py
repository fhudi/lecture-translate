import os
import json
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
        "video_id": "LQ7IXyQSXBk",
        "vqa_video_id": "LQ7IXyQSXBk"
        },
    "anat-2": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/anat-2_-_IvK-UGOI5ZQ_-_transcripts.xlsx",
        "video_id": "IvK-UGOI5ZQ",
        "vqa_video_id": "IvK-UGOI5ZQ"
        },
    "bio-1": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/bio-1_-_pUH2_SGnm9g_-_transcripts.xlsx",
        "video_id": "pUH2_SGnm9g",
        "vqa_video_id": "pUH2_SGnm9g"
        },
    "bio-2": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/bio-2_-_dEh_Zk9d0sE_-_transcripts.xlsx",
        "video_id": "dEh_Zk9d0sE",
        "vqa_video_id": "dEh_Zk9d0sE"
        },
    "bio-3": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/bio-3_-_u7AnKjGOEgI_-_transcripts.xlsx",
        "video_id": "u7AnKjGOEgI",
        "vqa_video_id": "u7AnKjGOEgI",
        },
    "bio-4": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/bio-4_-_gb0arNyiLCI_-_transcripts.xlsx",
        "video_id": "gb0arNyiLCI",
        "vqa_video_id": "gb0arNyiLCI"
        },
    "dental": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/dental_-_uLz-Ur_Drk4_-_transcripts.xlsx",
        "video_id": "uLz-Ur_Drk4",
        "vqa_video_id": "uLz-Ur_Drk4"
        },
    "intmd_microeconomics": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/intmd_microeconomics_-_rhOpABGewYA_-_transcripts.xlsx",
        "video_id": "rhOpABGewYA",
        "vqa_video_id": "02.rhOpABGewYA"
        },
    "intro_to_cs_programming_python": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/intro_to_cs_programming_python_-_tsMrUdynbQQ_-_transcripts.xlsx",
        "video_id": "tsMrUdynbQQ",
        "vqa_video_id": "17.tsMrUdynbQQ"
        },
    "intro_to_neural_computation": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/intro_to_neural_computation_-_K1pxJVdqlxw_-_transcripts.xlsx",
        "video_id": "K1pxJVdqlxw",
        "vqa_video_id": "05.K1pxJVdqlxw"
        },
    "intro_to_physics_20century": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/intro_to_physics_20century_-_Fce9xf_kjyc_-_transcripts.xlsx",
        "video_id": "Fce9xf_kjyc",
        "vqa_video_id": "08.Fce9xf_kjyc"
        },
    "ml-1": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/ml-1_-_L1TiP9P55-8_-_transcripts.xlsx",
        "video_id": "L1TiP9P55-8",
        "vqa_video_id": "L1TiP9P55-8"
        },
    "psy-1": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/psy-1_-_j2-9yymHfeU_-_transcripts.xlsx",
        "video_id": "j2-9yymHfeU",
        "vqa_video_id": "j2-9yymHfeU"
        },
    "psy-2": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/psy-2_-_uNrB2uG_9gs_-_transcripts.xlsx",
        "video_id": "uNrB2uG_9gs",
        "vqa_video_id": "uNrB2uG_9gs"
        },
    "speaking": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/speaking_-_KL_I9eE2aGk_-_transcripts.xlsx",
        "video_id": "KL_I9eE2aGk",
        "vqa_video_id": "KL_I9eE2aGk"
        },
    "theory_of_computation": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/theory_of_computation_-_aVv9WXwW95w_-_transcripts.xlsx",
        "video_id": "aVv9WXwW95w",
        "vqa_video_id": "17.aVv9WXwW95w"
        },
    "urban_energy_systems_and_policy": {
        "transcript_xlsx": "/large/datas/nina-n/testset/testdata_original/urban_energy_systems_and_policy_-_3WXWESx6jWY_-_transcripts.xlsx",
        "video_id": "3WXWESx6jWY",
        "vqa_video_id": "14.3WXWESx6jWY"
        }
}

def vqa_output(model):
    output_dir = "/large/datas/nina-n/NL-ken/input_jsonl_vqa"
    output_model_dir = os.path.join(output_dir, model)
    os.makedirs(output_model_dir, exist_ok=True)

    slide_text_alignment = "/large/datas/nina-n/NL-ken/transcript_img_combined_json"

    vqa_root = Path("/large/datas/nina-n/NL-ken/cogagent_vqa_quant8")

    for subject, paths in subjects.items():
        slide_json_path = f'{slide_text_alignment}/{subject}_transcript_slide.json'
        with open(slide_json_path, "r", encoding="utf-8") as json_file:
            slide_json = json.load(json_file)
    
        output_file = f"{output_dir}/{model}/{subject}_vqa_{model}.jsonl"
        print(f"Creating {output_file}")

        subject_video_key = f"{subject}/{subjects[subject]['video_id']}"
        if subject_video_key not in slide_json:
            print(f"No data: {subject_video_key}")
            continue
        
        testset = pd.read_excel(paths["transcript_xlsx"], engine="openpyxl")
        request_count = 0

        with open(output_file, "w", encoding="utf-8") as f:
            for transcript_i, row in testset.iterrows():
                transcript_en = row["transcript"]
                start_timestamp_ori = row["timestamp"]
                start_timestamp = start_timestamp_ori.split(" -->")[0]
                
                valid_image_count = 0

                contents = slide_json[subject_video_key].get(start_timestamp, None)
                img_files = []
                vqa_text = ""
                if contents:
                    img_files = [Path(img.strip()) for img in contents.get("images", []) if img.strip()]
                    for img_path in img_files:
                        scene_num = img_path.stem
                        if "vqa_video_id" in subjects[subject]:
                            if subjects[subject]["vqa_video_id"] not in scene_num:
                                scene_num = scene_num.replace(subjects[subject]["video_id"], subjects[subject]["vqa_video_id"], 1)
                            vqa_dir = subjects[subject]["vqa_video_id"]
                        else:
                            vqa_dir = subjects[subject]["video_id"]

                        vqa_path = (vqa_root / vqa_dir / f"{scene_num}_quant8.json")
                        if not vqa_path.exists():
                            print(f"Not found: {vqa_path}")
                            continue
                        with open(vqa_path, "r", encoding="utf-8") as vqa_file:
                            vqa_data = json.load(vqa_file)
                        vqa_text += (json.dumps(vqa_data, ensure_ascii=False, indent=2) + "\n\n")
                        valid_image_count += 1

                for language_code, language_name in target_languages.items():
                    system_prompt = f"""\
                    You are a translation system for lecture video transcripts.
                    Translate the spoken transcript into the specified target language.

                    In addition to the transcript, you are given auxiliary textual context generated from the corresponding presentation slides.
                    The context consists of question-answer pairs describing the slides.
                    This context is provided to supplement semantic understanding and disambiguate technical terms, but it is not part of the content to be translated.

                    The context is as follows:

                    {vqa_text}
                    """

                    user_prompt = f"""\
                    Translate the following from {lang_src} to {language_name}.
                    Please return only the translation.

                    {lang_src}:
                    "{transcript_en}"

                    {language_name}:
                    """
                    
                    content_list = [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ]

                    request_json = {
                        "custom_id": f"{subject}_vqa_{model}_{transcript_i}_{language_code}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model,
                            "messages": content_list,
                        }
                    }

                    f.write(json.dumps(request_json) + "\n")
                    request_count += 1
        print("File created")