import os
import json
from pathlib import Path
import base64

slide_base_path = "/large/datas/frederikus-h/lectrans-testset/25.11.slides_annot/slide_translation_251214"

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
    "anat-1": "LQ7IXyQSXBk", 
    "anat-2": "IvK-UGOI5ZQ",
    "bio-1": "pUH2_SGnm9g",
    "bio-2": "dEh_Zk9d0sE",
    "bio-3": "u7AnKjGOEgI",
    "bio-4": "gb0arNyiLCI",
    "dental": "uLz-Ur_Drk4",
    "intmd_microeconomics": "rhOpABGewYA",
    "intro_to_cs_programming_python": "tsMrUdynbQQ",
    "intro_to_neural_computation": "K1pxJVdqlxw",
    "intro_to_physics_20century": "Fce9xf_kjyc",
    "ml-1": "L1TiP9P55-8",
    "psy-1": "j2-9yymHfeU",
    "psy-2": "uNrB2uG_9gs",
    "speaking": "KL_I9eE2aGk",
    "theory_of_computation": "aVv9WXwW95w",
    "urban_energy_systems_and_policy": "3WXWESx6jWY"
    }

gpt_4x = [
    "gpt-4.1-2025-04-14",
    "o4-mini-2025-04-16"
]

gpt_5x = [
    # "gpt-5-2025-08-07",
    "gpt-5-mini-2025-08-07",
    # "gpt-5-nano-2025-08-07"
]

def encode_image_to_base64(image_path: Path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def translate_image(model):
    output_dir = "/large/datas/nina-n/NL-ken/input_jsonl_imageonly"
    output_model_dir = os.path.join(output_dir, model)
    os.makedirs(output_model_dir, exist_ok=True)

    for subject, video_id in subjects.items():
        image_dir = Path(slide_base_path) / subject / video_id
        if not image_dir.exists():
            print(f"Not found")
        output_file = f"{output_dir}/{model}/{subject}_imageonly_{model}.jsonl"
        print(f"Creating {output_file}")

        image_files = sorted(image_dir.glob("*.jpg"))
        request_count = 0

        with open(output_file, "w", encoding="utf-8") as f:
            for image in image_files:
                image_base64 = encode_image_to_base64(image)

                # get scene number
                file_name = image.stem
                if "Scene" in file_name:
                    scene_number = file_name.split("Scene-")[1]
                else:
                    scene_number = "N"

                for language_code, language_name in target_languages.items():
                    system_prompt = f"""\
                    You are a translation system for lecture slide images.
                    Translate the text on the image into the specified target language.
                    """

                    user_prompt = f"""\
                    Translate the following from {lang_src} to {language_name}.
                    Please return only the translation.

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

                    content_list.append({
                        "role": "user",
                        "content":
                        [{
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }}]
                    })

                    request_json = {
                        "custom_id": f"{subject}_imageonly_{model}_{scene_number}_{language_code}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model,
                            "messages": content_list,
                            # "max_tokens": 2048
                            # "max_completion_tokens": 2048
                        }
                    }

                    f.write(json.dumps(request_json) + "\n")
                    request_count += 1
            print("File created")

def translate_slide_text(model):
    output_dir = "/large/datas/nina-n/NL-ken/input_jsonl_slidetextonly"
    output_model_dir = os.path.join(output_dir, model)
    os.makedirs(output_model_dir, exist_ok=True)

    for subject, video_id in subjects.items():
        json_dir = Path(slide_base_path) / subject / video_id
        if not json_dir.exists():
            print(f"Not found")
        output_file = f"{output_dir}/{model}/{subject}_slidetextonly_{model}.jsonl"
        print(f"Creating {output_file}")

        json_files = sorted(json_dir.glob("*.json"))
        request_count = 0

        with open(output_file, "w", encoding="utf-8") as f:
            for json_file in json_files:
                with open(json_file, "r", encoding="utf-8") as ocr_file:
                    ocr_data = json.load(ocr_file)

                # get scene number
                file_name = json_file.stem
                if "Scene" in file_name:
                    scene_number = file_name.split("Scene-")[1]
                else:
                    scene_number = "N"

                slides_text = ""                       
                for item in ocr_data:
                    if "value" in item and "text" in item["value"]:
                        for ocr_text in item["value"]["text"]:
                            slides_text += ocr_text + "\n"

                for language_code, language_name in target_languages.items():
                    system_prompt = f"""\
                    You are a translation system for lecture slide texts.
                    Translate the slide text into the specified target language.
                    """

                    user_prompt = f"""\
                    Translate the following from {lang_src} to {language_name}.
                    Please return only the translation.

                    {lang_src}:
                    "{slides_text}"

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
                        "custom_id": f"{subject}_slidetextonly_{model}_{scene_number}_{language_code}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model,
                            "messages": content_list,
                            # "max_tokens": 2048
                            # "max_completion_tokens": 2048
                        }
                    }

                    f.write(json.dumps(request_json) + "\n")
                    request_count += 1
            print("File created")

translate_image("gpt-4.1-2025-04-14")
translate_image("o4-mini-2025-04-16")
translate_image("gpt-5-mini-2025-08-07")

'''translate_slide_text("gpt-4.1-2025-04-14")
translate_slide_text("o4-mini-2025-04-16")
translate_slide_text("gpt-5-mini-2025-08-07")'''