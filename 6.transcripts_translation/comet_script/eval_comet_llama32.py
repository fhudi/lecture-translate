from comet import download_model, load_from_checkpoint
import json
from pathlib import Path
import torch
import os
import numpy

reference_json_path = Path("/lab_path/NL-ken/transcript_translate_annotated_by_sbj_all_text.json")
with open(reference_json_path, "r", encoding="utf-8") as ref_file:
    ref_json = json.load(ref_file)

use_gpu = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

mt_base_path = Path("/lab_path/NL-ken/output_json")
comet_base_path = Path("/lab_path/NL-ken/comet_results")
comet_data_path = Path("/lab_path/NL-ken/comet_data")

exp_types = [
    "nocontext",
    "slideimage",
    "slidetext",
    # "imageonly",
    # "slidetextonly"
]

models = [
    # "gpt-4.1-2025-04-14",
    # "o4-mini-2025-04-16",
    # "gpt-5-mini-2025-08-07",
    # "gemma-3-12b-it",
    "llama32"
]

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

languages = ["ja", "zh", "th", "id", "es", "fr", "de"]

model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)

def input_data(model, exp_type, language):
    data = []

    for subject, video_id in subjects.items():
        mt_path = (mt_base_path / exp_type / model / f"{subject}_{video_id}.en2any.{model}.json")
        if not mt_path.exists():
            continue        

        with open(mt_path, "r", encoding="utf-8") as mt_file:
            mt_json = json.load(mt_file)

        ref_key = f"{subject}/{video_id}"
        if ref_key not in ref_json:
            continue

        ref_subject = ref_json[ref_key]

        for timestamp, mt_langs in mt_json.items():
            if language not in mt_langs:
                continue
            if timestamp not in ref_subject:
                continue
            if language not in ref_subject[timestamp]:
                continue

            src_text = ref_subject[timestamp]["en"]
            ref_text = ref_subject[timestamp][language]

            mt_list = mt_langs[language]
            mt_text = mt_list[0]
            mt_text = mt_text.strip('"')

            data.append({
                "src": src_text,
                "mt": mt_text,
                "ref": ref_text
            })
    
    return data

for model in models:
    for exp_type in exp_types:
        result_scores = {}
        comet_input_data = {}

        for language in languages:
            comet_data = input_data(model, exp_type, language)
            if not comet_data:
                continue

            data_path = comet_data_path / f"{model}_{exp_type}_{language}_data.json"
            with open(data_path, "w", encoding="utf-8") as data_file:
                json.dump(comet_data, data_file, ensure_ascii=False, indent=2)

            comet_output = comet_model.predict(comet_data, batch_size=8)
            # result_scores[language] = comet_score
            result_scores[language] = {
                "system_score": float(comet_output.system_score),
                "scores": comet_output.scores
            }
            
            score_path = comet_base_path / f"{model}_{exp_type}_scores.json"
            with open(score_path, "w", encoding="utf-8") as score_file:
                json.dump(result_scores, score_file, ensure_ascii=False, indent=2)