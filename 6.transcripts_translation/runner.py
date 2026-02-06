import os
import json
from pathlib import Path
from llama32 import Llama32Model

target_languages = {
    "ja": "Japanese",
    "es": "Spanish",
    "fr": "French",
    "id": "Indonesian",
    "th": "Thai",
    "zh": "Chinese",
    "de": "German"
}

def get_basedir_path():
    fd = None
    match os.uname()[1]:
        case "lectrans":
            return Path("/large/datas/frederikus-h/lectrans-testset")
        case u if u.startswith("elm"):
            return Path("~/_self_research/24.10.lecture_translate/lectrans-testset").expanduser()
        case _:
            raise NotImplementedError

fp_test_transl_slide = get_basedir_path() / "25.11.slides_annot/slide_translation_annot_by_sbj_all_text.json"
fp_test_transl_transcript = get_basedir_path() / "25.11.transcript_translate/transcript_translate_annotated_by_sbj_all_text.json"

with open(fp_test_transl_slide, "r") as f:
    datas_slide_text = json.load(f)

with open(fp_test_transl_transcript, "r") as f:
    datas_transcript_text = json.load(f)

model = Llama32Model()
model.load_model()

for k, v in datas_transcript_text.items():
    sbj, lec_id = k.split('/',1)
    preds_one_lecture = {}
    for ts, texts in v.items():
        cur_preds = {}
        for lang_tgt, lang_target in target_languages.items():
            cur_preds[lang_tgt] = model.eval_instance("English", lang_target, texts["en"])
        preds_one_lecture[ts] = cur_preds
    with open(f"{sbj}_{lec_id}.llama32_pred.json", "w", encoding="utf8") as f:
        json.dump(preds_one_lecture, f, ensure_ascii=False, indent=2)
