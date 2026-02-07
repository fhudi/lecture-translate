import os
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

LANGUAGES = {
    "en": "English",
    "ja": "Japanese",
    "id": "Indonesian",
    "th": "Thai",
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
    "de": "German"
}


def _get_basedir_path():
    fd = None
    match os.uname()[1]:
        case "lectrans":
            return Path("/large/datas/frederikus-h/lectrans-testset")
        case u if u.startswith("elm"):
            return Path("~/_self_research/24.10.lecture_translate/lectrans-testset").expanduser()
        case _:
            raise NotImplementedError


fp_test_transl_slide = _get_basedir_path() / "25.11.slides_annot/slide_translation_annot_by_sbj_all_text.json"
fp_test_transl_transcript = _get_basedir_path() / "25.11.transcript_translate/transcript_translate_annotated_by_sbj_all_text.json"


class BaseModel:
    def __init__(self, *args, **kwargs):
        self.model = None

    def load_model(self, *args, **kwargs):
        raise NotImplementedError

    def eval_instance(self, lang_src, lang_tgt, transcript_src, context_img=None, context_text=None):
        raise NotImplementedError
    
    def evaluate(self, output_suffix=None, lang_src="en"):
        _cur_time_ = datetime.now().strftime("%y%m%d_%H%M%S")

        with open(fp_test_transl_slide, "r") as f:
            datas_slide_text = json.load(f)

        with open(fp_test_transl_transcript, "r") as f:
            datas_transcript_text = json.load(f)

        if self.model is None:
            self.model = self.load_model()

        lang_source = LANGUAGES[lang_src]  # "English"
        target_languages = {
            k: v for k, v in LANGUAGES.items() if k != lang_src
        }

        outdir = Path("preds")
        outdir.mkdir(parents=True, exist_ok=True)

        for k, v in datas_transcript_text.items():
            sbj, lec_id = k.split('/',1)
            preds_one_lecture = {}
            for ts, texts in tqdm(v.items(), desc=f"{sbj}/{lec_id}"):
                cur_preds = {}
                for lang_tgt, lang_target in target_languages.items():
                    cur_preds[lang_tgt] = model.eval_instance(lang_source, lang_target, texts[lang_src])
                preds_one_lecture[ts] = cur_preds
                break
            
            with open(outdir / f"{sbj}_{lec_id}.{lang_src}2any.{output_suffix}.{_cur_time_}.json", "w", encoding="utf8") as f:
                json.dump(preds_one_lecture, f, ensure_ascii=False, indent=2)
            break



if __name__ == "__main__":
    pass

