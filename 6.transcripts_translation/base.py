import os
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from constants import LANGUAGES


def _get_basedir_path():
    match os.uname()[1]:
        case "lectrans":
            return Path("/large/datas/frederikus-h/lectrans-testset")
        case u if u.startswith("elm") or u.startswith("pine"):
            return Path(
                "~/_self_research/24.10.lecture_translate/lectrans-testset"
            ).expanduser()
        case _:
            raise NotImplementedError


fp_test_transl_slide = (
    _get_basedir_path()
    / "25.11.slides_annot/slide_translation_annot_by_sbj_all_text.json"
)
fp_test_transl_transcript = (
    _get_basedir_path()
    / "25.11.transcript_translate/transcript_translate_annotated_by_sbj_all_text.json"
)
fp_test_asr_transcript = (
    _get_basedir_path()
    / "25.11.transcript_translate/transcript_asr_output_by_sbj_all_text.json"
)
fp_test_transcript_slide_align = (
    _get_basedir_path() / "25.11.transcript_translate/transcript_slide_alignment.json"
)
fd_basedir_slides = _get_basedir_path() / "25.11.slides_annot/slide_translation_251214"


class BaseModel:
    def __init__(self, *args, **kwargs):
        self.model = None

    def load_model(self, *args, **kwargs):
        raise NotImplementedError

    def eval_instance(
        self,
        lang_src,
        lang_tgt,
        transcript_src,
        context_img=None,
        context_text=None,
        is_check_input_only=False,
    ):
        raise NotImplementedError

    def eval_instance_slide_translate(
        self,
        lang_src,
        lang_tgt,
        input_src,
        output_template,
        context_img=None,
        is_check_input_only=False,
    ):
        raise NotImplementedError

    def evaluate(
        self, output_suffix=None, context=None, lang_src="en", dummy=False, rg_idx=None,
        use_non_corrected_asr_output=False,
    ):
        with open(fp_test_transl_slide, "r") as f:
            datas_slide_text = json.load(f)

        with open(fp_test_transl_transcript, "r") as f:
            datas_transcript_text = json.load(f)

        with open(fp_test_asr_transcript, "r") as f:
            datas_transcript_asr = json.load(f)

        with open(fp_test_transcript_slide_align, "r") as f:
            datas_alignment = json.load(f)

        if self.model is None:
            self.model = self.load_model()

        lang_source = LANGUAGES[lang_src]  # "English"
        target_languages = {k: v for k, v in LANGUAGES.items() if k != lang_src}

        lang_asr = "asr"
        _lang_src = lang_asr if use_non_corrected_asr_output else lang_src

        # ===================== #
        # Do Preds (Transcipts) #
        # ===================== #

        _cur_time_ = datetime.now().strftime("%y%m%d_%H%M%S")
        outdir = Path(f"preds_{output_suffix}_{context}_{_cur_time_}")
        outdir.mkdir(parents=True, exist_ok=True)

        st, ed = rg_idx or (0, len(datas_transcript_text))
        for idx, (k, v) in enumerate(datas_transcript_text.items()):
            if (idx < st) or (idx >= ed):
                continue
            kwargs = {
                "is_check_input_only": dummy,
            }

            if (
                context == "slidetxt"
            ):  # [transcript translate]  Preparing text from slide text
                context_text = []
                for slide_idx, slide_bbs in enumerate(datas_slide_text[k].values()):
                    context_text.append(
                        "\n".join(
                            [t for ts in slide_bbs[lang_src].values() for t in ts]
                        )
                    )
                kwargs["context_text"] = "\n\n\n".join(context_text)

            sbj, lec_id = k.split("/", 1)
            preds_one_lecture = {}
            for ts, texts in tqdm(v.items(), desc=f"{sbj}/{lec_id}"):
                if (
                    context == "slideimg"
                ):  # [transcript translate]  Preparing text from slide text
                    kwargs["context_img"] = [
                        fd_basedir_slides / sbj / lec_id / f"{fn}.jpg"
                        for fn in datas_alignment[k][ts]["slides"]
                    ]
                cur_preds = {}
                for lang_tgt, lang_target in target_languages.items():
                    if use_non_corrected_asr_output:
                        input_text = datas_transcript_asr[k][ts][_lang_src]
                    else:
                        input_text = texts[lang_src]
                    cur_preds[lang_tgt] = self.eval_instance(
                        lang_source, lang_target, input_text, **kwargs
                    )
                preds_one_lecture[ts] = cur_preds
                if dummy:
                    break
            with open(
                outdir / f"{sbj}_{lec_id}.{_lang_src}2any.{output_suffix}.json",
                "w",
                encoding="utf8",
            ) as f:
                json.dump(preds_one_lecture, f, ensure_ascii=False, indent=2)
            if dummy:
                break

        # print(kwargs)

    def evaluate_slide_translate(
        self, output_suffix=None, context=None, lang_src="en", dummy=False, rg_idx=None
    ):
        with open(fp_test_transl_slide, "r") as f:
            datas_slide_text = json.load(f)

        if self.model is None:
            self.model = self.load_model()

        lang_source = LANGUAGES[lang_src]  # "English"
        target_languages = {k: v for k, v in LANGUAGES.items() if k != lang_src}

        # ================= #
        # Do Preds (Slides) #
        # ================= #

        _cur_time_ = datetime.now().strftime("%y%m%d_%H%M%S")
        outdir = Path(f"preds_slides_{output_suffix}_{context}_{_cur_time_}")
        outdir.mkdir(parents=True, exist_ok=True)

        st, ed = rg_idx or (0, len(datas_slide_text))
        for idx, (sbj_lecid, all_scenes) in enumerate(datas_slide_text.items()):
            if (idx < st) or (idx >= ed):
                continue
            kwargs = {
                "is_check_input_only": dummy,
            }

            sbj, lec_id = sbj_lecid.split("/", 1)
            preds_one_lecture = {}
            for scene_id, texts in tqdm(all_scenes.items(), desc=f"{sbj}/{lec_id}"):
                if context == "slideimg":
                    kwargs["slide_img_paths"] = [
                        fd_basedir_slides / sbj / lec_id / f"{scene_id}.jpg"
                    ]
                cur_preds = {}
                for lang_tgt, lang_target in target_languages.items():
                    # Per BB version
                    cur_lang_preds = {}
                    for bb_id, bb_texts in texts[lang_src].items():
                        input_src = bb_texts[0]
                        output_template = ""
                        cur_lang_preds[bb_id] = self.eval_instance_slide_translate(
                            lang_source,
                            lang_target,
                            input_src,
                            output_template,
                            **kwargs,
                        )
                    cur_preds[lang_tgt] = cur_lang_preds

                    # All-BB in one go version
                    # # {
                    # #     "{bb_1}": "{text_src_1}",
                    # #     "{bb_2}": "{text_src_2}",
                    # # }
                    # # assert all(len(v) == 1 for v in texts[lang_src].values())  # True
                    # input_src = {k: v[0] for k, v in texts[lang_src].items()}
                    #
                    # # {
                    # #     "{bb_1}": "",
                    # #     "{bb_2}": "",
                    # # }
                    # output_template = {k: "" for k in input_src.keys()}
                    #
                    # cur_preds[lang_tgt] = self.eval_instance_slide_translate(
                    #     lang_source, lang_target, input_src, output_template, **kwargs
                    # )

                preds_one_lecture[scene_id] = cur_preds
                with open(
                    outdir / f"{sbj}_{lec_id}.{lang_src}2any.{output_suffix}.jsonl",
                    "a",
                    encoding="utf8",
                ) as f:
                    json.dump((scene_id, cur_preds), f, ensure_ascii=False)
                    f.write("\n")
                if dummy:
                    break
            with open(
                outdir / f"{sbj}_{lec_id}.{lang_src}2any.{output_suffix}.json",
                "w",
                encoding="utf8",
            ) as f:
                json.dump(preds_one_lecture, f, ensure_ascii=False, indent=2)
            if dummy:
                break


if __name__ == "__main__":
    pass
