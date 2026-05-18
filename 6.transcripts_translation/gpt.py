import json
import base64
from pathlib import Path
from tqdm import tqdm
# import tiktoken

from base import (
    fp_test_transl_slide, fp_test_transl_transcript, fp_test_asr_transcript, lang_asr,
    fp_test_transcript_slide_align,
    fd_basedir_slides, fp_test_transl_slide,
)
from constants import (
    SYS_PROMPT_STRUCTURED_OUTPUT,
    USER_PROMPT_STRUCTURED_OUTPUT,

    SYS_PROMPT_TRANSL_SLIDE_TXT_ALL_BB_STRUCTURED_OUTPUT,
    # SYS_PROMPT_TRANSL_SLIDE_TXT_W_IMG,
    USER_PROMPT_TRANSL_SLIDE_TXT_ALL_BB_STRUCTURED_OUTPUT,
)
from constants import LANGUAGES


lang_src = "en"
lang_source = "English"
target_languages = {
    k: v for k, v in LANGUAGES.items() if k != lang_src
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
    # "gpt-4.1-2025-04-14",
    "o4-mini-2025-04-16",
]

gpt_5x = [
    # "gpt-5-2025-08-07",
    # "gpt-5-mini-2025-08-07",
    # "gpt-5-nano-2025-08-07",
]


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def translate_transcript_json(
        model,
        context=None,
        # dummy=False,
        # rg_idx=None,
        use_non_corrected_asr_output=False,
):
    print("Model:", model)

    with open(fp_test_transl_slide, "r") as f:
        datas_slide_text = json.load(f)

    with open(fp_test_transl_transcript, "r") as f:
        datas_transcript_text = json.load(f)

    with open(fp_test_asr_transcript, "r") as f:
        datas_transcript_asr = json.load(f)

    with open(fp_test_transcript_slide_align, "r") as f:
        datas_alignment = json.load(f)

    lang_source = LANGUAGES[lang_src]  # "English"
    target_languages = {k: v for k, v in LANGUAGES.items() if k != lang_src}

    _lang_src = lang_asr if use_non_corrected_asr_output else lang_src

    # ============================ #
    # Create JSON API (Transcipts) #
    # ============================ #

    output_dir = Path(f"ins_transcripts/{model}_{context or 'nocontext'}")
    output_dir.mkdir(exist_ok=True, parents=True)

    __idx_2_ts_all__ = dict()

    for k, v in datas_transcript_text.items():
        context_text = []
        if context == "slidetxt":  # [transcript translate]  Preparing text from slide text
            for slide_idx, slide_bbs in enumerate(datas_slide_text[k].values()):
                context_text.append(
                    "\n".join(
                        [t for ts in slide_bbs[lang_src].values() for t in ts]
                    )
                )
            # context_text = "\n\n\n".join(context_text)

        sbj, lec_id = k.split("/", 1)
        # subject, video_id = subject_video.split("/", 1)
        output_file = output_dir / f"transcripttransl_{context or 'nocontext'}_{sbj}_{model}.jsonl"

        print(f"Creating {output_file}")
        with open(output_file, "w", encoding="utf-8"):
            pass

        __idx_2_ts__ = __idx_2_ts_all__.setdefault(sbj, dict())
        for _idx, (ts, texts) in enumerate(tqdm(v.items(), desc=f"{sbj}/{lec_id}")):
            if context == "slideimg":  # [transcript translate]  Preparing text from slide text
                context_img = [
                    encode_image(
                        fd_basedir_slides / sbj / lec_id / f"{fn}.jpg"
                    ) for fn in datas_alignment[k][ts]
                ]
            else:
                context_img = None
            system_prompt = SYS_PROMPT_STRUCTURED_OUTPUT
            if use_non_corrected_asr_output:
                input_text = datas_transcript_asr[k][ts][_lang_src]
            else:
                input_text = texts[lang_src]
            user_prompt = USER_PROMPT_STRUCTURED_OUTPUT.format(
                context_text=context_text,
                transcript_src=input_text,
            )

            content_list = [
                {
                    "role": "system",
                    "content": system_prompt
                }
            ]
            if context_img:
                content_list.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": user_prompt,
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{context_img}"
                        },
                    ]
                })
            else:
                content_list.append({
                    "role": "user",
                    "content": user_prompt
                })

            __idx_2_ts__[f"t{_idx:03}"] = ts
            request_json = {
                "custom_id": f"transcripttransl_{context or 'nocontext'}_{sbj}_{model}_{lec_id}_t{_idx:03}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": model,
                    "input": content_list,
                }
            }

            with open(output_file, "a", encoding="utf-8") as f_out:
                f_out.write(json.dumps(request_json) + "\n")

    with open("__idx_2_ts_all__.json", "w", encoding="utf8") as f:
        json.dump(__idx_2_ts_all__, f, ensure_ascii=False, indent=2)



def translate_slide_text_json(model, include_slide_img=False):
    print("Model:", model)
    # encoding = tiktoken.encoding_for_model(model)
    # total_tokens_all = 0

    subtask = "withslideimg" if include_slide_img else "nocontext"
    output_dir = Path(f"ins_slides/{model}_{subtask}")
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(fp_test_transl_slide, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    for subject_video, scenes in all_data.items():
        # total_tokens = 0
        subject, video_id = subject_video.split("/", 1)
        output_file = output_dir / f"slidetransl_{subtask}_{subject}_{model}.jsonl"

        print(f"Creating {output_file}")
        with open(output_file, "w", encoding="utf-8"):
            pass

        for scene_id, bb_dict in scenes.items():
            if include_slide_img:
                image_base64 = encode_image(
                    fd_basedir_slides / subject / video_id / f"{scene_id}.jpg")
            else:
                image_base64 = None

            input_sources = []
            for bb_id, slides_texts in bb_dict["en"].items():
                slides_text = slides_texts[0]    # In LecTrans pipeline, there is only a single text per BB.
                input_sources.append(f"\"{bb_id}\": \"\"\"{slides_text}\"\"\"")
            input_src = ',\n'.join(input_sources)

            system_prompt = SYS_PROMPT_TRANSL_SLIDE_TXT_ALL_BB_STRUCTURED_OUTPUT
            for lang_tgt, lang_target in target_languages.items():
                user_prompt = USER_PROMPT_TRANSL_SLIDE_TXT_ALL_BB_STRUCTURED_OUTPUT.format(
                    # lang_src=lang_source,
                    lang_tgt=lang_target,
                    input_src=input_src,
                )

                content_list = [
                    {
                        "role": "system",
                        "content": system_prompt
                    }
                ]

                if include_slide_img and image_base64 is not None:
                    content_list.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": user_prompt,
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        ]
                    })
                else:
                    content_list.append({
                        "role": "user",
                        "content": user_prompt
                    })

                request_json = {
                    "custom_id": f"slidetransl_{subtask}_{subject}_{model}_{scene_id}_{lang_tgt}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": model,
                        # "messages": content_list,
                        "input": content_list,
                        # "response_format": {"type": "json_object"},
                        # "max_completion_tokens": 8192,
                    }
                }

                # JSONL append
                with open(output_file, "a", encoding="utf-8") as f_out:
                    f_out.write(json.dumps(request_json) + "\n")

                # # Approximate input token count
                # for message in content_list:
                #     if isinstance(message["content"], list):
                #         for item in message["content"]:
                #             if item["type"] == "text":
                #                 total_tokens += len(encoding.encode(item["text"]))
                #             elif item["type"] == "image_url":
                #                 # URL text counts as tokens, for base64
                #                 total_tokens += len(encoding.encode(item["image_url"]["url"]))
                #     else:
                #         total_tokens += len(encoding.encode(message["content"]))

        # print(f"Token count:  {total_tokens:12,}  ({subject})")
        # total_tokens_all += total_tokens

    # print(f">>> {model} Token count:  {total_tokens_all:12,}")




if __name__ == "__main__":
    for cur_model in gpt_4x + gpt_5x:
        translate_transcript_json(cur_model, use_non_corrected_asr_output=True)
        translate_transcript_json(cur_model, use_non_corrected_asr_output=True, context="slidetxt")
        translate_transcript_json(cur_model, use_non_corrected_asr_output=True, context="slideimg")

    # for cur_model in gpt_4x + gpt_5x:
    #     translate_slide_text_json(cur_model, include_slide_img=False)
    # for cur_model in gpt_5x + gpt_4x:
    #     translate_slide_text_json(cur_model, include_slide_img=True)
