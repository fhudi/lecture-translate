import json
import base64
from pathlib import Path
# import tiktoken

from base import fd_basedir_slides, fp_test_transl_slide
from constants import (
    SYS_PROMPT_TRANSL_SLIDE_TXT_PER_BB_STRUCTURED_OUTPUT,
    # SYS_PROMPT_TRANSL_SLIDE_TXT_W_IMG,
    USER_PROMPT_TRANSL_SLIDE_TXT_PER_BB_STRUCTURED_OUTPUT,
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
    "gpt-4.1-2025-04-14",
    "o4-mini-2025-04-16"
]

gpt_5x = [
    # "gpt-5-2025-08-07",
    "gpt-5-mini-2025-08-07",
    # "gpt-5-nano-2025-08-07"
]


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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

            system_prompt = SYS_PROMPT_TRANSL_SLIDE_TXT_PER_BB_STRUCTURED_OUTPUT
            for lang_tgt, lang_target in target_languages.items():
                user_prompt = USER_PROMPT_TRANSL_SLIDE_TXT_PER_BB_STRUCTURED_OUTPUT.format(
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
        translate_slide_text_json(cur_model, include_slide_img=False)
    # for cur_model in gpt_5x + gpt_4x:
    #     translate_slide_text_json(cur_model, include_slide_img=True)
