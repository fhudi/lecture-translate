from base import BaseModel
from PIL import Image
from constants import (
    SYS_PROMPT,
    SYS_PROMPT_SLIDE_TXT,
    SYS_PROMPT_SLIDE_IMG,
    USER_PROMPT,
    SYS_PROMPT_TRANSL_SLIDE_TXT_PER_BB,
    SYS_PROMPT_TRANSL_SLIDE_TXT_W_IMG,
    USER_PROMPT_TRANSL_SLIDE_TXT_PER_BB,
)


class Llama32Model(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = kwargs.get(
            "model_id", "meta-llama/Llama-3.2-11B-Vision-Instruct"
        )
        self.sys_prompt = {
            "base": SYS_PROMPT,
            "slide_txt": SYS_PROMPT_SLIDE_TXT,
            "slide_img": None,
        }
        self.user_prompt = USER_PROMPT
        self.model = None
        self.processor = None
        self.device = None

        self.sys_prompt_slide_transl = {
            "base": SYS_PROMPT_TRANSL_SLIDE_TXT_PER_BB,
            "slide_img": SYS_PROMPT_TRANSL_SLIDE_TXT_W_IMG,
        }
        self.user_prompt_slide_transl = USER_PROMPT_TRANSL_SLIDE_TXT_PER_BB

    def load_model(self, *args, **kwargs):
        if self.model is not None:
            return self.model

        import torch
        from transformers import MllamaForConditionalGeneration, AutoProcessor

        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.device = self.model.device

        return self.model

    def eval_instance(
        self,
        lang_src,
        lang_tgt,
        transcript_src,
        context_img=None,
        context_text=None,
        is_check_input_only=False,
    ):
        _device = self.model.device if self.model else self.device
        _user_prompt = {
            "type": "text",
            "text": self.user_prompt.format(
                lang_src=lang_src,
                lang_tgt=lang_tgt,
                transcript_src=transcript_src,
            ),
        }

        images = None
        if context_img is not None:
            # _sys_prompt = self.sys_prompt["slide_img"].format(slide_img=context_img)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.sys_prompt["slide_img"]},
                        *[{"type": "image"} for _ in range(len(context_img))],
                        _user_prompt,
                    ],
                },
            ]
            images = [Image.open(fp) for fp in context_img]

        elif context_text is not None:
            messages = [
                # {"role": "system", "content": [
                #     {"type": "text", "text": _sys_prompt},
                # ]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.sys_prompt["slide_txt"].format(
                                slide_txt=context_text
                            ),
                        },
                        _user_prompt,
                    ],
                },
            ]

        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.sys_prompt["base"]},
                        _user_prompt,
                    ],
                },
            ]

        inputs_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        inputs = self.processor(images or None, inputs_text, return_tensors="pt").to(
            _device
        )
        if images is not None:
            for i in images:
                i.close()

        if is_check_input_only:  # outputs = inputs
            return self.processor.decode(inputs["input_ids"], skip_special_tokens=False)

        outputs = self.model.generate(**inputs, max_new_tokens=2048)
        outputs_decoded = self.processor.decode(
            outputs[:, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )

        return outputs_decoded

    def eval_instance_slide_translate(
        self,
        lang_src,
        lang_tgt,
        input_src,
        output_template,
        slide_img_paths=None,  # should only be a single image
        is_check_input_only=False,
    ):
        _device = self.model.device if self.model else self.device
        _user_prompt = {
            "type": "text",
            "text": self.user_prompt_slide_transl.format(
                lang_src=lang_src,
                lang_tgt=lang_tgt,
                input_src=input_src,
                output_template=output_template,
            ),
        }

        images = None
        if slide_img_paths is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.sys_prompt_slide_transl["slide_img"],
                        },
                        *[{"type": "image"} for _ in range(len(slide_img_paths))],
                        _user_prompt,
                    ],
                },
            ]
            images = [Image.open(fp) for fp in slide_img_paths]

        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.sys_prompt_slide_transl["base"]},
                        _user_prompt,
                    ],
                },
            ]

        inputs_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        inputs = self.processor(images or None, inputs_text, return_tensors="pt").to(
            _device
        )
        if images is not None:
            for i in images:
                i.close()

        if is_check_input_only:  # outputs = inputs
            return self.processor.decode(inputs["input_ids"], skip_special_tokens=False)

        outputs = self.model.generate(**inputs, max_new_tokens=2048)
        outputs_decoded = self.processor.decode(
            outputs[:, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )

        return outputs_decoded
