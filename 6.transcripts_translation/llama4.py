from base import BaseModel
from PIL import Image
from constants import (
    SYS_PROMPT,
    SYS_PROMPT_SLIDE_TXT,
    SYS_PROMPT_SLIDE_IMG,
    USER_PROMPT,
    SYS_PROMPT_TRANSL_SLIDE_TXT_ALL_BB_STRUCTURED_OUTPUT,
    USER_PROMPT_TRANSL_SLIDE_TXT_ALL_BB_STRUCTURED_OUTPUT,
)
from llama32 import Llama32Model


class Llama4Model(Llama32Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # BaseModel.__init__(*args, **kwargs)
        # Llama32Model.__init__(*args, **kwargs)

        self.model_id = kwargs.get(
            "model_id", "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        )
        self.sys_prompt = {
            "base": SYS_PROMPT,
            "slide_txt": SYS_PROMPT_SLIDE_TXT,
            "slide_img": SYS_PROMPT_SLIDE_IMG,
        }
        self.user_prompt = USER_PROMPT
        self.model = None
        self.processor = None
        self.device = None

        self.sys_prompt_slide_transl = {
            # "base": SYS_PROMPT_TRANSL_SLIDE_TXT_PER_BB,
            # "slide_img": SYS_PROMPT_TRANSL_SLIDE_TXT_W_IMG,
            "base": SYS_PROMPT_TRANSL_SLIDE_TXT_ALL_BB_STRUCTURED_OUTPUT,
            "slide_img": SYS_PROMPT_TRANSL_SLIDE_TXT_ALL_BB_STRUCTURED_OUTPUT,
        }
        self.user_prompt_slide_transl = USER_PROMPT_TRANSL_SLIDE_TXT_ALL_BB_STRUCTURED_OUTPUT

    def load_model(self, *args, **kwargs):
        if self.model is not None:
            return self.model

        import torch
        from transformers import (
            Llama4ForConditionalGeneration,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        # quant_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     # bnb_4bit_compute_dtype=torch.bfloat16,
        #     # bnb_4bit_use_double_quant=True,
        #     # bnb_4bit_quant_type="nf4"
        # )

        self.model = Llama4ForConditionalGeneration.from_pretrained(
            self.model_id,
            # quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.device = self.model.device

        return self.model

    # def eval_instance(
    #     self,
    #     lang_src,
    #     lang_tgt,
    #     transcript_src,
    #     context_img=None,
    #     context_text=None,
    #     is_check_input_only=False,
    # ):
    #     return Llama32Model.eval_instance(
    #         self, lang_src, lang_tgt, transcript_src, context_img, context_text, is_check_input_only,
    #     )



    def eval_instance_slide_translate(
        self,
        lang_src,
        lang_tgt,
        input_src,
        output_template,
        slide_img_paths=None,  # should only be a single image
        is_check_input_only=False,
    ):
        raise NotImplementedError

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": [_sys_prompt]},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.user_prompt.format(
                            lang_src=lang_src,
                            lang_tgt=lang_tgt,
                            transcript_src=transcript_src,
                        ),
                    },
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=2048)
        outputs_decoded = self.processor.decode(
            outputs[:, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )
        # outputs_decoded = self.tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]

        return outputs_decoded
