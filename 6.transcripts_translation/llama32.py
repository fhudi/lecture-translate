from base import BaseModel
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor, AutoTokenizer

_SYS_PROMPT_ = """\
You are a translation system for lecture video transcripts.
Translate the spoken transcript into the specified target language.
"""

_USER_PROMPT_ = """\
Translate the following from {lang_src} to {lang_tgt}.
Please return only the translation.

{lang_src}:
"{transcript_src}"

{lang_tgt}:
"""

class Llama32Model(BaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        self.sys_prompt = _SYS_PROMPT_
        self.user_prompt = _USER_PROMPT_

    def load_model(self, *args, **kwargs):
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def eval_instance(self, lang_src, lang_tgt, transcript_src, context_img=None, context_text=None):
        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": self.sys_prompt},
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": self.user_prompt.format(
                    lang_src=lang_src, lang_tgt=lang_tgt, transcript_src=transcript_src,
                )},
            ]},
        ]

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, 
            return_dict=True, return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(**inputs.to(self.model.device), max_new_tokens=1024)
        outputs_decoded = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        return outputs_decoded[0]

