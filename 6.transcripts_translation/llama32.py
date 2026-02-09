from base import BaseModel
from PIL import Image

_SYS_PROMPT_ = """\
You are a translation system for lecture video transcripts.
Translate the spoken transcript into the specified target language.

"""

_SYS_PROMPT_SLIDE_TXT_ = """\
You are a translation system for lecture video transcripts.
Translate the spoken transcript into the specified target language.

In addition to the transcript, you are given auxiliary textual context extracted from the corresponding presentation slides.
This context is provided to supplement semantic understanding and disambiguate technical terms, but it is not part of the content to be translated.

The context is as follows, encapsulated with triple ticks:

```
{slide_txt}
```


"""

_SYS_PROMPT_SLIDE_IMG_ = """\
You are a translation system for lecture video transcripts.
Translate the spoken transcript into the specified target language.

In addition to the transcript, you are given auxiliary image context from the corresponding presentation slides.
This context is provided to supplement semantic understanding and disambiguate technical terms, but it is not part of the content to be translated.

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
        self.model_id = kwargs.get("model_id", "meta-llama/Llama-3.2-11B-Vision-Instruct")
        self.sys_prompt = {"base": _SYS_PROMPT_, "slide_txt": _SYS_PROMPT_SLIDE_TXT_, "slide_img": None}
        self.user_prompt = _USER_PROMPT_

    def load_model(self, *args, **kwargs):
        if self.model is not None:
            return self.model

        import torch
        from transformers import MllamaForConditionalGeneration, AutoProcessor, AutoTokenizer
        
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        return self.model

    def eval_instance(self, lang_src, lang_tgt, transcript_src, context_img=None, context_text=None):
        _user_prompt = {"type": "text", "text": self.user_prompt.format(
            lang_src=lang_src, lang_tgt=lang_tgt, transcript_src=transcript_src,
        )}
        images = None

        if context_img is not None:
            # _sys_prompt = self.sys_prompt["slide_img"].format(slide_img=context_img)
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": self.sys_prompt["slide_img"]},
                    *[{"type": "image"} for _ in range(len(context_img))],
                    _user_prompt,
                ]},
            ]
            images = [Image.open(fp) for fp in context_img]
        
        elif context_text is not None:
            messages = [
                # {"role": "system", "content": [
                #     {"type": "text", "text": _sys_prompt},
                # ]},
                {"role": "user", "content": [
                    {"type": "text", "text": self.sys_prompt["slide_txt"].format(slide_txt=context_text)},
                    _user_prompt,
                ]},
            ]
        
        else:
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": self.sys_prompt["base"]},
                    _user_prompt,
                ]},
            ]

        inputs_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True,
        )
        inputs = self.processor(
            images or None, inputs_text, return_tensors="pt"
        ).to(self.model.device)
        if images is not None:
            for i in images:
                i.close()

        outputs = self.model.generate(**inputs.to(self.model.device), max_new_tokens=2048)
        outputs_decoded = self.processor.decode(outputs[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        # outputs_decoded = self.tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]

        return outputs_decoded

