from base import BaseModel
from PIL import Image
from constants import *

class DummyInputCheckModel(BaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = kwargs.get("model_id", "meta-llama/Llama-3.2-11B-Vision-Instruct")
        self.sys_prompt = {"base": _SYS_PROMPT_, "slide_txt": _SYS_PROMPT_SLIDE_TXT_, "slide_img": _SYS_PROMPT_SLIDE_IMG_}
        self.user_prompt = _USER_PROMPT_

    def load_model(self, *args, **kwargs):
        if self.model is not None:
            return self.model

        from transformers import AutoProcessor, AutoTokenizer
        
        class _:
            def __init__(self):
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = _()

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        return self.model

    def eval_instance(self, lang_src, lang_tgt, transcript_src, context_img=None, context_text=None):
        _user_prompt = {"type": "text", "text": self.user_prompt.format(
            lang_src=lang_src, lang_tgt=lang_tgt, transcript_src=transcript_src,
        )}
        # print(_user_prompt)
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
        
        print(messages)

        inputs_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True,
        )
        inputs = self.processor(
            images, inputs_text, return_tensors="pt"
        ).to(self.model.device)
        if images is not None:
            for i in images:
                i.close()

        # outputs = inputs
        outputs_decoded = self.processor.decode(inputs["input_ids"], skip_special_tokens=False)

        return outputs_decoded

