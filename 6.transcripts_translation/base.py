class BaseModel:
    def __init__(self, *args, **kwargs):
        self.model_id = None

    def load_model(self, *args, **kwargs):
        raise NotImplementedError

    def eval_instance(self, lang_src, lang_tgt, transcript_src, context_img=None, context_text=None):
        raise NotImplementedError


if __name__ == "__main__":
    pass

