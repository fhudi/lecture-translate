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


SYS_PROMPT = """\
You are a translation system for lecture video transcripts.
Translate the spoken transcript into the specified target language.

"""

SYS_PROMPT_SLIDE_TXT = """\
You are a translation system for lecture video transcripts.
Translate the spoken transcript into the specified target language.

In addition to the transcript, you are given auxiliary textual context extracted from the corresponding presentation slides.
This context is provided to supplement semantic understanding and disambiguate technical terms, but it is not part of the content to be translated.

The context is as follows, encapsulated with triple ticks:

```
{slide_txt}
```


"""

SYS_PROMPT_SLIDE_IMG = """\
You are a translation system for lecture video transcripts.
Translate the spoken transcript into the specified target language.

In addition to the transcript, you are given auxiliary image context from the corresponding presentation slides.
This context is provided to supplement semantic understanding and disambiguate technical terms, but it is not part of the content to be translated.


"""

USER_PROMPT = """\
Translate the following from {lang_src} to {lang_tgt}.
Please return only the translation.

{lang_src}:
"{transcript_src}"

{lang_tgt}:
"""

SYS_PROMPT_TRANSL_SLIDE_TXT = """\
You are a translation system for lecture presentation slides.
Translate the text extracted from slides into the specified target language.
"""

SYS_PROMPT_TRANSL_SLIDE_IMG = """\
You are a translation system for lecture presentation slides.
Translate the text in the attached slides into the specified target language.
"""
