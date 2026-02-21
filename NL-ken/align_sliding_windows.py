import os
import json
from pathlib import Path

def get_basedir_path():
    fd = None
    match os.uname()[1]:
        case "lectrans":
            return Path("/large/datas/frederikus-h/lectrans-testset")
        case u if u.startswith("elm"):
            return Path("~/_self_research/24.10.lecture_translate/lectrans-testset").expanduser()
        case _:
            raise NotImplementedError

fp_test_transl_slide = get_basedir_path() / "25.11.slides_annot/slide_translation_annot_by_sbj_all_text.json"
fp_test_transl_transcript = get_basedir_path() / "25.11.transcript_translate/transcript_translate_annotated_by_sbj_all_text.json"
# fp_test_transcript_slide_align = get_basedir_path() / "25.11.transcript_translate/transcript_slide_alignment.json"
fp_test_transcript_slide_align = get_basedir_path() / "25.11.transcript_translate/transcript_slide_alignment.sliding_windows.json"

with open(fp_test_transl_slide, "r") as f:
    datas_slide_text = json.load(f)

with open(fp_test_transl_transcript, "r") as f:
    datas_transcript_text = json.load(f)




###################
# Sliding Windows #
###################

import pandas as pd
from datetime import timedelta

def ts_from_str(ts):
    return timedelta(**{k: float(v) for k, v in zip(["hours", "minutes", "seconds"], ts.split(':',3))})


transcript_to_slides = {}
for k, v in datas_transcript_text.items():
    _v = datas_slide_text[k]
    sbj, lec_id = k.split('/',1)
    with open(
        (get_basedir_path() / "25.11.slides_annot/annot_by_sbj" / sbj / f"{lec_id}_-_scene_id_to_timestamp.csv"),
        "r", encoding="utf8"
    ) as f:
        scene2ts = pd.read_csv(f)
    cur_sid2ts = {
        r.scene_id.rstrip(".json"):
            r.scene_timestamp.rstrip(".jpg").rsplit('-',1)[0]
        for _, r in scene2ts.iterrows()
    }
    for sid in _v.keys():
        assert sid in cur_sid2ts

    arr_trans = list(v.keys())
    def _ts_remap(_):
        """Manually remap empty timestamp on the transcript data"""
        return (
            "0:38:12.000" if (k=="ml-1/L1TiP9P55-8" and _=="") else
            "0:06:30.172" if (k=="psy-1/j2-9yymHfeU" and _=="") else
            _
        )
    arr_trans = {
        st: (ts_from_str(_ts_remap(st)), ts_from_str(_ts_remap(ed))
             if ed is not None else (ts_from_str(_ts_remap(st)) + timedelta(minutes=1)))
        for st, ed in zip(arr_trans, arr_trans[1:] + [None])
    }
    arr_slides = [(k, ts_from_str(cur_sid2ts[k])) for k in _v.keys()] + [(None, timedelta(hours=20))]
    # assumption: videos are less than 20 hours long

    cur_t2s = {}
    sarr_cur_id = 0
    for k_ts, (st, ed) in arr_trans.items():
        cur_t2s[k_ts] = []
        while arr_slides[sarr_cur_id+1][1] <= st:
            sarr_cur_id += 1
        cur_t2s[k_ts].append(arr_slides[sarr_cur_id][0])
        while arr_slides[sarr_cur_id+1][1] <= ed:
            sarr_cur_id += 1
            cur_t2s[k_ts].append(arr_slides[sarr_cur_id][0])

    transcript_to_slides[k] = cur_t2s

with open(fp_test_transcript_slide_align, "w", encoding="utf8") as f:
    json.dump(transcript_to_slides, f, ensure_ascii=False, indent=2)

