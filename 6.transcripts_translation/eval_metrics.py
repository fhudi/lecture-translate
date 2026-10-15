import ast
import sacrebleu
import json
from pathlib import Path
from pythainlp.tokenize import word_tokenize
from nltk.translate.nist_score import corpus_nist
from tqdm import tqdm

import os
def _get_basedir_path():
    match os.uname()[1]:
        case "lectrans":
            return Path("/large/datas/frederikus-h/lectrans-testset")
        case u if u.startswith("elm") or u.startswith("pine"):
            return Path(
                "~/_self_research/24.10.lecture_translate/lectrans-testset"
            ).expanduser()
        case _:
            raise NotImplementedError


def correcting_lang_id(lang):
    return {
        "jp": "ja"
    }.get(lang, lang)


def tokenize_thai(text):
    return " ".join(word_tokenize(text, engine="newmm"))


def get_bleu_kwargs(lang):
    kwargs = {}
    match lang:
        case "ja":
            kwargs["tokenize"] = "ja-mecab"
        case "zh":
            kwargs["tokenize"] = "zh"
        case "th":
            kwargs["tokenize"] = "none"
    return kwargs


def get_tokenized(text, language):
    match language:
        case "th":
            f_tokenize = tokenize_thai
        case "ja":
            f_tokenize = sacrebleu.metrics.BLEU(tokenize="ja-mecab").tokenizer
        case "zh":
            f_tokenize = sacrebleu.metrics.BLEU(tokenize="zh").tokenizer
        case _:
            f_tokenize = sacrebleu.metrics.BLEU(tokenize="13a").tokenizer
    return f_tokenize(text or "").split()


comet_model = None
cometkiwi_model = None
def comet_predict(preds, golds, srcs, batch_size=128):
    global comet_model, cometkiwi_model
    if comet_model is None or cometkiwi_model is None:
        from comet import download_model, load_from_checkpoint
        model_path = download_model("Unbabel/wmt22-comet-da")
        modelqe_path = download_model("Unbabel/wmt22-cometkiwi-da")
        print("COMET models downloaded")
        comet_model = load_from_checkpoint(model_path)
        cometkiwi_model = load_from_checkpoint(modelqe_path)
        assert comet_model and cometkiwi_model, "COMET models not loaded"
        print("COMET models loaded")
    data, data_qe = [], []
    for src, pred, gold in zip(srcs, preds, golds):
        data.append({"src": src,"mt": pred, "ref": gold})
        data_qe.append({"src": src,"mt": pred})
    score = comet_model.predict(data, batch_size=batch_size)
    score_qe = cometkiwi_model.predict(data_qe, batch_size=batch_size)
    return score, score_qe


def _get_triplets_transcript_(src, gold, pred, lang):
    global sbj_lecid_pair, instance_id    # `instance_id` is timestamp
    if (
        (sbj_lecid_pair, instance_id) == ('intro_to_physics_20century/Fce9xf_kjyc', '0:10:08.636') and
        isinstance(preds[sbj_lecid_pair][instance_id], str)
    ):
        pred = preds[sbj_lecid_pair][instance_id]    # either parsing error or answer error, treating return as language-agnostic
    else:
        pred = preds[sbj_lecid_pair][instance_id].get(lang, "")
    if isinstance(pred, list):
        if len(pred) > 1:
            print(len(pred), instance_id, lang, sbj)
            pred = " ".join(pred)
        else:
            pred = pred[0] if len(pred) else ""
    gold_tokenized = get_tokenized(gold, lang)
    pred_tokenized = get_tokenized(pred, lang)
    if lang == "th":
        gold = " ".join(gold_tokenized)    # tokenize_thai(gold)
        pred = " ".join(pred_tokenized)    # tokenize_thai(pred)
    pred = '""' if (pred is None) or (len(pred) == 0) else pred
    if isinstance(pred, list) and len(pred) == 1:
        pred = pred[0]
    if isinstance(pred, list):
        pred = '\n'.join(pred)
    return [src], [gold], [pred], [gold_tokenized], [pred_tokenized]


def _get_triplets_slide_(all_bb_src, all_bb_gold, all_bb_pred, lang):
    global sbj_lecid_pair, instance_id    # `instance_id` is scene_id
    # currently, we assume the sequential order after sorting are always aligned
    _src = ['\n'.join(bb_text_list) for bb_key, bb_text_list in sorted(all_bb_src.items())]
    _gold = ['\n'.join(bb_text_list) for bb_key, bb_text_list in sorted(all_bb_gold.items())]
    _pred = ['\n'.join(bb_text_list) for bb_key, bb_text_list in sorted(all_bb_pred.items())]
    _gold_tokenized = [get_tokenized(_, lang) for _ in _gold]
    _pred_tokenized = [get_tokenized(_, lang) for _ in _pred]
    if lang == "th":
        _gold = [' '.join(_) for _ in _gold_tokenized]    # tokenize_thai(gold)
        _pred = [' '.join(_) for _ in _pred_tokenized]    # tokenize_thai(pred)
    return _src, _gold, _pred, _gold_tokenized, _pred_tokenized


def get_scores(golds, preds, compute_comet=False, batch_size=128, task="transcripts"):
    global \
        sbj_lecid_pair, cur_gold, scores_per_subject,\
        instance_id, gold_all_langs, scores_per_span, scores_per_lang,\
        lang, src, gold, pred, kwargs, score, score_qe,\
        gold_tokenized, pred_tokenized

    scores_all = {}
    for sbj_lecid_pair, cur_gold in golds.items():
        sbj, lec_id = sbj_lecid_pair.split('/', 1)

        # Debugging only
        # if sbj_lecid_pair not in preds:
        #     continue

        scores_per_subject = {
            "per_language": {},
            "per_instance": {},
        }
        scores_per_lang = {}
        for instance_id, gold_all_langs in cur_gold.items():
            scores_per_span = {}
            gold_all_langs = {correcting_lang_id(k): v for k, v in gold_all_langs.items()}
            for lang, gold in gold_all_langs.items():
                lang = correcting_lang_id(lang)
                if lang == 'en':
                    continue
                match task:
                    case "transcripts":
                        try:
                            src, gold, pred, gold_tokenized, pred_tokenized = \
                                _get_triplets_transcript_(gold_all_langs['en'], gold, preds[sbj_lecid_pair][instance_id], lang)
                        except KeyError as e:
                            raise KeyError(f"{e}: ({sbj_lecid_pair}, {instance_id}, {lang}, )")
                    case "slides":
                        src, gold, pred, gold_tokenized, pred_tokenized = \
                            _get_triplets_slide_(gold_all_langs['en'], gold_all_langs[lang], preds[sbj_lecid_pair][instance_id][lang], lang)
                        if not (len(src) == len(gold) == len(pred)):
                            src, gold, pred = map('\n'.join, [src, gold, pred])
                            gold_tokenized = [[__ for _ in gold_tokenized for __ in _]]
                            pred_tokenized = [[__ for _ in pred_tokenized for __ in _]]
                    case _:
                        raise NotImplementedError(f"Not implemented get_scores(task='{task}')")
                _n = min(5, max(map(len, pred_tokenized)))
                scores_per_span[lang] = [
                    sacrebleu.corpus_bleu(pred, [gold], **get_bleu_kwargs(lang)).score,
                    sacrebleu.corpus_chrf(pred, [gold]).score,
                    corpus_nist([[_] for _ in gold_tokenized], pred_tokenized, n=_n) if _n > 0 else .0,
                    # cer_metric.compute(predictions=[pred], references=[gold]),
                    # wer_metric.compute(predictions=[pred_tokenized], references=[gold_tokenized]),
                    # [scorer.score(g, p).fmeasure for g, p in zip(gold, pred)],
                ] if not compute_comet else []
                _scores_per_lang = scores_per_lang.setdefault(lang, dict())
                for k, v in [
                    ("pred", pred), ("gold", gold), ("src", src),
                    ("pred_tokenized", pred_tokenized), ("gold_tokenized", gold_tokenized),
                    ("instance_id", [instance_id]), ("instance_len", [len(gold)]),
                ]:
                    _scores_per_lang.setdefault(k, []).extend(v)        # micro-averaged within same instance_id
            scores_per_subject["per_instance"][instance_id] = scores_per_span

        scores_per_subject["per_language"] = {
            lang: ([
                sacrebleu.corpus_bleu(_d["pred"], [_d["gold"]], **get_bleu_kwargs(lang)).score,
                sacrebleu.corpus_chrf(_d["pred"], [_d["gold"]]).score,
                corpus_nist([[_] for _ in _d["gold_tokenized"]], _d["pred_tokenized"], n=5),
            ] if not compute_comet else [])
            for lang, _d in scores_per_lang.items()
        }

        if compute_comet:
            print("computing COMET")
            for lang, _d in scores_per_lang.items():
                score, score_qe = comet_predict(_d["pred"], _d["gold"], _d["src"], batch_size)
                scores_per_subject["per_language"][lang].append(score["system_score"])
                scores_per_subject["per_language"][lang].append(score_qe["system_score"])
                _cur_id, s, s_qe = 0, score["scores"], score_qe["scores"]
                for instance_id, instance_len in zip(_d["instance_id"], _d["instance_len"]):
                    _sps = scores_per_subject["per_instance"][instance_id][lang]
                    _sps.append(sum(s[_cur_id:_cur_id+instance_len])/instance_len)
                    _sps.append(sum(s_qe[_cur_id:_cur_id+instance_len])/instance_len)
                    _cur_id += instance_len

        scores_all[sbj_lecid_pair] = scores_per_subject
    return scores_all


# %%time
# fd_outs_basedir = _get_basedir_path() / "lecture-translate/6.transcripts_translation/outs_transcripts"
# fd_evals_basedir = _get_basedir_path() / "lecture-translate/6.transcripts_translation/evals_transcripts"

compute_comet = bool(ast.literal_eval(os.getenv("__EVAL_USE_COMET__", '1')))
_bs_ = int(os.getenv("__EVAL_USE_COMET_BS__", 128))

# _suffix_name_ = os.getenv("__EVAL_SUFFIX_NAME__", "transcripts_annotd")
# _suffix_name_ = os.getenv("__EVAL_SUFFIX_NAME__", "transcripts_aligned")
# _suffix_name_ = os.getenv("__EVAL_SUFFIX_NAME__", "transcripts_asrraw")
_suffix_name_ = os.getenv("__EVAL_SUFFIX_NAME__", "slides_aligned")
_task_ = _suffix_name_.split("_", 1)[0]

_st_ = int(os.getenv("__EVAL_SPLIT_ST__", 0)) or None
_ed_ = int(os.getenv("__EVAL_SPLIT_ED__", 1)) or None

print(compute_comet, _suffix_name_, _task_, _st_, _ed_)

match _task_:
    case "transcripts":
        fp_reference = _get_basedir_path() / "25.11.transcript_translate/transcript_translate_annotated_by_sbj_all_text.json"
    case "slides":
        fp_reference = _get_basedir_path() / "25.11.slides_annot/slide_translation_annot_by_sbj_all_text.json"
    case _:
        raise NotImplementedError(f"Not implemented task='{_task_}'")
assert fp_reference.exists(), f"File not found: {fp_reference}"
with open(fp_reference, 'r', encoding="utf8") as f:
    golds = json.load(f)

fd_outs_basedir = _get_basedir_path() / f"lecture-translate/6.transcripts_translation/outs_{_suffix_name_}"
fd_evals_basedir = fd_outs_basedir.with_name(f"evals_{_suffix_name_}" + ("_comet" if compute_comet else ""))
fd_preds_list = sorted(filter(
    lambda fd: (
        # fd.name.startswith("o4-mini") and
        # fd.name.startswith("glm46v-flash") and
        fd.name.endswith(("nocontext", "slideimg", "slidetxt"))),
    fd_outs_basedir.iterdir()
    ))[_st_:_ed_]
print("Prediction List:", *[_.name for _ in fd_preds_list], sep="\n")
for _, fd_preds in enumerate(tqdm(fd_preds_list)):
    if not fd_preds.is_dir():
        continue
    preds = {}
    for k, v in golds.items():
        sbj, lec_id = k.split('/', 1)
        fp_preds = sorted(fd_preds.glob("*.json"))
        for fp in fp_preds:
            matching_file = (
                fp.name.startswith(f"{sbj}_{lec_id}") or
                (fd_preds.name.startswith(("gpt-4.1", "gpt-5-mini", "o4-mini"))
                 and fp.name.startswith(f"{sbj}_"))
            )
            if matching_file:
                assert k not in preds, f"{fd_preds}: {k} not found"
                with open(fp, 'r', encoding="utf8") as f:
                    preds[k] = json.load(f)

    scores = get_scores(golds, preds, compute_comet=compute_comet, batch_size=_bs_, task=_task_)
    fd_evals_basedir.mkdir(exist_ok=True)
    with open(fd_evals_basedir / f"{fd_preds.name}.json", 'w', encoding="utf8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

