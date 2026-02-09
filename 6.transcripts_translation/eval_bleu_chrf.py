from sacrebleu.metrics import BLEU, CHRF
import json
from pathlib import Path
from pythainlp.tokenize import word_tokenize
from constants import LANGUAGES

reference_json = Path("/large/datas/frederikus-h/lectrans-testset/25.11.transcript_translate/transcript_translate_annotated_by_sbj_all_text.json")
output_base_path = Path("/large/datas/nina-n/NL-ken/output_json")
result_base_path = Path("/large/datas/nina-n/NL-ken/bleu_chrf/results")

exp_types = [
    "nocontext",
    "slideimage",
    "slidetext",
    # "imageonly",
    # "slidetextonly"
]

models = [
    # "gpt-4.1-2025-04-14",
    # "o4-mini-2025-04-16",
    # "gpt-5-mini-2025-08-07",
    "gemma-3-12b-it",
    # "llama32"
]

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

languages = list(LANGUAGES.keys())

with open(reference_json, "r", encoding="utf-8") as json_path:
    reference_all = json.load(json_path)

def select_tokenizer(language):
    if language == "ja":
        return BLEU(tokenize="ja-mecab")
    elif language == "zh":
        return BLEU(tokenize="zh")
    else:
        return BLEU(tokenize="13a")

def tokenize_thai(text):
    return " ".join(word_tokenize(text, engine="newmm"))

chrf = CHRF()

def evaluate(model, exp_type, language):
    results = {}
    for subject, video_id in subjects.items():
        sys = []
        refs = []
        # mt_path = (output_base_path / exp_type / model / f"{subject}_{exp_type}_{model}.json")
        # mt_path = (output_base_path / exp_type / model / f"{subject}_{video_id}.en2any.{model}.json")
        
        # for gemma3
        mt_path = (output_base_path / exp_type / model / f"{subject}_{video_id}.en2any.gemma3it12b.json")
        if not mt_path.exists():
            continue
        with open(mt_path, "r", encoding="utf-8") as mt_file:
            mt_json = json.load(mt_file)

        ref_key = f"{subject}/{video_id}"
        if ref_key not in reference_all:
            continue

        ref_subject = reference_all[ref_key]

        for timestamp, mt_langs in mt_json.items():
            if language not in mt_langs:
                continue
            if timestamp not in ref_subject:
                continue
            if language not in ref_subject[timestamp]:
                continue

            hyp_text = mt_langs[language][0]
            ref_text = ref_subject[timestamp][language]

            if language == "th":
                hyp_text = tokenize_thai(hyp_text)
                ref_text = tokenize_thai(ref_text)

            sys.append(hyp_text)
            refs.append(ref_text)
    
        if len(sys) == 0:
            return None, None, 0, 0
    
        if language == "th":
            bleu = BLEU(tokenize="none")
        else:
            bleu = select_tokenizer(language)
        
        bleu_score = bleu.corpus_score(sys, [refs])
        chrf_score = chrf.corpus_score(sys, [refs])
    
        results[f"{subject}/{video_id}"] = (bleu_score, chrf_score, len(sys), len(refs))

    return results

for model in models:
    for exp_type in exp_types:
        '''print(f"Context: {exp_type}, Model: {model}")
        for language in languages:
            bleu_result, chrf_result, hyp_num, ref_num = evaluate(model, exp_type, language)
            if bleu_result is None:
                continue
            print(f"Language: {language}")
            print(bleu_result)
            print(chrf_result)
            print(f"Hyp_num: {hyp_num}, Ref_num: {ref_num}\n")
            # print()'''
        
        result_dict = {
            "model": model,
            "context": exp_type,
            "results": {}
        }

        for language in languages:
            # bleu_result, chrf_result, hyp_num, ref_num = evaluate(model, exp_type, language)
            # if bleu_result is None:
            #     continue

            results_per_subject = evaluate(model, exp_type, language)
            if not results_per_subject:
                continue

            bleu_result = sum(_[0].score for _ in results_per_subject.values()) / len(results_per_subject)
            chrf_result = sum(_[1].score for _ in results_per_subject.values()) / len(results_per_subject)

            result_dict["results"][language] = {
                "BLEU": str(bleu_result),
                "BLEU_score": bleu_result.score,
                "ChrF": str(chrf_result),
                "ChrF_score": chrf_result.score
            }

        output_json_path = result_base_path / f"{model}_{exp_type}.json"
        with open(output_json_path, "w", encoding="utf-8") as output_json:
            json.dump(result_dict, output_json, ensure_ascii=False, indent=2)
        
        print(f"Saved: {output_json_path}")
