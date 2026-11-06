# Evaluation metrics for ASR
# https://huggingface.co/learn/audio-course/en/chapter5/evaluation

import json
from evaluate import load
import pandas as pd

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
    "urban_energy_systems_and_policy": "3WXWESx6jWY",
}

wer_metric = load("wer")
cer_metric = load("cer")

# ASR transcript
asr_json = "/large/datas/frederikus-h/lectrans-testset/25.11.transcript_translate/transcript_asr_output_by_sbj_all_text.json"
# corrected transcript
annotated_json = "/large/datas/frederikus-h/lectrans-testset/25.11.transcript_translate/transcript_translate_annotated_by_sbj_all_text.json"

with open(asr_json, "r", encoding="utf-8") as asr_file:
    asr_data = json.load(asr_file)

with open(annotated_json, "r", encoding="utf-8") as annot_file:
    annot_data = json.load(annot_file)

reference_all = []
prediction_all = []

reference_subject = {}
prediction_subject = {}

for full_key in asr_data:
    if full_key not in annot_data:
        continue
    subject = full_key.split("/")[0]

    if subject not in prediction_subject:
        prediction_subject[subject] = []
        reference_subject[subject] = []

    for timestamp in asr_data[full_key]:
        if timestamp not in annot_data[full_key]:
            continue

        pred = asr_data[full_key][timestamp].get("asr", "").strip()
        ref = annot_data[full_key][timestamp].get("en", "").strip()

        if pred == "" or ref == "":
            continue

        reference_all.append(ref)
        prediction_all.append(pred)

        reference_subject[subject].append(ref)
        prediction_subject[subject].append(pred)

# wer = wer_metric.compute(references=[reference], predictions=[prediction])

'''overall_wer = wer_metric.compute(references=reference_all, predictions=prediction_all)
overall_cer = cer_metric.compute(references=reference_all, predictions=prediction_all)

print(overall_wer)
print(overall_cer)'''

# calculate subject-wise results
rows = []
for subject in prediction_subject:
    ref = reference_subject[subject]
    pred = prediction_subject[subject]

    wer = wer_metric.compute(references=ref, predictions=pred)
    cer = cer_metric.compute(references=ref, predictions=pred)
    rows.append({
        "subject": subject,
        "CER": round(cer, 2),
        "WER": round(wer, 2),
    })

subject_df = pd.DataFrame(rows)
print(subject_df)