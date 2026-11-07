
# Lecture Translation

## Requirements
 
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)
- Python (version specified in `environment.*.yml`)


## Installation
 
Create and activate the conda environment:
 
```bash
conda env create -f env-conda.*.yml
conda activate *
```

 
 
## Project Structure

> [!NOTE]
> This section is under construction.
 
```
lectrans/
├── env-conda.*.yml   # Conda environments
├── 1.ytdlp_and_dataset_split/     # Data sourcing and splits (§3.1, §3.3.1, §3.5 of the manuscript).
├── 2.pyscenedetect_stats/         # Scene detection (§3.3.2, Appendix E)
├── 3.transcripts_extraction/      # Transcript ASR (§3.3.1, §3.3.4, §3.4.1)
├── 4.slides_extraction/           # Slide OCR (§3.3.2, §3.4.2, Appendix J)
├── 5.lectrans-testset/            # The benchmark test dataset (§3.3.3, §3.3.3, §3.4, §3.5, Appendix G, Appendix H)
├── 6.translation_evaluation/      # Experiments (§4–6)
└── README.md    # This file
```
 
## License

The **LecTrans** benchmark dataset is released under the **Creative Commons Attribution–NonCommercial–ShareAlike 4.0** (CC BY-NC-SA 4.0).
Please also refer to paper Ethical Considerations section.


## How to Cite

```bibtex
@inproceedings{hudi-etal-2026-lectrans,
    title = "{L}ec{T}rans: A Multimodal Translation Benchmark for Academic Lectures",
    author = "Hudi, Frederikus  and
      Nanakorn, Nina  and
      Vasselli, Justin  and
      Sakai, Yusuke  and
      Kamigaito, Hidetaka  and
      Tamura, Akihiro  and
      Watanabe, Taro",
    editor = "Zhang, Yue  and
      Xu, Ruifeng  and
      Purwarianti, Ayu  and
      Rong, Tong  and
      Suzuki, Jun",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {IJCNLP}-{AACL} 2026",
    month = nov,
    year = "2026",
    address = "Hengqin, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-aacl.93/",
    pages = "1872--1897",
    ISBN = "979-8-89176-444-6",
}
```
