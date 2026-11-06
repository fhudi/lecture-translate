
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
 
## Usage

> [!NOTE]
> This section is under construction.
 
```bash
./6.transcripts_translation/runner.sh
```

 
 
## Project Structure

> [!NOTE]
> This section is under construction.
 
```
lectrans/
├── env-conda.*.yml   # Conda environments
├── 6.transcripts_translation/              # evaluating translation tasks
└── README.md
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
