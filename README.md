# DataRubrics

<p align="left">
  <a href="https://arxiv.org/abs/2506.01789">
    <img src="https://img.shields.io/badge/arXiv-b31b1b.svg?style=flat&logo=arxiv&logoColor=white" alt="arXiv"/>
  </a>
  <a href="https://datarubrics.github.io">
    <img src="https://img.shields.io/badge/🌐-Website-blue.svg" alt="Website"/>
  </a>
  <a href="https://github.com/datarubrics/datarubrics/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/rubricreward/r3?color=blue" alt="License"/>
  </a>
</p>

<p align="center">
  <img src="./assets/data_rubrics_logo_orange.png" alt="DataRubrics Logo" width="22%"/>
</p>

<p align="center">
  <img src="./assets/aspects.png" alt="DataRubrics Dimensions" width="80%"/>
</p>

**TL;DR**: **DataRubrics** is a structured framework for assessing the quality of both human- and model-generated datasets. Leveraging recent advances in LLM-based evaluation.

## 📦 Contents

+ [🤔 Why DataRubrics?](#-why-datarubrics)
+ [⚙️ Setup Instructions](#-setup-instruction)
+ [🚀 Use DataRubrics](#-use-datarubrics)
+ [📚 Citation](#-citation)

## 🤔 Why DataRubrics?

We advocate for a more systematic approach to evaluating datasets—moving beyond datasheets and checklists, which are often neither easily decomposable into qualitative and quantitative metrics nor particularly useful for conference paper reviewers. 

## ⚙️ Setup Instructions

For performing OCR, do `pip install -r requirements_olmocr.txt`.

On the other hand, for performing rubric generation, do `pip install -r requirements_inference.txt`.

## 🚀 Use DataRubrics

Below are the instructions to run benchmark classifier, OCR, and inference for rubric generation.

### Running Benchmark Classifier

You may want to look at `src/data_generation/novel_benchmark_classifier.py` for the arguments. Use `data/configs/r3_qwen3_14b_4k.json` as your `model_config` to reproduce our result. 

### Running OCR

1. Check `src/data_generation/serve_vllm.sh` and change the model paths accordingly
2. Run `bash serve_vllm.sh`
3. Change the VLLM ports according to the serve script
4. Run `python3 ocr_ray.py` using the appropriate arguments. The CSV path (`--selected_csv_path`) should be the appropriate conference to be extracted. For example, it can be `data/csv/sampled_filtered_year_neurips_papers.csv`.

**Note:** The CSVs that you use to reproduce our results are `sampled_filtered_year_*_papers.csv`. However, you can immediately use the OCR results by unzipping `data/ocr_results/sampled_conference.zip`.

### Running inference

After running the OCR or unzipping `data/ocr_results/sampled_conference.zip`, you may perform inference for rubric generation with different methods.

#### Inference without serving

1. Simply run `python3 -m src.inference.inference` with appropriate arguments.

Some important arguments:
- `--input_folder` should be the folder path where OCR results are stored (e.g. from unzipping `data/ocr_results/sampled_conference.zip`).
- `--model_config` can follow from one of the JSONs at `data/configs`.
- `--output_file` should be the JSON output file path name.

For example, to reproduce GPT-4.1 mini results at `data/rubric_gen_results`, you can do `python3 -m src.inference.inference --model_config data/configs/gpt41mini.json --input_folder data/ocr_results/sampled_filtered_year_neurips_conference --output_file data/rubric_gen_results/gpt41mini_result_neurips.json`.

#### Serving the infer server - VLLM

1. Modify the serving script `src/inference/serve_infer_vllm.sh` to have the right TP size, model path, etc.
2. Serve the vllm server with `bash serve_infer_vllm.sh`
3. Look into `src/inference/ray_models.py`, `_initialize_sglang_urls()`, and ensure that the API endpoints match up (port number and base address) with the serve script
4. Run inference with `python3 -m src.inference.ray_inference`

Some important arguments:
- `--input_folder` should be the folder path where OCR results are stored (e.g. from unzipping `data/ocr_results/sampled_conference.zip`).
- `--model_config` can follow from  `data/configs/qwen3_32b_served_config.json` by changing the appropriate arguments.
- `--output_file` should be the JSON output file path name.

#### Serving the infer server - SGLANG (Recommended)

Basically the same steps as above, but instead of running `bash serve_infer_vllm.sh`, run `bash serve_infer_sglang.sh`

Remember to double check on `src/inference/ray_models.py`, `_initialize_sglang_urls()`, to ensure that the ports line up. 

## 📚 Citation

If you found our work helpful, please cite our work using the following citation!

```bibtex
@article{winata2025datasheets,
  title={Datasheets Aren’t Enough: DataRubrics for Automated Quality Metrics and Accountability},
  author={Winata, Genta Indra and Anugraha, David and Liu, Emmy and Aji, Alham Fikri and Hung, Shou-Yi, Parashar, Aditya and Irawan, Patrick Amadeus and Zhang, Ruochen and Yong, Zheng-Xin and Cruz, Jan Christian Blaise and Muennighoff, Niklas and Kim, Seungone and Zhao, Hanyang and Kar, Sudipta and Suryoraharjo, Kezia Erina and Adilazuarda, M. Farid and Lee, En-Shiun Annie and Purwarianti, Ayu and Wijaya, Derry Tanti and Choudhury, Monojit},
  journal={arXiv preprint arXiv:2506.01789},
  year={2025}
}
```

If you have any questions, you can open a [GitHub Issue](https://github.com/datarubrics/datarubrics/issues)!
