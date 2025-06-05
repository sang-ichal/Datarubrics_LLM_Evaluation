# Rnuning OCR

## Serve and infer with vllm

1. Check `serve_vllm.sh` and change the model paths accordingly
2. Run `bash serve_vllm.sh`
3. At the top of the file `ocr_ray.py`, change the selected CSV file path
4. Change the VLLM ports according to the serve script
5. Run `python ocr_ray.py`