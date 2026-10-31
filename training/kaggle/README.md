# Running the v0.7 data generation on Kaggle

Kaggle's free tier gives a GPU notebook (2× T4, 16 GB each) for about 30 hours a week once your
account is **phone-verified**.

## One-time setup
1. Create a Kaggle account and verify your phone number (Settings → Phone verification).
2. Zip the project code (without `.venv`, `data`, `models`):
   `git archive` isn't available here, so use the PowerShell line below. It writes `aegis-code.zip`.
   ```powershell
   Compress-Archive -Path aegisframe, reframebench, training, pyproject.toml -DestinationPath aegis-code.zip -Force
   ```
3. On Kaggle: **Datasets → New dataset**, upload `aegis-code.zip`, and name it `aegis-code` (private).

## Each run
New notebook → Settings: **Accelerator: GPU T4 ×2**, **Internet: on**. Add the `aegis-code` dataset. Then:

```python
!pip install -q "vllm" pydantic pyyaml pillow numpy httpx
!cp -r /kaggle/input/aegis-code/* /kaggle/working/ && cd /kaggle/working

# 1) a 20-group smoke run: check refusal rate and quality before spending GPU hours
!cd /kaggle/working && python -m reframebench.datagen_v2.generate --groups 20 --backend vllm \
    --model <TEACHER> --vllm-kwargs '{"tensor_parallel_size": 2, "max_model_len": 8192}' \
    --out /kaggle/working/aegis_v2_smoke --judge

# 2) the full run, about 12,000 groups, about 22,000 samples after rejects; resumable
!cd /kaggle/working && python -m reframebench.datagen_v2.generate --groups 12000 --seed 7 --backend vllm \
    --model <TEACHER> --vllm-kwargs '{"tensor_parallel_size": 2, "max_model_len": 8192}' \
    --batch 256 --out /kaggle/working/aegis_v2 --judge
```
Download `/kaggle/working/aegis_v2/` (records, rejects, raw) from the notebook's Output tab into
`data/aegis_v2/` locally.

**Choosing `<TEACHER>`.** Use an open-weight instruct model with a permissive licence (Apache-2.0 or MIT)
that fits 2× T4 (32 GB total), e.g. an AWQ / GPTQ 4-bit build of a 14B–32B model. Check the exact repo
name, licence and size on its Hugging Face card at run time.

For the out-of-distribution split, run a smaller second pass (`--groups 1500 --seed 99`) with a teacher
from a **different model family**, then `finalize(out, ood_teachers=[...])`.

## After generation (locally)
```python
from reframebench.datagen_v2.generate import finalize
finalize("data/aegis_v2", ood_teachers=["<second teacher name>"])
```
This removes near-duplicates and writes `dataset.jsonl` with train / val / test_id / test_ood splits.
