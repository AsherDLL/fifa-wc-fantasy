# Hugging Face dataset release checklist (post-final)

The dataset release is prepared by `scripts/export_hf_dataset.py` and
published manually AFTER the World Cup final (19 July 2026), at the same
time the repository goes public. Do not publish earlier.

## Steps

1. Wait for the final to complete and for the last daily snapshot to
   land (collector cron, or run it once more by hand):

   ```bash
   .venv/bin/python -m fifa_fantasy.collector
   .venv/bin/python -m fifa_fantasy.external
   ```

2. Stage the release directory:

   ```bash
   .venv/bin/python scripts/export_hf_dataset.py   # -> dist/hf-dataset/
   ```

3. Review before upload:
   - open `dist/hf-dataset/README.md`; confirm the YAML front matter and
     the CC-BY-4.0 / GPL-3.0 split reads correctly;
   - confirm nothing from `data/training/` (vaastav mirror), news or
     scraping caches, or third-party `data/external/` files was staged
     (the script only stages the five whitelisted groups; spot-check
     anyway);
   - no secrets or tokens anywhere (`grep -ri "api_key\|token" dist/`).

4. Upload (huggingface_hub is intentionally not a project dependency):

   ```bash
   pip install huggingface_hub
   hf auth login
   hf upload <account>/fifa-wc-2026-fantasy dist/hf-dataset \
       --repo-type dataset
   ```

5. Only flip the HF dataset repo to public once the GitHub repository is
   public, then cross-link: add the HF badge/link to the repo README and
   the GitHub URL is already in the dataset card.
