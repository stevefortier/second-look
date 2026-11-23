# LoRA adapters from idea-bundle runs

Each folder holds one adapter archive split into parts, because GitHub limits files to 100 MB.
Results for a run are in `studies/ideas/<run>/RESULTS.md` on `main`.

```sh
cd 003-birds                      # any run folder
cat part-* > ideas-003-birds-lora.zip
sha256sum -c SHA256SUMS
unzip ideas-003-birds-lora.zip    # a PEFT adapter directory
```
