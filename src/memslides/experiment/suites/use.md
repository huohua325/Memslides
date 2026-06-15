# Experiment Suite Usage

The public package includes one built-in smoke suite, `smoke_minimal`, so users
can verify the experiment runner without sorting through internal benchmark
configs. Run it from the repository root after configuring a ready local Service
Profile or a private config selected with `MEMSLIDES_CONFIG_FILE`.

```bash
conda activate memslides

python -u -m memslides.experiment run smoke_minimal \
  --output-base .memslides/experiments \
  --parallel 1
```

The suite creates a single one-slide deck with web research, image generation,
and memory injection disabled. Do not commit private model paths, API keys, logs,
or generated experiment outputs.
