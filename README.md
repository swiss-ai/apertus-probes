# apertus-probes

Code and notebooks for the hallucination (error prediction) probe project for DSL #16.

## Quickstart

Below are the basic steps to get up and running with this repository:

### 1. Optional: Copy environment configuration
If you need to use a custom configuration, copy the sample config:

```bash
cp probes.toml ~/.edf/probes.toml
```

### 2. Run the compute node with environment: 
```bash
  srun -A infra01 --environment=$HOME/.edf/probes.toml --pty bash
```
### 3. Load the Dataset

Currently, only the `sms_spam` dataset is supported. Run:

```bash
python load_datasets.py
```

This will download and cache the dataset in the expected format.

### 4. Run the Caching Script

To extract and cache model activations for the dataset, run:

```bash
bash cache.sh
```
> **Note:** If you get a permission error, make the script executable:
> ```bash
> chmod +x cache.sh
> ```

### 5. Post-process the Cached Data

Aggregate and organize the activations for probe training:

```bash
bash postprocess.sh
```

### 6. Train Probes

Train linear probes on the cached activations:

```bash
bash train_probes.sh
```

----


