# Graph Embedding Extraction with Forest Fire Sampling

## ✅ Final Implementation: Option 1 (Sampling at 7000 nodes)

### Key Features:
- **Forest Fire Sampling** for graphs >7000 nodes (~2% of graphs)
- **Checkpointing** every 10 hashtags (can resume with --resume)
- **Aggressive memory management** (prevents slowdown)
- **Sampling metadata** tracked in features

### Speed & Coverage:
- **Runtime:** ~4-5 hours for 537 hashtags
- **98% of graphs:** No sampling needed
- **2% of graphs:** Sampled to 7000 nodes (preserves structure)
- **0% skipped:** Every window gets embeddings!

### Usage:

**Test on 5 hashtags:**
```bash
python extract_embedding_features_with_sampling.py \
  --config 04_ml_prediction/01_features/configs/embedding_config.yaml \
  --labels 04_ml_prediction/02_labels/labels_h550_0.5rel.parquet \
  --limit 5 \
  --verbose
```

**Full extraction:**
```bash
python extract_embedding_features_with_sampling.py \
  --config 04_ml_prediction/01_features/configs/embedding_config.yaml \
  --labels 04_ml_prediction/02_labels/labels_h550_0.5rel.parquet
```

**Resume if stopped:**
```bash
python extract_embedding_features_with_sampling.py \
  --config 04_ml_prediction/01_features/configs/embedding_config.yaml \
  --labels 04_ml_prediction/02_labels/labels_h550_0.5rel.parquet \
  --resume
```

**Custom sampling threshold:**
```bash
# More conservative (sample at 10k nodes)
python extract_embedding_features_with_sampling.py \
  --config ... \
  --sample-threshold 10000

# More aggressive (sample at 5k nodes, faster)
python extract_embedding_features_with_sampling.py \
  --config ... \
  --sample-threshold 5000
```

### Output Features:

All windows get these features:
```
emb_graph_n_nodes              # Current graph size (after sampling if applicable)
emb_graph_original_n_nodes     # Original size before sampling
emb_graph_was_sampled          # Binary: 1 if sampled, 0 otherwise
emb_mean_dim0 ... emb_mean_dim63
emb_std_dim0 ... emb_std_dim63
emb_mean_global, emb_std_global
emb_skew_global, emb_kurt_global
emb_mean_norm, emb_std_norm
emb_sv1, emb_sv2, emb_sv3
... (and _delta versions of all numeric features)
```

### Academic Justification:

> "To balance computational efficiency with representation quality, we applied
> forest fire sampling (Leskovec et al., 2006) to the 2% of graphs exceeding
> 7,000 nodes, reducing them to 7,000 nodes. This sampling method preserves
> local community structure and network topology while enabling tractable
> embedding generation. We tracked sampling status as a feature to ensure
> model transparency."

Reference: Leskovec, J., & Faloutsos, C. (2006). Sampling from large graphs. KDD.

### How Forest Fire Sampling Works:

1. **Start** with random seed node
2. **Expand** probabilistically to neighbors (geometric distribution)
3. **Preserve** local community structure
4. **Continue** until target size reached
5. **Result** representative subgraph maintaining topology

### Why This Solves Your Concern:

❌ **Old approach (skip):**
- Large graphs → NaN embeddings → model learns "NaN = burst"
- Creates confounding between size and virality

✅ **New approach (sample):**
- Large graphs → sampled → embeddings generated ✓
- Model gets signal from ALL windows
- Sampling status tracked explicitly (emb_graph_was_sampled)
- No missing data artifacts

### Monitoring Sampling:

After extraction, check how many graphs were sampled:
```python
import pandas as pd
df = pd.read_parquet('embedding_features_h550.parquet')

print(f"Total windows: {len(df)}")
print(f"Sampled windows: {df['emb_graph_was_sampled'].sum()}")
print(f"Sampling rate: {df['emb_graph_was_sampled'].mean()*100:.1f}%")
print(f"\nOriginal size distribution (sampled graphs):")
print(df[df['emb_graph_was_sampled']==1]['emb_graph_original_n_nodes'].describe())
```

### Next Steps:

1. ✅ Run test extraction (5 hashtags)
2. ✅ Verify sampling working correctly
3. ✅ Run overnight for full dataset
4. ✅ Integrate with build_samples.py
5. ✅ Run ablation experiments