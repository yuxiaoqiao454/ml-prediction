# Complete Workflow: Parallel Embedding Extraction

## 🚀 **Recommended Approach: Extract Separately + Merge**

**Time savings: 13.9 hours instead of 27.8 hours!**

---

## 📋 **Step-by-Step Guide:**

### **Step 1: Run Both Extractions in Parallel**

Open **TWO terminals** and run simultaneously:

#### **Terminal 1: Audience Projection**
```bash
python 04_ml_prediction/01_features/scripts/extract_embedding_features.py \
  --config 04_ml_prediction/01_features/configs/embedding_audience.yaml \
  --labels 04_ml_prediction/02_labels/labels_h550_0.5rel.parquet \
  --prefix emb_aud \
  --resume
```

#### **Terminal 2: Bipartite**
```bash
python 04_ml_prediction/01_features/scripts/extract_embedding_features.py \
  --config 04_ml_prediction/01_features/configs/embedding_bipartite.yaml \
  --labels 04_ml_prediction/02_labels/labels_h550_0.5rel.parquet \
  --prefix emb_bip \
  --resume
```

**Expected time: ~14 hours each (but running in parallel!)**

---

### **Step 2: Merge the Two Parquets**

After both finish:

```bash
python merge_embeddings.py \
  --audience 04_ml_prediction/01_features/outputs/embedding_features_h550_audience.parquet \
  --bipartite 04_ml_prediction/01_features/outputs/embedding_features_h550_bipartite.parquet \
  --output 04_ml_prediction/01_features/outputs/embedding_features_h550_combined.parquet \
  --verbose
```

**Expected time: ~2 minutes**

**Output:**
```
================================================================================
Merge Statistics
================================================================================
Total windows: 16679
  Both embeddings: 15234 (91.3%)
  Audience only: 892 (5.3%)
  Bipartite only: 553 (3.3%)

Total features: 558
  Audience: 279
  Bipartite: 279

✓ Saved merged embeddings to embedding_features_h550_combined.parquet
  Size: 87.3 MB
```

---

### **Step 3: Build Dataset**

```yaml
# dataset_config.yaml
input:
  embedding_features: "04_ml_prediction/01_features/outputs/embedding_features_h550_combined.parquet"

features:
  use_embeddings: true
  
output:
  dataset_version: "h550_0.5rel_emb_both"
```

```bash
python build_samples.py --config dataset_config.yaml
```

---

### **Step 4: Fix train.py**

```python
# In prepare_features():
feature_cols = [c for c in train_df.columns 
               if c.startswith('ts_') or c.startswith('net_') 
               or c.startswith('emb_aud_') or c.startswith('emb_bip_')]
```

---

### **Step 5: Train & Compare!**

```bash
# Train with both embedding types
python train.py \
  --config random_forest.yaml \
  --model random_forest_default \
  --dataset-version h550_0.5rel_emb_both
```

**Check feature importance:**
```
feature,importance
ts_mentions_mean,0.123
net_degree_mean,0.089
emb_aud_mean_global,0.067      # ✓ Audience embeddings
emb_aud_sv1,0.054              # ✓
emb_bip_mean_global,0.048      # ✓ Bipartite embeddings
emb_bip_sv1,0.041              # ✓
...
```

---

## 🎯 **Feature Naming Convention:**

### **Audience Projection (emb_aud_*):**
```
emb_aud_graph_n_nodes
emb_aud_mean_dim0 ... emb_aud_mean_dim63
emb_aud_mean_global
emb_aud_sv1, emb_aud_sv2, emb_aud_sv3
emb_aud_mean_global_delta
...
```

### **Bipartite (emb_bip_*):**
```
emb_bip_graph_n_nodes
emb_bip_mean_dim0 ... emb_bip_mean_dim63
emb_bip_mean_global
emb_bip_sv1, emb_bip_sv2, emb_bip_sv3
emb_bip_mean_global_delta
...
```

**Total: ~560 embedding features!**

---

## 💡 **Why This Approach is Better:**

### **1. Speed:**
- ⚡ Run in parallel → **50% time savings**
- 14 hours instead of 28 hours

### **2. Resilience:**
- ✅ If audience extraction fails at 90%, bipartite is safe
- ✅ Can use `--resume` independently
- ✅ Don't lose all progress if one crashes

### **3. Flexibility:**
- 🔄 Can re-run just one graph type
- 📊 Can compare: audience-only vs bipartite-only vs both
- 🧪 Easy to experiment with different configs

### **4. Clarity:**
- 📝 Clear separation: emb_aud_* vs emb_bip_*
- 🔍 Easy to identify which features come from which graph
- 📈 Can analyze feature importance separately

---

## 📊 **Ablation Study:**

After extraction, you can easily compare:

### **Experiment 1: Baseline**
```yaml
use_embeddings: false
dataset_version: "h550_0.5rel"
```

### **Experiment 2: Audience Only**
```yaml
use_embeddings: true
embedding_features: "embedding_features_h550_audience.parquet"
dataset_version: "h550_0.5rel_emb_aud"
```

### **Experiment 3: Bipartite Only**
```yaml
use_embeddings: true
embedding_features: "embedding_features_h550_bipartite.parquet"
dataset_version: "h550_0.5rel_emb_bip"
```

### **Experiment 4: Both**
```yaml
use_embeddings: true
embedding_features: "embedding_features_h550_combined.parquet"
dataset_version: "h550_0.5rel_emb_both"
```

---

## 🐛 **Troubleshooting:**

**Q: One terminal crashes midway?**  
→ Just restart that terminal with `--resume` flag

**Q: Merge script says dimensions don't match?**  
→ Make sure both used same config (dim=64, walks=15, etc.)

**Q: Feature importance still shows only ts_* ?**  
→ Update train.py to include `emb_aud_` and `emb_bip_` prefixes

**Q: Can I run on different machines?**  
→ Yes! Run audience on machine A, bipartite on machine B, merge later

---

## ✅ **Summary:**

1. ✅ Run TWO extractions in parallel (different terminals)
2. ✅ Use prefixes: `emb_aud_*` and `emb_bip_*`
3. ✅ Merge with simple script (~2 min)
4. ✅ Update train.py to recognize both prefixes
5. ✅ Train and compare all 4 experiments!

**Not confusing at all - actually cleaner and much faster!** 🚀