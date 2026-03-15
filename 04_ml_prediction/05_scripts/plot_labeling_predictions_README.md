# Labeling & Prediction Visualization Tool

## 📊 Quick Start

### Example 1: Single Rule + Single Model
```bash
python 04_ml_prediction/05_scripts/plot_labeling_predictions.py \
  --hashtag superbowl \
  --timeseries comments \
  --rules h550_0.5rel \
  --models exp_049_lightgbm_h550_0.5rel \
  --output 04_ml_prediction/06_plots/superbowl_single_relaxed.png \
  --relaxed
```

### Example 2: Compare Multiple Rules
```bash
python plot_labeling_predictions.py \
  --hashtag inmyfeelings \
  --timeseries comments \
  --rules h550_0.5rel h550_cpinside \
  --models exp_040_random_forest_h550_0.5rel \
  --output 04_ml_prediction/06_plots/inmyfeelings_rules.png
```

### Example 3: Compare Multiple Models
```bash
python 04_ml_prediction/05_scripts/plot_labeling_predictions.py \
  --hashtag 4thofjuly \
  --timeseries comments \
  --rules h550_0.5rel \
  --models exp_046_random_forest_h550_0.5rel exp_047_xgboost_h550_0.5rel \
  --output 04_ml_prediction/06_plots/4thofjuly_models.png
```

### Example 4: Full Comparison Matrix (3×3)
```bash
python 04_ml_prediction/05_scripts/plot_labeling_predictions.py \
  --hashtag christmaslights \
  --timeseries comments \
  --rules h550_cpinside h550_0.5rel \
  --models exp_065_xgboost_h550_cpinside exp_047_xgboost_h550_0.5rel \
  --output 04_ml_prediction/06_plots/christmaslights_full.png \
  --figsize 14 4 \
  --dpi 200
```

### Example 5: Labels Only (No Predictions)
```bash
python 04_ml_prediction/05_scripts/plot_labeling_predictions.py \
  --hashtag canada150 \
  --timeseries comments \
  --rules h550_cpinside h550_0.5rel h550_0.5rel_75abs \
  --output 04_ml_prediction/06_plots/canada150_labels.png
```

### Example 6: Auto-Discover All Available
```bash
# Will auto-discover all labeling rules and models
python plot_labeling_predictions.py \
  --hashtag valentinesday \
  --timeseries comments \
  --output plots/valentinesday_all.png
```

---

## 🎨 Visual Legend

### Label Plots (Column 1):
- **Black line**: Timeseries (raw counts)
- **Blue shading**: Burst window (label=1)
- **Gray shading**: No label available (NA)
- **Red ✗**: Change point detected
- **Black dashed lines**: Window boundaries

### Prediction Plots (Columns 2+):
- **Green shading**: True Positive (TP) - Correctly predicted burst
- **Red shading**: False Positive (FP) - False alarm
- **Yellow shading**: False Negative (FN) - Missed burst
- **No shading**: True Negative (TN) - Correctly predicted non-burst
- **Gray shading**: No prediction (NA)

---

## 📂 File Requirements

The script expects these files to exist:

1. **Timeseries CSVs:**
   - `02t_timeseries/csvs/hashtag_timeseries_mentions_norm_smooth.csv`
   - `02t_timeseries/csvs/hashtag_timeseries_comments_norm_smooth.csv`

2. **Window slices:**
   - `02t_timeseries/window_slices/{hashtag}.parquet`

3. **Labels:**
   - `04_ml_prediction/02_labels/labels_{rule}.parquet`

4. **Predictions:**
   - `04_ml_prediction/04_models/trained/{experiment_id}/test_predictions.csv`

---

## 🔧 Arguments

```
--hashtag       Hashtag to plot (required)
--timeseries    'mentions' or 'comments' (default: mentions)
--rules         List of labeling rules (auto-discover if omitted)
--models        List of experiment IDs (labels only if omitted)
--output        Output file path (show interactive if omitted)
--figsize       Subplot size in inches (default: 12 3)
--dpi           Output resolution (default: 150)
```

---

## 💡 Tips

1. **Start small**: Test with 1 rule + 1 model first
2. **Large matrices**: For 3×3 or bigger, use `--figsize 14 4` and `--dpi 200`
3. **Labels only**: Omit `--models` to just compare labeling rules
4. **Auto-discover**: Omit `--rules` to see all available rules
5. **High-res**: Use `--dpi 300` for publication-quality figures

---

## 🐛 Troubleshooting

**"No timeseries data found"**
→ Check hashtag spelling (case-insensitive)

**"Labels file not found"**
→ Verify labeling rule name matches file: `labels_{rule}.parquet`

**"Predictions file not found"**
→ Check experiment ID exactly matches folder name

**Plot too crowded**
→ Increase `--figsize` or reduce number of rules/models

**Text labels overlapping**
→ Script auto-adjusts based on space, but you can increase figure width

---

## 📸 Example Output

For a 2×3 matrix (2 rules × 2 models + 1 label column):

```
┌─────────────┬──────────────┬──────────────┐
│ Labels      │ Model A      │ Model B      │
│ Rule 1      │ Predictions  │ Predictions  │
├─────────────┼──────────────┼──────────────┤
│ Labels      │ Model A      │ Model B      │
│ Rule 2      │ Predictions  │ Predictions  │
└─────────────┴──────────────┴──────────────┘
```

Each subplot shows:
- Full timeseries (black line)
- Window boundaries (dashed lines)
- Color-coded shadings (blue/green/red/yellow/gray)
- Change points (red ✗ in label plots)
- Text annotations (Burst/TP/FP/FN/TN/NA)
- Legend (top-left corner)

---

## 🚀 Python API

You can also use it programmatically:

```python
from plot_labeling_predictions import plot_labeling_comparison

plot_labeling_comparison(
    hashtag='4thofjuly',
    timeseries_type='mentions',
    labeling_rules=['h550_0.5rel'],
    models=['exp_040_random_forest_h550_0.5rel'],
    output_path='plots/my_plot.png',
    figsize_per_subplot=(14, 4),
    dpi=200
)
```