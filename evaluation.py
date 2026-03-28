import os
import warnings
warnings.filterwarnings("ignore")
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib
 
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import preprocessor, X_train, X_test, y_train, y_test

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_or_warn(path):
    if os.path.exists(path):
        return joblib.load(path)
    raise FileNotFoundError(
        f"Model file '{path}' not found.\n"
    )

lr_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
 
models = {
    "Linear Regression": lr_pipeline,
    "Ridge": load_or_warn("ridge.pkl"),
    "Lasso": load_or_warn("lasso.pkl"),
    "Decision Tree": load_or_warn("decision_tree.pkl"),
    "KNN": load_or_warn("knn.pkl"),
    "Random Forest": load_or_warn("random_forest.pkl"),
}


# Evaluate model
results = {}
predictions = {}
 
for name, model in models.items():
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MAE": mae, "RMSE": rmse, "R²": r2}
    print(f"  {name:<22}  MAE=${mae:>10,.0f}  RMSE=${rmse:>10,.0f}  R²={r2:.4f}")
 
results_df = (
    pd.DataFrame(results).T
    .sort_values("RMSE")
    .rename_axis("Model")
    .reset_index()
)
print()

# Helper: extract feature names from any pipeline in this project

def get_feature_names(pipeline):
    pre = pipeline.named_steps["preprocess"]
    num_feats = list(pre.transformers_[0][2])          
    ohe = pre.transformers_[1][1]                 
    cat_feats = ohe.get_feature_names_out(
        ["City", "Property Type"]
    ).tolist()
    return num_feats + cat_feats


# Comparison table
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.axis("off")
 
disp = results_df.copy()
disp["MAE"] = disp["MAE"].apply(lambda v: f"${v:,.0f}")
disp["RMSE"] = disp["RMSE"].apply(lambda v: f"${v:,.0f}")
disp["R²"] = disp["R²"].apply(lambda v: f"{v:.4f}")
 
row_colors = [
    ["#D4EDDA" if i == 0 else ("#EAF2FF" if i % 2 == 1 else "#FFFFFF")] * 4
    for i in range(len(disp))
]
table = ax.table(
    cellText = disp.values,
    colLabels = disp.columns.tolist(),
    cellLoc = "center",
    loc = "center",
    cellColours = row_colors,
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.3, 2.1)
for j in range(len(disp.columns)):
    table[(0, j)].set_facecolor("#2C5F9E")
    table[(0, j)].set_text_props(color="white", fontweight="bold")
 
ax.set_title(
    "Model Performance Comparison  (↑ green = best RMSE)",
    fontsize=13, fontweight="bold", pad=10
)
plt.tight_layout()
path = f"{PLOTS_DIR}/comparison_table.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()


# Metric bar charts (MAE / RMSE / R^2)

model_order = results_df["Model"].tolist()   # already sorted by RMSE
bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(model_order))]
 
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Model Metric Comparison (sorted by RMSE)", fontsize=14, fontweight="bold")
 
for ax, metric, ylabel in zip(
    axes,
    ["MAE", "RMSE", "R²"],
    ["Mean Absolute Error ($)", "Root Mean Squared Error ($)", "R² Score"],
):
    vals = [results[m][metric] for m in model_order]
    bars = ax.bar(model_order, vals, color=bar_colors, edgecolor="white", linewidth=0.6)
    ax.set_title(ylabel, fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order, rotation=30, ha="right", fontsize=9)
    if metric in ("MAE", "RMSE"):
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
        ax.bar_label(bars, labels=[f"${v/1e3:.0f}k" for v in vals],
                     padding=3, fontsize=8)
    else:
        ax.set_ylim(0, 1.08)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
 
plt.tight_layout()
path = f"{PLOTS_DIR}/metric_bars.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
 
# Actual vs. Predicted (2 × 3 grid)

rng = np.random.default_rng(42)
y_test_arr = np.array(y_test)
sample_idx = rng.choice(len(y_test_arr), size=min(2000, len(y_test_arr)), replace=False)
y_samp = y_test_arr[sample_idx]
PRICE_LIM = (0, 3_000_000)
 
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Actual vs. Predicted House Prices", fontsize=15, fontweight="bold", y=1.01)
 
for ax, (name, color) in zip(axes.flat, zip(model_order, PALETTE)):
    y_pred_samp = predictions[name][sample_idx]
    ax.scatter(y_samp, y_pred_samp, alpha=0.3, s=10, color=color, rasterized=True)
    ax.plot(PRICE_LIM, PRICE_LIM, "k--", linewidth=1.2, label="Perfect fit")
    r2 = results[name]["R²"]
    mae = results[name]["MAE"]
    ax.set_title(f"{name}\nR²={r2:.4f}   MAE=${mae:,.0f}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Actual Price", fontsize=9)
    ax.set_ylabel("Predicted Price", fontsize=9)
    ax.set_xlim(*PRICE_LIM)
    ax.set_ylim(*PRICE_LIM)
    fmt = mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.legend(fontsize=8, loc="upper left")
 
plt.tight_layout()
path = f"{PLOTS_DIR}/actual_vs_predicted.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()

# Residual distributions (2 × 3 grid)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Residual (Error) Distributions", fontsize=15, fontweight="bold", y=1.01)
 
for ax, (name, color) in zip(axes.flat, zip(model_order, PALETTE)):
    residuals = y_test_arr - predictions[name]
    clip_val = np.percentile(np.abs(residuals), 99)
    res_clip = np.clip(residuals, -clip_val, clip_val)
 
    ax.hist(res_clip, bins=60, color=color, edgecolor="white",
            linewidth=0.4, density=True)
    ax.axvline(0,                  color="black", linestyle="--", linewidth=1.2, label="Zero error")
    ax.axvline(np.mean(residuals), color="red",   linestyle="-",  linewidth=1.2,
               label=f"Mean: ${np.mean(residuals):,.0f}")
    ax.set_title(name, fontsize=10, fontweight="bold")
    ax.set_xlabel("Residual ($)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
    ax.legend(fontsize=8)
 
plt.tight_layout()
path = f"{PLOTS_DIR}/residual_distributions.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()

# Feature importances (Random Forest & Decision Tree)

TOP_N = 20
 
for model_name, pkl_color in [("Random Forest", PALETTE[4]), ("Decision Tree", PALETTE[3])]:
    pipeline = models[model_name]
    feature_names = get_feature_names(pipeline)
    importances = pipeline.named_steps["model"].feature_importances_
 
    top_idx = np.argsort(importances)[-TOP_N:]
    top_imp = importances[top_idx]
    top_feats = [feature_names[i] for i in top_idx]
 
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top_feats, top_imp, color=pkl_color, edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlim(0, top_imp.max() * 1.20)
    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)", fontsize=11)
    ax.set_title(f"{model_name} – Top {TOP_N} Feature Importances",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    slug = model_name.lower().replace(" ", "_")
    path = f"{PLOTS_DIR}/feature_importance_{slug}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
 
# Console summary

for _, row in results_df.iterrows():
    r = results[row["Model"]]
    print(f"  {row['Model']:<22}  MAE=${r['MAE']:>10,.0f} "
          f"RMSE=${r['RMSE']:>10,.0f}  R²={r['R²']:.4f}")
 
best = results_df.iloc[0]["Model"]
print(f"\n  Best model: {best}  "
      f"(RMSE=${results[best]['RMSE']:,.0f}, R²={results[best]['R²']:.4f})")