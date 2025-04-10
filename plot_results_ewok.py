import json
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch

def load_group_accuracies(path, group_name):
    with open(path, 'r') as f:
        data = json.load(f)
    results = data["results"]
    subtasks = data["group_subtasks"][group_name]
    accs = {
        task: results[task]["acc,none"]
        for task in subtasks if task in results
    }
    return accs

def plot_accuracy_comparison(accs1, accs2, common_tasks, label1, label2, output_file, title_prefix):
    accs1_vals = [accs1[task] for task in common_tasks]
    accs2_vals = [accs2[task] for task in common_tasks]

    x = np.arange(len(common_tasks))
    width = 0.35

    light_green = '#88B04B'
    dark_green = '#006400'
    label2_colors = [light_green if accs2[task] > accs1[task] else dark_green for task in common_tasks]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, accs1_vals, width, label=label1, color='#1f77b4')
    ax.bar(x + width/2, accs2_vals, width, label=label2, color=label2_colors)

    ax.set_ylabel('Accuracy')

    count_light_green = label2_colors.count(light_green)
    count_dark_green = label2_colors.count(dark_green)
    total = len(label2_colors)

    if count_light_green > count_dark_green:
        winner_str = f"{label2} ↑: {count_light_green}/{total}  |  {label1} ↓: {count_dark_green}/{total}"
    elif count_light_green < count_dark_green:
        winner_str = f"{label1} ↑: {count_dark_green}/{total}  |  {label2} ↓: {count_light_green}/{total}"
    else:
        winner_str = f"{label2}: {count_light_green}/{total}  |  {label1}: {count_dark_green}/{total}"

    ax.set_title(f'{title_prefix} Accuracy Comparison — {winner_str}')

    ax.set_xticks(x)
    ax.set_xticklabels(
        [t.replace("ewok_", "").replace("_filtered", "") for t in common_tasks],
        rotation=90
    )

    legend_elements = [
        Patch(facecolor='#1f77b4', label=label1),
        Patch(facecolor=light_green, label=f"{label2} ↑"),
        Patch(facecolor=dark_green, label=f"{label2} ↓")
    ]
    ax.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        fontsize=10,
        frameon=False
    )

    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✅ Saved plot to: {output_file}")

def compare_accuracies(json1_path, json2_path, label1="Model 1", label2="Model 2", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True) 

    group_name = "ewok_filtered"
    title = "EWOK"

    accs1 = load_group_accuracies(json1_path, group_name)
    accs2 = load_group_accuracies(json2_path, group_name)
    common_tasks = sorted(set(accs1.keys()) & set(accs2.keys()))

    if not common_tasks:
        print(f"⚠️ No common tasks found for group: {group_name}")
        return

    # 🛠️ Same style as Blimp: ewok_128_15300_filtered.png
    filename = f"ewok_256_15300_{group_name.replace('ewok_', '')}.png"
    output_path = os.path.join(output_dir, filename)
    plot_accuracy_comparison(accs1, accs2, common_tasks, label1, label2, output_path, title.capitalize())

# ✅ Example usage
compare_accuracies(
    "results/ewok/ntp_lingua_causal_256_15300/ewok_results.json",
    "results/ewok/lingua_3fh_causal_256_15300/ewok_results.json",
    label1="NTP (dim=256)",
    label2="MTP (n_heads=3, dim=256)",
    output_dir="ewok_visualization"
)
