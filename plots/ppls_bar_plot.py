import string

import matplotlib.pyplot as plt

plt.style.use(
    [
        "grid",
        "science",
        "notebook",
        "mylegend",
        "./mplstyles/presentation.mplstyle",
    ],
)

plt.rcParams["xtick.minor.visible"] = False

ppls = {
    "Finetuned head": 32.59,
    "Original head": 38.56,
    "SAE encoder + new head": 42.27,
    "Retrained head": 106.65,
    "SAE recon + original head": 243.18,
    "LoRA*": 84.7,
    "CB-LLM*": 116.22,
}
letters = list(string.ascii_lowercase)
x_labels = [f"({letters[i]})" for i in range(len(ppls))]

fig, ax = plt.subplots(1, 1)
ax.bar(
    x=range(len(ppls)),
    height=list(ppls.values()),
    tick_label=x_labels,
)
ax.bar(
    x=list(ppls.keys()).index("SAE encoder + new head"),
    height=ppls["SAE encoder + new head"],
)
ax.set_ylabel("Perplexity")

legend_parts = [f"({letters[i]}) {name}" for i, name in enumerate(ppls.keys())]
items_per_line = (len(legend_parts) + 1) // 3  # Distribute evenly across 3 lines

line1 = "                   ".join(legend_parts[:items_per_line])
line2 = "    ".join(legend_parts[items_per_line : 2 * items_per_line])
line3 = "   ".join(legend_parts[2 * items_per_line :])

legend_text = f"{line1}\n{line2}\n{line3}"
plt.text(
    x=0.1,
    y=-0.25,
    s=legend_text,
    ha="left",
    fontsize=16,
    transform=plt.gca().transAxes,
)

fig.tight_layout()
fig.savefig("plots/ppls.png", transparent=True, dpi=200)

plt.show()
