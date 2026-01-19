import json
from pathlib import Path

import matplotlib.pyplot as plt

plt.style.use(
    ["grid", "science", "notebook", "mylegend", "./mplstyles/presentation.mplstyle"],
)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


with Path("plots/SAE_stats.json").open() as f:
    activation_counts = json.load(f)["activation_counts"]


# fig, ax = plt.subplots(1, 1)
#
# # Histogram of activation counts
# ax.hist(activation_counts, bins=50, edgecolor="black")
# ax.set_xlabel("Number of Samples Activated")
# ax.set_ylabel("Number of Latents")
# ax.set_title("Distribution of Latent Activation Frequencies")
# ax.set_yscale("log")
#
# fig.tight_layout()

dead_latents = activation_counts.count(0)
print(
    f"Dead latents: {dead_latents}/{len(activation_counts)} "
    f"({dead_latents / len(activation_counts):.2%})",
)

fig, ax = plt.subplots(1, 1)

# Sorted activation counts
sorted_counts = sorted(activation_counts, reverse=True)
ax.plot(sorted_counts)
# ax.axvline(len(activation_counts) - dead_latents, ls="--", linewidth=1)
ax.axvspan(
    xmin=len(activation_counts) - dead_latents,
    xmax=len(activation_counts),
    color=colors[1],
    alpha=0.6,
    linewidth=0,
)
ax.set_xlabel("Latent Index (sorted)")
ax.set_ylabel("Number of Samples Activated")
ax.set_xscale("log")
ax.set_title("Latent Activation Counts (Sorted)")

fig.tight_layout()
fig.savefig("plots/latent_activations.png", transparent=True, dpi=200)
plt.show()
