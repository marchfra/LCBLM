import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.style.use(
    ["grid", "science", "notebook", "mylegend", "./mplstyles/presentation.mplstyle"],
)

with Path("plots/SAE-head-losses.json").open() as f:
    losses = json.load(f)

training_losses = losses["training_losses"]
validation_losses = losses["validation_losses"]
best_epoch = validation_losses.index(min(validation_losses))

fig, ax = plt.subplots()

ax.plot(range(1, len(training_losses) + 1), training_losses, label="Training")
ax.plot(range(1, len(validation_losses) + 1), validation_losses, label="Validation")
ax.scatter(best_epoch + 1, training_losses[best_epoch])
ax.scatter(best_epoch + 1, validation_losses[best_epoch])

ax.xaxis.set_major_locator(MaxNLocator("auto", integer=True))
ax.xaxis.set_minor_locator(MaxNLocator(len(training_losses) + 1, integer=True))

# Add best epoch as a minor tick with label
current_minor_ticks = list(ax.get_xticks(minor=True))
if best_epoch + 1 not in current_minor_ticks:
    current_minor_ticks.append(best_epoch + 1)
    current_minor_ticks.sort()
ax.set_xticks(current_minor_ticks, minor=True)

# Set label only for the best epoch tick
minor_labels = [
    str(best_epoch + 1) if x == best_epoch + 1 else "" for x in current_minor_ticks
]
ax.set_xticklabels(minor_labels, minor=True)

# Check if best epoch label conflicts with major tick labels
major_ticks = ax.get_xticks()

# Estimate label width based on number of digits
# Rough estimate: each digit is about 2% of the x-axis range
digit_width = 0.02
x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
best_label_width = len(str(int(best_epoch + 1))) * digit_width * x_range

has_conflict = False
for tick in major_ticks:
    tick_label_width = len(str(int(tick))) * digit_width * x_range
    min_distance = (best_label_width + tick_label_width) / 2
    if abs(best_epoch + 1 - tick) < min_distance:
        has_conflict = True
        break

# Only move label inside plot if there's a conflict
if has_conflict:
    ax.tick_params(axis="x", which="minor", pad=-20)

ax.set_xlabel("Epoch")
ax.set_ylabel("CE Loss")
ax.set_title("Learning Curves")
ax.legend()

fig.tight_layout()

fig.savefig("plots/learning-curves.png", transparent=True, dpi=200)
plt.show()
