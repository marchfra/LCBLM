import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from trainvox import edit_telegram_media, send_telegram_photo


def plot_learning_curves(  # noqa: PLR0913
    training_losses: list[float],
    validation_losses: list[float] | None,
    best_epoch: int | None,
    tg_token: str | None = None,
    tg_chat_id: str | None = None,
    msg_id: int | None = None,
) -> None:
    """Plot and optionally send learning curves via Telegram.

    Args:
        training_losses: list of training losses. Every entry corresponds to one epoch
        validation_losses: list of validation losses. Every entry corresponds to one
            epoch
        best_epoch: the epoch with the lowest validation loss. Can even be an epoch that
            you want to highlight for whatever reason. If None and validation_losses is
            provided, will automatically be calculated from validation_losses
        tg_token: the token of the Telegram bot
        tg_chat_id: the unique identifier for the target chat
        msg_id: the id of the Telegram message to edit. If not supplied will send a new
            message

    Raises:
        ValueError: if training_losses and validation_losses have different lengths

    """
    if validation_losses is not None and len(training_losses) != len(validation_losses):
        raise ValueError(
            "training_losses and validation_losses must have the same length",
        )

    # Try to infer best epoch
    if best_epoch is None and validation_losses is not None:
        best_epoch = validation_losses.index(min(validation_losses))

    fig, ax = plt.subplots()

    ax.plot(range(1, len(training_losses) + 1), training_losses, label="Training")
    if validation_losses is not None:
        ax.plot(
            range(1, len(validation_losses) + 1),
            validation_losses,
            label="Validation",
        )
    if best_epoch is not None:
        ax.scatter(best_epoch + 1, training_losses[best_epoch])
        if validation_losses is not None:
            ax.scatter(best_epoch + 1, validation_losses[best_epoch])

        _best_epoch_tick(ax, best_epoch)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Learning Curves")
    ax.legend()

    fig.tight_layout()
    fig.savefig("learning_curves.png", dpi=300)

    # Send plot on Telegram
    if tg_token and tg_chat_id:
        try:
            if msg_id is None:
                send_telegram_photo(
                    token=tg_token,
                    chat_id=tg_chat_id,
                    photo_path="learning_curves.png",
                    caption=r"Next\-token classifier learning curves",
                )
            else:
                edit_telegram_media(
                    token=tg_token,
                    chat_id=tg_chat_id,
                    photo_path="learning_curves.png",
                    message_id=msg_id,
                )
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Failed to send Telegram photo: {e}")

    plt.show()


def _best_epoch_tick(ax: Axes, best_epoch: int) -> None:
    """Add a label and minor tick at `best_epoch` on the supplied axes."""
    # Add best epoch as a minor tick
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
    digit_width = 0.02  # rough estimate: each digit is about 2% of the x-axis range
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    best_label_width = len(str(int(best_epoch + 1))) * digit_width * x_range

    has_conflict = False
    for tick in ax.get_xticks():
        tick_label_width = len(str(int(tick))) * digit_width * x_range
        min_distance = (best_label_width + tick_label_width) / 2
        if abs(best_epoch + 1 - tick) < min_distance:
            has_conflict = True
            break

    # Only move label inside plot if there's a conflict
    if has_conflict:
        ax.tick_params(axis="x", which="minor", pad=-20)
