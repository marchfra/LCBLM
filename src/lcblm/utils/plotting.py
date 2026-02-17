from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from trainvox import edit_telegram_media, send_telegram_photo

from lcblm._logging import utils_logger as logger


def plot_learning_curves(
    training_losses: list[float],
    validation_losses: list[float] | None = None,
    title: str = "Learning Curves",
    best_epoch: int | None = None,
    best_epoch_method: Literal["tick", "arrow"] = "tick",
) -> tuple[Figure, Axes]:
    """Create a learning curves plot.

    Args:
        training_losses: List of training losses. Every entry corresponds to one epoch.
        validation_losses: List of validation losses. Every entry corresponds to one
            epoch.
        title: The title of the plot.
        best_epoch: The epoch with the lowest validation loss. Can even be an epoch that
            you want to highlight for whatever reason. If None and validation_losses is
            provided, will automatically be calculated from validation_losses.
        best_epoch_method: The method to use to highlight the best epoch. 'tick' will
            place a new tick and label at the best epoch, while 'arrow' will annotate
            the best epoch with an arrow.

    Returns:
        The Figure and Axes objects.

    Raises:
        ValueError: If training_losses and validation_losses have different lengths.

    """
    if validation_losses is not None and len(training_losses) != len(validation_losses):
        msg = "training_losses and validation_losses must have the same length"
        raise ValueError(msg)

    methods = {
        "tick": _highlight_best_epoch_with_tick,
        "arrow": _highlight_best_epoch_with_arrow,
    }

    if best_epoch_method not in methods:
        msg = "Method not supported. Use 'tick' or 'arrow'"
        raise ValueError(msg)

    fig, ax = plt.subplots()

    ax.plot(range(1, len(training_losses) + 1), training_losses, label="Training")
    if validation_losses is not None:
        ax.plot(
            range(1, len(validation_losses) + 1),
            validation_losses,
            label="Validation",
        )

    _configure_integer_ticks(ax)

    # Auto-infer best epoch
    if best_epoch is None and validation_losses is not None:
        best_epoch = validation_losses.index(min(validation_losses))

    if best_epoch is not None:
        methods[best_epoch_method](
            fig=fig,
            ax=ax,
            best_epoch=best_epoch,
            training_losses=training_losses,
            validation_losses=validation_losses,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()

    return fig, ax


@contextmanager
def learning_curves_plot(
    training_losses: list[float],
    validation_losses: list[float] | None = None,
    title: str = "Learning Curves",
    best_epoch: int | None = None,
    best_epoch_method: Literal["tick", "arrow"] = "tick",
) -> Generator[tuple[Figure, Axes], None, None]:
    """Context manager for creating learning curves plots.

    Args:
        training_losses: List of training losses. Every entry corresponds to one epoch.
        validation_losses: List of validation losses. Every entry corresponds to one
            epoch.
        title: The title of the plot.
        best_epoch: The epoch with the lowest validation loss. Can even be an epoch that
            you want to highlight for whatever reason. If None and validation_losses is
            provided, will automatically be calculated from validation_losses.
        best_epoch_method: The method to use to highlight the best epoch. 'tick' will
            place a new tick and label at the best epoch, while 'arrow' will annotate
            the best epoch with an arrow.

    Yields:
        The Figure and Axes objects.

    Raises:
        ValueError: If training_losses and validation_losses have different lengths.

    Example:
        >>> with learning_curves_plot(train_losses, val_losses) as (fig, ax):
        ...     fig.savefig("output.png", dpi=300)

    """
    fig, ax = plot_learning_curves(
        training_losses=training_losses,
        validation_losses=validation_losses,
        title=title,
        best_epoch=best_epoch,
        best_epoch_method=best_epoch_method,
    )

    try:
        yield fig, ax
    finally:
        plt.close(fig)


def send_learning_curves_to_telegram(
    image_path: str | Path,
    tg_token: str,
    tg_chat_id: str,
    caption: str | None = None,
    msg_id: int | None = None,
) -> int | None:
    """Send a learning curves plot on Telegram.

    Args:
        image_path: Path to the image file to send.
        tg_token: The token of the Telegram bot.
        tg_chat_id: The unique identifier for the target chat.
        caption: The caption of the plot on Telegram.
        msg_id: The ID of the Telegram message to edit. If not supplied will send a new
            message.

    Returns:
        The Telegram message ID, or None if sending failed.

    """
    image_path = Path(image_path)

    # Send plot on Telegram
    if tg_token and tg_chat_id:
        try:
            if msg_id is None:
                return send_telegram_photo(
                    token=tg_token,
                    chat_id=tg_chat_id,
                    photo_path=image_path,
                    caption=caption,
                )
            return edit_telegram_media(
                photo_path=image_path,
                message_id=msg_id,
                token=tg_token,
                chat_id=tg_chat_id,
                caption=caption,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Failed to send Telegram photo: {e}")

    return None


def _configure_integer_ticks(ax: Axes) -> None:
    """Set up integer major ticks with appropriately spaced minor ticks."""
    ax.xaxis.set_major_locator(MaxNLocator("auto", integer=True))
    major_ticks = ax.xaxis.get_majorticklocs()
    if len(major_ticks) >= 2:  # noqa: PLR2004
        # Compute minor ticks after major ticks are placed
        major_spacing = int(major_ticks[1] - major_ticks[0])
        # Find a nice divisor of major_spacing for minor ticks
        divisors = [
            d
            for d in [5, 4, 3, 2]
            if major_spacing % d == 0 and major_spacing // d >= 1
        ]
        n_minor = divisors[0] if divisors else major_spacing

        ax.xaxis.set_minor_locator(AutoMinorLocator(n_minor))


def _add_best_epoch_tick(ax: Axes, best_epoch: int) -> None:
    """Add best epoch as a minor tick with label."""
    current_minor_ticks = list(ax.get_xticks(minor=True))
    if best_epoch not in current_minor_ticks:
        current_minor_ticks.append(best_epoch)
        current_minor_ticks.sort()
    ax.set_xticks(current_minor_ticks, minor=True)

    minor_labels = [
        str(best_epoch) if x == best_epoch else "" for x in current_minor_ticks
    ]
    ax.set_xticklabels(minor_labels, minor=True)


def _has_label_conflict(fig: Figure, ax: Axes, best_epoch: int) -> bool:
    """Check if best epoch label conflicts with a major tick label."""
    fig.canvas.draw()

    best_tick_bbox = None
    for tick in ax.xaxis.get_minor_ticks():
        if tick.get_loc() == best_epoch:
            best_tick_bbox = tick.label1.get_window_extent()
            break

    if best_tick_bbox is None:
        return False

    return any(
        best_tick_bbox.overlaps(tick.label1.get_window_extent())
        for tick in ax.xaxis.get_major_ticks()
        if tick.label1.get_text()
    )


def _highlight_best_epoch_with_tick(
    fig: Figure,
    ax: Axes,
    best_epoch: int,
    training_losses: list[float],
    validation_losses: list[float] | None = None,
) -> None:
    """Add scatter points and ticks for best epoch.

    If there's a conflict between the new tick label and pre-existing labels, move the
    new tick inside the plot and extend the plot's y-range so that the label doesn't
    overlap the learning curves.
    """
    best_epoch += 1  # convert to 1-indexed

    ax.scatter(best_epoch, training_losses[best_epoch - 1])
    if validation_losses is not None:
        ax.scatter(best_epoch, validation_losses[best_epoch - 1])

    _add_best_epoch_tick(ax, best_epoch)

    if _has_label_conflict(fig, ax, best_epoch):
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.05 * y_range, y_max)
        ax.tick_params(axis="x", which="minor", pad=-20)


def _highlight_best_epoch_with_arrow(
    fig: Figure,  # noqa: ARG001
    ax: Axes,
    best_epoch: int,
    training_losses: list[float],
    validation_losses: list[float] | None = None,
) -> None:
    """Add scatter points and ticks for best epoch.

    If there's a conflict between the new tick label and pre-existing labels, replace
    the label with an arrow annotation.
    """
    best_epoch += 1  # convert to 1-indexed

    ax.scatter(best_epoch, training_losses[best_epoch - 1])
    if validation_losses is not None:
        ax.scatter(best_epoch, validation_losses[best_epoch - 1])

    _add_best_epoch_tick(ax, best_epoch)

    # Hide the bottom label
    for tick in ax.xaxis.get_minor_ticks():
        if tick.get_loc() == best_epoch:
            tick.label1.set_visible(False)
            break

    # Annotate with arrow pointing to the training scatter point
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.annotate(
        str(best_epoch),
        xy=(best_epoch, training_losses[best_epoch - 1]),
        xytext=(best_epoch, training_losses[best_epoch - 1] - 0.15 * y_range),
        ha="center",
        va="top",
        fontsize=ax.xaxis.get_major_ticks()[0].label1.get_fontsize(),
        arrowprops={"arrowstyle": "-|>", "color": "black"},
    )


def set_plt_style(
    styles: list[str] | None = None,
    style_path: str | Path | None = None,
) -> None:
    """Set Matplotlib styles.

    Args:
        styles: List of styles to use. If None, grid, science, notebook, and mylegend
            styles will be loaded.
        style_path: The path where the .mplstyle files are stored. If None, tries to
            load styles from the system Matplotlib. If the style files are not found
            there, tries local "mplstyles" directory.

    """
    if styles is None:
        styles = ["grid", "science", "notebook", "mylegend"]

    if style_path is None:
        # Look in default style location
        with suppress(OSError):
            plt.style.use(styles)
            return

        # Look in local mplstyles directory
        with suppress(OSError):
            plt.style.use([Path("mplstyles") / f"{style}.mplstyle" for style in styles])
            return

        logger.warning("Styles not found, using default matplotlib style.")
        plt.style.use("default")
    else:
        style_path = Path(style_path)
        if style_path.exists():
            try:
                plt.style.use(
                    [str(style_path / f"{style}.mplstyle") for style in styles],
                )
            except FileNotFoundError:
                logger.warning("Style files not found, using default matplotlib style.")
                plt.style.use("default")
            else:
                return
        else:
            logger.warning("Styles path doesn't exist, using default matplotlib style.")
            plt.style.use("default")
