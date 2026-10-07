import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "output_dir" / "gpt2-1b-russian" / "trainer_state.json"
OUTPUT_PATH = ROOT / "figures" / "training_curves.png"

# Chance-level cross-entropy for the ruGPT-3 tokenizer, ln(50257).
CHANCE_LOSS = float(np.log(50257))

INK = "#1C1917"
MUTED = "#78716C"
GRID = "#E7E5E4"
PAPER = "#F7F5F2"
PANEL = "#FFFcf8"
LOSS_COLOR = "#9A3412"
GRAD_COLOR = "#1D4E89"
LR_COLOR = "#3F6212"
EVAL_COLOR = "#7C2D12"
CHANCE_COLOR = "#A8A29E"


def load_history(path):
    state = json.loads(path.read_text())
    train, evals = [], []
    for row in state["log_history"]:
        if "loss" in row:
            train.append(row)
        elif "eval_loss" in row:
            evals.append(row)
    if not train:
        raise SystemExit(f"No training steps found in {path}")
    return state, train, evals


def rolling_mean(values, window):
    if window <= 1 or len(values) < window:
        return np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def style_axis(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10, fontweight="medium")
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_facecolor(PANEL)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D6D3D1")
    ax.spines["bottom"].set_color("#D6D3D1")
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))


def main():
    state, train, evals = load_history(STATE_PATH)
    steps = np.array([row["step"] for row in train])
    loss = np.array([row["loss"] for row in train], dtype=float)
    grad_norm = np.array([row["grad_norm"] for row in train], dtype=float)
    learning_rate = np.array([row["learning_rate"] for row in train], dtype=float)

    window = 9
    smooth_steps = steps[window - 1 :]
    smooth_loss = rolling_mean(loss, window)
    smooth_grad = rolling_mean(grad_norm, window)

    eval_steps = [row["step"] for row in evals]
    eval_loss = [row["eval_loss"] for row in evals]
    best_eval = min(eval_loss) if eval_loss else None

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
        }
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.4, 9.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0, 0.85], "hspace": 0.28},
    )

    axes[0].fill_between(steps, loss, CHANCE_LOSS, color=LOSS_COLOR, alpha=0.08, linewidth=0)
    axes[0].plot(steps, loss, color=LOSS_COLOR, alpha=0.28, linewidth=1.0, label="train loss, шаг")
    axes[0].plot(
        smooth_steps,
        smooth_loss,
        color=LOSS_COLOR,
        linewidth=2.1,
        label=f"скользящее среднее, окно {window}",
    )
    axes[0].axhline(
        CHANCE_LOSS,
        color=CHANCE_COLOR,
        linewidth=1.0,
        linestyle=(0, (3, 2)),
        label=f"случайный уровень, ln|V| = {CHANCE_LOSS:.2f}",
    )
    if eval_steps:
        axes[0].scatter(
            eval_steps,
            eval_loss,
            s=46,
            color=EVAL_COLOR,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
            label="eval loss",
        )
        for step, value in zip(eval_steps, eval_loss):
            axes[0].annotate(
                f"{value:.3f}",
                (step, value),
                textcoords="offset points",
                xytext=(6, -11),
                fontsize=8,
                color=EVAL_COLOR,
            )
    style_axis(axes[0], "Кросс-энтропия", "loss")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK)
    axes[0].set_ylim(10.15, 11.45)

    axes[1].plot(steps, grad_norm, color=GRAD_COLOR, alpha=0.28, linewidth=1.0, label="grad norm, шаг")
    axes[1].plot(smooth_steps, smooth_grad, color=GRAD_COLOR, linewidth=2.1, label=f"скользящее среднее, окно {window}")
    axes[1].axhline(float(np.median(grad_norm)), color=GRAD_COLOR, alpha=0.45, linewidth=0.9, linestyle=(0, (2, 2)))
    style_axis(axes[1], "Норма градиента", "grad norm")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)

    axes[2].plot(steps, learning_rate * 1e5, color=LR_COLOR, linewidth=2.1)
    axes[2].fill_between(steps, learning_rate * 1e5, color=LR_COLOR, alpha=0.12, linewidth=0)
    style_axis(axes[2], "Расписание learning rate", r"learning rate, $\times 10^{-5}$")
    axes[2].set_xlabel("Шаг обучения", color=MUTED, fontsize=9)

    best_text = f"{best_eval:.4f}" if best_eval is not None else "—"
    fig.suptitle(
        "Обучение Qwen3, 961 млн параметров, русский Wikipedia",
        x=0.07,
        ha="left",
        fontsize=14,
        color=INK,
        fontweight="medium",
    )
    fig.text(
        0.07,
        0.935,
        (
            f"101 шаг из 485 016  ·  404 обучающих примера  ·  "
            f"лучший eval loss {best_text}  ·  MPS, bf16, SDPA"
        ),
        ha="left",
        fontsize=8.5,
        color=MUTED,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")
    print(f"Steps: {int(steps[0])}–{int(steps[-1])}")
    print(f"Train loss: {loss[0]:.4f} -> {loss[-1]:.4f}")
    if best_eval is not None:
        print(f"Best eval loss: {best_eval:.6f} at step {state.get('best_global_step')}")


if __name__ == "__main__":
    main()
