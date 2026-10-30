#!/usr/bin/env python3
"""Two-panel Figure 1 with the manuscript's tueplots typography.

PDF is used in LaTeX; SVG retains editable text and named vector objects.
Centaur artwork is placed unmodified, with attribution in the caption.
Task prompts are shortened; displayed answers are illustrative.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from textwrap import wrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle
from figure_helpers import fit_label_widths
from paper_plot_style import COLORS, paper_style

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 850, 332
# Preserve approximately six-point model labels at manuscript textwidth.
FONT_SCALE = 740 / 453.6
INK = "#222222"
CARD = "#414C5E"


def response_means() -> tuple[dict, dict]:
    """Use the same held-out-target means as the manuscript's allocation figure."""
    source = ROOT / "results/giving_taking/figure_inputs/prediction_shifts_by_seed.csv"
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    task_means = {}
    for task in ("Giving", "Taking"):
        task_rows = [row for row in rows if row["task"] == task]
        frozen = {float(row["frozen_expected_tokens"]) for row in task_rows}
        if len(frozen) != 1:
            raise ValueError(f"Expected one frozen mean for {task}")
        task_means[task] = {"frozen": frozen.pop()}
        for condition in ("us", "china", "pooled"):
            values = [row for row in task_rows if row["condition"] == condition]
            if sorted(row["seed"] for row in values) != ["0", "1", "2"]:
                raise ValueError(f"Expected seeds 0, 1, 2 for {task}/{condition}")
            task_means[task][condition] = mean(float(row["expected_tokens"]) for row in values)
    provenance = {
        "source": str(source.relative_to(ROOT)),
        "method": "Equal average over three seeds; source values average 25 held-out targets, conditional on responses 0-20.",
        "means_tokens": task_means,
        "us_minus_china_tokens": {
            task: values["us"]-values["china"] for task, values in task_means.items()
        },
        "curve_shape": "Illustrative Gaussian, common width 0.45 tokens; not an estimated response density or uncertainty interval.",
        "display_range_tokens": [4, 11.5],
        "human_inset": "Qualitative Giving direction only: China below US in both source waves; curve positions and widths are illustrative.",
        "human_inset_source": "results/giving_taking/frozen/human_allocation_summary.csv",
    }
    return task_means, provenance


def draw(output: Path) -> None:
    task_means, provenance = response_means()
    style = paper_style(HEIGHT / 72)
    style.update({"svg.fonttype": "none", "svg.hashsalt": "centaur-pipeline-panels",
                  "figure.constrained_layout.use": False})
    with plt.rc_context(style):
        fig = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72), facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
        ax.set_axis_off()
        width_checks = []

        def label(x, y, value, gid, size=9, color=INK, ha="center", weight="normal", width=None,
                  style="normal", rotation=0):
            artist = ax.text(x, y, value, fontsize=size*FONT_SCALE, ha=ha, va="center", color=color,
                             fontweight=weight, fontstyle=style, linespacing=1.35,
                             rotation=rotation, gid=gid)
            if width is not None:
                width_checks.append((artist, width))
            return artist

        def box(x, y, w, h, fill, gid, edge="none", radius=2, lw=.45, **kwargs):
            patch = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         facecolor=fill, edgecolor=edge, linewidth=lw, gid=gid, **kwargs)
            ax.add_patch(patch)
            return patch

        def rule(xs, ys, gid, color="#8B929C", lw=.55):
            ax.plot(xs, ys, lw=lw, color=color, gid=gid, solid_capstyle="butt")

        def arrow(x1, y1, x2, y2, gid, color="#8B929C"):
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->",
                         mutation_scale=7, linewidth=.6, color=color,
                         shrinkA=0, shrinkB=0, gid=gid))

        def person(x, y, color, gid):
            ax.add_patch(Circle((x, y-1.68), 1.64, facecolor=color, edgecolor="none",
                                gid=f"{gid}-head"))
            shoulders = MplPath(
                [(x-3.04, y+3.52), (x-3.04, y+1.2), (x-1.76, y+.4), (x, y+.4),
                 (x+1.76, y+.4), (x+3.04, y+1.2), (x+3.04, y+3.52),
                 (x-3.04, y+3.52), (x-3.04, y+3.52)],
                [MplPath.MOVETO] + [MplPath.CURVE4]*6 + [MplPath.LINETO, MplPath.CLOSEPOLY])
            ax.add_patch(PathPatch(shoulders, facecolor=color, edgecolor="none",
                                   gid=f"{gid}-shoulders"))

        def star(x, y, radius, color, gid, angle=-np.pi/2):
            angles = angle + np.arange(10)*np.pi/5
            radii = np.where(np.arange(10)%2 == 0, radius, radius*.382)
            vertices = np.column_stack((x+radii*np.cos(angles), y+radii*np.sin(angles)))
            ax.add_patch(Polygon(vertices, facecolor=color, edgecolor="none", gid=gid))

        def flag(x, y, width, country, gid):
            height = width*(10/19 if country == "us" else 2/3)
            ax.add_patch(Rectangle((x, y), width, height,
                                  facecolor="white" if country == "us" else "#DE2910",
                                  edgecolor="none", gid=f"{gid}-field"))
            if country == "us":
                for i in range(0, 13, 2):
                    ax.add_patch(Rectangle((x, y+i*height/13), width, height/13,
                                          facecolor="#B22234", edgecolor="none",
                                          gid=f"{gid}-stripe-{i}"))
                cw, ch = width*.4, height*7/13
                ax.add_patch(Rectangle((x, y), cw, ch, facecolor="#3C3B6E", edgecolor="none",
                                      gid=f"{gid}-canton"))
                for row in range(9):
                    for col in range(6 if row%2 == 0 else 5):
                        star(x+cw*((col+1)/7 if row%2 == 0 else (col+1.5)/7),
                             y+ch*(row+1)/10, height*.027, "white", f"{gid}-star-{row}-{col}")
            else:
                star(x+width/6, y+height/4, height*.15, "#FFDE00", f"{gid}-star-main")
                for i, (sx, sy) in enumerate([(10, 2), (12, 4), (12, 7), (10, 9)]):
                    cx, cy = x+width*sx/30, y+height*sy/20
                    angle = np.arctan2(y+height/4-cy, x+width/6-cx)
                    star(cx, cy, height*.05, "#FFDE00", f"{gid}-star-{i}", angle=angle)
            ax.add_patch(Rectangle((x, y), width, height, facecolor="none", edgecolor="#B6BCC5",
                                  linewidth=.25, gid=f"{gid}-outline"))

        # a: concrete task examples, with the explanation in the caption.
        label(3, 10, "a", "panel-a", size=10, weight="bold")
        card_width = (WIDTH - 54) / 3
        cards = [
            (17, "Social allocation: giving variant", ["You start with 20 tokens.", "Other person: farmer.",
                            "How many tokens would you give?"], "You enter", "10"),
            (17+card_width+11, "Categorization", ["Which option is most closely", "related to teacup?",
                                    "A: glass       B: kettle"], "You choose", "A"),
            (17+2*(card_width+11), "Risky choice", ["A: 10 points with 25% chance.", "B: 1 point with 75% chance.",
                               "Otherwise 0 points."], "You choose", "A"),
        ]
        for x, title, lines, prefix, answer in cards:
            key = title.lower().replace(" / ", "-")
            label(x + card_width/2, 10, title, f"task-{key}")
            box(x, 20, card_width, 80, CARD, f"prompt-card-{key}", radius=3)
            for i, value in enumerate(lines):
                label(x + 8, 33 + i*16, value, f"prompt-{key}-{i}", size=7.2,
                      color="white", ha="left", width=card_width-16)
            rule([x+8, x+card_width-8], [78, 78], f"response-rule-{key}", color="#798391", lw=.35)
            label(x + 8, 89, prefix, f"response-prefix-{key}", size=8, color="white", ha="left")
            label(x + card_width-46, 89, f"<<{answer}>>", f"response-{key}", size=8.3,
                  color="#BDE9DF", weight="bold")

        # b: a shared frozen pair followed by its local population-adapter options.
        label(3, 117, "b", "panel-b", size=10, weight="bold")
        label(76, 117, "Fine-tuning population", "training-heading", size=8.0, width=140)
        label(432, 117, "Centaur + one population adapter", "model-heading", size=8.5, width=280)
        label(762, 117, "Evaluation", "evaluation-heading", size=8.5)
        response_x = np.linspace(23, 103, 160)
        for condition, center in [("china", 48), ("us", 77)]:
            density = 13*np.exp(-.5*((response_x-center)/11.5)**2)
            ax.fill_between(response_x, 151, 151-density, color=COLORS[condition], alpha=.12,
                            linewidth=0, gid=f"responses-{condition}-fill")
            ax.plot(response_x, 151-density, color=COLORS[condition], linewidth=.8,
                    gid=f"responses-{condition}-curve")
        rule([23, 103], [151, 151], "responses-axis", color="#90959D", lw=.4)
        for y, condition, sample in [(176, "us", "US"), (228, "china", "China"),
                                     (280, "pooled", "Pooled")]:
            color = COLORS[condition]
            box(17, y-18, 92, 36, "#FAFAFB", f"sample-{condition}", edge=color, lw=.7)
            if condition == "pooled":
                flag(25, y-11, 17, "us", "flag-pooled-us")
                flag(25, y, 17, "china", "flag-pooled-china")
            else:
                flag(25, y-(18*(10/19 if condition == "us" else 2/3))/2, 18,
                     condition, f"flag-{condition}")
            person_colors = ([COLORS["us"]]*6 + [COLORS["china"]]*6
                             if condition == "pooled" else [color]*12)
            for i, person_color in enumerate(person_colors):
                person(57+(i%4)*14, y-10+(i//4)*10, person_color, f"person-{condition}-{i}")
            label(63, y+25, sample, f"sample-label-{condition}", size=6.6, width=92)

        input_y = 228
        box(120, input_y-67, 90, 134, CARD, "input-prompt-shared", radius=2)
        # Reuse panel a's exact example, with line breaks for the narrower box.
        prompt_lines = [line for sentence in cards[0][2] for line in wrap(sentence, width=14)]
        for i, value in enumerate(prompt_lines):
            label(126, input_y-54+i*14, value, f"input-prompt-shared-{i}",
                  size=6.5, color="white", ha="left", width=78)
        rule([126, 204], [273, 273], "input-response-rule", color="#798391", lw=.35)
        label(126, 284, cards[0][3], "input-response-prefix", size=6.5,
              color="white", ha="left", width=62)
        label(196, 284, "<<", "input-response-delimiter", size=6.5,
              color="#BDE9DF", weight="bold", width=18)
        arrow(214, input_y, 220, input_y, "prompt-to-embedding-shared")
        box(222, input_y-67, 26, 134, "#E4E6E9", "token-embedding-shared", radius=2)
        label(235, input_y, "Token\nembedding", "token-embedding-label-shared", size=6,
              rotation=90, width=24)
        arrow(250, input_y, 256, input_y, "embedding-to-model-shared")

        mascot = plt.imread(ROOT / "paper/figures/assets/centaur.png")
        mascot_h = 18
        mascot_w = mascot_h*mascot.shape[1]/mascot.shape[0]
        for layer, x in enumerate((262, 446), start=1):
            label(x+51, 139, "Frozen", f"frozen-heading-{layer}", size=7, width=102)
            rule([x, x, x+102, x+102], [154, 150, 150, 154], f"frozen-bracket-{layer}",
                 color="#9AA0A9", lw=.6)
            box(x-3, input_y-67, 74, 134, "none", f"layer-outline-{layer}",
                edge="#8B929C", radius=2.5, lw=.65, linestyle=(0, (1, 2)))
            for offset, component, name in [(-34, "attention", "Self-attention"),
                                          (34, "feedforward", "Feedforward")]:
                cy = input_y+offset
                key = f"layer-{layer}-{component}"
                box(x, cy-30, 68, 60, "#D6D8DC", key, edge="#AEB4BE", radius=1.8, lw=.55)
                label(x+34, cy, name, f"{key}-label", size=6.8, width=64)
                label(x+73, cy, "+", f"centaur-plus-{key}", size=6.5, color="#6B727D")
                box(x+78, cy-30, 24, 60, "#AEB4BE", f"frozen-centaur-adapter-{key}", radius=1.8)
                ax.imshow(mascot, extent=(x+90-mascot_w/2, x+90+mascot_w/2,
                                         cy+mascot_h/2, cy-mascot_h/2),
                          aspect="equal", interpolation="antialiased", zorder=3, gid=f"centaur-{key}")

            # Each local bank has one attention/feedforward pair. The chosen
            # population is the same in both representative layers.
            branch_x, bank_x = x+110, x+129
            rule([x+105, branch_x], [input_y, input_y], f"frozen-to-alternatives-{layer}")
            rule([branch_x, branch_x], [176, 280], f"alternative-branches-{layer}")
            label(bank_x+15, 139, "Trainable", f"trainable-heading-{layer}", size=7, width=64)
            rule([bank_x, bank_x, bank_x+30, bank_x+30], [154, 150, 150, 154],
                 f"trainable-bracket-{layer}", color="#9AA0A9", lw=.6)
            for y, condition in [(176, "us"), (228, "china"), (280, "pooled")]:
                color = COLORS[condition]
                rule([branch_x, branch_x+6], [y, y], f"branch-{layer}-{condition}")
                label(branch_x+11, y, "+", f"population-plus-{layer}-{condition}",
                      size=7, color="#6B727D")
                box(bank_x, y-15, 30, 30, "#FAFAFB", f"adapter-bank-{layer}-{condition}",
                    edge=color, lw=.55)
                for offset, component in [(-7, "attention"), (7, "feedforward")]:
                    module_x, module_y = bank_x+8, y+offset-5
                    key = f"population-{condition}-{layer}-{component}"
                    module = box(module_x, module_y, 14, 10, color, key, radius=1.3)
                    if condition == "pooled":
                        for j, cc in enumerate((COLORS["us"], COLORS["china"])):
                            patch = Rectangle((module_x+j*7, module_y), 7, 10, facecolor=cc,
                                              edgecolor="none", gid=f"{key}-{j}")
                            patch.set_clip_path(module)
                            ax.add_patch(patch)

            # Parameter type is distinct from whether the parameter is updated:
            # Centaur's original adapter is LoRA too, although it is frozen here.
            for left, right, title, key in [
                (x, x+68, "Model weights", "model-weights"),
                (x+78, bank_x+30, "LoRA adapters", "lora-adapters"),
            ]:
                rule([left, left, right, right], [303, 307, 307, 303],
                     f"{key}-bracket-{layer}", color="#9AA0A9", lw=.6)
                label((left+right)/2, 318, title, f"{key}-heading-{layer}",
                      size=5.8, width=right-left)
        label(434, input_y, "...", "intervening-layers-shared", size=7)

        label(642, 139, "Output", "output-heading", size=7, width=48)
        # Lighter versions of the population hues remain legible on the input fill.
        output_colors = {"us": "#92BDE3", "china": "#E8A58F", "pooled": "#BCAFDB"}
        # Illustrative responses to the displayed item, not empirical predictions.
        example_outputs = {"us": 8, "china": 5, "pooled": 6}
        for y, condition in [(176, "us"), (228, "china"), (280, "pooled")]:
            arrow(609, y, 615, y, f"adapter-to-output-{condition}")
            box(618, y-21, 48, 42, CARD, f"output-{condition}")
            label(642, y, str(example_outputs[condition]),
                  f"output-example-{condition}", size=9, color=output_colors[condition])
            rule([666, 674], [y, y], f"evaluate-{condition}", lw=.55)
        rule([674, 674], [176, 280], "shared-evaluation-protocol")
        arrow(674, 195, 687, 195, "held-out-evaluation")
        arrow(674, 280, 687, 280, "new-task-evaluation")

        def schematic_plot(baseline, title, means, gid):
            centers = {condition: 696+(value-4)*132/7.5 for condition, value in means.items()}
            label(762, baseline-47, title, f"{gid}-title", size=6.0 if gid == "new-task" else 6.6, width=174 if gid == "new-task" else 164)
            xs = np.linspace(696, 828, 260)
            for condition in ["us", "china", "pooled", "frozen"]:
                density = 34*np.exp(-.5*((xs-centers[condition])/(.45*132/7.5))**2)
                if condition in ("us", "china"):
                    ax.fill_between(xs, baseline, baseline-density, color=COLORS[condition],
                                    alpha=.09, linewidth=0, gid=f"{gid}-{condition}-fill")
                ax.plot(xs, baseline-density, color=COLORS[condition],
                        lw=.9 if condition != "frozen" else 1,
                        linestyle=(0, (2, 1.5)) if condition == "frozen" else "-",
                        gid=f"{gid}-{condition}-curve")
            rule([696, 828], [baseline, baseline], f"{gid}-axis", color="#90959D", lw=.4)
        schematic_plot(214, "Same task, new participants & items", task_means["Giving"], "trained-task")
        schematic_plot(307, "New task variant, no further fine-tuning", task_means["Taking"], "new-task")
        for x, y, condition, name in [(700, 226, "frozen", "Frozen Centaur"), (785, 226, "us", "US"),
                                    (700, 238, "china", "China"), (785, 238, "pooled", "Pooled")]:
            ax.plot([x, x+7], [y, y], color=COLORS[condition], lw=.9,
                    linestyle=(0, (2, 1.5)) if condition == "frozen" else "-",
                    gid=f"evaluation-legend-line-{condition}")
            label(x+10, y, name, f"evaluation-legend-label-{condition}", ha="left", size=6.2)

        fitted_labels = fit_label_widths(fig, ax, width_checks, WIDTH)
        if fitted_labels:
            provenance["fitted_labels"] = fitted_labels

        provenance["layout"] = "Tall common prompt, embedding, frozen weights, and frozen Centaur adapters; unchanged trainable adapter dimensions. Aligned Frozen and Trainable top brackets identify update status; lower Model weights and LoRA adapters brackets identify parameter type, including frozen Centaur LoRA. Dark output cards match the input. One population adapter is selected consistently across all layers."
        provenance["displayed_output"] = "Illustrative integer responses to the displayed farmer item; not recorded model predictions or population means."
        provenance["example_outputs_tokens"] = example_outputs

        output.mkdir(parents=True, exist_ok=True)
        stem = output / "finetuning_pipeline"
        stem.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
        fig.savefig(stem.with_suffix(".pdf"), metadata={
            "Title": "Population-specific fine-tuning of Centaur",
            "Subject": "Centaur illustration by Milena Rmus; official Centaur project",
            "CreationDate": None, "ModDate": None})
        fig.savefig(stem.with_suffix(".svg"), metadata={
            "Date": None, "Description": "Editable SVG. Centaur illustration by Milena Rmus, "
                                         "https://marcelbinz.github.io/centaur."})
        fig.savefig(stem.with_suffix(".png"), dpi=300)
        plt.close(fig)
        for suffix in ("pdf", "svg", "png"):
            print(stem.with_suffix(f".{suffix}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/figures")
    draw(parser.parse_args().output_dir)
