# Change Log — ML-bike (CS-C3240)

What changed in the project, newest first. Add an entry whenever you change something
another team member would be surprised by: file moves, notebook edits, new results,
environment requirements.

---

## 2026-10-06 — Stage 2 analysis, repo tidy-up

### Repo structure

- All Stage 1 deliverables moved into `ML_bike_stage1/`: the report (`.docx`, `.pdf`),
  the merged submission (`Stage1_Submission.pdf`), the notebook, its PDF/zip exports and
  the markdown export with its images. Nothing was lost.
- **Before committing**, use `git add -A`. Git currently reports these as deleted files
  plus untracked new ones; `-A` makes it record them as renames instead.
- `.gitignore` now ignores `.idea/` (the line was present but commented out, which is why
  PyCharm's files kept showing up in `git status`).
  Note: `.idea/.gitignore` is *still tracked*, and ignore rules never apply to tracked
  files. To finish the job: `git rm -r --cached .idea`.

### Notebooks

- `ML_bike.ipynb` was run end to end and its **outputs are now saved in the file**, so you
  can read the Stage 2 results (sections 8–11) without rerunning anything. A full rerun
  takes about 3.5 minutes.
- `ML_bike_stage1/ML_bike_stage1.ipynb`: fixed markdown list formatting. Markdown needs a
  blank line between a paragraph and the list that follows it; without one, every renderer
  runs the items together into a paragraph. This affected the "Steps", "ML problem
  formulation", "Data" and weather-column lists, which came out as run-on text in the PDF.
  **Only blank lines were added** — no text, code, outputs or execution counts changed.
  Anyone re-exporting the notebook now gets correctly formatted lists.

### Stage 2 results so far

Validation = the 2024 season (2,508,945 trips). MAE and MedAE are in minutes.

| Model | MAE | MedAE | vs constant |
|---|---|---|---|
| Random guess (resample a real trip) | 20.90 | 7.68 | −46.2% |
| Constant: mean | 15.35 | 8.75 | −7.4% |
| Constant: median (baseline) | 14.29 | 5.30 | — |
| Linear, squared error | 15.27 | 8.29 | −6.8% |
| **Linear, Huber** | **14.21** | 5.56 | +0.58% |
| Linear, absolute error (LAD) | 14.24 | 5.28 | +0.35% |
| Boosted trees, squared error | 16.03 | 8.08 | −12.2% |
| Boosted trees, absolute error | 14.24 | 5.26 | +0.35% |

Three findings:

1. **The loss matters far more than the model family.** Every robust-loss model lands at
   MAE 14.2–14.3; every squared-error model lands at 15.3–16.0. The gap between loss
   groups is ~30x the spread within a group. Boosted trees fitted with squared error are
   *worse than predicting a constant*.
2. **Squared-error fits are unstable.** Removing the longest 0.1% of training trips (400
   rows) moves their predictions by 17–23%, and moves them *toward* the robust answer.
   Huber moves 0.14%, LAD 0.60%. So the squared-error answer depends entirely on an
   arbitrary cut-off; the robust one needs no cut-off.
3. **The features themselves are the limit.** A lookup table of per-cell medians — the most
   flexible model these five features allow, able to represent any interaction — reaches
   MAE 14.25, i.e. 0.28% better than a constant. Across cells the conditional median
   varies with an IQR of 2.05 min, while the spread *within* a cell is 10.25 min. The
   signal is about a fifth of the noise.

Read together: going from random guessing to a constant is worth 6.6 min of MAE; going
from a constant to the best possible model on these features is worth under 0.1 min. The
five features account for roughly **1% of the total achievable improvement over random**.

### Environment reminders

- `data/weather_daily.csv` is daily FMI data (Helsinki Kaisaniemi, `fmisid=100971`):
  `temp_c` and `precip_mm`. FMI codes "no measurable rain" as `-1`; the notebook clips it
  to 0. Delete the file and the notebook re-downloads it.
- `requirements.txt` includes `certifi`, which the FMI download needs on Windows —
  without it, Python cannot verify the TLS certificate and the download fails.
- If `pip install` fails once with "Access is denied" on a `.pyd` file, just run it again;
  that is antivirus scanning, not a broken environment.

---

### How to add an entry

Put the newest entry directly under the intro, with today's date and a short title.
Say what changed and what a teammate has to *do* about it (rerun something, pull first,
install something), not just that a file was touched.
