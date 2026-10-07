# Change Log — ML-bike (CS-C3240)

What changed in the project, newest first. Add an entry whenever you change something
another team member would be surprised by: file moves, notebook edits, new results,
environment requirements.

---

## 2026-10-07 — Stage 2: the model now knows the route; three models; notebook rewritten

> This supersedes the "Stage 2 results so far" section of the 2026-10-06 entry. Those
> numbers were measured without route information.

### The big change: route length is now a feature

Stage 1 left distance out entirely, so the model could not "cheat". That mixed up two
different things:

- the **recorded distance of the trip itself** is only known after the ride ends — it
  stays excluded;
- the **route** (start and end station) is known before the ride, and our own problem
  statement assumes the rider knows their destination — it is now included.

How route length works:

- It is the median distance ridden on the same station pair in **earlier** data, never
  the trip's own distance.
- Training trips are **cross-fitted**: a 2022 trip gets its route length from 2023 trips,
  and vice versa, so no trip ever sees its own distance.
- A route needs at least 3 earlier trips. Otherwise it falls back to the departure
  station's median, then to the overall median.
- We also added a **round-trip flag** (3.2% of trips return to their starting station).
- The label is now modelled as **log(duration)**, so every feature acts as a percentage
  change.

Why: without the route, time and weather captured 1.5% of the achievable improvement.
With it, the typical error on 2024 drops from 5.30 to 1.86 minutes.

### Models: down to three

| Model | Role |
|---|---|
| Linear, squared error | control for the loss comparison |
| **Linear, Huber** | main model — selected |
| Boosted trees, absolute error | the non-linear alternative |

Each pair differs in exactly one thing. Two reference rows sit alongside them: a
**constant median**, and **physics** (route length × one typical pace, no machine
learning at all).

**Removed:** constant mean, LAD (it was under-trained), boosted trees with squared error,
and the RMSE and R² metrics.

The selection rule was declared before any results: **lowest validation MAE**. All
comparisons now come with 95% confidence intervals from a day-block bootstrap, on both
2024 and 2025.

### Cleaning

- Dropped **1,264 moves to or from workshop and test stations** (`997` Workshop Helsinki,
  `999` test stations). These are maintenance moves, not rides.
- Labels are now stored as 64-bit floats; 32-bit labels broke the log/exp round-trip
  check that scikit-learn runs.

### Notebook

- **`ML_bike.ipynb` was rewritten in place** — 14 sections, executed, outputs saved. A
  full rerun takes about 2–3 minutes.
- The previous version (7 models, no route) is still in git, in commit `7cc258f`
  ("update version n + n"):
  `git show 7cc258f:ML_bike.ipynb > ML_bike_old.ipynb`
- An accidentally duplicated cell from the old version is gone.
- `ML_bike_stage1/` is untouched — it is the submitted Stage 1 appendix.

### Results on the 2025 test season (never used before this run)

| Model | MAE | MedAE | Within 2 min |
|---|---|---|---|
| Constant median | 14.89 | 5.43 | 18.6% |
| Physics (no ML) | 11.95 | 2.14 | 48.0% |
| Linear, squared error | 11.70 | 2.35 | 43.3% |
| **Linear, Huber (selected)** | **11.55** | **1.91** | **51.5%** |
| Boosted trees, absolute error | 11.60 | 1.89 | 51.8% |

Findings:

1. **Route length does almost everything** — 99.8% of the improvement. The physics row
   alone achieves 61 of the 65 percentage points of improvement in typical error.
2. **Huber still beats squared error on a log scale:** +6.4 points within 2 minutes on
   2024, +8.2 on 2025. Squared error predicts ordinary trips about half a minute too long.
3. **Trees and linear are equal in practice.** Differences of ~3 seconds in MAE or 0.3
   points within 2 minutes, with the sign flipping between metrics.
4. **Time matters a little, through trip purpose rather than traffic.** The morning commute
   is the fastest time of day; weekends are 4–5% slower on the same route.
5. **Weather does not help.** Rain makes trips slightly *faster and shorter*, not slower.

### Stage 2 report

- **The file to submit is `ML_bike_stage2/Stage2_Submission.pdf`.** It contains the
  4-page report followed by the notebook as the code appendix (32 pages; the appendix
  does not count toward the limit).
- `Stage2_Report_Draft.docx` / `.pdf` is the report on its own, kept in case it needs
  editing.
- **Hard limit: 4 pages, excluding the appendix.** Anything beyond page 4 is not read or
  graded. The report currently ends about 60% of the way down page 4, so any edit must
  keep it there.
- It follows the course outline and the peer-review rubric
  (`ML_bike_stage2/PeerReview_assignmentDescription-2-2026_updated.pdf`), with sections
  1 Introduction, 2 Problem Formulation, 3 Methods, 4 Results, 5 Conclusions, an
  unnumbered *Use of AI* section, 6 References and 7 Appendix. The introduction ends with
  the section-by-section overview the rubric asks for (Q1.2).
- It is anonymous, since grading is by peer review: no names, and "Anonymous" as the
  author in both the Word and PDF metadata. The notebook was checked for names, emails
  and user paths, and none were found.

- **2026-10-08: report rewritten for readability.**
  - It now uses plain language, and the Introduction and Methods contain no results.
  - Results carries the evidence, as four claims in order: route length does most of the
    work, ML adds a small but real gain, the Huber loss matters, and trees don't help.
  - A small feature table (Table 3) shows that weather adds nothing.
  - Same numbers, same notebook, still 4 pages: the report now ends halfway down page 4.
    `Stage2_Submission.pdf` has been rebuilt.

### Notebook: training errors added

- Step 8 has one new cell that scores every predictor on the training sample it was fitted
  on (rubric Q4.1 asks for training *and* validation errors). No existing result changed.
- Training errors are lower than validation errors for *every* predictor, including the
  constant. So the gap reflects the 2024 season being harder, not overfitting.

### What you need to do

- **Check the *Use of AI* section** before submitting. It must describe what the group
  actually did; edit it if anything is inaccurate.
- **Deadline: 7 Oct 2026, 23:59.** Late submissions lose 30% of the submission points.
- **Don't reuse the Stage 1 sentence "a rider is a little slower in the rain".** The data
  contradicts it.
- **The `ML_bike.ipynb` in commit "Update 5" is an intermediate run.** The final version
  has the full interpretation text and a corrected test-set plot: the old per-hour plot
  wrongly suggested a 1-minute bias. Commit the current file, not that one.
- `summarize.md` has been rewritten to match these results.

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

### Stage 2 results so far — superseded by 2026-10-07 (measured without the route)

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
