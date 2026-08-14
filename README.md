# Zekan

**Find out if your model is secretly cheating — and how much it's cheating.**

When you build a machine-learning model, information can accidentally leak into it that it wouldn't have in the real world. The model looks great in testing, then falls apart in production. This is called **data leakage**, and it's one of the most common and expensive mistakes in machine learning.

Most tools just tell you *whether* there's a leak. Zekan tells you what actually matters:

- **How much** the leak is costing you (measured as a drop in your model's real score)
- **Which feature** is causing it
- **Whether it's real** or just statistical noise
- **What Zekan did *not* check** — so a clean result never gives you false confidence

Zekan runs entirely on your own machine. Your data never leaves your computer.

---

## Does Zekan fit my project?

Zekan works when **all** of these are true:

- You're predicting a **yes/no outcome** (e.g. will this customer churn? will this patient be readmitted?).
- Your data is a **table** (rows and columns — CSV or Parquet).
- The same thing appears **more than once over time** — e.g. one customer with several monthly snapshots. Zekan uses this time structure to detect leaks.

If your data has one row per item with no repeats over time, Zekan's core check won't apply yet.

---

## The idea in one line

Zekan trains your model the normal way, then trains an **honest** version with the leak removed, and reports the difference. That difference is the performance that was fake — the part that would vanish in production.

---

## Install

You'll need Python 3.10 or newer.

```bash
# Get the code
git clone https://github.com/91Sakthivel/Zekan-Audit.git
cd Zekan-Audit

# Set up an isolated environment (recommended)
python -m venv .venv

# Turn it on:
#   Windows (PowerShell):
.venv\Scripts\Activate.ps1
#   macOS / Linux:
source .venv/bin/activate

# Install Zekan
pip install -e ".[dev]"
```

Once installed, you can type `zekan` as a command.

---

## Try it in 60 seconds (no setup)

A ready-made example ships with Zekan, so you can see it work before using your own data:

```bash
zekan audit --data examples/churn_instacart/fixture.csv --config examples/churn_instacart/zekan.yml
```

You'll get a plain-language verdict telling you what Zekan found.

---

## A small worked example

Let's walk through that example so the pieces make sense. The data (`fixture.csv`) is a tiny customer-churn table — a few columns, one row per customer per month:

| customer_id | snapshot_date | tenure_months | spend_last_30d | leaky_col | churned |
|---|---|---|---|---|---|
| 1001 | 2023-01-31 | 5 | 42.0 | ... | 0 |
| 1001 | 2023-02-28 | 6 | 38.5 | ... | 1 |
| 1002 | 2023-01-31 | 22 | 110.0 | ... | 0 |

Here's how each column maps to what Zekan asks you:

- **`customer_id`** → the **entity_id**. It's the same customer appearing across several months — that's the "recurring over time" structure Zekan needs.
- **`snapshot_date`** → the **prediction_time**. Each row is a monthly snapshot.
- **`churned`** → the **target**. The yes/no thing we're predicting (1 = churned, 0 = stayed).
- **`spend_last_30d`, `tenure_months`** → normal features. Fair game — they'd be known at prediction time.
- **`leaky_col`** → declared **forbidden**. In this example it's a stand-in for a column that gives away the answer, so we tell Zekan it must not be used.

The matching config (`zekan.yml`) simply writes those choices down:

```yaml
contract:
  entity_id: customer_id
  prediction_time: snapshot_date
  target: churned
  available_features_until: snapshot_date
  forbidden_after_prediction:
    - leaky_col
```

When you run the audit, Zekan checks whether that forbidden column (or any other) is secretly leaking, measures how much, and prints a verdict — for this clean example, **TRUSTED**. Swap in your own data and your own column names, and it's the same three steps below.

---

## Use it on your own data (3 steps)

### Step 1 — Let Zekan set up a config for you

```bash
zekan init --data your_data.csv
```

Zekan reads your file, lists your columns, and asks you a few questions. For each one you can type **the column name or its number** — whichever is easier. It asks for:

- **entity_id** — the column that identifies *one thing* tracked over time (e.g. `customer_id`).
- **prediction_time** — the column with the date/time of each row (e.g. `snapshot_date`).
- **target** — the yes/no thing you're predicting (e.g. `churned`).
- **available_features_until** — usually the same as your time column. It means "only information known up to this point is fair to use."
- **forbidden_after_prediction** — any columns that would *give away the answer* if the model used them. Leave empty if you're not sure; you can add them later. (You can list several, by name or number.)

This writes a file called `zekan.yml`.

> **Not sure what counts as "forbidden"?** A forbidden feature is anything that couldn't actually be known at the moment you'd make the prediction — for example, a field that's only filled in *after* the outcome happens. If in doubt, leave it empty and run the audit; Zekan also screens for leaks you didn't declare.

### Step 2 — Run the audit

```bash
zekan audit --data your_data.csv --config zekan.yml
```

That's it — no other editing needed. Zekan runs the full check and prints a verdict.

### Step 3 — Read the verdict

Zekan gives you one of four results:

| Verdict | What it means |
|---|---|
| **TRUSTED** | No leakage found *in what you declared and what Zekan checks*. (Not a guarantee the whole pipeline is perfect — Zekan tells you its scope.) |
| **RISKY** / **FAILED** | A real, confirmed leak — with how much it's costing you and which feature is responsible. |
| **INCONCLUSIVE** | The result wasn't stable enough to trust. |

---

## Getting a machine-readable result (for automation)

To use Zekan in a CI pipeline or script, add `--json`:

```bash
zekan audit --data your_data.csv --config zekan.yml --json
```

This prints structured JSON (the human-readable text goes to the error stream instead), so another tool can read the verdict automatically.

---

## Commands at a glance

| Command | What it does |
|---|---|
| `zekan init` | Ask a few questions and write a config for you. |
| `zekan audit` | Run the leakage audit and print a verdict. |
| `zekan diff` | Compare two audits to see if leakage got better or worse. |
| `zekan report` | Produce a report from an audit. |
| `zekan benchmark` | Run Zekan's built-in test suite. |

Useful `zekan audit` options (run `zekan audit --help` for all of them):

| Option | What it does |
|---|---|
| `--json` | Machine-readable output for automation. |
| `--stability` | Run several times with different random seeds; downgrade to INCONCLUSIVE if the result isn't stable. |
| `--dry-run` | Check your config is valid without running the full audit. |
| `--estimator NAME` | Choose the model type (`histgb` by default). |

---

## Honest limitations

Zekan is under active development, and we'd rather tell you its limits than oversell it:

- It's been validated on real data (a large public hospital-readmission dataset) through experiments designed in advance — but broader testing across more datasets is still in progress.
- It needs data where the same entity recurs over time; it doesn't fit ordinary one-row-per-item datasets yet.
- Version 1 focuses on yes/no predictions on tabular data.

Every verdict Zekan gives includes a note about what it did and did not check. That honesty is the point.

---

## License

MIT — see [LICENSE](LICENSE). Free to use, modify, and share.
