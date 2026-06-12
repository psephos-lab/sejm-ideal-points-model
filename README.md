# Sejm Ideal Points — a Bayesian spatial voting model

Estimating the positions of Polish *Sejm* deputies (10th term) on the **main axis of
division** recovered from their roll-call votes, using a Bayesian spatial voting model
fit by MCMC. An educational walk through computational statistics — that happened to
fill a real gap: a public, per-MP ideal-point map that Poland lacked.

🌍 **Interactive site:** https://mapasejmu.org &nbsp;·&nbsp; 📄 **Write-up:** [`article/ARTICLE.md`](article/ARTICLE.md)

## What it does

Each deputy gets a latent position `x_i`; each vote gets a discrimination `β_j` and a
threshold `α_j`, with `P(yea) = Φ(β_j·x_i − α_j)` (two-parameter IRT / probit). The
posterior is sampled with a **Gibbs sampler (Albert–Chib data augmentation)**.

The recovered axis is deliberately **not** labelled "left–right": roll-call votes do
carry ideological signal, but rendering an ideological verdict is a judgment this
project leaves to the reader — it supplies the **main axis of division** as data. It
most plausibly reflects the **government–opposition** split. The write-up covers the rest of the story: why NUTS
collapses on this geometry and Gibbs wins, a parameter-expansion fix, a negative result
on marginal data augmentation, a data-pipeline bug worth remembering, and a
dimensionality analysis (the chamber is ~1-dimensional).

![Ideal points by club](article/figures/ideal_points.png)

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python fetch_data.py        # roll-call from api.sejm.gov.pl -> data/*.pkl
python run.py               # 1-D Gibbs sampler -> results/ (ideal_points.csv, figures)
python make_site_data.py    # export site JSON
```
Optional: `run_2d.py` (2-D), `pilot_dims.py` (how many dimensions?), `interpret_dim2.py`.

## Repository

| Path | What |
|---|---|
| `fetch_data.py` | Sejm API client (probes each proceeding; handles the multi-day numbering bug) |
| `gibbs.py` / `gibbs_nd.py` | Albert–Chib Gibbs sampler (1-D / N-D) |
| `model.py` | NUTS path — kept to document *why* it fails here |
| `run.py`, `run_2d.py` | pipelines (1-D / 2-D) |
| `make_site_data.py`, `make_history_data.py`, `make_model_params.py` | export site JSON |
| `pilot_dims.py`, `interpret_dim2.py`, `visualize.py` | dimensionality, interpretation, plots |
| `article/ARTICLE.md`, `REFERENCES.md` | write-up and bibliography |
| `data/term10_rollcall_v2.pkl` | input snapshot · `results/*.csv`, `results/*.png` — outputs |

**Note:** raw MCMC draws (`results/draws*.npz`, ~2 GB) are **not** committed — rerun the
pipeline to regenerate them. Everything is reproducible from the public Sejm API.

## Disclaimer

An "ideal point" is a position in vote space, **not** a judgment of a politician;
uncertainty grows for deputies who vote rarely. This is an educational / analytical
project on computational statistics, not a political ranking.

## License

Code: MIT. Text and figures: CC-BY-4.0. Roll-call data: public (Sejm API).
