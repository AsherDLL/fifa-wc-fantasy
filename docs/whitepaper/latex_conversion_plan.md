# LaTeX conversion plan (post-tournament)

A working note for how we'll turn the Markdown sections into a publishable
LaTeX paper after the WC final.

## Toolchain

- **pandoc** with a custom LaTeX template for the markdown-to-tex pass
- `latexmk -pdf` for compilation
- `chktex` and `lacheck` for static analysis
- `bibtex` for the reference list

Install on Linux Mint:

```bash
sudo apt install pandoc texlive-full chktex
```

## Conversion steps

1. **Concatenate sections** in order into a single `paper.md` file.

   ```bash
   cat docs/whitepaper/sections/{00..13,AA}_*.md > /tmp/paper.md
   ```

2. **Run pandoc** with a template that emits clean LaTeX:

   ```bash
   pandoc /tmp/paper.md \
     -o paper.tex \
     --standalone \
     --template=docs/whitepaper/latex_template.tex \
     --bibliography=docs/whitepaper/references.bib \
     --csl=docs/whitepaper/ieee.csl \
     -V documentclass=article \
     -V geometry:margin=2.5cm \
     -V fontsize=11pt
   ```

3. **Hand-polish** the LaTeX output:
   - Replace pandoc tables with `\toprule`, `\midrule`, `\bottomrule` from `booktabs`
   - Convert code blocks to `lstlisting` with a clean style
   - Replace algorithm pseudocode with `algorithm2e` environments
   - Add `subcaption` for multi-panel figures
   - Validate every `\cite{}` against `references.bib`
   - Add a title page with author, date, and abstract

4. **Compile and review:**

   ```bash
   latexmk -pdf paper.tex
   ```

5. **Submit** to a sports analytics venue (e.g. KDD Sports Analytics
   workshop, MIT Sloan Sports Analytics Conference) or post as a
   preprint on arXiv under `cs.LG` or `stat.AP`.

## LaTeX template skeleton (to draft)

Save under `docs/whitepaper/latex_template.tex`:

```latex
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{algorithm2e}
\usepackage{listings}
\usepackage{hyperref}
\usepackage{cite}
% ... more packages

\title{A Multi-Backend Predictive Framework for FIFA World Cup Fantasy Football}
\author{Asher Davila}
\date{\today}

\begin{document}
\maketitle
\begin{abstract}
$body$
\end{abstract}
\end{document}
```

## Figures we'll need

- **Figure 1**: System architecture diagram (collector to external to
  features to model to optimizer to results)
- **Figure 2**: Held-out RMSE per backend per position (bar chart)
- **Figure 3**: Per-matchday user score vs field median (line chart),
  with annotations for transfer decisions
- **Figure 4**: Captain decision dispersion (model vs user, with
  realised points marker)
- **Figure 5**: Country-Elo gap for each R32 fixture (horizontal bar)
- **Figure 6** (optional): Monte Carlo distribution of one matchday's
  total points under three captain choices (density plot)

Each figure must reference the script under `scripts/` that generates
it.

## Submission targets to consider

- **KDD Sports Analytics workshop** (annual, Aug-ish)
- **MIT Sloan Sports Analytics Conference** (Feb-Mar, more applied)
- **IEEE MLSP** (machine learning for signal processing, accepts sports
  applications)
- **arXiv preprint** under `cs.LG` (no review, immediate publication)

For a master's-degree-quality write-up, the arXiv preprint plus a
sports-analytics workshop is the right scope. Top-tier conference
submission would require more controlled experiments and a larger
empirical study.
