# Manuscript

- `main.md` — readable Markdown draft (source of truth for prose).
- `main.tex` — self-contained LaTeX article (mirrors `main.md`).
- `references.bib` — BibTeX entries.

## Build

```powershell
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Before submission

- References were checked against OpenAlex on 2026-09-13. Conferences often have
  an arXiv preprint indexed first (e.g., MERT, CLaMP, VICReg, V-JEPA,
  Wang & Isola), so the published venue in `references.bib` is kept but should be
  confirmed. The MERT author list is truncated with `and others` (20 authors).
  Book entries (Krippendorff; Godøy & Leman; Meyer; Efron & Tibshirani) are not
  covered by OpenAlex and were not machine-verified.
- The human ontology and representation comparison are preregistered but not yet
  collected; the empirical claims must be added only after the study is run.
- Preliminary numbers in Section 6 use provisional transformation hypotheses and
  must be labeled as engineering validation, not evidence.
