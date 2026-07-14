ROOT := $(abspath ..)
PY   := $(ROOT)/.venv/bin/python

.PHONY: all pdf tables figures check final-check clean

all: references.bib tables pdf

references.bib: $(ROOT)/docs/whitepaper/sections/13_references.bib.md
	awk '/^```bibtex/{f=1;next} /^```/{f=0} f' $< > $@

tables:
	$(PY) $(ROOT)/scripts/paper_tables.py --out generated

figures:
	cd $(ROOT) && $(PY) -m fifa_fantasy.report

pdf:
	tectonic --keep-logs --keep-intermediates paper.tex

check: pdf
	@echo "POSTFINAL placeholders remaining:"
	@grep -rc 'POSTFINAL' sections/ paper.tex || true
	@! grep -q 'There were undefined references' paper.log
	@! grep -q "didn't find a database entry" paper.log

final-check: check
	@! grep -rq 'POSTFINAL' sections/ paper.tex
	@! grep -riq 'GLM 5.2\|conversation log' sections/
	@git -C $(ROOT) diff --quiet -- paper/generated paper/references.bib \
	  || (echo 'generated fragments stale: run make tables'; exit 1)

clean:
	rm -f paper.pdf *.log *.aux *.bbl *.blg *.xdv
