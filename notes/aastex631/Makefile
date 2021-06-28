main.pdf: main.tex references.bib
	lualatex -halt-on-error main.tex
	bibtex main.aux
	lualatex -halt-on-error  main.tex
	lualatex -halt-on-error  main.tex
