main.pdf: main.tex references.bib
	pdflatex -halt-on-error main.tex
	bibtex main.aux
	pdflatex -halt-on-error  main.tex
	pdflatex -halt-on-error  main.tex
