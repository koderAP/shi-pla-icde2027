.PHONY: all clean

TEXINPUTS := $(CURDIR)/../IEEE-conference-template-062824:

all: ieee.pdf main.pdf

ieee.pdf: ieee.tex refs.bib
	TEXINPUTS="$(TEXINPUTS)" latexmk -pdf -interaction=nonstopmode -halt-on-error ieee.tex

main.pdf: ieee.pdf
	ln -sfn ieee.pdf main.pdf

clean:
	latexmk -C ieee.tex
	rm -f main.pdf
