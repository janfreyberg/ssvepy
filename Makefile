# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = ssvepy
SOURCEDIR     = ./doc
GH_PAGES_SOURCES = Makefile doc
BUILDDIR      = doc/_build
# BUILDDIR      = ..

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

gh-pages:
			git checkout gh-pages
			rm -rf *
			git checkout master $(GH_PAGES_SOURCES)
			git reset HEAD
			make html
			mv -fv $(BUILDDIR)/html/* ./
			mv -fv $(BUILDDIR)/html/_sources/* ./_sources/
			mv -fv $(BUILDDIR)/html/_static/* ./_static/
			mv -fv $(BUILDDIR)/html/_modules/* ./_modules/
			rm -rf $(SOURCEDIR) build
			> .nojekyll
			git add -A
			git ci -m "Generated gh-pages for `git log master -1 --pretty=short --abbrev-commit`" && git push origin gh-pages ; git checkout master
