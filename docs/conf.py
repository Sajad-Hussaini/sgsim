project = 'SGSIM'
copyright = '2025, Sajad Hussaini'
author = 'Sajad Hussaini'
release = '1.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # To understand NumPy style docstrings
    'sphinx.ext.viewcode',
    'myst_parser',
    'nbsphinx']

nbsphinx_allow_errors = True
nbsphinx_execute = 'never'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autodoc_typehints = "none"