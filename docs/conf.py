project = 'SGSIM'
copyright = '2025, Sajad Hussaini'
author = 'Sajad Hussaini'
from sgsim import __version__ as release

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'myst_parser',
    'sphinx_copybutton',
    'nbsphinx',
]

# Notebook settings
nbsphinx_allow_errors = True
nbsphinx_execute = 'never'

# General configuration
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML output options
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
}

# Syntax highlighting
pygments_style = 'sphinx'

# Autodoc settings (CLEAN LOOK)
autodoc_default_options = {
    'members': True,
    'undoc-members': False,        # Hide undocumented members
    'show-inheritance': True,
    'member-order': 'bysource',    # Keep source order
    'exclude-members': '__weakref__, __dict__, __module__',  # Hide noise
}
autodoc_typehints = 'none'         # Keep this - prevents type hint clutter
autodoc_typehints_format = 'short'

# Napoleon settings (NumPy style)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True
napoleon_include_init_with_doc = False  # Don't duplicate __init__ docs
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = False

# Math settings
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'

# Copy button settings
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
