---
trigger: always_on
description: "Description of the formatting rule"
---

# Agent Rules

- **Stack**: Jupyter Book `==2.1.6` (MyST ecosystem). Config: `myst.yml`, `toc.yml`. No `_config.yml`/`_toc.yml`. Do NOT downgrade jupyter-book.
- **Monorepo**: Contains book (`Chapters/`, `index.md`) & package (`executable_engineering`).
- **Folder Structure**: When creating a new section or chapter directory, the introductory file for that section must be placed *inside* its corresponding subfolder (e.g., `direct_methods/direct_methods.ipynb`), not outside in the parent directory. This keeps component logic modular and prevents filename collisions.
- **Package**: PyPI & local name is `executable_engineering`. Local dev: `-e ./executable_engineering`. Colab: `pip install executable_engineering`. Build: `python -m build ./executable_engineering`.
- **Interactivity**: Core philosophy: interactive numerical methods. PREFER `plotly` for all plotting. Use `ipywidgets`, `numpy`, `scipy`. (Use `matplotlib` only when static output is strictly required).
- **Cross-Environment Execution**: 
  - All pages must include a Colab markdown badge immediately below the main `# Title` in the first markdown cell: `[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/themintlab/ExecutableEngineering/blob/main/...)`
  - Do not assume packages are pre-installed in the cloud. Avoid heavy C-compiled dependencies unsupported by Pyodide/JupyterLite.
  - All required `import` statements (e.g., `numpy`, `plotly`) MUST be grouped together in a single code block located directly after the main `# Title` and Colab badge. This block must include the `executable_engineering` setup and must NOT be hidden. Example:
    ```python
    import numpy as np
    import plotly.graph_objects as go
    
    try:
        import executable_engineering as exe
    except ImportError:
        %pip install -q executable_engineering
        import executable_engineering as exe
    ```
- **Git**: NEVER commit `_build/`, `.venv/`, `.ipynb_checkpoints/`, or `__pycache__/`.
- **CI/CD**: GitHub Actions deploys to GitHub Pages from `main`. Do not commit built HTML files.
- **License**: Content = CC-BY-NC-4.0. Code = MIT.

### Course Context & Math
- **Nature of Course**: Numerical Methods for undergraduate engineers. Focus on algorithmic thinking, computational efficiency, and executable code mirroring math.
- **Math Formatting**: Use robust, standard MathJax (avoid custom macros).
  - *Vectors*: `\mathbf{x}`. *Matrices*: `\mathbf{A}`. *Scalars*: `x`.
  - *Derivatives/Operators*: Upright Roman (`\mathrm{d}x`).
  - *Equations*: Equations should be numbered using MyST syntax: `$$ math $$ (label)`. Ensure they have horizontal scroll if at all possible.
  - *Equation Arrays*: Use `\begin{aligned} ... \end{aligned}` inside `$$ ... $$` for robust multi-line alignment that remains compatible with MyST labels. Do not use the obsolete `eqnarray`.
  - *File Naming*: All files and directories within the book content (e.g., inside Chapters/) must be named using `snake_case` (all lowercase, no spaces). This ensures maximum compatibility with command-line tools and URLs.
  - *Page Titles*: Every Markdown or Jupyter Notebook file must contain exactly one top-level header (e.g., `# Title`) at the very top of the document. Jupyter Book uses this H1 header for the sidebar index and page title.

### Pedagogical Style Guide (Colab Presentation Mode)
- **Chunking**: Chunk information using heavy Markdown headers (e.g., `#`, `##`, `###`) to create logical presentation slides for Google Colab Presentation mode.
- **Notebook Cells**: Separate distinct ideas into their own Markdown cells. Keep markdown concise, focusing on intuition rather than walls of text.
- **Code Simplicity & Readability**: Keep Python code as simple and concise as possible so it fits cleanly on a single presentation slide without scrolling.
- **No Docstrings**: Forgo standard documentation and docstrings in the notebooks unless absolutely necessary for the lesson.
- **Abstraction**: Abstract heavy boilerplate (like plotting setup) into the `executable_engineering` Python package to keep the notebook focused on the core math/concept.
- **Focus**: Emphasize comprehension over rote memorization. Target audience: 2nd-year undergraduate engineering physics students. Rely heavily on geometric interpretation and visual proofs.