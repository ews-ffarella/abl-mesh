---
applyTo: "**"
---

I want to program a CFD mesher as decribed in this article (file literature/garallo/main_arxiv.tex)
You can also use the pdf and figures.

you will use python 3.12 and above

you will not create pull requests of issues, just give code

prefer class based implementations, with a verbosity level (int) member to control debugging output

you will implement the paper a precisely as possible. All deviations should be reported

you will use libraries matplotlib scipy pyvista

you will provide visualizations functions to view the mesh, and the mesh statistics
for 3d visualizations, use pyvista, for 2d, prefer matplotlib

You can also comment code. When editing code, to do not delete existing comments if they are still relevant

Always explain me the changes. Always add a description when you propose changes: rational, implementation choice, limitations, and especially deviations / conformity to orignal paper

Coding style:

- use python 3.12 and above type annotations (tuple instead of Tuple, dict instead of Dict, etc)
- respect the maximum line length of 100 characters for code (see ruff.toml), and 120 for docstrings.
- Write docstrings for all public classes and methods using Google style.
- Write module based documentation using docstrings, also using Google style.
- Use pathlib instead of os.path
- Avoid using wildcard imports (e.g. from module import \*).
- Use absolute imports
- Comment your code generously, explaining the rationale behind complex or non-obvious decisions.
- Always reference to the original paper when implementing algorithms or methods described therein.
- Document any deviations from the original paper and explain the reasons for these changes.
- Using contextlib.suppress instead of try: ... except: ...
