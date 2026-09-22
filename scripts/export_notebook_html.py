from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "phase1_backtesting.ipynb"
HTML = PROJECT_ROOT / "notebooks" / "phase1_backtesting.html"


subprocess.run(
    [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "html",
        "--HTMLExporter.embed_images=True",
        '--HTMLExporter.mathjax_url=',
        '--HTMLExporter.require_js_url=',
        str(NOTEBOOK),
    ],
    cwd=PROJECT_ROOT,
    check=True,
)

document = HTML.read_text(encoding="utf-8")

# nbconvert's Lab template ships an optional Mermaid loader even when the
# notebook contains no Mermaid diagram. Remove that unused remote loader so the
# saved report remains self-contained and opens without an internet connection.
document, substitutions = re.subn(
    r'(<!-- End of mathjax configuration -->)<script type="module">.*?</script>',
    r"\1",
    document,
    count=1,
    flags=re.DOTALL,
)
if substitutions != 1:
    raise RuntimeError("Could not locate the optional Mermaid loader in exported HTML")

document = re.sub(r'<script src="">\s*</script>', "", document)
active_remote_dependency = re.search(
    r'<(?:script|img|iframe|link)\b[^>]*(?:src|href)=["\']https?://|import\(["\']https?://',
    document,
    flags=re.IGNORECASE,
)
if active_remote_dependency:
    raise RuntimeError(f"Export still contains a remote dependency near byte {active_remote_dependency.start()}")

HTML.write_text(document, encoding="utf-8")
print(HTML)
