#!/bin/bash
# Build the PDF report. Pandoc has no LaTeX engine on this machine, so the
# route is Markdown -> styled HTML -> headless Chrome's print-to-PDF.
set -euo pipefail
cd "$(dirname "$0")/.."

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
[ -x "$CHROME" ] || { echo "Chrome not found at $CHROME"; exit 1; }

echo "regenerating figures ..."
.venv/bin/python scripts/make_figures.py
.venv/bin/python scripts/make_report_figures.py
.venv/bin/python scripts/make_report_images.py

echo "markdown -> html ..."
# implicit_figures turns a lone image in a paragraph into <figure> + <figcaption>.
pandoc report/MIDAS_report.md \
  -f markdown+implicit_figures \
  -t html5 --standalone --css=style.css \
  -o report/MIDAS_report.html

echo "html -> pdf ..."
"$CHROME" --headless --disable-gpu --no-pdf-header-footer \
  --virtual-time-budget=20000 \
  --print-to-pdf="$PWD/report/MIDAS_report.pdf" \
  "file://$PWD/report/MIDAS_report.html" 2>/dev/null

echo "wrote report/MIDAS_report.pdf"
