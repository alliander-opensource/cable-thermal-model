# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0
"""Add wheel zoom and drag-to-pan to a Pyreverse Mermaid HTML file.

Usage:
    python scripts/add_mermaid_zoom.py classes_cable_thermal_model.html
    python scripts/add_mermaid_zoom.py classes_cable_thermal_model.html packages_cable_thermal_model.html
    python scripts/add_mermaid_zoom.py classes_cable_thermal_model.html --in-place
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MERMAID_DIV_OPEN = re.compile(
    r"""<div\b(?=[^>]*\bclass\s*=\s*["'][^"']*\bmermaid\b[^"']*["'])[^>]*>""",
    re.IGNORECASE,
)
DIV_CLOSE = re.compile(r"</div\s*>", re.IGNORECASE)
HTML_OPEN = re.compile(r"<html\b[^>]*>", re.IGNORECASE)
HEAD_OPEN = re.compile(r"<head\b[^>]*>", re.IGNORECASE)
HEAD_CLOSE = re.compile(r"</head\s*>", re.IGNORECASE)
BODY_OPEN = re.compile(r"<body\b[^>]*>", re.IGNORECASE)
BODY_CLOSE = re.compile(r"</body\s*>", re.IGNORECASE)
MERMAID_SCRIPT = re.compile(
    r"""<script\b(?=[^>]*\bsrc\s*=\s*["'][^"']*mermaid[^"']*["'])[^>]*>\s*<\s*/\s*script(?:\s+[^>]*)?\s*>""",
    re.IGNORECASE,
)
PINNED_MERMAID_SCRIPT = '<script src="https://cdn.jsdelivr.net/npm/mermaid@12.2.0/dist/mermaid.min.js"></script>'

HEAD_CONTENT = """\
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pyreverse Mermaid Diagram</title>
  <style>
    html,
    body {
      width: 100%;
      height: 100%;
      margin: 0;
      overflow: hidden;
    }

    #diagram-viewport {
      position: fixed;
      inset: 0;
      overflow: hidden;
      cursor: grab;
      touch-action: pinch-zoom;
    }

    #diagram-viewport.is-panning {
      cursor: grabbing;
    }

    #diagram {
      width: 100%;
      height: 100%;
    }

    #diagram svg {
      display: block;
      width: 100%;
      height: 100%;
      max-width: none !important;
      user-select: none;
    }
  </style>
"""

VIEWPORT_OPEN = (
    '<main id="diagram-viewport" data-mermaid-zoom="1" '
    'aria-label="Diagram viewport. Scroll to zoom and drag to pan.">\n'
    '  <div id="diagram" class="mermaid">'
)

ZOOM_SCRIPT = """\
<script>
  mermaid.initialize({ startOnLoad: false });
  mermaid.run({ querySelector: "#diagram" }).then(() => {
    const viewport = document.querySelector("#diagram-viewport");
    const svg = document.querySelector("#diagram svg");

    if (!(viewport instanceof HTMLElement) || !(svg instanceof SVGSVGElement)) {
      throw new Error("Mermaid did not render the diagram.");
    }

    const initialViewBox = svg.viewBox.baseVal;
    if (initialViewBox.width <= 0 || initialViewBox.height <= 0) {
      throw new Error("The rendered diagram has no usable viewBox.");
    }

    let zoom = 1;
    let pan = null;

    viewport.addEventListener("wheel", (event) => {
      event.preventDefault();

      const matrix = svg.getScreenCTM();
      if (!matrix) {
        throw new Error("The rendered diagram has no screen transform.");
      }

      const pointer = svg.createSVGPoint();
      pointer.x = event.clientX;
      pointer.y = event.clientY;
      const anchor = pointer.matrixTransform(matrix.inverse());
      const viewBox = svg.viewBox.baseVal;
      const nextZoom = Math.min(8, Math.max(0.25, zoom * Math.exp(-event.deltaY * 0.001)));
      const ratio = nextZoom / zoom;

      svg.setAttribute(
        "viewBox",
        `${anchor.x - (anchor.x - viewBox.x) / ratio} ` +
          `${anchor.y - (anchor.y - viewBox.y) / ratio} ` +
          `${viewBox.width / ratio} ${viewBox.height / ratio}`,
      );
      zoom = nextZoom;
    }, { passive: false });

    viewport.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || !svg.contains(event.target)) {
        return;
      }

      const matrix = svg.getScreenCTM();
      if (!matrix) {
        throw new Error("The rendered diagram has no screen transform.");
      }

      const viewBox = svg.viewBox.baseVal;
      pan = {
        pointerId: event.pointerId,
        clientX: event.clientX,
        clientY: event.clientY,
        x: viewBox.x,
        y: viewBox.y,
        width: viewBox.width,
        height: viewBox.height,
        scaleX: matrix.a,
        scaleY: matrix.d,
      };
      viewport.setPointerCapture(event.pointerId);
      viewport.classList.add("is-panning");
      event.preventDefault();
    });

    viewport.addEventListener("pointermove", (event) => {
      if (!pan || event.pointerId !== pan.pointerId) {
        return;
      }

      const deltaX = (event.clientX - pan.clientX) / pan.scaleX;
      const deltaY = (event.clientY - pan.clientY) / pan.scaleY;
      svg.setAttribute(
        "viewBox",
        `${pan.x - deltaX} ${pan.y - deltaY} ${pan.width} ${pan.height}`,
      );
    });

    const endPan = (event) => {
      if (pan && event.pointerId === pan.pointerId) {
        pan = null;
        viewport.classList.remove("is-panning");
      }
    };

    viewport.addEventListener("pointerup", endPan);
    viewport.addEventListener("pointercancel", endPan);
  });
</script>
"""


def add_zoom_support(html: str) -> str:
    """Wrap the Pyreverse Mermaid diagram and add viewport interaction handlers."""
    if 'data-mermaid-zoom="1"' in html or re.search(r'<main\b[^>]*\bid="diagram-viewport"', html, re.IGNORECASE):
        return html

    mermaid_divs = list(MERMAID_DIV_OPEN.finditer(html))
    if len(mermaid_divs) != 1:
        raise ValueError(f"Expected one Pyreverse Mermaid div; found {len(mermaid_divs)}.")

    mermaid_scripts = list(MERMAID_SCRIPT.finditer(html))
    if len(mermaid_scripts) != 1:
        raise ValueError(f"Expected one Pyreverse Mermaid script tag; found {len(mermaid_scripts)}.")

    html_open = HTML_OPEN.search(html)
    body_open = BODY_OPEN.search(html)
    body_close = BODY_CLOSE.search(html)
    if not html_open or not body_open or not body_close:
        raise ValueError("Expected an HTML document with html and body elements.")

    diagram_open = mermaid_divs[0]
    if not (body_open.end() < diagram_open.start() < body_close.start()):
        raise ValueError("The Mermaid diagram must be inside the document body.")

    diagram_close = DIV_CLOSE.search(html, diagram_open.end())
    if not diagram_close or diagram_close.start() > body_close.start():
        raise ValueError("Could not find the closing tag for the Mermaid div.")

    html = (
        html[: diagram_open.start()]
        + VIEWPORT_OPEN
        + html[diagram_open.end() : diagram_close.start()]
        + "  </div>\n</main>"
        + html[diagram_close.end() :]
    )
    html = MERMAID_SCRIPT.sub(PINNED_MERMAID_SCRIPT, html, count=1)

    head_open = HEAD_OPEN.search(html)
    if head_open:
        head_close = HEAD_CLOSE.search(html, head_open.end())
        if not head_close:
            raise ValueError("Found an opening head tag without a closing tag.")
        html = html[: head_close.start()] + HEAD_CONTENT + html[head_close.start() :]
    else:
        html_open = HTML_OPEN.search(html)
        if not html_open:
            raise ValueError("Could not find the opening html tag.")
        html = html[: html_open.end()] + "\n<head>\n" + HEAD_CONTENT + "</head>" + html[html_open.end() :]

    body_close = BODY_CLOSE.search(html)
    if not body_close:
        raise ValueError("Could not find the closing body tag.")
    return html[: body_close.start()] + ZOOM_SCRIPT + html[body_close.start() :]


def parse_args() -> argparse.Namespace:
    """Parse the input, output, and in-place options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="one or more Pyreverse-generated HTML files",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output file (default: <input>_zoom.html)",
    )
    output_group.add_argument(
        "--in-place",
        action="store_true",
        help="replace the input file instead of writing a separate output",
    )
    args = parser.parse_args()
    if args.output and len(args.inputs) != 1:
        parser.error("--output can only be used with one input file.")
    return args


def main() -> int:
    """Enhance a Pyreverse-generated HTML file."""
    args = parse_args()
    exit_code = 0
    for source in args.inputs:
        if args.in_place:
            destination = source
        elif args.output:
            destination = args.output
        else:
            destination = source.with_name(f"{source.stem}_zoom{source.suffix}")

        try:
            if not source.is_file():
                raise FileNotFoundError(f"Input file does not exist: {source}")
            if not args.in_place and destination.resolve() == source.resolve():
                raise ValueError("Use --in-place to overwrite the input file.")

            html = source.read_text(encoding="utf-8")
            enhanced_html = add_zoom_support(html)
            destination.write_text(enhanced_html, encoding="utf-8")
        except (OSError, ValueError) as error:
            print(f"Error processing {source}: {error}", file=sys.stderr)
            exit_code = 1
            continue

        print(f"Wrote zoom-enabled diagram to {destination}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
