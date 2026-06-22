import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  Presentation,
  PresentationFile,
} from "/Users/aritra/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const previewRoot = path.join(here, "_previews");
const slideDir = path.join(here, "slides");
const previewDir = path.join(slideDir, "preview");
const layoutDir = path.join(slideDir, "layout");
const qaDir = path.join(slideDir, "qa");
const pptxPath = path.join(here, "dijet_rho_unfolding_jacobian_reg_slides.pptx");

const W = 1280;
const H = 720;
const palette = {
  bg: "#f8fafc",
  ink: "#0f172a",
  muted: "#475569",
  faint: "#e2e8f0",
  panel: "#ffffff",
  accent: "#0f766e",
  warn: "#b45309",
};

const slides = [
  {
    title: "Dijet rho unfolding",
    subtitle: "2018, Jacobian statistical propagation, ratio-curvature regularization",
    kind: "title",
  },
  {
    title: "Run configuration and scope",
    kind: "bullets",
    bullets: [
      "Command: python scripts/run_unfolding.py --channel dijet --observable rho --year 2018 --jacobian --regularization ratio_curvature",
      "Output: outputs/dijet/2018/rho/unfolding_jacobian_reg/",
      "Resolved tau: ungroomed 0.7907, groomed 0.6954",
      "Uncertainties are partial: input-data statistics plus available response variations.",
      "Excluded by design for dijet/trijet inputs: response MC statistics and HERWIG/model uncertainty.",
    ],
  },
  {
    title: "Unfolded results: summary overview",
    kind: "twoImages",
    images: [
      ["Ungroomed", "ungroomed_summary.png"],
      ["Groomed", "groomed_summary.png"],
    ],
  },
  {
    title: "Unfolded results: full unrolled spectra",
    kind: "twoImages",
    images: [
      ["Ungroomed", "unfold/unfolded_unrolled_2d_ungroomed.png"],
      ["Groomed", "unfold/unfolded_unrolled_2d_groomed.png"],
    ],
  },
  {
    title: "Response matrices",
    kind: "twoImages",
    images: [
      ["Ungroomed", "unfold/response_ungroomed.png"],
      ["Groomed", "unfold/response_groomed.png"],
    ],
  },
  {
    title: "Normalized statistical correlations",
    kind: "twoImages",
    images: [
      ["Ungroomed", "unfold/correlation_ungroomed.png"],
      ["Groomed", "unfold/correlation_groomed.png"],
    ],
  },
  {
    title: "L-curve regularization scans",
    kind: "twoImages",
    images: [
      ["Ungroomed tau = 0.7907", "unfold/lcurve_ungroomed.png"],
      ["Groomed tau = 0.6954", "unfold/lcurve_groomed.png"],
    ],
  },
  {
    title: "Unfolded panels: ungroomed",
    kind: "grid",
    images: [
      ["200-290 GeV", "unfold/unfolded_basic_ungroomed_0.png"],
      ["290-400 GeV", "unfold/unfolded_basic_ungroomed_1.png"],
      ["400-570 GeV", "unfold/unfolded_basic_ungroomed_2.png"],
      ["570-760 GeV", "unfold/unfolded_basic_ungroomed_3.png"],
      ["> 760 GeV", "unfold/unfolded_basic_ungroomed_4.png"],
    ],
  },
  {
    title: "Unfolded panels: groomed",
    kind: "grid",
    images: [
      ["200-290 GeV", "unfold/unfolded_basic_groomed_0.png"],
      ["290-400 GeV", "unfold/unfolded_basic_groomed_1.png"],
      ["400-570 GeV", "unfold/unfolded_basic_groomed_2.png"],
      ["570-760 GeV", "unfold/unfolded_basic_groomed_3.png"],
      ["> 760 GeV", "unfold/unfolded_basic_groomed_4.png"],
    ],
  },
  {
    title: "Grouped uncertainty summary: ungroomed",
    kind: "grid",
    images: [
      ["200-290 GeV", "uncertainties/summary_grouped_ungroomed_0.png"],
      ["290-400 GeV", "uncertainties/summary_grouped_ungroomed_1.png"],
      ["400-570 GeV", "uncertainties/summary_grouped_ungroomed_2.png"],
      ["570-760 GeV", "uncertainties/summary_grouped_ungroomed_3.png"],
      ["> 760 GeV", "uncertainties/summary_grouped_ungroomed_4.png"],
    ],
  },
  {
    title: "Grouped uncertainty summary: groomed",
    kind: "grid",
    images: [
      ["200-290 GeV", "uncertainties/summary_grouped_groomed_0.png"],
      ["290-400 GeV", "uncertainties/summary_grouped_groomed_1.png"],
      ["400-570 GeV", "uncertainties/summary_grouped_groomed_2.png"],
      ["570-760 GeV", "uncertainties/summary_grouped_groomed_3.png"],
      ["> 760 GeV", "uncertainties/summary_grouped_groomed_4.png"],
    ],
  },
  {
    title: "Statistical uncertainty fractions",
    kind: "twoImages",
    images: [
      ["Ungroomed, 400-570 GeV", "uncertainties/stat_fraction_ungroomed_2.png"],
      ["Groomed, 400-570 GeV", "uncertainties/stat_fraction_groomed_2.png"],
    ],
  },
  {
    title: "Fake-rate and efficiency diagnostics",
    kind: "twoImages",
    images: [
      ["Ungroomed, 400-570 GeV", "fakerates_ungroomed_2.png"],
      ["Groomed, 400-570 GeV", "fakerates_groomed_2.png"],
    ],
  },
  {
    title: "Purity and stability checks",
    kind: "twoImages",
    images: [
      ["Ungroomed, 400-570 GeV", "purity_stability_ungroomed_2.png"],
      ["Groomed, 400-570 GeV", "purity_stability_groomed_2.png"],
    ],
  },
];

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function addText(slide, text, position, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = style;
  return shape;
}

async function addImage(slide, relPath, position, alt) {
  const filePath = path.join(previewRoot, relPath);
  const blob = await fs.readFile(filePath);
  slide.images.add({
    blob,
    contentType: "image/png",
    alt,
    fit: "contain",
    position,
  });
}

function addHeader(slide, title) {
  addText(slide, title, { left: 48, top: 28, width: 980, height: 44 }, {
    fontSize: 30,
    bold: true,
    color: palette.ink,
  });
  addText(slide, "dijet 2018 rho | Jacobian + ratio-curvature regularization", {
    left: 920,
    top: 34,
    width: 312,
    height: 28,
  }, {
    fontSize: 12,
    color: palette.muted,
    alignment: "right",
  });
}

async function drawPlotSlide(presentation, spec) {
  const slide = presentation.slides.add();
  slide.background.fill = palette.bg;
  addHeader(slide, spec.title);

  if (spec.kind === "twoImages") {
    const frames = [
      { left: 52, top: 104, width: 566, height: 548 },
      { left: 662, top: 104, width: 566, height: 548 },
    ];
    for (let i = 0; i < spec.images.length; i++) {
      const [label, relPath] = spec.images[i];
      addText(slide, label, { left: frames[i].left, top: 76, width: frames[i].width, height: 24 }, {
        fontSize: 16,
        bold: true,
        color: palette.accent,
      });
      await addImage(slide, relPath, frames[i], `${spec.title}: ${label}`);
    }
  } else if (spec.kind === "grid") {
    const frames = [
      { left: 40, top: 94, width: 380, height: 250 },
      { left: 450, top: 94, width: 380, height: 250 },
      { left: 860, top: 94, width: 380, height: 250 },
      { left: 230, top: 390, width: 380, height: 250 },
      { left: 670, top: 390, width: 380, height: 250 },
    ];
    for (let i = 0; i < spec.images.length; i++) {
      const [label, relPath] = spec.images[i];
      addText(slide, label, { left: frames[i].left, top: frames[i].top - 22, width: frames[i].width, height: 20 }, {
        fontSize: 13,
        bold: true,
        color: palette.accent,
        alignment: "center",
      });
      await addImage(slide, relPath, frames[i], `${spec.title}: ${label}`);
    }
  }
  addFooter(slide);
}

function addFooter(slide) {
  slide.shapes.add({
    geometry: "line",
    position: { left: 48, top: 674, width: 1184, height: 0 },
    line: { style: "solid", fill: palette.faint, width: 1 },
  });
  addText(slide, "Generated from outputs/dijet/2018/rho/unfolding_jacobian_reg", {
    left: 52,
    top: 684,
    width: 650,
    height: 18,
  }, {
    fontSize: 10,
    color: palette.muted,
  });
}

async function main() {
  await fs.mkdir(previewDir, { recursive: true });
  await fs.mkdir(layoutDir, { recursive: true });
  await fs.mkdir(qaDir, { recursive: true });

  const presentation = Presentation.create({ slideSize: { width: W, height: H } });

  for (const spec of slides) {
    if (spec.kind === "title") {
      const slide = presentation.slides.add();
      slide.background.fill = palette.bg;
      addText(slide, spec.title, { left: 72, top: 150, width: 900, height: 82 }, {
        fontSize: 56,
        bold: true,
        color: palette.ink,
      });
      addText(slide, spec.subtitle, { left: 76, top: 250, width: 980, height: 44 }, {
        fontSize: 24,
        color: palette.muted,
      });
      addText(slide, "Key outputs: unfolded spectra, response/correlation matrices, L-curves, grouped uncertainties, and diagnostics.", {
        left: 76,
        top: 340,
        width: 980,
        height: 36,
      }, {
        fontSize: 20,
        color: palette.accent,
      });
      addFooter(slide);
    } else if (spec.kind === "bullets") {
      const slide = presentation.slides.add();
      slide.background.fill = palette.bg;
      addHeader(slide, spec.title);
      let top = 126;
      for (const bullet of spec.bullets) {
        addText(slide, bullet, { left: 100, top, width: 1050, height: 54 }, {
          fontSize: 21,
          color: bullet.startsWith("Excluded") ? palette.warn : palette.ink,
        });
        addText(slide, "•", { left: 70, top, width: 22, height: 30 }, {
          fontSize: 24,
          color: palette.accent,
        });
        top += 80;
      }
      addFooter(slide);
    } else {
      await drawPlotSlide(presentation, spec);
    }
  }

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(
      path.join(previewDir, `${stem}.png`),
      await presentation.export({ slide, format: "png", scale: 1 }),
    );
    await fs.writeFile(
      path.join(layoutDir, `${stem}.layout.json`),
      await (await slide.export({ format: "layout" })).text(),
    );
  }

  await writeBlob(
    path.join(slideDir, "contact_sheet.webp"),
    await presentation.export({ format: "webp", montage: true, scale: 1 }),
  );
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(pptxPath);

  await fs.writeFile(
    path.join(slideDir, "source-notes.txt"),
    [
      "Source: outputs/dijet/2018/rho/unfolding_jacobian_reg generated by scripts/run_unfolding.py.",
      "Command: python scripts/run_unfolding.py --channel dijet --observable rho --year 2018 --jacobian --regularization ratio_curvature.",
      "Run summary records ROOT 6.39.99, 23 systematics including nominal, tau 0.7907 ungroomed and 0.6954 groomed.",
      "Caveat: uncertainties are partial; response MC statistics and HERWIG/model uncertainty are excluded for this channel-input workflow.",
      "All plot images are local preview PNGs generated by outputs/build_rho_gallery.py from the unfolding PDFs.",
      "",
    ].join("\n"),
  );
  await fs.writeFile(
    path.join(qaDir, "visual-qa.txt"),
    [
      "Rendered all slides to PNG with artifact-tool.",
      "Checked by script: generated slide previews, layout JSON, contact sheet, and PPTX.",
      "Slides intentionally embed plot images as figures; titles, labels, and notes are editable deck objects.",
      "",
    ].join("\n"),
  );
  console.log(pptxPath);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
