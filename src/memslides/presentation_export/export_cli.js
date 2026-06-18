"use strict";

const fs = require("node:fs");
const path = require("node:path");

const glob = require("fast-glob");
const minimist = require("minimist");

const { configurePresentation, addSlideFromDom } = require("./pptx_writer");
const { extractSlideDom, withBrowser } = require("./dom_extract");

const DIAGNOSTICS_PREFIX = "__MEMSLIDES_PPTX_EXPORT_DIAGNOSTICS__=";

function usage() {
  return [
    "Usage:",
    "  node export_cli.js --html_dir <dir> --output <file.pptx> --layout <16:9|4:3|A1|A2|A3|A4>",
    "  node export_cli.js --html <file> [--html <file2>] --output <file.pptx> --layout <16:9|4:3|A1|A2|A3|A4>",
    "",
    "Options:",
    "  --validate                 Extract and map inputs without writing a PPTX.",
    "  --report <file.json>       Write per-slide export statistics and rasterization diagnostics.",
    "  --skip-layout-validation   Accepted for compatibility; export now records layout issues as warnings.",
  ].join("\n");
}

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

function normalizeHtmlArgs(value) {
  if (!value) {
    return [];
  }
  const items = Array.isArray(value) ? value : String(value).split(",");
  return items.map((item) => path.resolve(String(item).trim())).filter(Boolean);
}

function collectHtmlFiles(args) {
  const htmlDir = args.html_dir || args["html-dir"];
  const explicitFiles = normalizeHtmlArgs(args.html);

  if (htmlDir && explicitFiles.length) {
    fail("Use either --html_dir or --html, not both.");
  }

  if (htmlDir) {
    const resolvedDir = path.resolve(String(htmlDir));
    if (!fs.existsSync(resolvedDir) || !fs.statSync(resolvedDir).isDirectory()) {
      fail(`HTML directory not found: ${resolvedDir}`);
    }
    return glob.sync("*.html", { cwd: resolvedDir, absolute: true }).sort();
  }

  return explicitFiles;
}

function emitDiagnostics(htmlFile, diagnostics) {
  const clean = (diagnostics || []).filter(Boolean);
  if (!clean.length) {
    return;
  }
  process.stderr.write(
    `${DIAGNOSTICS_PREFIX}${JSON.stringify({
      html_file: htmlFile || "",
      diagnostics: clean,
    })}\n`,
  );
}

function countBy(items, keyFn) {
  const counts = {};
  for (const item of items || []) {
    const key = keyFn(item) || "unknown";
    counts[key] = (counts[key] || 0) + 1;
  }
  return counts;
}

function slideReport(slideData, warnings, slideIndex) {
  const elements = slideData.elements || [];
  const rasterized = elements
    .filter((element) => element.snapshotId || element.preferRaster)
    .map((element) => ({
      kind: element.kind || "",
      tag: element.tag || "",
      visual_kind: element.visualKind || "",
      snapshot_captured: Boolean(element.snapshotPath),
      background_only: Boolean(element.backgroundOnly),
      box_px: {
        x: Math.round(Number(element.box?.x || 0)),
        y: Math.round(Number(element.box?.y || 0)),
        w: Math.round(Number(element.box?.width || 0)),
        h: Math.round(Number(element.box?.height || 0)),
      },
    }));
  return {
    slide_index: slideIndex,
    html_file: slideData.htmlFile || "",
    title: slideData.title || "",
    visual_mode: slideData.visualMode || "",
    element_counts: countBy(elements, (element) => element.kind),
    rasterized_regions: rasterized,
    warnings: warnings || [],
  };
}

async function exportSlides({ args, htmlFiles }) {
  const layoutName = String(args.layout || "16:9");
  const outputFile = args.output ? path.resolve(String(args.output)) : "";
  const reportFile = args.report ? path.resolve(String(args.report)) : "";
  const validateOnly = Boolean(args.validate);
  if (!htmlFiles.length) {
    fail(usage());
  }
  for (const htmlFile of htmlFiles) {
    if (!fs.existsSync(htmlFile)) {
      fail(`HTML file not found: ${htmlFile}`);
    }
  }
  if (!validateOnly && !outputFile) {
    fail("Missing --output for PPTX generation.");
  }

  const pptx = configurePresentation(layoutName);
  const allWarnings = [];
  const reports = [];
  const tempDirs = [];
  try {
    await withBrowser(async (browser) => {
      for (const [index, htmlFile] of htmlFiles.entries()) {
        const slideData = await extractSlideDom(browser, htmlFile, layoutName);
        tempDirs.push(...(slideData.assetTempDirs || []));
        if (!validateOnly) {
          const result = addSlideFromDom(pptx, slideData);
          allWarnings.push(...result.warnings);
          reports.push(slideReport(slideData, result.warnings, index + 1));
        } else {
          allWarnings.push(...(slideData.warnings || []));
          reports.push(slideReport(slideData, slideData.warnings || [], index + 1));
        }
      }
    });

    if (allWarnings.length) {
      emitDiagnostics("", allWarnings);
    }

    if (validateOnly) {
      if (reportFile) {
        fs.mkdirSync(path.dirname(reportFile), { recursive: true });
        fs.writeFileSync(reportFile, JSON.stringify({ layout: layoutName, slides: reports }, null, 2), "utf-8");
      }
      return;
    }

    fs.mkdirSync(path.dirname(outputFile), { recursive: true });
    await pptx.writeFile({ fileName: outputFile });
    if (reportFile) {
      fs.mkdirSync(path.dirname(reportFile), { recursive: true });
      fs.writeFileSync(reportFile, JSON.stringify({ layout: layoutName, output_file: outputFile, slides: reports }, null, 2), "utf-8");
    }
  } finally {
    for (const dir of tempDirs) {
      if (dir) {
        fs.rmSync(dir, { recursive: true, force: true });
      }
    }
  }
}

function reportError(error) {
  const diagnostics = Array.isArray(error?.pptxExportDiagnostics)
    ? error.pptxExportDiagnostics
    : [
        {
          severity: "error",
          code: "pptx_export_failed",
          message: error?.message || String(error),
          html_file: error?.htmlFile || "",
          source: "memslides_presentation_export",
        },
      ];
  emitDiagnostics(error?.htmlFile || "", diagnostics);
  process.stderr.write(`${error?.stack || error?.message || String(error)}\n`);
  process.exit(1);
}

async function main() {
  const args = minimist(process.argv.slice(2));
  await exportSlides({
    args,
    htmlFiles: collectHtmlFiles(args),
  });
}

main().catch(reportError);
