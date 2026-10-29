const childProcess = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

function run(command, args, options = {}) {
  const result = childProcess.spawnSync(command, args, {
    encoding: "utf-8",
    stdio: ["ignore", "pipe", "pipe"],
    ...options,
  });
  if (result.status !== 0) {
    const detail = [result.stdout, result.stderr].filter(Boolean).join("\n").trim();
    throw new Error(`${command} ${args.join(" ")} failed\n${detail}`);
  }
  return result;
}

function writeFixture(tmpDir) {
  const generatedDir = path.join(tmpDir, "generated_visuals");
  fs.mkdirSync(generatedDir, { recursive: true });

  const png = Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAGXRFWHRTb2Z0d2FyZQBNZW1TbGlkZXMgQ2hlY2sAW1rXvAAAAFRJREFUeJztzjEBwCAQwLCDe5f9d7AkdpA8gukm2tvZ2QAA7w1+zQAAwH8BAQgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAEC3AvGngEBbGIDtAAAAABJRU5ErkJggg==",
    "base64",
  );
  fs.writeFileSync(path.join(tmpDir, "fixture.png"), png);

  fs.writeFileSync(
    path.join(generatedDir, "memory-flowchart.svg"),
    `<svg xmlns="http://www.w3.org/2000/svg" width="420" height="180" viewBox="0 0 420 180">
      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" fill="#2563eb"/>
        </marker>
      </defs>
      <rect width="420" height="180" rx="18" fill="#eff6ff"/>
      <rect x="24" y="46" width="112" height="64" rx="14" fill="#dbeafe" stroke="#2563eb" stroke-width="3"/>
      <rect x="154" y="46" width="112" height="64" rx="14" fill="#dcfce7" stroke="#16a34a" stroke-width="3"/>
      <rect x="284" y="46" width="112" height="64" rx="14" fill="#fef3c7" stroke="#d97706" stroke-width="3"/>
      <line x1="136" y1="78" x2="154" y2="78" stroke="#2563eb" stroke-width="4" marker-end="url(#arrow)"/>
      <line x1="266" y1="78" x2="284" y2="78" stroke="#2563eb" stroke-width="4" marker-end="url(#arrow)"/>
      <text x="80" y="84" text-anchor="middle" font-family="Arial" font-size="16" font-weight="700" fill="#0f172a">Profile</text>
      <text x="210" y="84" text-anchor="middle" font-family="Arial" font-size="16" font-weight="700" fill="#0f172a">Working</text>
      <text x="340" y="84" text-anchor="middle" font-family="Arial" font-size="16" font-weight="700" fill="#0f172a">Tool</text>
    </svg>`,
    "utf-8",
  );

  fs.writeFileSync(
    path.join(generatedDir, "metric-table.svg"),
    `<svg xmlns="http://www.w3.org/2000/svg" width="460" height="150" viewBox="0 0 460 150">
      <rect width="460" height="150" fill="#ffffff"/>
      <rect width="460" height="38" fill="#dbeafe"/>
      <line x1="0" y1="38" x2="460" y2="38" stroke="#0f172a" stroke-width="2"/>
      <line x1="0" y1="78" x2="460" y2="78" stroke="#94a3b8" stroke-width="1.5"/>
      <line x1="145" y1="0" x2="145" y2="150" stroke="#94a3b8" stroke-width="1.5"/>
      <line x1="300" y1="0" x2="300" y2="150" stroke="#94a3b8" stroke-width="1.5"/>
      <text x="12" y="25" font-family="Arial" font-size="17" font-weight="700" fill="#0f172a">SVG table</text>
      <text x="158" y="25" font-family="Arial" font-size="17" font-weight="700" fill="#0f172a">Focus</text>
      <text x="314" y="25" font-family="Arial" font-size="17" font-weight="700" fill="#0f172a">Evidence</text>
      <text x="12" y="64" font-family="Arial" font-size="15" fill="#0f172a">Local edit</text>
      <text x="158" y="64" font-family="Arial" font-size="15" fill="#0f172a">Scoped change</text>
      <text x="314" y="64" font-family="Arial" font-size="15" fill="#0f172a">Diff stability</text>
      <text x="12" y="112" font-family="Arial" font-size="15" fill="#0f172a">Memory</text>
      <text x="158" y="112" font-family="Arial" font-size="15" fill="#0f172a">Reuse</text>
      <text x="314" y="112" font-family="Arial" font-size="15" fill="#0f172a">Fewer retries</text>
    </svg>`,
    "utf-8",
  );

  const html = `<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>MemSlides PPTX export fixture</title>
  <style>
    html, body { width:1280px; height:720px; margin:0; overflow:hidden; font-family:Arial, sans-serif; background:#f8fafc; }
    .panel { position:absolute; left:44px; top:38px; width:500px; height:170px; padding:24px; border:3px solid #2563eb; border-radius:18px; background:#ffffff; box-sizing:border-box; }
    h1 { margin:0 0 10px 0; color:#111827; font-size:35px; line-height:1.05; }
    p { margin:0; color:#374151; font-size:19px; line-height:1.32; }
    code { font-family:Consolas, monospace; background:#e5e7eb; padding:1px 4px; border-radius:4px; }
    .zh { position:absolute; left:48px; top:228px; width:500px; font-size:19px; color:#0f172a; line-height:1.28; }
    .nested { position:absolute; left:70px; top:282px; width:390px; color:#1f2937; font-size:20px; line-height:1.25; }
    .nested ol { margin-top:6px; }
    .gradient-card { position:absolute; left:48px; top:455px; width:350px; height:142px; padding:18px; box-sizing:border-box; border-radius:26px 10px 26px 10px; background:linear-gradient(135deg,#dbeafe,#fef3c7); box-shadow:0 18px 32px rgba(15,23,42,.18); }
    .gradient-card h2 { margin:0 0 8px 0; font-size:24px; color:#111827; }
    .gradient-card p { font-size:16px; color:#1f2937; }
    .forced-raster { position:absolute; left:424px; top:475px; width:124px; height:74px; display:flex; align-items:center; justify-content:center; text-align:center; color:#ffffff; font-weight:700; font-size:16px; border-radius:18px; background:#7c3aed; filter:drop-shadow(0 8px 14px rgba(76,29,149,.35)); }
    .plain-img { position:absolute; left:612px; top:44px; width:118px; height:118px; border:2px solid #0f766e; object-fit:contain; }
    .cover-frame { position:absolute; left:760px; top:44px; width:174px; height:118px; border-radius:30px; overflow:hidden; box-shadow:0 10px 22px rgba(15,23,42,.2); }
    .cover-frame img { width:100%; height:100%; object-fit:cover; object-position:70% 50%; filter:saturate(1.2); }
    .flowchart-img { position:absolute; left:962px; top:38px; width:260px; height:112px; object-fit:cover; border-radius:16px; }
    table { position:absolute; left:612px; top:190px; width:598px; border-collapse:collapse; font-size:16px; color:#111827; table-layout:fixed; }
    td, th { border:2px solid #94a3b8; padding:8px 10px; background:#ffffff; line-height:1.16; vertical-align:middle; }
    th { background:#dbeafe; font-weight:700; }
    td:nth-child(1), th:nth-child(1) { width:130px; }
    td:nth-child(2), th:nth-child(2) { width:184px; }
    td:nth-child(3), th:nth-child(3) { width:284px; }
    .simple-svg { position:absolute; left:612px; top:400px; width:260px; height:190px; }
    .caption { position:absolute; left:610px; top:598px; width:290px; font-size:16px; color:#4b5563; }
    .svg-table { position:absolute; left:910px; top:404px; width:310px; height:112px; object-fit:contain; }
    .checklist { position:absolute; left:912px; top:546px; width:282px; height:112px; margin:0; padding:14px 16px 14px 36px; box-sizing:border-box; border-radius:18px; background:#ecfeff; box-shadow:0 12px 24px rgba(8,145,178,.18); color:#164e63; font-size:16px; line-height:1.25; }
    .checklist li { margin:4px 0; padding-left:4px; }
  </style>
</head>
<body>
  <section class="panel">
    <h1>Structured Export</h1>
    <p>Text remains <strong>editable</strong>, inline <em>runs</em> survive, and <code>tool memory</code> keeps its style.</p>
  </section>
  <p class="zh">&#29992;&#25143;&#30011;&#20687;&#12289;&#35774;&#35745; memory &#21644; tool memory &#25991;&#26412;&#24212;&#20445;&#25345;&#21487;&#32534;&#36753;&#12290;</p>
  <ul class="nested">
    <li>Editable list item
      <ol start="3">
        <li>Nested ordered item</li>
      </ol>
    </li>
    <li>Second list item</li>
  </ul>
  <section class="gradient-card">
    <h2>Gradient card</h2>
    <p>The complex background is rasterized locally while this text remains editable.</p>
  </section>
  <div class="forced-raster" data-memslides-pptx-export="raster">Forced raster region</div>
  <img class="plain-img" src="fixture.png" alt="fixture">
  <div class="cover-frame"><img src="fixture.png" alt="cover image"></div>
  <img class="flowchart-img" src="generated_visuals/memory-flowchart.svg" alt="memory flowchart">
  <table>
    <tr><th>Scenario</th><th>Evaluation focus</th><th>Measurement</th></tr>
    <tr><td rowspan="2">Local edits</td><td>Stability and minimal collateral change</td><td>Token/line diff vs previous version</td></tr>
    <tr><td colspan="2">Verification bullet: unchanged slides remain stable and readable.</td></tr>
    <tr><td>Tool usage</td><td>Disciplined calls</td><td>External calls per turn; time to first useful draft</td></tr>
  </table>
  <svg class="simple-svg" viewBox="0 0 260 190">
    <rect x="12" y="16" width="116" height="62" fill="#fee2e2" stroke="#dc2626" stroke-width="4"></rect>
    <circle cx="188" cy="52" r="34" fill="#dcfce7" stroke="#16a34a" stroke-width="4"></circle>
    <line x1="36" y1="128" x2="220" y2="128" stroke="#7c3aed" stroke-width="6"></line>
    <text x="34" y="174" fill="#111827" font-size="23">SVG text</text>
  </svg>
  <div class="caption">Simple SVG nodes are exported as editable PPT shapes/text.</div>
  <img class="svg-table" src="generated_visuals/metric-table.svg" alt="linked svg table">
  <ul class="checklist">
    <li>Checklist card with painted items</li>
    <li>Rasterized as one local region</li>
    <li>Readable instead of overflowing</li>
  </ul>
</body>
</html>`;
  fs.writeFileSync(path.join(tmpDir, "slide_01.html"), html, "utf-8");
}

function verifyWithPython(pptxPath) {
  const python = process.env.PYTHON || "python";
  const script = `
from pathlib import Path
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
path = Path(r'''${pptxPath.replace(/\\/g, "\\\\")}''')
prs = Presentation(path)
if len(prs.slides) != 1:
    raise SystemExit(f"expected 1 slide, got {len(prs.slides)}")
slide = prs.slides[0]
shapes = list(slide.shapes)
text_shapes = [s for s in shapes if getattr(s, "has_text_frame", False) and s.text.strip()]
tables = [s for s in shapes if getattr(s, "has_table", False)]
pictures = [s for s in shapes if s.shape_type == MSO_SHAPE_TYPE.PICTURE]
text_blob = "\\n".join(s.text for s in text_shapes)
if len(shapes) < 12:
    raise SystemExit(f"expected multiple editable and raster elements, got {len(shapes)} shape(s)")
if not text_shapes:
    raise SystemExit("expected editable text shapes")
if "Structured Export" not in text_blob:
    raise SystemExit("expected ordinary slide text to remain editable")
if "tool memory" not in text_blob:
    raise SystemExit("expected inline run text to remain editable")
if "Nested ordered item" not in text_blob:
    raise SystemExit("expected nested list text to remain editable")
if "SVG text" not in text_blob:
    raise SystemExit("expected simple inline SVG text to remain editable in auto mode")
if not tables:
    raise SystemExit("expected ordinary HTML table to remain native/editable in default auto mode")
if len(pictures) < 6:
    raise SystemExit(f"expected fixture image plus rasterized visual/CSS/list/image regions; got {len(pictures)} picture(s)")
if "SVG table" in text_blob:
    raise SystemExit("expected generated SVG table image to be rasterized, not editable text")
if len(shapes) <= len(pictures) + 2:
    raise SystemExit("output looks too close to a screenshot-only deck")
print(f"auto_shapes={len(shapes)} text={len(text_shapes)} tables={len(tables)} pictures={len(pictures)}")
`;
  return run(python, ["-c", script]);
}

function verifyEditableModeWithPython(pptxPath) {
  const python = process.env.PYTHON || "python";
  const script = `
from pathlib import Path
from pptx import Presentation
path = Path(r'''${pptxPath.replace(/\\/g, "\\\\")}''')
prs = Presentation(path)
slide = prs.slides[0]
shapes = list(slide.shapes)
text_shapes = [s for s in shapes if getattr(s, "has_text_frame", False) and s.text.strip()]
tables = [s for s in shapes if getattr(s, "has_table", False)]
text_blob = "\\n".join(s.text for s in text_shapes)
table_blob = "\\n".join(cell.text for table_shape in tables for row in table_shape.table.rows for cell in row.cells)
if "SVG table" not in text_blob:
    raise SystemExit("expected linked SVG text to be editable in editable mode")
if "Local edits" not in text_blob and "Local edits" not in table_blob:
    raise SystemExit("expected HTML table content to remain available in editable mode")
print(f"editable_mode_shapes={len(shapes)} text={len(text_shapes)} tables={len(tables)}")
`;
  return run(python, ["-c", script]);
}

function readReport(reportPath) {
  return JSON.parse(fs.readFileSync(reportPath, "utf-8"));
}

function reportCodes(report) {
  return new Set(
    (report.slides || [])
      .flatMap((slide) => slide.warnings || [])
      .map((warning) => warning.code)
      .filter(Boolean),
  );
}

function assertReportIncludes(report, expectedCodes) {
  const codes = reportCodes(report);
  for (const code of expectedCodes) {
    if (!codes.has(code)) {
      throw new Error(`expected report diagnostics to include ${code}; got ${Array.from(codes).sort().join(", ")}`);
    }
  }
  const rasterized = (report.slides || []).flatMap((slide) => slide.rasterized_regions || []);
  if (rasterized.length < expectedCodes.length) {
    throw new Error(`expected several rasterized regions in report, got ${rasterized.length}`);
  }
}

function assertEditableReport(report) {
  const codes = reportCodes(report);
  if (codes.has("table_rasterized")) {
    throw new Error("editable mode should not rasterize the HTML table by default");
  }
  if (codes.has("css_region_rasterized") || codes.has("image_style_rasterized") || codes.has("complex_list_rasterized")) {
    throw new Error("editable mode should not auto-rasterize CSS, image-style, or complex-list regions");
  }
}

function main() {
  const appRoot = path.resolve(__dirname, "..");
  const cliPath = path.join(appRoot, "src", "memslides", "presentation_export", "export_cli.js");
  if (!fs.existsSync(cliPath)) {
    throw new Error(`PPTX export CLI not found at ${cliPath}`);
  }
  require("fast-glob");
  require("minimist");
  require("playwright");
  require("pptxgenjs");
  require("sharp");

  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "memslides-pptx-export-check-"));
  try {
    writeFixture(tmpDir);
    const htmlPath = path.join(tmpDir, "slide_01.html");

    const outputPath = path.join(tmpDir, "fixture.pptx");
    const reportPath = path.join(tmpDir, "fixture-report.json");
    const result = run(process.execPath, [
      cliPath,
      "--html",
      htmlPath,
      "--output",
      outputPath,
      "--report",
      reportPath,
      "--layout",
      "16:9",
    ], { cwd: appRoot });
    if (!fs.existsSync(outputPath)) {
      throw new Error(`PPTX output not created at ${outputPath}`);
    }
    const verify = verifyWithPython(outputPath);
    assertReportIncludes(readReport(reportPath), [
      "flowchart_rasterized",
      "css_region_rasterized",
      "image_style_rasterized",
      "complex_list_rasterized",
      "visual_rasterized",
    ]);
    process.stdout.write(result.stderr || "");
    process.stdout.write(verify.stdout || "");

    const editableOutputPath = path.join(tmpDir, "fixture-editable.pptx");
    const editableReportPath = path.join(tmpDir, "fixture-editable-report.json");
    const editableResult = run(process.execPath, [
      cliPath,
      "--html",
      htmlPath,
      "--output",
      editableOutputPath,
      "--report",
      editableReportPath,
      "--layout",
      "16:9",
    ], {
      cwd: appRoot,
      env: {
        ...process.env,
        MEMSLIDES_PPTX_EXPORT_VISUAL_MODE: "editable",
      },
    });
    const editableVerify = verifyEditableModeWithPython(editableOutputPath);
    assertEditableReport(readReport(editableReportPath));
    process.stdout.write(editableResult.stderr || "");
    process.stdout.write(editableVerify.stdout || "");
    process.stdout.write("PPTX export runtime ok\n");
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

try {
  main();
} catch (error) {
  process.stderr.write(`${error?.stack || error?.message || String(error)}\n`);
  process.exit(1);
}
