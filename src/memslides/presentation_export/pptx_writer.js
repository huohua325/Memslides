"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { fileURLToPath } = require("node:url");

const pptxgen = require("pptxgenjs");

const {
  fillFromCss,
  lineFromStyle,
  paddingFromStyle,
  parseCssColor,
  parseCssLength,
  positionFromBox,
  pxToPt,
  rounded,
  shapeOptionsFromStyle,
  tableBorderFromStyle,
  textOptionsFromStyle,
} = require("./style_mapper");
const { layoutSpec } = require("./dom_extract");

function configurePresentation(layoutName) {
  const spec = layoutSpec(layoutName);
  const pptx = new pptxgen();
  if (spec.pptx === "A1" || spec.pptx === "A2" || spec.pptx === "A3" || spec.pptx === "A4") {
    pptx.defineLayout({ name: spec.pptx, width: spec.widthIn, height: spec.heightIn });
    pptx.layout = spec.pptx;
  } else {
    pptx.layout = spec.pptx;
  }
  pptx.author = "MemSlides";
  pptx.company = "MemSlides";
  pptx.subject = "Generated presentation";
  pptx.title = "MemSlides presentation";
  pptx.lang = "en-US";
  return pptx;
}

function shapeType(pptx, name) {
  const mapping = {
    ellipse: pptx.ShapeType.ellipse,
    line: pptx.ShapeType.line,
    rect: pptx.ShapeType.rect,
    roundRect: pptx.ShapeType.roundRect,
  };
  return mapping[name] || pptx.ShapeType.rect;
}

function sortedElements(elements) {
  return [...(elements || [])].sort((a, b) => {
    const za = Number(a.zIndex || 0);
    const zb = Number(b.zIndex || 0);
    if (za !== zb) {
      return za - zb;
    }
    return Number(a.domIndex || 0) - Number(b.domIndex || 0);
  });
}

function scaleForSlide(slideData) {
  const spec = slideData.layout || layoutSpec("16:9");
  const body = slideData.body || {};
  const bodyWidth = Math.max(1, Number(body.widthPx || spec.widthPx));
  const bodyHeight = Math.max(1, Number(body.heightPx || spec.heightPx));
  return {
    x: spec.widthIn / bodyWidth,
    y: spec.heightIn / bodyHeight,
    widthIn: spec.widthIn,
    heightIn: spec.heightIn,
  };
}

function dataUriToAddImage(value) {
  return { data: value };
}

function sourceToImage(value, htmlFile) {
  const src = String(value || "").trim();
  if (!src) {
    return null;
  }
  if (src.startsWith("data:image/")) {
    return dataUriToAddImage(src);
  }
  if (src.startsWith("file://")) {
    try {
      return { path: fileURLToPath(src) };
    } catch (_) {
      return null;
    }
  }
  if (/^[a-z]+:\/\//i.test(src)) {
    return null;
  }
  const candidate = path.isAbsolute(src)
    ? src
    : path.resolve(path.dirname(String(htmlFile || process.cwd())), src);
  return fs.existsSync(candidate) ? { path: candidate } : null;
}

function sourceToLocalPath(value, htmlFile) {
  const src = String(value || "").trim();
  if (!src || src.startsWith("data:")) {
    return "";
  }
  if (src.startsWith("file://")) {
    try {
      return fileURLToPath(src);
    } catch (_) {
      return "";
    }
  }
  if (/^[a-z]+:\/\//i.test(src)) {
    return "";
  }
  const candidate = path.isAbsolute(src)
    ? src
    : path.resolve(path.dirname(String(htmlFile || process.cwd())), src);
  return fs.existsSync(candidate) ? candidate : "";
}

function addPlaceholder(slide, pptx, pos, label = "Missing image") {
  slide.addShape(shapeType(pptx, "rect"), {
    ...pos,
    fill: { color: "F3F4F6" },
    line: { color: "9CA3AF", width: 0.75, dashType: "dash" },
  });
  slide.addText(label, {
    ...pos,
    color: "6B7280",
    fontFace: "Aptos",
    fontSize: 9,
    align: "center",
    valign: "mid",
    margin: 0.02,
    fit: "shrink",
  });
}

function addShape(slide, pptx, element, pos) {
  const opts = {
    ...pos,
    ...shapeOptionsFromStyle(element.style || {}),
  };
  slide.addShape(shapeType(pptx, element.shape || "rect"), opts);
}

function clampTextBoxPosition(pos) {
  return {
    ...pos,
    w: Math.max(0.05, Number(pos.w || 0)),
    h: Math.max(0.05, Number(pos.h || 0)),
  };
}

function bulletOptionsForElement(element) {
  const depth = Math.min(8, Math.max(0, Number(element.listDepth || 0)));
  const indent = 14 + depth * 16;
  const listStyle = String(element.listStyleType || "").toLowerCase();
  if (element.list === "number") {
    const numberType = listStyle.includes("roman")
      ? "romanLcPeriod"
      : listStyle.includes("alpha") || listStyle.includes("letter")
        ? "alphaLcPeriod"
        : "arabicPeriod";
    return {
      type: "number",
      style: numberType,
      numberType,
      indent,
      startAt: Math.max(1, Number(element.listIndex || element.listStart || 1)),
      numberStartAt: Math.max(1, Number(element.listIndex || element.listStart || 1)),
    };
  }
  if (element.list !== "bullet") {
    return null;
  }
  if (listStyle.includes("circle")) {
    return { characterCode: "25E6", indent };
  }
  if (listStyle.includes("square")) {
    return { characterCode: "25A0", indent };
  }
  if (listStyle.includes("none")) {
    return null;
  }
  return { characterCode: "2022", indent };
}

function addText(slide, element, pos) {
  const extra = {};
  const bullet = bulletOptionsForElement(element);
  if (bullet) {
    extra.bullet = bullet;
  }
  if (element.overflowRisk) {
    extra.margin = 0;
  }
  const opts = {
    ...clampTextBoxPosition(pos),
    ...textOptionsFromStyle(element.style || {}, { ...extra, isTextBox: false, text: element.text || "" }),
  };
  if (element.list) {
    opts.indentLevel = Math.min(8, Math.max(0, Number(element.listDepth || 0)));
  }
  if (element.overflowRisk && opts.fontSize) {
    opts.fontSize = Math.max(5, rounded(Number(opts.fontSize) * 0.88, 2));
    opts.margin = 0;
    if (opts.lineSpacing) {
      opts.lineSpacing = Math.max(1, rounded(Number(opts.lineSpacing) * 0.9, 2));
    }
  }
  const runs = Array.isArray(element.runs)
    ? element.runs
        .filter((run) => String(run?.text || "").length > 0)
        .map((run) => ({
          text: String(run.text || ""),
          options: textOptionsFromStyle(run.style || element.style || {}, {
            text: run.text || "",
            margin: 0,
          }),
        }))
    : [];
  if (element.overflowRisk) {
    for (const run of runs) {
      if (run.options?.fontSize) {
        run.options.fontSize = Math.max(5, rounded(Number(run.options.fontSize) * 0.88, 2));
      }
      run.options.margin = 0;
      if (run.options?.lineSpacing) {
        run.options.lineSpacing = Math.max(1, rounded(Number(run.options.lineSpacing) * 0.9, 2));
      }
    }
  }
  slide.addText(runs.length ? runs : String(element.text || ""), opts);
}

function cellOptions(style) {
  const textOpts = textOptionsFromStyle(style || {}, { margin: paddingFromStyle(style || {}, 3) });
  const fill = fillFromCss(style?.backgroundColor);
  const opts = {
    color: textOpts.color,
    fontFace: textOpts.fontFace,
    fontSize: textOpts.fontSize,
    bold: textOpts.bold,
    italic: textOpts.italic,
    underline: textOpts.underline,
    align: textOpts.align,
    valign: textOpts.valign,
    margin: textOpts.margin,
    fit: "shrink",
  };
  if (textOpts.lineSpacing) {
    opts.lineSpacing = textOpts.lineSpacing;
  }
  if (fill.transparency < 100) {
    opts.fill = fill;
  }
  const border = tableBorderFromStyle(style || {});
  if (border) {
    opts.border = border;
  }
  return opts;
}

function tableCellFallbackTextPos(pos, style) {
  const margin = paddingFromStyle(style || {}, 3);
  const margins = Array.isArray(margin) ? margin : [margin, margin, margin, margin];
  const top = Number(margins[0] || 0) / 72;
  const right = Number(margins[1] || 0) / 72;
  const bottom = Number(margins[2] || 0) / 72;
  const left = Number(margins[3] || 0) / 72;
  return {
    x: pos.x + left,
    y: pos.y + top,
    w: Math.max(0.03, pos.w - left - right),
    h: Math.max(0.03, pos.h - top - bottom),
  };
}

function addTableFallback(slide, pptx, element, scale, warnings, reason = "fallback") {
  warnings.push({
    severity: "warning",
    code: "editable_fallback_used",
    message: `Table exported as editable cell shapes/text because ${reason}.`,
    html_file: element.htmlFile || "",
    source: "memslides_presentation_export",
    fallback_kind: "table_cell_fallback",
  });
  for (const row of element.rows || []) {
    for (const cell of row) {
      const pos = positionFromBox(cell.box, scale.x, scale.y);
      const shapeOpts = {
        ...pos,
        fill: fillFromCss(cell.style?.backgroundColor),
        line: lineFromStyle(cell.style || {}),
      };
      slide.addShape(shapeType(pptx, "rect"), shapeOpts);
      if (cell.text) {
        slide.addText(cell.text, {
          ...tableCellFallbackTextPos(pos, cell.style || {}),
          ...textOptionsFromStyle(cell.style || {}, { margin: 0, text: cell.text || "" }),
          fit: "shrink",
        });
      }
    }
  }
}

function tableLooksTooDense(element) {
  for (const row of element.rows || []) {
    for (const cell of row) {
      const textLength = String(cell.text || "").length;
      if (!textLength) {
        continue;
      }
      const fontSize = parseCssLength(cell.style?.fontSize, 16);
      const area = Math.max(1, Number(cell.box?.width || 0) * Number(cell.box?.height || 0));
      if (textLength * fontSize * fontSize * 0.42 > area) {
        return true;
      }
      if (Number(cell.box?.width || 0) < 36 && textLength > 8) {
        return true;
      }
    }
  }
  return false;
}

function addTable(slide, pptx, element, scale, warnings) {
  const rows = (element.rows || []).filter((row) => row.length);
  if (!rows.length) {
    return;
  }
  if (element.preferRaster) {
    const pos = positionFromBox(element.box, scale.x, scale.y);
    if (element.snapshotPath && fs.existsSync(element.snapshotPath)) {
      slide.addImage({ path: element.snapshotPath, ...pos });
      return;
    }
    warnings.push({
      severity: "warning",
      code: "editable_fallback_used",
      message: "Table image capture was unavailable; falling back to structured table export.",
      html_file: element.htmlFile || "",
      source: "memslides_presentation_export",
      fallback_kind: "table_raster_fallback",
    });
  }
  if (tableLooksTooDense(element)) {
    addTableFallback(slide, pptx, element, scale, warnings, "cell text is too dense for reliable native table layout");
    return;
  }
  const pos = positionFromBox(element.box, scale.x, scale.y);
  const tableRows = rows.map((row) =>
    row.map((cell) => ({
      text: String(cell.text || ""),
      options: {
        ...cellOptions(cell.style || {}),
        colspan: Number(cell.colspan || 1) > 1 ? Number(cell.colspan || 1) : undefined,
        rowspan: Number(cell.rowspan || 1) > 1 ? Number(cell.rowspan || 1) : undefined,
      },
    })),
  );
  const colW = (element.colWidthsPx || [])
    .map((width) => rounded(Number(width || 0) * scale.x, 3))
    .filter((width) => width > 0);
  const rowH = (element.rowHeightsPx || [])
    .map((height) => rounded(Math.max(10, Number(height || 0)) * scale.y, 3))
    .filter((height) => height > 0);
  try {
    slide.addTable(tableRows, {
      ...pos,
      ...(colW.length ? { colW } : {}),
      ...(rowH.length ? { rowH } : {}),
      border: { type: "solid", color: "D1D5DB", pt: 0.5 },
      margin: 0.04,
      autoFit: false,
    });
  } catch (error) {
    addTableFallback(slide, pptx, element, scale, warnings, `native addTable failed: ${error.message || error}`);
  }
}

function decodeXmlEntities(value) {
  return String(value || "")
    .replace(/&#x([0-9a-f]+);/gi, (_, hex) => String.fromCodePoint(Number.parseInt(hex, 16)))
    .replace(/&#(\d+);/g, (_, number) => String.fromCodePoint(Number.parseInt(number, 10)))
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">");
}

function parseSvgAttributes(raw) {
  const attrs = {};
  const pattern = /([:\w-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')/g;
  let match;
  while ((match = pattern.exec(String(raw || ""))) !== null) {
    attrs[match[1]] = decodeXmlEntities(match[2] ?? match[3] ?? "");
  }
  return attrs;
}

function parseSvgNumber(value, fallback = 0) {
  const parsed = Number.parseFloat(String(value || ""));
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseSvgDocument(svgText) {
  const svgMatch = /<svg\b([^>]*)>/i.exec(svgText);
  if (!svgMatch) {
    return null;
  }
  const svgAttrs = parseSvgAttributes(svgMatch[1]);
  let minX = 0;
  let minY = 0;
  let width = parseSvgNumber(svgAttrs.width, 0);
  let height = parseSvgNumber(svgAttrs.height, 0);
  const viewBox = String(svgAttrs.viewBox || "").trim().split(/[\s,]+/).map(Number);
  if (viewBox.length === 4 && viewBox.every(Number.isFinite)) {
    [minX, minY, width, height] = viewBox;
  }
  if (!(width > 0 && height > 0)) {
    return null;
  }
  const nodes = [];
  const elementPattern = /<(rect|circle|ellipse|line|text)\b([^>]*)>([\s\S]*?)<\/\1>|<(rect|circle|ellipse|line)\b([^>]*?)\/>/gi;
  let match;
  while ((match = elementPattern.exec(svgText)) !== null) {
    const tag = String(match[1] || match[4] || "").toLowerCase();
    const attrs = parseSvgAttributes(match[2] || match[5] || "");
    const text = tag === "text" ? decodeXmlEntities(String(match[3] || "").replace(/<[^>]+>/g, "")).replace(/\s+/g, " ").trim() : "";
    nodes.push({ tag, attrs, text });
  }
  return {
    minX,
    minY,
    width,
    height,
    nodes,
  };
}

function containedBox(pos, sourceWidth, sourceHeight) {
  const sourceRatio = sourceWidth / Math.max(1, sourceHeight);
  const boxRatio = Number(pos.w || 1) / Math.max(0.01, Number(pos.h || 1));
  if (Math.abs(sourceRatio - boxRatio) < 0.001) {
    return { ...pos };
  }
  if (sourceRatio > boxRatio) {
    const h = pos.w / sourceRatio;
    return { x: pos.x, y: pos.y + (pos.h - h) / 2, w: pos.w, h };
  }
  const w = pos.h * sourceRatio;
  return { x: pos.x + (pos.w - w) / 2, y: pos.y, w, h: pos.h };
}

function svgColor(value, fallback = "111111") {
  const parsed = parseCssColor(value);
  return parsed ? parsed.hex : fallback;
}

function addLinkedSvgBasic(slide, pptx, element, pos, warnings) {
  const localPath = sourceToLocalPath(element.src, element.htmlFile);
  if (!localPath || path.extname(localPath).toLowerCase() !== ".svg") {
    return false;
  }
  const fit = String(element.style?.objectFit || "").toLowerCase();
  if (fit === "cover") {
    return false;
  }
  let parsed;
  try {
    parsed = parseSvgDocument(fs.readFileSync(localPath, "utf-8"));
  } catch (_) {
    return false;
  }
  if (!parsed || !parsed.nodes.length) {
    return false;
  }
  const target = containedBox(pos, parsed.width, parsed.height);
  const sx = target.w / parsed.width;
  const sy = target.h / parsed.height;
  const pxScale = (target.w * 96) / parsed.width;
  const mapX = (value) => target.x + (parseSvgNumber(value, 0) - parsed.minX) * sx;
  const mapY = (value) => target.y + (parseSvgNumber(value, 0) - parsed.minY) * sy;
  const mapW = (value) => Math.max(0.01, parseSvgNumber(value, 0) * sx);
  const mapH = (value) => Math.max(0.01, parseSvgNumber(value, 0) * sy);

  for (const node of parsed.nodes) {
    const attrs = node.attrs || {};
    if (node.tag === "rect") {
      slide.addShape(shapeType(pptx, "rect"), {
        x: mapX(attrs.x),
        y: mapY(attrs.y),
        w: mapW(attrs.width),
        h: mapH(attrs.height),
        fill: { color: svgColor(attrs.fill, "FFFFFF"), transparency: attrs.fill === "none" ? 100 : 0 },
        line: attrs.stroke && attrs.stroke !== "none"
          ? { color: svgColor(attrs.stroke), width: Math.max(0.25, pxToPt(parseSvgNumber(attrs["stroke-width"], 1) * pxScale)) }
          : { color: "FFFFFF", transparency: 100 },
      });
    } else if (node.tag === "circle" || node.tag === "ellipse") {
      const cx = parseSvgNumber(attrs.cx, 0);
      const cy = parseSvgNumber(attrs.cy, 0);
      const rx = node.tag === "circle" ? parseSvgNumber(attrs.r, 0) : parseSvgNumber(attrs.rx, 0);
      const ry = node.tag === "circle" ? parseSvgNumber(attrs.r, 0) : parseSvgNumber(attrs.ry, 0);
      slide.addShape(shapeType(pptx, "ellipse"), {
        x: target.x + (cx - rx - parsed.minX) * sx,
        y: target.y + (cy - ry - parsed.minY) * sy,
        w: Math.max(0.01, rx * 2 * sx),
        h: Math.max(0.01, ry * 2 * sy),
        fill: { color: svgColor(attrs.fill, "FFFFFF"), transparency: attrs.fill === "none" ? 100 : 0 },
        line: attrs.stroke && attrs.stroke !== "none"
          ? { color: svgColor(attrs.stroke), width: Math.max(0.25, pxToPt(parseSvgNumber(attrs["stroke-width"], 1) * pxScale)) }
          : { color: "FFFFFF", transparency: 100 },
      });
    } else if (node.tag === "line") {
      slide.addShape(shapeType(pptx, "line"), {
        x: mapX(attrs.x1),
        y: mapY(attrs.y1),
        w: mapX(attrs.x2) - mapX(attrs.x1),
        h: mapY(attrs.y2) - mapY(attrs.y1),
        line: { color: svgColor(attrs.stroke), width: Math.max(0.25, pxToPt(parseSvgNumber(attrs["stroke-width"], 1) * pxScale)) },
      });
    } else if (node.tag === "text" && node.text) {
      const fontSizePx = parseSvgNumber(attrs["font-size"], 14);
      const fontSizePt = Math.max(3, pxToPt(fontSizePx * pxScale));
      const x = mapX(attrs.x);
      const baselineY = mapY(attrs.y);
      const textW = Math.max(0.08, (parsed.width - parseSvgNumber(attrs.x, 0)) * sx);
      const textH = Math.max(0.04, (fontSizePx * 1.45) * sy);
      let textX = x;
      const anchor = String(attrs["text-anchor"] || "start").toLowerCase();
      if (anchor === "middle") {
        textX = x - textW / 2;
      } else if (anchor === "end") {
        textX = x - textW;
      }
      slide.addText(node.text, {
        x: textX,
        y: baselineY - textH * 0.82,
        w: textW,
        h: textH,
        fontFace: attrs["font-family"] ? attrs["font-family"].split(",")[0].replace(/^["']|["']$/g, "").trim() : "Aptos",
        fontSize: rounded(fontSizePt, 2),
        color: svgColor(attrs.fill, "111111"),
        bold: Number.parseInt(String(attrs["font-weight"] || "400"), 10) >= 600 || String(attrs["font-weight"] || "").toLowerCase() === "bold",
        margin: 0,
        fit: "shrink",
        breakLine: false,
      });
    }
  }
  warnings.push({
    severity: "warning",
    code: "linked_svg_editable",
    message: "Local SVG image exported as editable basic PPTX elements.",
    html_file: element.htmlFile || "",
    source: "memslides_presentation_export",
    src: element.src || "",
  });
  return true;
}

function addImage(slide, pptx, element, pos, warnings) {
  let image = null;
  if (element.snapshotPath && fs.existsSync(element.snapshotPath)) {
    image = { path: element.snapshotPath };
  } else {
    if (element.preferRaster) {
      warnings.push({
        severity: "warning",
        code: "editable_fallback_used",
        message: "Raster image capture was unavailable; falling back to source image or placeholder.",
        html_file: element.htmlFile || "",
        source: "memslides_presentation_export",
        fallback_kind: "image_raster_fallback",
        visual_kind: element.visualKind || "",
      });
    }
    image = element.loaded === false ? null : sourceToImage(element.src, element.htmlFile);
  }
  if (!image) {
    warnings.push({
      severity: "warning",
      code: "image_placeholder",
      message: "Image source could not be embedded; placeholder written instead.",
      html_file: element.htmlFile || "",
      source: "memslides_presentation_export",
      src: element.src || "",
    });
    addPlaceholder(slide, pptx, pos, element.alt || "Missing image");
    return;
  }
  if (!element.preferRaster && addLinkedSvgBasic(slide, pptx, element, pos, warnings)) {
    return;
  }
  const fit = String(element.style?.objectFit || "").toLowerCase();
  const sizingType = fit === "cover" ? "cover" : "contain";
  const imageOptions = {
    ...image,
    ...pos,
    altText: element.alt || "",
  };
  if (fit === "contain" || fit === "cover" || fit === "scale-down" || fit === "none") {
    imageOptions.sizing = {
      type: sizingType,
      x: pos.x,
      y: pos.y,
      w: pos.w,
      h: pos.h,
    };
  } else if (!element.snapshotPath && Number(element.naturalWidth || 0) > 0 && Number(element.naturalHeight || 0) > 0) {
    const naturalRatio = Number(element.naturalWidth) / Number(element.naturalHeight);
    const boxRatio = Number(pos.w || 1) / Math.max(0.01, Number(pos.h || 1));
    if (Math.abs(naturalRatio - boxRatio) > 0.04) {
      imageOptions.sizing = {
        type: "contain",
        x: pos.x,
        y: pos.y,
        w: pos.w,
        h: pos.h,
      };
      warnings.push({
        severity: "warning",
        code: "image_aspect_fit",
        message: "Image aspect ratio differs from the HTML box; exported with contain sizing to avoid distortion.",
        html_file: element.htmlFile || "",
        source: "memslides_presentation_export",
        src: element.src || "",
      });
    }
  }
  slide.addImage({
    ...imageOptions,
  });
}

function addSvgBasic(slide, pptx, element, pos) {
  const style = element.style || {};
  if (element.tag === "text") {
    slide.addText(element.text || "", {
      ...pos,
      ...textOptionsFromStyle(style || {}),
      fit: "shrink",
    });
    return;
  }
  if (element.tag === "line") {
    const stroke = parseCssColor(style.stroke || style.color) || { hex: "111111", alpha: 1 };
    slide.addShape(shapeType(pptx, "line"), {
      ...pos,
      line: {
        color: stroke.hex,
        width: Math.max(0.25, parseCssLength(style.strokeWidth, 1) * 0.75),
      },
    });
    return;
  }
  const fill = parseCssColor(style.fill || style.backgroundColor);
  const stroke = parseCssColor(style.stroke || style.borderTopColor);
  slide.addShape(shapeType(pptx, element.tag === "rect" ? "rect" : "ellipse"), {
    ...pos,
    fill: fill ? { color: fill.hex } : { color: "FFFFFF", transparency: 100 },
    line: stroke
      ? { color: stroke.hex, width: Math.max(0.25, parseCssLength(style.strokeWidth, 1) * 0.75) }
      : { color: "FFFFFF", transparency: 100 },
  });
}

function addSlideFromDom(pptx, slideData) {
  const slide = pptx.addSlide();
  const scale = scaleForSlide(slideData);
  const runtimeWarnings = [];
  const bodyFill = fillFromCss(slideData.body?.style?.backgroundColor);
  slide.background = { color: bodyFill.transparency < 100 ? bodyFill.color : "FFFFFF" };
  if (bodyFill.transparency < 100) {
    slide.addShape(shapeType(pptx, "rect"), {
      x: 0,
      y: 0,
      w: scale.widthIn,
      h: scale.heightIn,
      fill: bodyFill,
      line: { color: bodyFill.color, transparency: 100 },
    });
  }

  for (const element of sortedElements(slideData.elements)) {
    const withFile = { ...element, htmlFile: slideData.htmlFile };
    const pos = positionFromBox(element.box, scale.x, scale.y);
    try {
      if (element.kind === "shape") {
        addShape(slide, pptx, withFile, pos);
      } else if (element.kind === "text") {
        addText(slide, withFile, pos);
      } else if (element.kind === "image") {
        addImage(slide, pptx, withFile, pos, runtimeWarnings);
      } else if (element.kind === "table") {
        addTable(slide, pptx, withFile, scale, runtimeWarnings);
      } else if (element.kind === "svg-basic") {
        addSvgBasic(slide, pptx, withFile, pos);
      }
    } catch (error) {
      runtimeWarnings.push({
        severity: "warning",
        code: "element_export_failed",
        message: `Skipped ${element.kind || "element"} because pptxgenjs rejected it: ${error.message || error}`,
        html_file: slideData.htmlFile,
        source: "memslides_presentation_export",
      });
    }
  }
  return {
    slide,
    warnings: [...(slideData.warnings || []), ...runtimeWarnings],
  };
}

module.exports = {
  addSlideFromDom,
  configurePresentation,
  scaleForSlide,
};
