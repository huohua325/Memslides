"use strict";

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { pathToFileURL } = require("node:url");

const { chromium } = require("playwright");

const PAGE_TIMEOUT_MS = Number(process.env.MEMSLIDES_PPTX_EXPORT_TIMEOUT_MS || 120000);
const VISUAL_MODE = normalizeVisualMode(process.env.MEMSLIDES_PPTX_EXPORT_VISUAL_MODE || "auto");

const LAYOUT_SPECS = {
  "16:9": { widthPx: 1280, heightPx: 720, widthIn: 13.333, heightIn: 7.5, pptx: "LAYOUT_WIDE" },
  "4:3": { widthPx: 960, heightPx: 720, widthIn: 10, heightIn: 7.5, pptx: "LAYOUT_4X3" },
  A1: { widthPx: 2244, heightPx: 3178, widthIn: 23.39, heightIn: 33.11, pptx: "A1" },
  A2: { widthPx: 1587, heightPx: 2244, widthIn: 16.54, heightIn: 23.39, pptx: "A2" },
  A3: { widthPx: 1122, heightPx: 1587, widthIn: 11.69, heightIn: 16.54, pptx: "A3" },
  A4: { widthPx: 794, heightPx: 1123, widthIn: 8.27, heightIn: 11.69, pptx: "A4" },
};

function layoutSpec(layoutName) {
  return LAYOUT_SPECS[String(layoutName || "16:9")] || LAYOUT_SPECS["16:9"];
}

function normalizeVisualMode(value) {
  const normalized = String(value || "auto").trim().toLowerCase();
  if (normalized === "editable" || normalized === "raster" || normalized === "auto") {
    return normalized;
  }
  return "auto";
}

function makeWarning(code, message, htmlFile, extra = {}) {
  return {
    severity: "warning",
    code,
    message,
    html_file: htmlFile,
    source: "memslides_presentation_export",
    ...extra,
  };
}

async function waitForRenderablePage(page) {
  await page.evaluate(async () => {
    if (document.fonts && document.fonts.ready) {
      try {
        await document.fonts.ready;
      } catch (_) {
        // Font readiness is helpful, not fatal.
      }
    }
    const images = Array.from(document.images || []);
    await Promise.all(
      images.map((img) => {
        if (img.complete) {
          return Promise.resolve();
        }
        return new Promise((resolve) => {
          img.addEventListener("load", resolve, { once: true });
          img.addEventListener("error", resolve, { once: true });
        });
      }),
    );
  });
}

function normalizeRect(rect) {
  return {
    x: Number(rect?.x || 0),
    y: Number(rect?.y || 0),
    width: Math.max(0, Number(rect?.width || 0)),
    height: Math.max(0, Number(rect?.height || 0)),
  };
}

function cssString(value) {
  return String(value || "").replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

async function captureRasterTargets(page, slideData, htmlFile) {
  const targets = Array.isArray(slideData.rasterTargets) ? slideData.rasterTargets : [];
  if (!targets.length) {
    return slideData;
  }

  const elementsBySnapshot = new Map(
    (slideData.elements || [])
      .filter((element) => element.snapshotId)
      .map((element) => [element.snapshotId, element]),
  );
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "memslides-pptx-visual-"));
  slideData.assetTempDir = tempDir;
  slideData.assetTempDirs = [...(slideData.assetTempDirs || []), tempDir];

  for (const target of targets) {
    const id = String(target.id || "");
    const element = elementsBySnapshot.get(id);
    if (!id || !element) {
      continue;
    }
    const outputPath = path.join(tempDir, `${id}.png`);
    try {
      if (target.backgroundOnly) {
        const cloneId = `memslides-raster-bg-${id}`;
        await page.evaluate(
          ({ id: targetId, cloneId: browserCloneId }) => {
            const source = document.querySelector(`[data-memslides-raster-id="${targetId}"]`);
            if (!source) {
              throw new Error(`Raster source not found: ${targetId}`);
            }
            const rect = source.getBoundingClientRect();
            const computed = window.getComputedStyle(source);
            const clone = document.createElement("div");
            clone.setAttribute("data-memslides-raster-clone", browserCloneId);
            clone.style.position = "fixed";
            clone.style.left = `${rect.left}px`;
            clone.style.top = `${rect.top}px`;
            clone.style.width = `${rect.width}px`;
            clone.style.height = `${rect.height}px`;
            clone.style.pointerEvents = "none";
            clone.style.zIndex = "2147483647";
            clone.style.boxSizing = "border-box";
            clone.style.backgroundColor = computed.backgroundColor;
            clone.style.backgroundImage = computed.backgroundImage;
            clone.style.backgroundRepeat = computed.backgroundRepeat;
            clone.style.backgroundSize = computed.backgroundSize;
            clone.style.backgroundPosition = computed.backgroundPosition;
            clone.style.borderTop = computed.borderTop;
            clone.style.borderRight = computed.borderRight;
            clone.style.borderBottom = computed.borderBottom;
            clone.style.borderLeft = computed.borderLeft;
            clone.style.borderRadius = computed.borderRadius;
            clone.style.boxShadow = computed.boxShadow;
            clone.style.opacity = computed.opacity;
            clone.style.filter = computed.filter;
            document.body.appendChild(clone);
          },
          { id, cloneId },
        );
        try {
          await page.locator(`[data-memslides-raster-clone="${cssString(cloneId)}"]`).first().screenshot({
            path: outputPath,
            omitBackground: true,
            timeout: Math.min(PAGE_TIMEOUT_MS, 30000),
          });
        } finally {
          await page.evaluate(({ cloneId: browserCloneId }) => {
            document.querySelector(`[data-memslides-raster-clone="${browserCloneId}"]`)?.remove();
          }, { cloneId });
        }
      } else {
        await page.locator(`[data-memslides-raster-id="${cssString(id)}"]`).first().screenshot({
          path: outputPath,
          omitBackground: true,
          timeout: Math.min(PAGE_TIMEOUT_MS, 30000),
        });
      }
      if (fs.existsSync(outputPath) && fs.statSync(outputPath).size > 0) {
        element.snapshotPath = outputPath;
        element.loaded = true;
        slideData.warnings.push(
          makeWarning(
            target.code || "visual_rasterized",
            target.message || "Element exported as a local image for visual fidelity.",
            htmlFile,
            {
              tag: target.tag || element.tag || "",
              visual_kind: target.visualKind || element.visualKind || "",
              background_only: Boolean(target.backgroundOnly),
            },
          ),
        );
      }
    } catch (error) {
      slideData.warnings.push(
        makeWarning(
          "raster_capture_failed",
          `Could not capture element as an image; structured fallback will be used where possible: ${error.message || error}`,
          htmlFile,
          {
            tag: target.tag || element.tag || "",
            visual_kind: target.visualKind || element.visualKind || "",
            background_only: Boolean(target.backgroundOnly),
          },
        ),
      );
    }
  }
  return slideData;
}

async function extractSlideDom(browser, htmlFile, layoutName = "16:9") {
  const resolvedHtml = path.resolve(String(htmlFile));
  if (!fs.existsSync(resolvedHtml)) {
    const error = new Error(`HTML file not found: ${resolvedHtml}`);
    error.pptxExportDiagnostics = [
      {
        severity: "error",
        code: "html_missing",
        message: `HTML file not found: ${resolvedHtml}`,
        html_file: resolvedHtml,
        source: "memslides_presentation_export",
      },
    ];
    throw error;
  }

  const spec = layoutSpec(layoutName);
  const page = await browser.newPage({
    viewport: { width: spec.widthPx, height: spec.heightPx },
    deviceScaleFactor: 1,
  });

  try {
    await page.goto(pathToFileURL(resolvedHtml).href, {
      waitUntil: "networkidle",
      timeout: PAGE_TIMEOUT_MS,
    });
    await waitForRenderablePage(page);
    const slideData = await page.evaluate(
      ({ htmlFile: browserHtmlFile, expected, visualMode }) => {
        const TEXT_BLOCK_TAGS = new Set([
          "P",
          "H1",
          "H2",
          "H3",
          "H4",
          "H5",
          "H6",
          "LI",
          "TD",
          "TH",
          "BUTTON",
          "LABEL",
          "FIGCAPTION",
          "BLOCKQUOTE",
          "PRE",
          "CODE",
        ]);
        const INLINE_TEXT_TAGS = new Set(["SPAN", "A", "B", "STRONG", "I", "EM", "U", "SMALL", "SUP", "SUB"]);
        const BLOCK_CHILD_SELECTOR = "p,h1,h2,h3,h4,h5,h6,ul,ol,li,table,section,article,header,footer,nav,aside,figure";
        const SVG_BASIC_TAGS = new Set(["rect", "circle", "ellipse", "line", "text"]);
        const warnings = [];
        const rasterTargets = [];

        function warn(code, message, extra = {}) {
          warnings.push({
            severity: "warning",
            code,
            message,
            html_file: browserHtmlFile,
            source: "memslides_presentation_export",
            ...extra,
          });
        }

        function numeric(value, fallback = 0) {
          const parsed = Number.parseFloat(String(value || ""));
          return Number.isFinite(parsed) ? parsed : fallback;
        }

        function rectFor(element, rootRect) {
          const rect = element.getBoundingClientRect();
          return {
            x: rect.left - rootRect.left,
            y: rect.top - rootRect.top,
            width: rect.width,
            height: rect.height,
            left: rect.left,
            top: rect.top,
            right: rect.right,
            bottom: rect.bottom,
          };
        }

        function styleFor(element) {
          const computed = window.getComputedStyle(element);
          return {
            display: computed.display,
            visibility: computed.visibility,
            opacity: computed.opacity,
            position: computed.position,
            zIndex: computed.zIndex,
            width: computed.width,
            height: computed.height,
            color: computed.color,
            backgroundColor: computed.backgroundColor,
            backgroundImage: computed.backgroundImage,
            borderTopWidth: computed.borderTopWidth,
            borderRightWidth: computed.borderRightWidth,
            borderBottomWidth: computed.borderBottomWidth,
            borderLeftWidth: computed.borderLeftWidth,
            borderTopStyle: computed.borderTopStyle,
            borderRightStyle: computed.borderRightStyle,
            borderBottomStyle: computed.borderBottomStyle,
            borderLeftStyle: computed.borderLeftStyle,
            borderTopColor: computed.borderTopColor,
            borderRightColor: computed.borderRightColor,
            borderBottomColor: computed.borderBottomColor,
            borderLeftColor: computed.borderLeftColor,
            borderRadius: computed.borderRadius,
            borderTopLeftRadius: computed.borderTopLeftRadius,
            borderTopRightRadius: computed.borderTopRightRadius,
            borderBottomRightRadius: computed.borderBottomRightRadius,
            borderBottomLeftRadius: computed.borderBottomLeftRadius,
            fontFamily: computed.fontFamily,
            fontSize: computed.fontSize,
            fontWeight: computed.fontWeight,
            fontStyle: computed.fontStyle,
            lineHeight: computed.lineHeight,
            textAlign: computed.textAlign,
            textDecoration: computed.textDecoration,
            textDecorationLine: computed.textDecorationLine,
            textTransform: computed.textTransform,
            verticalAlign: computed.verticalAlign,
            whiteSpace: computed.whiteSpace,
            overflowWrap: computed.overflowWrap,
            wordBreak: computed.wordBreak,
            paddingTop: computed.paddingTop,
            paddingRight: computed.paddingRight,
            paddingBottom: computed.paddingBottom,
            paddingLeft: computed.paddingLeft,
            transform: computed.transform,
            filter: computed.filter,
            backdropFilter: computed.backdropFilter,
            mixBlendMode: computed.mixBlendMode,
            fill: computed.fill,
            stroke: computed.stroke,
            strokeWidth: computed.strokeWidth,
            overflow: computed.overflow,
            overflowX: computed.overflowX,
            overflowY: computed.overflowY,
            objectFit: computed.objectFit,
            objectPosition: computed.objectPosition,
            boxShadow: computed.boxShadow,
            textShadow: computed.textShadow,
            backgroundRepeat: computed.backgroundRepeat,
            backgroundSize: computed.backgroundSize,
            backgroundPosition: computed.backgroundPosition,
            listStyleType: computed.listStyleType,
            listStylePosition: computed.listStylePosition,
            letterSpacing: computed.letterSpacing,
            marginTop: computed.marginTop,
            marginRight: computed.marginRight,
            marginBottom: computed.marginBottom,
            marginLeft: computed.marginLeft,
          };
        }

        function visible(style, box) {
          return (
            style.display !== "none" &&
            style.visibility !== "hidden" &&
            Number.parseFloat(style.opacity || "1") > 0.01 &&
            box.width > 0.5 &&
            box.height > 0.5
          );
        }

        function explicitExportMode(element) {
          const owner = element.closest("[data-memslides-pptx-export]");
          const value = String(owner?.getAttribute("data-memslides-pptx-export") || "").trim().toLowerCase();
          if (value === "raster" || value === "editable" || value === "auto") {
            return value;
          }
          return "";
        }

        function effectiveVisualMode(element) {
          const explicit = explicitExportMode(element);
          if (explicit === "raster" || explicit === "editable") {
            return explicit;
          }
          return visualMode;
        }

        function isActiveCss(value) {
          const text = String(value || "").trim().toLowerCase();
          return Boolean(text && text !== "none" && text !== "normal" && text !== "auto" && text !== "0px" && text !== "rgba(0, 0, 0, 0)");
        }

        function hasGradient(style) {
          return String(style?.backgroundImage || "").toLowerCase().includes("gradient(");
        }

        function hasShadow(style) {
          return isActiveCss(style?.boxShadow);
        }

        function hasFilter(style) {
          return isActiveCss(style?.filter) || isActiveCss(style?.backdropFilter) || String(style?.mixBlendMode || "").toLowerCase() !== "normal";
        }

        function radiusParts(style) {
          return [
            style?.borderTopLeftRadius,
            style?.borderTopRightRadius,
            style?.borderBottomRightRadius,
            style?.borderBottomLeftRadius,
          ].map((value) => String(value || "0px").trim().toLowerCase());
        }

        function hasRadius(style) {
          return radiusParts(style).some((value) => value && value !== "0px" && value !== "0");
        }

        function hasComplexRadius(style) {
          const joined = String(style?.borderRadius || "").toLowerCase();
          const parts = radiusParts(style).filter((value) => value && value !== "0px" && value !== "0");
          return joined.includes("%") || joined.includes("/") || new Set(parts).size > 1;
        }

        function hasClipping(style) {
          return ["hidden", "clip"].includes(String(style?.overflow || "").toLowerCase()) ||
            ["hidden", "clip"].includes(String(style?.overflowX || "").toLowerCase()) ||
            ["hidden", "clip"].includes(String(style?.overflowY || "").toLowerCase());
        }

        function clippedByRoundedAncestor(element) {
          let parent = element.parentElement;
          while (parent && parent !== document.body && parent !== document.documentElement) {
            const parentStyle = styleFor(parent);
            if (hasClipping(parentStyle) && hasRadius(parentStyle)) {
              return true;
            }
            parent = parent.parentElement;
          }
          return false;
        }

        function sourceFingerprint(src, alt = "", className = "") {
          let text = `${src || ""} ${alt || ""} ${className || ""}`.toLowerCase();
          try {
            text = decodeURIComponent(text);
          } catch (_) {
            // Decoding is best-effort; the raw source is still useful.
          }
          return text.replace(/\\/g, "/");
        }

        function visualKindFor(src, alt = "", className = "") {
          const text = sourceFingerprint(src, alt, className);
          if (/(^|[\/_\-\s])flow-?chart([\/_\-\s.]|$)/.test(text) || /(^|[\/_\-\s])flow([\/_\-\s.]|$)/.test(text)) {
            return "flowchart";
          }
          if (/(^|[\/_\-\s])diagram([\/_\-\s.]|$)/.test(text)) {
            return "diagram";
          }
          if (/(^|[\/_\-\s])chart([\/_\-\s.]|$)/.test(text) || /(^|[\/_\-\s])plot([\/_\-\s.]|$)/.test(text)) {
            return "chart";
          }
          if (/(^|[\/_\-\s])table([\/_\-\s.]|$)/.test(text)) {
            return "table";
          }
          if (text.includes("/generated_visuals/")) {
            return "generated_visual";
          }
          return "";
        }

        function shouldRasterizeImage(element, src, style) {
          const mode = effectiveVisualMode(element);
          if (mode === "editable") {
            return "";
          }
          if (mode === "raster") {
            return "image";
          }
          const kind = visualKindFor(src, element.getAttribute("alt") || "", element.className || "");
          const isSvg = /\.svg(?:$|[?#])/i.test(String(src || ""));
          if (kind) {
            return kind;
          }
          if (
            String(style?.objectFit || "").toLowerCase() === "cover" ||
            hasFilter(style) ||
            hasShadow(style) ||
            hasComplexRadius(style) ||
            clippedByRoundedAncestor(element)
          ) {
            return "image_style";
          }
          if (isSvg && svgSourceLooksComplex(src)) {
            return "svg";
          }
          return "";
        }

        function svgSourceLooksComplex(src) {
          return !/\.svg(?:$|[?#])/i.test(String(src || "")) ? false : sourceFingerprint(src).includes("/generated_visuals/");
        }

        function svgIsComplex(svg) {
          for (const node of Array.from(svg.querySelectorAll("*"))) {
            const tag = node.tagName.toLowerCase();
            if (!SVG_BASIC_TAGS.has(tag)) {
              return true;
            }
            if (tag === "text" && node.querySelector("*")) {
              return true;
            }
          }
          return false;
        }

        function rasterCodeFor(kind, tag) {
          if (kind === "table" || tag === "TABLE") {
            return "table_rasterized";
          }
          if (kind === "flowchart" || kind === "diagram") {
            return "flowchart_rasterized";
          }
          if (kind === "css_region") {
            return "css_region_rasterized";
          }
          if (kind === "image_style") {
            return "image_style_rasterized";
          }
          if (kind === "complex_list") {
            return "complex_list_rasterized";
          }
          return "visual_rasterized";
        }

        function registerRasterTarget(element, domIndex, kind, tag, options = {}) {
          const id = `r${domIndex}_${rasterTargets.length + 1}`;
          element.setAttribute("data-memslides-raster-id", id);
          if (options.skipDescendants) {
            element.setAttribute("data-memslides-raster-skip", "1");
          }
          const code = rasterCodeFor(kind, tag);
          rasterTargets.push({
            id,
            tag,
            visualKind: kind,
            code,
            backgroundOnly: Boolean(options.backgroundOnly),
            skipDescendants: Boolean(options.skipDescendants),
            message:
              code === "table_rasterized"
                ? "Table exported as a local image to avoid PowerPoint text overflow."
                : code === "css_region_rasterized"
                  ? "Complex CSS visual region exported as a local image background for fidelity."
                  : "Visual element exported as a local image for fidelity.",
          });
          return id;
        }

        function cssRegionRasterKind(element, style) {
          if (effectiveVisualMode(element) === "editable") {
            return "";
          }
          if (!hasPaint(style) && !hasGradient(style) && !hasShadow(style) && !hasComplexRadius(style)) {
            return "";
          }
          if (hasGradient(style) || hasShadow(style) || hasComplexRadius(style)) {
            return "css_region";
          }
          return "";
        }

        function fullElementRasterKind(element, style, tag) {
          const mode = effectiveVisualMode(element);
          if (mode === "editable") {
            return "";
          }
          if (mode === "raster" && !["BODY", "HTML"].includes(tag)) {
            return "visual";
          }
          if (hasFilter(style)) {
            return "visual";
          }
          return "";
        }

        function complexListKind(element, style, tag) {
          if (!["UL", "OL"].includes(tag) || effectiveVisualMode(element) === "editable") {
            return "";
          }
          const className = String(element.className || "").toLowerCase();
          if (/checklist|check-list|task-list|timeline|steps|cards/.test(className)) {
            return "complex_list";
          }
          const items = Array.from(element.children || []).filter((child) => child.tagName === "LI");
          return items.some((item) => {
            const itemStyle = styleFor(item);
            const itemClass = String(item.className || "").toLowerCase();
            return /card|panel|step|task/.test(itemClass) || hasPaint(itemStyle) || hasShadow(itemStyle) || hasFilter(itemStyle);
          }) ? "complex_list" : "";
        }

        function tableRasterKind(element, style) {
          const mode = effectiveVisualMode(element);
          if (mode === "editable") {
            return "";
          }
          if (mode === "raster") {
            return "table";
          }
          const className = String(element.className || "").toLowerCase();
          if (/raster|visual|heatmap|matrix|timeline|cards/.test(className)) {
            return "table";
          }
          if (hasGradient(style) || hasShadow(style) || hasFilter(style) || hasComplexRadius(style)) {
            return "table";
          }
          return "";
        }

        function hasPaint(style) {
          const bg = String(style.backgroundColor || "").toLowerCase();
          const backgroundPaint = bg && bg !== "transparent" && bg !== "rgba(0, 0, 0, 0)";
          const backgroundImagePaint = hasGradient(style);
          const widths = [
            numeric(style.borderTopWidth),
            numeric(style.borderRightWidth),
            numeric(style.borderBottomWidth),
            numeric(style.borderLeftWidth),
          ];
          const styles = [
            style.borderTopStyle,
            style.borderRightStyle,
            style.borderBottomStyle,
            style.borderLeftStyle,
          ];
          const borderPaint = widths.some((width, index) => width > 0 && !["none", "hidden"].includes(String(styles[index] || "").toLowerCase()));
          return backgroundPaint || backgroundImagePaint || borderPaint;
        }

        function directText(element) {
          return Array.from(element.childNodes || [])
            .filter((node) => node.nodeType === Node.TEXT_NODE)
            .map((node) => node.textContent || "")
            .join(" ")
            .replace(/\s+/g, " ")
            .trim();
        }

        function listItemOwnText(element) {
          const pieces = [];
          const inlineTags = new Set(["A", "B", "STRONG", "I", "EM", "U", "SMALL", "SUP", "SUB", "CODE", "SPAN", "BR"]);
          for (const node of Array.from(element.childNodes || [])) {
            if (node.nodeType === Node.TEXT_NODE) {
              pieces.push(node.textContent || "");
            } else if (node.nodeType === Node.ELEMENT_NODE) {
              if (node.tagName === "UL" || node.tagName === "OL") {
                continue;
              }
              if (node.tagName === "BR") {
                pieces.push("\n");
              } else if (inlineTags.has(node.tagName) || !node.querySelector(BLOCK_CHILD_SELECTOR)) {
                pieces.push(node.innerText || node.textContent || "");
              }
            }
          }
          return pieces.join(" ").replace(/\s+/g, " ").trim();
        }

        function textFor(element) {
          const tag = String(element.tagName || "").toUpperCase();
          if (element.closest("table") && !["TD", "TH"].includes(tag)) {
            return "";
          }
          if (tag === "LI") {
            return listItemOwnText(element);
          }
          if (TEXT_BLOCK_TAGS.has(tag)) {
            return (element.innerText || element.textContent || "").replace(/\s+/g, " ").trim();
          }
          if (INLINE_TEXT_TAGS.has(tag)) {
            const parentBlock = element.parentElement?.closest("p,h1,h2,h3,h4,h5,h6,li,td,th,button,label,figcaption,blockquote,pre");
            return parentBlock ? "" : (element.innerText || element.textContent || "").replace(/\s+/g, " ").trim();
          }
          if (tag === "DIV") {
            if (element.querySelector(BLOCK_CHILD_SELECTOR)) {
              return directText(element);
            }
            return (element.innerText || directText(element)).replace(/\s+/g, " ").trim();
          }
          return "";
        }

        function transformText(text, style) {
          const transform = String(style?.textTransform || "").toLowerCase();
          if (transform === "uppercase") return String(text || "").toUpperCase();
          if (transform === "lowercase") return String(text || "").toLowerCase();
          if (transform === "capitalize") return String(text || "").replace(/\b([a-z])/gi, (match) => match.toUpperCase());
          return String(text || "");
        }

        function inlineTextRunsFor(element, fallbackStyle) {
          if (!element.querySelector("b,strong,i,em,u,span,small,sup,sub,code,br")) {
            return [];
          }
          const runs = [];
          const inlineTags = new Set(["A", "B", "STRONG", "I", "EM", "U", "SMALL", "SUP", "SUB", "CODE", "SPAN", "BR"]);
          function appendRun(text, style) {
            let normalized = transformText(text, style).replace(/\s+/g, " ");
            if (!normalized) {
              return;
            }
            if (!runs.length) {
              normalized = normalized.replace(/^\s+/, "");
            }
            const previousText = runs.length ? runs[runs.length - 1].text : "";
            if (previousText.endsWith(" ") && normalized.startsWith(" ")) {
              normalized = normalized.replace(/^\s+/, "");
            }
            if (!normalized) {
              return;
            }
            const previous = runs[runs.length - 1];
            const signature = JSON.stringify({
              fontFamily: style.fontFamily,
              fontSize: style.fontSize,
              fontWeight: style.fontWeight,
              fontStyle: style.fontStyle,
              textDecoration: style.textDecoration,
              textDecorationLine: style.textDecorationLine,
              color: style.color,
            });
            if (previous && previous.signature === signature) {
              previous.text += normalized;
              return;
            }
            runs.push({ text: normalized, style, signature });
          }
          function walk(node, inheritedStyle) {
            if (node.nodeType === Node.TEXT_NODE) {
              appendRun(node.textContent || "", inheritedStyle);
              return;
            }
            if (node.nodeType !== Node.ELEMENT_NODE) {
              return;
            }
            if (node.tagName === "BR") {
              runs.push({ text: "\n", style: inheritedStyle, signature: "__break__" });
              return;
            }
            if (node !== element && !inlineTags.has(node.tagName)) {
              return;
            }
            const nodeStyle = node === element ? inheritedStyle : styleFor(node);
            for (const child of Array.from(node.childNodes || [])) {
              walk(child, nodeStyle);
            }
          }
          walk(element, fallbackStyle);
          if (runs.length) {
            runs[runs.length - 1].text = runs[runs.length - 1].text.replace(/\s+$/, "");
          }
          return runs.map(({ text, style }) => ({ text, style }));
        }

        function listDepthFor(element) {
          let depth = 0;
          let node = element.parentElement;
          while (node) {
            if (node.tagName === "UL" || node.tagName === "OL") {
              depth += 1;
            }
            node = node.parentElement;
          }
          return Math.max(0, depth - 1);
        }

        function listStartFor(listElement) {
          const start = Number.parseInt(String(listElement?.getAttribute("start") || "1"), 10);
          return Number.isFinite(start) ? start : 1;
        }

        function textOverflowRiskFor(element, style, box, text) {
          const fontSize = numeric(style.fontSize, 16);
          const lineHeight = numeric(style.lineHeight, Math.max(1, fontSize * 1.2));
          const scrollOverflowX = Math.max(0, Number(element.scrollWidth || 0) - Number(element.clientWidth || box.width || 0) - 1);
          const scrollOverflowY = Math.max(0, Number(element.scrollHeight || 0) - Number(element.clientHeight || box.height || 0) - 1);
          const hasCjk = /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]/.test(String(text || ""));
          const charWidth = fontSize * (hasCjk ? 0.94 : 0.52);
          const estimatedCharsPerLine = Math.max(1, Math.floor((box.width || 1) / Math.max(4, charWidth)));
          const longestWord = String(text || "").split(/\s+/).reduce((max, word) => Math.max(max, word.length), 0);
          const nowrap = /nowrap|pre/.test(String(style.whiteSpace || "").toLowerCase());
          const estimatedLines = nowrap ? 1 : Math.max(1, Math.ceil(String(text || "").length / estimatedCharsPerLine));
          const estimatedHeight = estimatedLines * lineHeight;
          const longWordRisk = longestWord * charWidth > Math.max(1, box.width) * 0.94;
          const lowHeightRisk = Math.max(1, box.height) < lineHeight * 0.92;
          return {
            overflowRisk:
              scrollOverflowX > 0 ||
              scrollOverflowY > 0 ||
              estimatedHeight > Math.max(1, box.height) * 1.08 ||
              (nowrap && String(text || "").length * charWidth > Math.max(1, box.width)) ||
              longWordRisk ||
              lowHeightRisk,
            scrollOverflowX,
            scrollOverflowY,
            estimatedLines,
            longestWord,
            nowrap,
            lowHeightRisk,
          };
        }

        function zIndexFor(style) {
          const parsed = Number.parseInt(style.zIndex, 10);
          return Number.isFinite(parsed) ? parsed : 0;
        }

        function imageSource(element) {
          const src = element.currentSrc || element.getAttribute("src") || "";
          if (src) {
            return src;
          }
          const style = window.getComputedStyle(element);
          const match = /url\((['"]?)(.*?)\1\)/.exec(style.backgroundImage || "");
          return match ? match[2] : "";
        }

        function tableData(table, rootRect, domIndex, style, snapshotId = "") {
          const rowElements = Array.from(table.rows || []);
          const rows = rowElements.map((row, rowIndex) =>
            Array.from(row.cells || []).map((cell) => {
              const cellStyle = styleFor(cell);
              const box = rectFor(cell, rootRect);
              const text = (cell.innerText || cell.textContent || "").replace(/\s+/g, " ").trim();
              const colspan = Math.max(1, Number(cell.colSpan || 1));
              const rowspan = Math.max(1, Number(cell.rowSpan || 1));
              return {
                text,
                box,
                style: cellStyle,
                colspan,
                rowspan,
                rowIndex,
                colIndex: 0,
              };
            }),
          );
          const occupied = [];
          const colWidthsPx = [];
          const rowHeightsPx = rowElements.map((row) => row.getBoundingClientRect().height);
          let colCount = 0;
          rows.forEach((row, rowIndex) => {
            occupied[rowIndex] = occupied[rowIndex] || [];
            let colIndex = 0;
            row.forEach((cell) => {
              while (occupied[rowIndex][colIndex]) {
                colIndex += 1;
              }
              cell.colIndex = colIndex;
              const eachColWidth = cell.box.width / Math.max(1, cell.colspan);
              for (let col = 0; col < cell.colspan; col += 1) {
                colWidthsPx[colIndex + col] = Math.max(colWidthsPx[colIndex + col] || 0, eachColWidth);
              }
              for (let rowOffset = 0; rowOffset < cell.rowspan; rowOffset += 1) {
                const targetRow = rowIndex + rowOffset;
                occupied[targetRow] = occupied[targetRow] || [];
                for (let colOffset = 0; colOffset < cell.colspan; colOffset += 1) {
                  occupied[targetRow][colIndex + colOffset] = true;
                }
              }
              colIndex += cell.colspan;
              colCount = Math.max(colCount, colIndex);
            });
          });
          const tableBox = rectFor(table, rootRect);
          const measuredColWidth = colWidthsPx.reduce((sum, width) => sum + Number(width || 0), 0);
          if (measuredColWidth > 0 && tableBox.width > 0) {
            const factor = tableBox.width / measuredColWidth;
            for (let index = 0; index < colWidthsPx.length; index += 1) {
              colWidthsPx[index] = colWidthsPx[index] * factor;
            }
          }
          const measuredRowHeight = rowHeightsPx.reduce((sum, height) => sum + Number(height || 0), 0);
          if (measuredRowHeight > 0 && tableBox.height > 0) {
            const factor = tableBox.height / measuredRowHeight;
            for (let index = 0; index < rowHeightsPx.length; index += 1) {
              rowHeightsPx[index] = rowHeightsPx[index] * factor;
            }
          }
          for (const row of rows) {
            for (const cell of row) {
              const fontSize = numeric(cell.style.fontSize, 16);
              const textLength = cell.text.length;
              const cellArea = Math.max(1, cell.box.width * cell.box.height);
              if (textLength > 0 && textLength * fontSize * fontSize * 0.4 > cellArea) {
                warn("table_cell_text_dense", "Table cell text is dense for its rendered cell area; PPTX export will shrink text if needed.", {
                  tag: "TABLE",
                  row: cell.rowIndex + 1,
                  col: cell.colIndex + 1,
                  text_length: textLength,
                  cell_width_px: Math.round(cell.box.width),
                  cell_height_px: Math.round(cell.box.height),
                });
              }
            }
          }
          return {
            kind: "table",
            tag: "table",
            domIndex,
            zIndex: zIndexFor(style),
            box: tableBox,
            style,
            rows,
            colWidthsPx: colWidthsPx.slice(0, colCount),
            rowHeightsPx,
            colCount,
            preferRaster: Boolean(snapshotId),
            visualMode,
            visualKind: snapshotId ? "table" : "",
            snapshotId,
          };
        }

        function svgElements(svg, rootRect, baseIndex, zIndex) {
          const exported = [];
          for (const node of Array.from(svg.querySelectorAll("*"))) {
            const tag = node.tagName.toLowerCase();
            if (!SVG_BASIC_TAGS.has(tag)) {
              if (tag === "path" || tag === "filter" || tag === "marker") {
                warn("unsupported_svg_node", `Skipped unsupported SVG <${tag}> node.`, { tag });
              }
              continue;
            }
            const style = styleFor(node);
            const box = rectFor(node, rootRect);
            if (!visible(style, box)) {
              continue;
            }
            exported.push({
              kind: "svg-basic",
              tag,
              domIndex: baseIndex + exported.length / 100,
              zIndex,
              box,
              style: {
                ...style,
                fill: node.getAttribute("fill") || style.fill || style.color,
                stroke: node.getAttribute("stroke") || style.stroke || style.color,
                strokeWidth: node.getAttribute("stroke-width") || style.strokeWidth || "1px",
              },
              text: tag === "text" ? (node.textContent || "").replace(/\s+/g, " ").trim() : "",
            });
          }
          return exported;
        }

        const body = document.body || document.documentElement;
        const html = document.documentElement;
        const bodyStyle = styleFor(body);
        const bodyRect = body.getBoundingClientRect();
        const cssWidth = numeric(bodyStyle.width, bodyRect.width);
        const cssHeight = numeric(bodyStyle.height, bodyRect.height);
        const bodyWidth = Math.max(1, cssWidth || bodyRect.width || expected.widthPx);
        const bodyHeight = Math.max(1, cssHeight || bodyRect.height || expected.heightPx);
        const scrollWidth = Math.max(body.scrollWidth || 0, html.scrollWidth || 0, bodyWidth);
        const scrollHeight = Math.max(body.scrollHeight || 0, html.scrollHeight || 0, bodyHeight);
        if (Math.abs(bodyWidth - expected.widthPx) > 2 || Math.abs(bodyHeight - expected.heightPx) > 2) {
          warn(
            "dimension_mismatch",
            `HTML dimensions ${Math.round(bodyWidth)}x${Math.round(bodyHeight)} differ from target ${expected.widthPx}x${expected.heightPx}; coordinates will be scaled.`,
            {
              expected_width_px: expected.widthPx,
              expected_height_px: expected.heightPx,
              actual_width_px: Math.round(bodyWidth),
              actual_height_px: Math.round(bodyHeight),
            },
          );
        }
        const overflowX = Math.max(0, scrollWidth - bodyWidth - 1);
        const overflowY = Math.max(0, scrollHeight - bodyHeight - 1);
        if (overflowX > 0 || overflowY > 0) {
          warn("content_overflow", "HTML content extends beyond the slide body; export will clip or scale affected elements.", {
            overflow_x_px: Math.round(overflowX),
            overflow_y_px: Math.round(overflowY),
          });
        }

        const rootRect = {
          left: bodyRect.left,
          top: bodyRect.top,
        };
        const elements = [];
        let domIndex = 0;

        for (const element of Array.from(body.querySelectorAll("*"))) {
          const tag = String(element.tagName || "").toUpperCase();
          if (["SCRIPT", "STYLE", "META", "LINK", "TITLE", "HEAD"].includes(tag)) {
            continue;
          }
          const skippedRasterRoot = element.closest("[data-memslides-raster-skip]");
          if (skippedRasterRoot && skippedRasterRoot !== element) {
            continue;
          }
          const style = styleFor(element);
          const box = rectFor(element, rootRect);
          if (!visible(style, box)) {
            continue;
          }
          const zIndex = zIndexFor(style);
          domIndex += 1;
          const mode = effectiveVisualMode(element);

          if (mode === "raster" && !["IMG", "SVG", "TABLE", "CANVAS", "VIDEO"].includes(tag)) {
            const snapshotId = registerRasterTarget(element, domIndex, "visual", tag, { skipDescendants: true });
            elements.push({
              kind: "image",
              tag: tag.toLowerCase(),
              domIndex,
              zIndex,
              box,
              style,
              src: "",
              alt: element.getAttribute("aria-label") || "",
              loaded: true,
              naturalWidth: Math.round(box.width),
              naturalHeight: Math.round(box.height),
              preferRaster: true,
              visualMode: mode,
              visualKind: "visual",
              snapshotId,
            });
            continue;
          }

          if (style.backgroundImage && style.backgroundImage.includes("gradient(")) {
            warn(
              "unsupported_gradient",
              mode === "editable"
                ? "Gradient background is approximated with the computed background color in editable mode."
                : "Gradient background will be rasterized when possible; structured fallback uses the computed background color.",
              { tag },
            );
          }
          if (style.filter && style.filter !== "none") {
            warn(
              "unsupported_filter",
              mode === "editable"
                ? "CSS filter is ignored in editable mode."
                : "CSS filter will be rasterized when possible; structured fallback ignores the filter.",
              { tag },
            );
          }
          if (style.backdropFilter && style.backdropFilter !== "none") {
            warn(
              "unsupported_backdrop_filter",
              mode === "editable"
                ? "CSS backdrop-filter is ignored in editable mode."
                : "CSS backdrop-filter will be rasterized when possible; structured fallback ignores it.",
              { tag },
            );
          }
          if (style.mixBlendMode && style.mixBlendMode !== "normal") {
            warn(
              "unsupported_blend_mode",
              mode === "editable"
                ? "CSS blend mode is ignored in editable mode."
                : "CSS blend mode will be rasterized when possible; structured fallback ignores it.",
              { tag },
            );
          }
          if (style.transform && style.transform !== "none") {
            warn("unsupported_transform", "Complex CSS transform is approximated by the element bounding box.", { tag });
          }

          if (tag === "CANVAS" || tag === "VIDEO") {
            if (mode !== "editable") {
              const snapshotId = registerRasterTarget(element, domIndex, tag.toLowerCase(), tag);
              elements.push({
                kind: "image",
                tag: tag.toLowerCase(),
                domIndex,
                zIndex,
                box,
                style,
                src: "",
                alt: element.getAttribute("aria-label") || "",
                loaded: true,
                naturalWidth: Math.round(box.width),
                naturalHeight: Math.round(box.height),
                preferRaster: true,
                visualMode: mode,
                visualKind: tag.toLowerCase(),
                snapshotId,
              });
              continue;
            }
            warn("unsupported_media_node", `Skipped unsupported <${tag.toLowerCase()}> node.`, { tag });
            continue;
          }

          if (tag === "TABLE") {
            const tableKind = tableRasterKind(element, style);
            const snapshotId = tableKind ? registerRasterTarget(element, domIndex, tableKind, tag) : "";
            elements.push(tableData(element, rootRect, domIndex, style, snapshotId));
            continue;
          }
          if (element.closest("table")) {
            continue;
          }
          if (element.closest("svg") && tag !== "SVG") {
            continue;
          }

          if (tag === "IMG") {
            const src = imageSource(element);
            const loaded = Boolean(element.complete && element.naturalWidth > 0 && element.naturalHeight > 0);
            const visualKind = shouldRasterizeImage(element, src, style);
            const snapshotId = visualKind ? registerRasterTarget(element, domIndex, visualKind, tag) : "";
            if (!src || !loaded) {
              warn("image_unavailable", `Image could not be loaded and will be replaced by a placeholder: ${src || "(missing src)"}`, {
                src,
              });
            }
            elements.push({
              kind: "image",
              tag: "img",
              domIndex,
              zIndex,
              box,
              style,
              src,
              alt: element.getAttribute("alt") || "",
              loaded,
              naturalWidth: element.naturalWidth || 0,
              naturalHeight: element.naturalHeight || 0,
              preferRaster: Boolean(snapshotId),
              visualMode,
              visualKind,
              snapshotId,
            });
            continue;
          }

          if (tag === "SVG") {
            const visualKind = mode === "raster" || svgIsComplex(element) ? "svg" : "";
            if (mode !== "editable" && visualKind) {
              const snapshotId = registerRasterTarget(element, domIndex, visualKind, tag);
              elements.push({
                kind: "image",
                tag: "svg",
                domIndex,
                zIndex,
                box,
                style,
                src: "",
                alt: element.getAttribute("aria-label") || "",
                loaded: true,
                naturalWidth: Math.round(box.width),
                naturalHeight: Math.round(box.height),
                preferRaster: true,
                visualMode: mode,
                visualKind,
                snapshotId,
              });
              continue;
            }
            const svgParts = svgElements(element, rootRect, domIndex, zIndex);
            if (!svgParts.length) {
              warn("unsupported_svg", "SVG contained no basic editable nodes to export.", { tag });
            }
            elements.push(...svgParts);
            continue;
          }

          const complexList = complexListKind(element, style, tag);
          if (complexList) {
            const snapshotId = registerRasterTarget(element, domIndex, complexList, tag, { skipDescendants: true });
            elements.push({
              kind: "image",
              tag: tag.toLowerCase(),
              domIndex,
              zIndex,
              box,
              style,
              src: "",
              alt: element.getAttribute("aria-label") || "",
              loaded: true,
              naturalWidth: Math.round(box.width),
              naturalHeight: Math.round(box.height),
              preferRaster: true,
              visualMode: mode,
              visualKind: complexList,
              snapshotId,
            });
            continue;
          }

          const fullRasterKind = fullElementRasterKind(element, style, tag);
          if (fullRasterKind) {
            const snapshotId = registerRasterTarget(element, domIndex, fullRasterKind, tag, { skipDescendants: true });
            elements.push({
              kind: "image",
              tag: tag.toLowerCase(),
              domIndex,
              zIndex,
              box,
              style,
              src: "",
              alt: element.getAttribute("aria-label") || "",
              loaded: true,
              naturalWidth: Math.round(box.width),
              naturalHeight: Math.round(box.height),
              preferRaster: true,
              visualMode: mode,
              visualKind: fullRasterKind,
              snapshotId,
            });
            continue;
          }

          const cssRegionKind = cssRegionRasterKind(element, style);
          const paintHandledByRaster = Boolean(cssRegionKind);
          if (cssRegionKind) {
            const snapshotId = registerRasterTarget(element, domIndex, cssRegionKind, tag, { backgroundOnly: true });
            elements.push({
              kind: "image",
              tag: tag.toLowerCase(),
              domIndex: domIndex - 0.35,
              zIndex,
              box,
              style,
              src: "",
              alt: element.getAttribute("aria-label") || "",
              loaded: true,
              naturalWidth: Math.round(box.width),
              naturalHeight: Math.round(box.height),
              preferRaster: true,
              visualMode: mode,
              visualKind: cssRegionKind,
              snapshotId,
              backgroundOnly: true,
            });
          }

          if (hasPaint(style) && !paintHandledByRaster) {
            elements.push({
              kind: "shape",
              tag: tag.toLowerCase(),
              domIndex: domIndex - 0.25,
              zIndex,
              box,
              style,
              shape: numeric(style.borderRadius) > Math.min(box.width, box.height) / 3 ? "roundRect" : "rect",
            });
          }

          const text = textFor(element);
          if (text) {
            const fontSize = numeric(style.fontSize, 16);
            const bottomGap = bodyHeight - (box.y + box.height);
            if (fontSize > 16 && bottomGap < 48) {
              warn("bottom_safe_zone", "Large text is close to the slide bottom edge; export continues without blocking.", {
                tag,
                bottom_gap_px: Math.round(bottomGap),
                font_size_px: Math.round(fontSize),
              });
            }
            const overflow = textOverflowRiskFor(element, style, box, text);
            if (overflow.overflowRisk) {
              warn("text_overflow_risk", "Text may overflow after PowerPoint font/layout substitution; export will shrink this text box when possible.", {
                tag,
                text_length: text.length,
                box_width_px: Math.round(box.width),
                box_height_px: Math.round(box.height),
                estimated_lines: overflow.estimatedLines,
                longest_word_length: overflow.longestWord,
                nowrap: overflow.nowrap,
                low_height: overflow.lowHeightRisk,
                scroll_overflow_x_px: Math.round(overflow.scrollOverflowX),
                scroll_overflow_y_px: Math.round(overflow.scrollOverflowY),
              });
            }
            const parent = element.parentElement;
            const isBullet = tag === "LI" && parent && parent.tagName === "UL";
            const isNumbered = tag === "LI" && parent && parent.tagName === "OL";
            elements.push({
              kind: "text",
              tag: tag.toLowerCase(),
              domIndex,
              zIndex,
              box,
              style,
              text: transformText(text, style),
              runs: inlineTextRunsFor(element, style),
              list: isBullet ? "bullet" : isNumbered ? "number" : "",
              listDepth: tag === "LI" ? listDepthFor(element) : 0,
              listStyleType: parent ? styleFor(parent).listStyleType || style.listStyleType || "" : style.listStyleType || "",
              listIndex: isNumbered && parent ? listStartFor(parent) + Array.from(parent.children).filter((child) => child.tagName === "LI").indexOf(element) : 0,
              listStart: isNumbered && parent ? listStartFor(parent) : 0,
              overflowRisk: overflow.overflowRisk,
            });
          }
        }

        return {
          htmlFile: browserHtmlFile,
          title: document.title || "",
          layout: expected,
          body: {
            widthPx: bodyWidth,
            heightPx: bodyHeight,
            scrollWidthPx: scrollWidth,
            scrollHeightPx: scrollHeight,
            style: bodyStyle,
          },
          warnings,
          elements,
          rasterTargets,
          visualMode,
        };
      },
      { htmlFile: resolvedHtml, expected: spec, visualMode: VISUAL_MODE },
    );
    return await captureRasterTargets(page, slideData, resolvedHtml);
  } finally {
    await page.close().catch(() => {});
  }
}

async function withBrowser(callback) {
  const launchOptions = { headless: true };
  const explicitExecutable =
    process.env.MEMSLIDES_CHROMIUM_EXECUTABLE || process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || "";
  if (explicitExecutable) {
    launchOptions.executablePath = explicitExecutable;
  }
  const browser = await chromium.launch(launchOptions);
  try {
    return await callback(browser);
  } finally {
    await browser.close().catch(() => {});
  }
}

module.exports = {
  LAYOUT_SPECS,
  extractSlideDom,
  layoutSpec,
  makeWarning,
  normalizeRect,
  normalizeVisualMode,
  withBrowser,
};
