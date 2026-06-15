"use strict";

const PX_PER_INCH = 96;
const PT_PER_PX = 72 / 96;

const NAMED_COLORS = new Map([
  ["black", "000000"],
  ["white", "FFFFFF"],
  ["red", "FF0000"],
  ["green", "008000"],
  ["blue", "0000FF"],
  ["transparent", null],
]);

function clamp(value, min, max) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return min;
  }
  return Math.min(max, Math.max(min, numeric));
}

function pxToIn(value) {
  return Number(value || 0) / PX_PER_INCH;
}

function pxToPt(value) {
  return Number(value || 0) * PT_PER_PX;
}

function rounded(value, digits = 4) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return 0;
  }
  const scale = 10 ** digits;
  return Math.round(numeric * scale) / scale;
}

function parseCssLength(value, fallback = 0) {
  if (value === undefined || value === null) {
    return fallback;
  }
  const text = String(value).trim().toLowerCase();
  if (!text || text === "auto" || text === "normal") {
    return fallback;
  }
  const numeric = Number.parseFloat(text);
  if (!Number.isFinite(numeric)) {
    return fallback;
  }
  if (text.endsWith("pt")) {
    return numeric / PT_PER_PX;
  }
  if (text.endsWith("in")) {
    return numeric * PX_PER_INCH;
  }
  if (text.endsWith("cm")) {
    return (numeric / 2.54) * PX_PER_INCH;
  }
  if (text.endsWith("mm")) {
    return (numeric / 25.4) * PX_PER_INCH;
  }
  return numeric;
}

function lengthPxToPt(value, fallback = 0) {
  return rounded(pxToPt(parseCssLength(value, fallback)), 2);
}

function componentToHex(value) {
  return clamp(Math.round(value), 0, 255).toString(16).padStart(2, "0").toUpperCase();
}

function parseCssColor(value) {
  if (value === undefined || value === null) {
    return null;
  }
  const text = String(value).trim();
  if (!text) {
    return null;
  }
  const lowered = text.toLowerCase();
  if (NAMED_COLORS.has(lowered)) {
    const hex = NAMED_COLORS.get(lowered);
    return hex ? { hex, alpha: 1 } : null;
  }

  const shortHex = /^#([0-9a-f]{3,4})$/i.exec(text);
  if (shortHex) {
    const chars = shortHex[1].split("");
    const hex = `${chars[0]}${chars[0]}${chars[1]}${chars[1]}${chars[2]}${chars[2]}`.toUpperCase();
    const alpha = chars[3] ? Number.parseInt(`${chars[3]}${chars[3]}`, 16) / 255 : 1;
    return alpha <= 0 ? null : { hex, alpha };
  }

  const longHex = /^#([0-9a-f]{6})([0-9a-f]{2})?$/i.exec(text);
  if (longHex) {
    const alpha = longHex[2] ? Number.parseInt(longHex[2], 16) / 255 : 1;
    return alpha <= 0 ? null : { hex: longHex[1].toUpperCase(), alpha };
  }

  const rgb = /^rgba?\((.+)\)$/i.exec(text);
  if (!rgb) {
    return null;
  }
  const parts = rgb[1].split(",").map((part) => part.trim());
  if (parts.length < 3) {
    return null;
  }
  const channels = parts.slice(0, 3).map((part) => {
    if (part.endsWith("%")) {
      return clamp((Number.parseFloat(part) / 100) * 255, 0, 255);
    }
    return clamp(Number.parseFloat(part), 0, 255);
  });
  const alpha = parts.length >= 4 ? clamp(Number.parseFloat(parts[3]), 0, 1) : 1;
  if (alpha <= 0.01) {
    return null;
  }
  return {
    hex: channels.map(componentToHex).join(""),
    alpha,
  };
}

function transparencyFromAlpha(alpha) {
  return clamp(Math.round((1 - Number(alpha || 1)) * 100), 0, 100);
}

function fillFromCss(value) {
  const parsed = parseCssColor(value);
  if (!parsed) {
    return { color: "FFFFFF", transparency: 100 };
  }
  return {
    color: parsed.hex,
    transparency: transparencyFromAlpha(parsed.alpha),
  };
}

function lineFromStyle(style) {
  const widthPx = Math.max(
    parseCssLength(style?.borderTopWidth, 0),
    parseCssLength(style?.borderRightWidth, 0),
    parseCssLength(style?.borderBottomWidth, 0),
    parseCssLength(style?.borderLeftWidth, 0),
  );
  const borderStyle = String(style?.borderTopStyle || style?.borderStyle || "").toLowerCase();
  const color = parseCssColor(
    style?.borderTopColor ||
      style?.borderRightColor ||
      style?.borderBottomColor ||
      style?.borderLeftColor,
  );
  if (!color || widthPx <= 0 || borderStyle === "none" || borderStyle === "hidden") {
    return { color: "FFFFFF", transparency: 100 };
  }
  const line = {
    color: color.hex,
    width: Math.max(0.25, pxToPt(widthPx)),
    transparency: transparencyFromAlpha(color.alpha),
  };
  if (borderStyle === "dashed") {
    line.dashType = "dash";
  } else if (borderStyle === "dotted") {
    line.dashType = "sysDot";
  }
  return line;
}

function containsCjk(text) {
  return /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]/.test(String(text || ""));
}

function fontFaceFromCss(value, text = "") {
  const first = String(value || "").split(",")[0] || "";
  const cleaned = first.replace(/^["']|["']$/g, "").trim();
  if (!containsCjk(text)) {
    return cleaned || "Aptos";
  }
  if (/noto sans cjk|microsoft yahei|simhei|simsun|pingfang|hiragino|meiryo|malgun/i.test(cleaned)) {
    return cleaned;
  }
  return "Microsoft YaHei";
}

function normalizeTextAlign(value) {
  const text = String(value || "").toLowerCase();
  if (text === "center" || text === "right" || text === "justify") {
    return text;
  }
  if (text === "end") {
    return "right";
  }
  return "left";
}

function normalizeVerticalAlign(value) {
  const text = String(value || "").toLowerCase();
  if (text === "middle" || text === "center") {
    return "mid";
  }
  if (text === "bottom") {
    return "bottom";
  }
  return "top";
}

function applyTextTransform(text, value) {
  const transform = String(value || "").toLowerCase();
  const input = String(text || "");
  if (transform === "uppercase") {
    return input.toUpperCase();
  }
  if (transform === "lowercase") {
    return input.toLowerCase();
  }
  if (transform === "capitalize") {
    return input.replace(/\b([a-z])/gi, (match) => match.toUpperCase());
  }
  return input;
}

function paddingFromStyle(style, fallbackPx = 2) {
  const top = lengthPxToPt(style?.paddingTop, fallbackPx);
  const right = lengthPxToPt(style?.paddingRight, fallbackPx);
  const bottom = lengthPxToPt(style?.paddingBottom, fallbackPx);
  const left = lengthPxToPt(style?.paddingLeft, fallbackPx);
  if (top === right && right === bottom && bottom === left) {
    return Math.max(0, top);
  }
  return [top, right, bottom, left].map((value) => Math.max(0, value));
}

function borderSideFromStyle(style, side) {
  const prefix = `border${side}`;
  const width = lengthPxToPt(style?.[`${prefix}Width`], 0);
  const borderStyle = String(style?.[`${prefix}Style`] || "").toLowerCase();
  const color = parseCssColor(style?.[`${prefix}Color`]);
  if (!color || width <= 0 || borderStyle === "none" || borderStyle === "hidden") {
    return null;
  }
  const border = {
    pt: Math.max(0.25, width),
    color: color.hex,
  };
  if (borderStyle === "dashed") {
    border.dash = "dash";
  } else if (borderStyle === "dotted") {
    border.dash = "sysDot";
  }
  return border;
}

function tableBorderFromStyle(style) {
  const sides = ["Top", "Right", "Bottom", "Left"].map((side) => borderSideFromStyle(style || {}, side));
  if (sides.some(Boolean)) {
    return sides;
  }
  const line = lineFromStyle(style || {});
  if (line.transparency >= 100) {
    return null;
  }
  return { type: "solid", color: line.color, pt: line.width || 0.5 };
}

function textOptionsFromStyle(style, extra = {}) {
  const color = parseCssColor(style?.color) || { hex: "111111", alpha: 1 };
  const fontSizePx = parseCssLength(style?.fontSize, 16);
  const lineHeightPx = parseCssLength(style?.lineHeight, Math.max(1, fontSizePx * 1.2));
  const opts = {
    fontFace: fontFaceFromCss(style?.fontFamily, extra.text || ""),
    fontSize: Math.max(4, rounded(pxToPt(fontSizePx), 2)),
    color: color.hex,
    transparency: transparencyFromAlpha(color.alpha),
    bold: Number.parseInt(String(style?.fontWeight || "400"), 10) >= 600 || String(style?.fontWeight || "").toLowerCase() === "bold",
    italic: String(style?.fontStyle || "").toLowerCase() === "italic",
    underline: String(style?.textDecorationLine || style?.textDecoration || "").toLowerCase().includes("underline"),
    align: normalizeTextAlign(style?.textAlign),
    valign: normalizeVerticalAlign(style?.verticalAlign),
    breakLine: false,
    fit: "shrink",
    margin: extra.margin !== undefined ? extra.margin : paddingFromStyle(style || {}, 2),
  };
  if (lineHeightPx > 0 && Number.isFinite(lineHeightPx)) {
    opts.lineSpacing = Math.max(1, rounded(pxToPt(lineHeightPx), 2));
  }
  if (extra.bullet) {
    opts.bullet = extra.bullet;
  }
  if (extra.isTextBox) {
    const fill = fillFromCss(style?.backgroundColor);
    const line = lineFromStyle(style || {});
    if (fill.transparency < 100) {
      opts.fill = fill;
    }
    if (line.transparency < 100) {
      opts.line = line;
    }
  }
  return opts;
}

function shapeOptionsFromStyle(style) {
  return {
    fill: fillFromCss(style?.backgroundColor),
    line: lineFromStyle(style || {}),
  };
}

function positionFromBox(box, scaleX, scaleY) {
  return {
    x: rounded(Number(box?.x || 0) * scaleX),
    y: rounded(Number(box?.y || 0) * scaleY),
    w: Math.max(0.01, rounded(Number(box?.width || 0) * scaleX)),
    h: Math.max(0.01, rounded(Number(box?.height || 0) * scaleY)),
  };
}

module.exports = {
  PX_PER_INCH,
  clamp,
  fillFromCss,
  lineFromStyle,
  parseCssColor,
  parseCssLength,
  applyTextTransform,
  borderSideFromStyle,
  containsCjk,
  fontFaceFromCss,
  lengthPxToPt,
  paddingFromStyle,
  positionFromBox,
  pxToIn,
  pxToPt,
  rounded,
  shapeOptionsFromStyle,
  tableBorderFromStyle,
  textOptionsFromStyle,
  transparencyFromAlpha,
};
