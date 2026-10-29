"""
高清PDF图片提取模块

利用PDF parser布局检测的figure边界框 + PyMuPDF高DPI页面渲染，
从学术PDF中提取高清完整figure/table（2000+px），替代嵌入图片提取。

策略（3层fallback）：
1. PDF parser model.json bbox + PyMuPDF page.get_pixmap(clip=rect, dpi=300)
2. PyMuPDF cluster_drawings() 自动检测 + 渲染
3. PyMuPDF extract_image() 提取嵌入光栅图（仅对非向量图有效）
"""
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# PDF parser model.json category_id 定义
CATEGORY_FIGURE = 3
CATEGORY_FIGURE_CAPTION = 4
CATEGORY_TABLE = 5
CATEGORY_TABLE_CAPTION = 6
CAPTION_LABEL_RE = re.compile(r"^(Figure|Table)\s+(\d+)\s*:")


@dataclass
class FigureInfo:
    """提取的figure/table结构化信息"""
    page_num: int
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) in page coords
    image_path: str
    width: int
    height: int
    category: str  # "figure" | "table"
    caption: str = ""
    pdf_parser_img_name: str = ""  # PDF parser缩略图文件名（用于替换映射）


def parse_caption_label(caption_text: str) -> tuple[str | None, int | None, str]:
    """Parse a `Figure N:` / `Table N:` caption label."""
    text = (caption_text or "").strip()
    match = CAPTION_LABEL_RE.match(text)
    if not match:
        return None, None, text
    kind = "figure" if match.group(1) == "Figure" else "table"
    return kind, int(match.group(2)), text


def make_figure_filename(
    page_num: int,
    category: str,
    fallback_index: int,
    caption_text: str = "",
    used_names: set[str] | None = None,
) -> str:
    """Build a stable filename that prefers the actual caption number when available."""
    label_kind, label_num, _ = parse_caption_label(caption_text)
    if label_kind == category and label_num is not None:
        candidate = f"p{page_num + 1}_{category}_{label_num}.png"
    else:
        candidate = f"p{page_num + 1}_{category}_{fallback_index}.png"

    if not used_names:
        return candidate
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate

    fallback = f"p{page_num + 1}_{category}_{fallback_index}.png"
    if fallback not in used_names:
        used_names.add(fallback)
        return fallback

    dedup = 1
    while True:
        alt = f"p{page_num + 1}_{category}_{fallback_index}_{dedup}.png"
        if alt not in used_names:
            used_names.add(alt)
            return alt
        dedup += 1


def match_caption_for_region(page, region, category: str, captions: list[dict] | None = None) -> str:
    """Find the nearest matching caption for a rendered figure/table region."""
    import fitz

    page_captions = captions if captions is not None else _find_figure_captions(page)
    best_caption = ""
    best_score: tuple[int, float, float] | None = None

    for cap in page_captions:
        if cap.get("category") != category:
            continue

        cap_rect = fitz.Rect(*cap["bbox"])
        overlap = min(region.x1, cap_rect.x1) - max(region.x0, cap_rect.x0)
        min_width = max(1.0, min(region.width, cap_rect.width))
        overlap_ratio = overlap / min_width
        if overlap_ratio < 0.2:
            continue

        below_gap = cap_rect.y0 - region.y1
        above_gap = region.y0 - cap_rect.y1
        if -12 <= below_gap <= 120:
            score = (0, abs(below_gap), -overlap_ratio)
        elif -12 <= above_gap <= 80:
            score = (1, abs(above_gap), -overlap_ratio)
        else:
            continue

        if best_score is None or score < best_score:
            best_score = score
            best_caption = cap.get("caption_text", "").strip()

    return best_caption


def extract_hd_images(
    pdf_path: str,
    output_dir: str,
    pdf_parser_model_json: str | None = None,
    pdf_parser_layout_json: str | None = None,
    pdf_parser_content_list_json: str | None = None,
    min_size: int = 100,
    dpi: int = 300,
) -> list[FigureInfo]:
    """
    从PDF中提取高清figure/table图片

    Args:
        pdf_path: PDF文件路径
        output_dir: 图片输出目录
        pdf_parser_model_json: PDF parser model.json路径（含figure bbox）
        pdf_parser_layout_json: PDF parser layout.json路径（新版在线API布局结果）
        pdf_parser_content_list_json: PDF parser content_list.json路径（含figure与img_path映射）
        min_size: 最小图片尺寸（宽或高），过滤小图标
        dpi: 渲染DPI，默认300

    Returns:
        list[FigureInfo]: 提取的figure信息列表
    """
    try:
        import fitz
    except ImportError:
        logger.error("PyMuPDF (fitz) not installed. Cannot extract HD images.")
        return []

    os.makedirs(output_dir, exist_ok=True)

    has_model_json = bool(pdf_parser_model_json and os.path.exists(pdf_parser_model_json))
    has_layout_json = bool(pdf_parser_layout_json and os.path.exists(pdf_parser_layout_json))

    # 策略1：PDF parser bbox + 高DPI渲染
    if has_model_json:
        figures = _extract_with_pdf_parser_bbox(
            pdf_path, output_dir, pdf_parser_model_json,
            pdf_parser_content_list_json, min_size, dpi,
        )
        if figures:
            logger.info(
                f"Strategy 1 (PDF parser bbox): extracted {len(figures)} HD images from {pdf_path}"
            )
            return figures
        if has_layout_json:
            logger.info(
                "Strategy 1 (PDF parser model.json bbox) yielded no figures; "
                "trying Strategy 1b (PDF parser layout.json)"
            )
        else:
            logger.warning(
                "Strategy 1 (PDF parser model.json bbox) yielded no figures; "
                "falling back to Strategy 2 (cluster_drawings)"
            )
    elif has_layout_json:
        logger.info("PDF parser model.json unavailable; using Strategy 1b (PDF parser layout.json)")

    # 策略1b：PDF parser layout.json + 高DPI渲染（新版在线API）
    if has_layout_json:
        figures = _extract_with_pdf_parser_layout(
            pdf_path, output_dir, pdf_parser_layout_json,
            pdf_parser_content_list_json, min_size, dpi,
        )
        if figures:
            logger.info(
                f"Strategy 1b (PDF parser layout): extracted {len(figures)} HD images from {pdf_path}"
            )
            return figures
        logger.warning(
            "Strategy 1b (PDF parser layout.json) yielded no figures; "
            "falling back to Strategy 2 (cluster_drawings)"
        )

    # 策略2：cluster_drawings + 渲染
    figures = _extract_with_cluster_drawings(pdf_path, output_dir, min_size, dpi)
    if figures:
        logger.info(
            f"Strategy 2 (cluster_drawings): extracted {len(figures)} HD images from {pdf_path}"
        )
        return figures
    logger.warning("Strategy 2 (cluster_drawings) yielded no figures, trying strategy 3")

    # 策略3：extract_image 提取嵌入光栅图
    figures = _extract_embedded_images(pdf_path, output_dir, min_size)
    logger.info(
        f"Strategy 3 (extract_image): extracted {len(figures)} HD images from {pdf_path}"
    )
    return figures


def _parse_pdf_parser_model_json(model_json_path: str) -> list[dict]:
    """
    解析PDF parser model.json，提取figure/table的bbox信息

    Returns:
        list[dict]: 每项包含 page_num, bbox, category, page_width, page_height
    """
    with open(model_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    regions = []
    for page_data in data:
        page_info = page_data.get("page_info", {})
        page_num = page_info.get("page_no", 0)
        page_width = page_info.get("width", 0)
        page_height = page_info.get("height", 0)

        for det in page_data.get("layout_dets", []):
            cat_id = det.get("category_id", -1)
            if cat_id not in (CATEGORY_FIGURE, CATEGORY_TABLE):
                continue

            poly = det.get("poly", [])
            if len(poly) < 8:
                continue

            # poly = [x0,y0, x1,y1, x2,y2, x3,y3] (四角坐标)
            xs = [poly[i] for i in range(0, 8, 2)]
            ys = [poly[i] for i in range(1, 8, 2)]
            bbox = (min(xs), min(ys), max(xs), max(ys))

            category = "figure" if cat_id == CATEGORY_FIGURE else "table"
            regions.append({
                "page_num": page_num,
                "bbox": bbox,
                "category": category,
                "page_width": page_width,
                "page_height": page_height,
            })

    return regions


def _parse_content_list_img_mapping(content_list_path: str) -> dict[tuple[int, str, int], str]:
    """
    解析PDF parser content_list.json，建立 (page_idx, category) -> img_path 的映射

    Returns:
        dict: { (page_idx, sequential_key): img_path_basename }
    """
    if not content_list_path or not os.path.exists(content_list_path):
        return {}

    with open(content_list_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = data.get("content_list") or data.get("items") or []
    if not isinstance(data, list):
        return {}

    mapping = {}
    fig_counter = {}  # per-page counter
    for item in data:
        if not isinstance(item, dict):
            continue
        content_type = item.get("type", "")
        if content_type not in ("image", "table"):
            continue

        page_idx = item.get("page_idx", 0)
        img_path = item.get("img_path", "")
        if not img_path:
            continue

        key = (page_idx, content_type)
        count = fig_counter.get(key, 0)
        fig_counter[key] = count + 1
        mapping[(page_idx, content_type, count)] = os.path.basename(img_path)

    return mapping


def _join_layout_text(lines: list[dict]) -> str:
    """Join text spans from a PDF parser layout block."""
    parts = []
    for line in lines or []:
        if not isinstance(line, dict):
            continue
        text = "".join(
            str(span.get("content") or span.get("text") or "")
            for span in line.get("spans", [])
            if isinstance(span, dict) and span.get("type") == "text"
        ).strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _layout_image_path(lines: list[dict]) -> str:
    """Extract the image_path recorded in a PDF parser layout image_body block."""
    for line in lines or []:
        if not isinstance(line, dict):
            continue
        for span in line.get("spans", []):
            if not isinstance(span, dict):
                continue
            if span.get("type") == "image" and span.get("image_path"):
                return os.path.basename(span["image_path"])
    return ""


def _parse_pdf_parser_layout_json(
    layout_json_path: str,
    content_list_path: str | None = None,
) -> list[dict]:
    """
    解析新版 PDF parser layout.json，提取 figure/table 的bbox、caption 和原始图片名。

    Returns:
        list[dict]: 每项包含 page_num, bbox, category, page_width, page_height, caption, pdf_parser_img_name
    """
    with open(layout_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        pages = data.get("pdf_info") or data.get("pages") or []
    elif isinstance(data, list):
        pages = data
    else:
        pages = []
    img_mapping = _parse_content_list_img_mapping(content_list_path)
    regions = []
    counters: dict[tuple[int, str], int] = {}

    for page_num, page_data in enumerate(pages):
        if not isinstance(page_data, dict):
            continue
        page_size = page_data.get("page_size", [0, 0])
        if not isinstance(page_size, (list, tuple)):
            page_size = [0, 0]
        page_width = page_size[0] if len(page_size) >= 2 else 0
        page_height = page_size[1] if len(page_size) >= 2 else 0
        blocks = page_data.get("preproc_blocks") or page_data.get("para_blocks") or []
        if not isinstance(blocks, list):
            continue

        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")
            if block_type not in ("image", "table"):
                continue

            bbox = block.get("bbox", [])
            if not isinstance(bbox, (list, tuple)):
                continue
            if len(bbox) < 4:
                continue

            category = "figure" if block_type == "image" else "table"
            caption_text = ""
            pdf_parser_img_name = ""
            rects = [bbox]

            sub_blocks = block.get("blocks", [])
            if not isinstance(sub_blocks, list):
                sub_blocks = []
            for sub_block in sub_blocks:
                if not isinstance(sub_block, dict):
                    continue
                sub_bbox = sub_block.get("bbox", [])
                if isinstance(sub_bbox, (list, tuple)) and len(sub_bbox) >= 4:
                    rects.append(sub_bbox)

                sub_type = sub_block.get("type", "")
                if sub_type in ("image_caption", "table_caption"):
                    text = _join_layout_text(sub_block.get("lines", []))
                    if text:
                        caption_text = text
                elif sub_type == "image_body":
                    path = _layout_image_path(sub_block.get("lines", []))
                    if path:
                        pdf_parser_img_name = path

            key = (page_num, block_type)
            count = counters.get(key, 0)
            counters[key] = count + 1
            if not pdf_parser_img_name:
                pdf_parser_img_name = img_mapping.get((page_num, block_type, count), "")

            x0 = min(rect[0] for rect in rects)
            y0 = min(rect[1] for rect in rects)
            x1 = max(rect[2] for rect in rects)
            y1 = max(rect[3] for rect in rects)
            regions.append({
                "page_num": page_num,
                "bbox": (x0, y0, x1, y1),
                "category": category,
                "page_width": page_width,
                "page_height": page_height,
                "caption": caption_text,
                "pdf_parser_img_name": pdf_parser_img_name,
            })

    return regions


def _extract_with_pdf_parser_bbox(
    pdf_path: str,
    output_dir: str,
    model_json_path: str,
    content_list_path: str | None,
    min_size: int,
    dpi: int,
) -> list[FigureInfo]:
    """策略1：用PDF parser检测的figure/table bbox区域，通过PyMuPDF高DPI渲染"""
    import fitz

    regions = _parse_pdf_parser_model_json(model_json_path)
    if not regions:
        return []

    img_mapping = _parse_content_list_img_mapping(content_list_path)

    doc = fitz.open(pdf_path)
    figures = []
    fig_counter = {}  # (page_num, category) -> count
    used_names: set[str] = set()
    page_captions_cache: dict[int, list[dict]] = {}

    for region in regions:
        page_num = region["page_num"]
        bbox = region["bbox"]
        category = region["category"]

        if page_num >= len(doc):
            continue

        page = doc[page_num]
        page_rect = page.rect
        page_captions = page_captions_cache.get(page_num)
        if page_captions is None:
            page_captions = _find_figure_captions(page)
            page_captions_cache[page_num] = page_captions

        # PDF parser model.json的坐标基于模型推理分辨率，需要映射到PDF页面坐标
        model_w = region["page_width"]
        model_h = region["page_height"]
        if model_w > 0 and model_h > 0:
            scale_x = page_rect.width / model_w
            scale_y = page_rect.height / model_h
            clip = fitz.Rect(
                bbox[0] * scale_x,
                bbox[1] * scale_y,
                bbox[2] * scale_x,
                bbox[3] * scale_y,
            )
        else:
            clip = fitz.Rect(*bbox)

        # 确保clip在页面范围内
        clip = clip & page_rect
        if clip.is_empty or clip.width < 5 or clip.height < 5:
            continue

        # 渲染
        pix = page.get_pixmap(clip=clip, dpi=dpi)

        if pix.width < min_size and pix.height < min_size:
            continue

        # 文件名：page_category_index.png
        key = (page_num, category)
        count = fig_counter.get(key, 0)
        fig_counter[key] = count + 1

        caption_text = match_caption_for_region(page, clip, category, captions=page_captions)
        img_filename = make_figure_filename(
            page_num,
            category,
            count,
            caption_text=caption_text,
            used_names=used_names,
        )
        img_path = os.path.join(output_dir, img_filename)
        pix.save(img_path)

        # 查找对应的PDF parser缩略图文件名
        pdf_parser_img = img_mapping.get((page_num, "image" if category == "figure" else "table", count), "")

        fig = FigureInfo(
            page_num=page_num,
            bbox=bbox,
            image_path=img_path,
            width=pix.width,
            height=pix.height,
            category=category,
            caption=caption_text,
            pdf_parser_img_name=pdf_parser_img,
        )
        figures.append(fig)
        logger.debug(f"Rendered {img_filename} ({pix.width}x{pix.height}) from page {page_num + 1}")

    doc.close()
    return figures


def _extract_with_pdf_parser_layout(
    pdf_path: str,
    output_dir: str,
    layout_json_path: str,
    content_list_path: str | None,
    min_size: int,
    dpi: int,
) -> list[FigureInfo]:
    """策略1b：用PDF parser layout.json中的布局区域，通过PyMuPDF高DPI渲染"""
    import fitz

    regions = _parse_pdf_parser_layout_json(layout_json_path, content_list_path)
    if not regions:
        return []

    doc = fitz.open(pdf_path)
    figures = []
    fig_counter: dict[tuple[int, str], int] = {}
    used_names: set[str] = set()

    for region in regions:
        page_num = region["page_num"]
        bbox = region["bbox"]
        category = region["category"]

        if page_num >= len(doc):
            continue

        page = doc[page_num]
        page_rect = page.rect
        layout_w = region.get("page_width", 0)
        layout_h = region.get("page_height", 0)
        if layout_w > 0 and layout_h > 0:
            scale_x = page_rect.width / layout_w
            scale_y = page_rect.height / layout_h
            clip = fitz.Rect(
                bbox[0] * scale_x,
                bbox[1] * scale_y,
                bbox[2] * scale_x,
                bbox[3] * scale_y,
            )
        else:
            clip = fitz.Rect(*bbox)

        clip = clip & page_rect
        if clip.is_empty or clip.width < 5 or clip.height < 5:
            continue

        pix = page.get_pixmap(clip=clip, dpi=dpi)
        if pix.width < min_size and pix.height < min_size:
            continue

        key = (page_num, category)
        count = fig_counter.get(key, 0)
        fig_counter[key] = count + 1

        caption_text = region.get("caption", "").strip()
        img_filename = make_figure_filename(
            page_num,
            category,
            count,
            caption_text=caption_text,
            used_names=used_names,
        )
        img_path = os.path.join(output_dir, img_filename)
        pix.save(img_path)

        figures.append(FigureInfo(
            page_num=page_num,
            bbox=(clip.x0, clip.y0, clip.x1, clip.y1),
            image_path=img_path,
            width=pix.width,
            height=pix.height,
            category=category,
            caption=caption_text,
            pdf_parser_img_name=region.get("pdf_parser_img_name", ""),
        ))
        logger.debug(
            f"Rendered layout-backed {img_filename} ({pix.width}x{pix.height}) from page {page_num + 1}"
        )

    doc.close()
    return figures


def _merge_nearby_rects(rects: list, margin: float = 30.0):
    """
    将距离较近的矩形合并为复合区域。

    反复扫描，将任意两个距离 < margin 的矩形合并，直到无法继续合并。
    """
    import fitz

    if not rects:
        return []

    merged = [fitz.Rect(r) for r in rects]
    changed = True
    while changed:
        changed = False
        new_merged = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            current = fitz.Rect(merged[i])
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                # 检查两个矩形是否足够近（扩展margin后相交）
                expanded = fitz.Rect(
                    current.x0 - margin, current.y0 - margin,
                    current.x1 + margin, current.y1 + margin,
                )
                if not (expanded & merged[j]).is_empty:
                    current = current | merged[j]  # union
                    used[j] = True
                    changed = True
            new_merged.append(current)
        merged = new_merged

    return merged


def _find_figure_captions(page) -> list[dict]:
    """
    在页面中查找 "Figure N:" / "Table N:" 形式的caption。

    Returns:
        list[dict]: 每项包含 label, category, bbox, caption_text
    """
    import re as _re

    captions = []
    blocks = page.get_text("dict")["blocks"]
    for block in blocks:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            text = "".join(span["text"] for span in line["spans"])
            m = _re.match(r"^(Figure|Table)\s+(\d+)\s*:", text)
            if m:
                category = "figure" if m.group(1) == "Figure" else "table"
                captions.append({
                    "label": f"{m.group(1)} {m.group(2)}",
                    "category": category,
                    "bbox": line["bbox"],  # (x0, y0, x1, y1)
                    "caption_text": text.strip(),
                })
    return captions


def _extract_with_cluster_drawings(
    pdf_path: str,
    output_dir: str,
    min_size: int,
    dpi: int,
) -> list[FigureInfo]:
    """
    策略2：caption锚定 + 图形元素区域检测

    优先方法（caption锚定）：
    1. 查找页面中的 "Figure N:" / "Table N:" caption文本
    2. 收集caption上方（同列区域内）的所有图形元素（向量cluster + 嵌入图片）
    3. 合并这些元素为figure区域，包含caption本身

    后备方法（纯图形合并）：
    若页面无caption但有显著图形区域，使用合并+文本覆盖率过滤
    """
    import fitz

    doc = fitz.open(pdf_path)
    figures = []
    img_count = 0
    used_names: set[str] = set()

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height

        # 收集页面上所有图形元素
        graphic_rects = []
        try:
            paths = page.get_drawings()
            if paths:
                clusters = page.cluster_drawings(drawings=paths)
                for box in clusters:
                    if box.width >= 15 or box.height >= 15:
                        graphic_rects.append(box)
        except Exception as e:
            logger.debug(f"cluster_drawings failed on page {page_num + 1}: {e}")

        try:
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                for ir in page.get_image_rects(xref):
                    if (not ir.is_empty and ir.width >= 10 and ir.height >= 10
                            and ir.x0 >= 0 and ir.y0 >= 0):
                        graphic_rects.append(ir)
        except Exception as e:
            logger.debug(f"get_image_rects failed on page {page_num + 1}: {e}")

        # 方法A：caption锚定
        captions = _find_figure_captions(page)
        caption_used_graphics = set()  # 记录已被caption方法使用的图形元素索引

        for cap in captions:
            cap_bbox = cap["bbox"]  # (x0, y0, x1, y1)
            cap_y_top = cap_bbox[1]
            cap_x0 = cap_bbox[0]
            cap_x1 = cap_bbox[2]

            # 确定figure所在的列范围（学术论文常为双栏）
            page_mid_x = page_rect.width / 2
            if cap_x1 <= page_mid_x + 20:
                # caption在左栏
                col_x0, col_x1 = 0, page_mid_x + 20
            elif cap_x0 >= page_mid_x - 20:
                # caption在右栏
                col_x0, col_x1 = page_mid_x - 20, page_rect.width
            else:
                # caption跨栏（全宽figure）
                col_x0, col_x1 = 0, page_rect.width

            # 收集caption上方、同列内的图形元素
            fig_elements = []
            for gi, gr in enumerate(graphic_rects):
                # 图形元素需要在caption上方，且水平位置与caption列重叠
                if gr.y1 <= cap_y_top + 10:  # 在caption上方（允许10pt重叠）
                    h_overlap = min(gr.x1, col_x1) - max(gr.x0, col_x0)
                    if h_overlap > 0:
                        fig_elements.append(gr)
                        caption_used_graphics.add(gi)

            if not fig_elements:
                # 没有图形元素在caption上方，尝试caption下方（某些论文caption在上方）
                for gi, gr in enumerate(graphic_rects):
                    if gr.y0 >= cap_y_top - 10:
                        h_overlap = min(gr.x1, col_x1) - max(gr.x0, col_x0)
                        if h_overlap > 0:
                            fig_elements.append(gr)
                            caption_used_graphics.add(gi)

            if not fig_elements:
                continue

            # 合并图形元素为figure区域
            merged = _merge_nearby_rects(fig_elements, margin=50.0)
            if not merged:
                continue

            # 取最大的合并区域
            best = max(merged, key=lambda r: r.width * r.height)

            # 扩展region以包含caption文本
            region = best | fitz.Rect(cap_bbox)

            # 确保region在页面范围内，并添加margin
            clip = fitz.Rect(
                max(0, region.x0 - 5),
                max(0, region.y0 - 5),
                min(page_rect.width, region.x1 + 5),
                min(page_rect.height, region.y1 + 15),  # 底部多留空间给多行caption
            )

            pix = page.get_pixmap(clip=clip, dpi=dpi)
            if pix.width < min_size and pix.height < min_size:
                continue

            img_filename = make_figure_filename(
                page_num,
                cap["category"],
                img_count,
                caption_text=cap["caption_text"],
                used_names=used_names,
            )
            img_path = os.path.join(output_dir, img_filename)
            pix.save(img_path)

            fig = FigureInfo(
                page_num=page_num,
                bbox=(clip.x0, clip.y0, clip.x1, clip.y1),
                image_path=img_path,
                width=pix.width,
                height=pix.height,
                category=cap["category"],
                caption=cap["caption_text"],
            )
            figures.append(fig)
            img_count += 1
            logger.debug(
                f"Caption-anchored {cap['label']}: {img_filename} ({pix.width}x{pix.height})"
            )

        # 方法B：无caption的纯图形区域（后备）
        # 收集未被caption方法使用的图形元素
        remaining = [
            gr for gi, gr in enumerate(graphic_rects)
            if gi not in caption_used_graphics
        ]
        if remaining:
            merged = _merge_nearby_rects(remaining, margin=40.0)
            text_blocks = page.get_text("blocks")

            for i, region in enumerate(merged):
                if region.width < 40 or region.height < 40:
                    continue
                area_ratio = (region.width * region.height) / page_area
                if area_ratio < 0.05:
                    continue

                # 文本覆盖率过滤
                region_area = region.width * region.height
                text_cover_area = 0
                for tb in text_blocks:
                    if tb[6] != 0:
                        continue
                    tb_rect = fitz.Rect(tb[:4])
                    overlap = region & tb_rect
                    if not overlap.is_empty:
                        text_cover_area += overlap.width * overlap.height
                text_coverage = text_cover_area / region_area if region_area > 0 else 0
                if text_coverage > 0.50:
                    continue

                clip = fitz.Rect(
                    max(0, region.x0 - 5),
                    max(0, region.y0 - 5),
                    min(page_rect.width, region.x1 + 5),
                    min(page_rect.height, region.y1 + 5),
                )
                pix = page.get_pixmap(clip=clip, dpi=dpi)
                if pix.width < min_size and pix.height < min_size:
                    continue

                img_filename = f"p{page_num + 1}_region_{img_count}.png"
                img_path = os.path.join(output_dir, img_filename)
                pix.save(img_path)

                fig = FigureInfo(
                    page_num=page_num,
                    bbox=(region.x0, region.y0, region.x1, region.y1),
                    image_path=img_path,
                    width=pix.width,
                    height=pix.height,
                    category="figure",
                )
                figures.append(fig)
                img_count += 1
                logger.debug(f"Uncaptioned region {img_filename} ({pix.width}x{pix.height})")

    doc.close()
    return figures


def _extract_embedded_images(
    pdf_path: str,
    output_dir: str,
    min_size: int,
) -> list[FigureInfo]:
    """策略3：提取PDF中嵌入的光栅图片（最后手段）"""
    import fitz

    doc = fitz.open(pdf_path)
    figures = []
    seen_xrefs = set()

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_info in image_list:
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                if width < min_size and height < min_size:
                    continue

                img_hash = hashlib.md5(image_bytes).hexdigest()[:12]
                img_filename = f"p{page_num + 1}_embed_{img_hash}.{image_ext}"
                img_path = os.path.join(output_dir, img_filename)

                with open(img_path, "wb") as f:
                    f.write(image_bytes)

                fig = FigureInfo(
                    page_num=page_num,
                    bbox=(0, 0, 0, 0),
                    image_path=img_path,
                    width=width,
                    height=height,
                    category="figure",
                )
                figures.append(fig)

            except Exception as e:
                logger.warning(f"Failed to extract embedded image xref={xref}: {e}")
                continue

    doc.close()
    return figures


def build_pdf_parser_to_hd_mapping(
    figures: list[FigureInfo],
    pdf_parser_images_dir: str | Path,
) -> dict[str, str]:
    """
    建立 PDF parser缩略图文件名 -> HD图片路径 的映射

    优先使用 FigureInfo.pdf_parser_img_name（来自content_list.json的精确映射）。
    若缺少精确映射，则宁可留空也不做顺序盲配，避免 Figure/Table 串义。

    Args:
        figures: extract_hd_images 返回的figure列表
        pdf_parser_images_dir: PDF parser缩略图目录

    Returns:
        dict: { pdf_parser_img_basename: hd_image_path }
    """
    pdf_parser_dir = Path(pdf_parser_images_dir)
    if not pdf_parser_dir.exists():
        return {}

    # 收集PDF parser图片，按文件名排序（保持出现顺序）
    pdf_parser_imgs = sorted([
        f.name for f in pdf_parser_dir.iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp")
    ])

    mapping = {}

    # 优先：通过content_list精确映射
    named_figures = [f for f in figures if f.pdf_parser_img_name]
    for fig in named_figures:
        if fig.pdf_parser_img_name in pdf_parser_imgs:
            mapping[fig.pdf_parser_img_name] = fig.image_path

    # 注意：不要再做“按顺序”盲配。
    # PDF parser markdown 中有些表格以 HTML table 形式保留，不会出现在图片引用里；
    # 若这里继续把未命中的 HD 图片按顺序硬配给剩余缩略图，极易把 Figure / Table 串义。
    unnamed_figures = [f for f in figures if not f.pdf_parser_img_name]
    unmatched_pdf_parser = [m for m in pdf_parser_imgs if m not in mapping]
    if unnamed_figures and unmatched_pdf_parser:
        logger.debug(
            "Skipping unsafe order-based PDF parser→HD fallback: %d unmatched markdown images, %d unnamed HD figures",
            len(unmatched_pdf_parser),
            len(unnamed_figures),
        )

    return mapping
