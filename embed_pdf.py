"""
embed_pdf.py  ─  版面感知 × 圖文強綁定 RAG 嵌入引擎
Layout-Aware × Figure-Anchored RAG Embedding Engine

╔══════════════════════════════════════════════════════════════════╗
║  架構總覽 / Architecture Overview                                 ║
╠══════════════════════════════════════════════════════════════════╣
║  STAGE 1 ─ Layout-Aware Text Extraction                          ║
║    • pdfplumber chars (x0,y0,x1,y1) → 自動偵測單/雙欄排版        ║
║    • 雙欄：先 crop 左欄 extract_text，再 crop 右欄，拼接           ║
║    • 排除圖表佔位 bbox，避免圖說文字混入正文段落                    ║
║                                                                  ║
║  STAGE 2 ─ Visual Element Extraction (圖文強綁定)                ║
║    • 正則精準偵測 Caption 行（"Figure X:…" / "Table X:…"）        ║
║      ─ Caption 行必須出現在行首，排除內文提及                       ║
║    • 以 caption bbox ±100pt 向上搜尋最近圖片物件 (pypdfium2)       ║
║    • 擷取圖表所在頁的局部裁切圖 → Gemini Vision 描述               ║
║    • 每個視覺元素產生獨立 Chunk，強注入結構化標籤：                  ║
║      [ELEMENT_TYPE:…][ID:…][CAPTION:…][DESCRIPTION:…]           ║
║                                                                  ║
║  STAGE 3 ─ Structured Table → Markdown                          ║
║    • pdfplumber extract_tables() → table_to_markdown()           ║
║    • 標準 | --- | 分隔行，LLM 理解率最佳                           ║
║                                                                  ║
║  STAGE 4 ─ Semantic / Priority Chunking                         ║
║    • 偵測章節標題（Abstract / Conclusion 等）做語意切塊邊界         ║
║    • Abstract & Conclusion：CHUNK_SIZE_PRIORITY = 1200           ║
║      overlap 加大至 200，確保完整保留                              ║
║    • 一般正文：CHUNK_SIZE_BODY = 800, overlap 160                 ║
║    • 視覺元素 Chunk：不再二次切塊，整塊保留                         ║
║                                                                  ║
║  STAGE 5 ─ Embedding → FAISS                                    ║
║    • provider="google"  → GoogleGenerativeAIEmbeddings           ║
║    • provider="huggingface" → HuggingFaceEmbeddings              ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pdfplumber
import pypdfium2 as pdfium
import pytesseract
from PIL import Image

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("embed_pdf")

# ══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════

RENDER_DPI   = 150          # 頁面光柵化 DPI
JPEG_QUALITY = 85

# ── Chunking parameters ───────────────────────────────────────────
CHUNK_SIZE_PRIORITY = 1200  # Abstract / Conclusion 用大塊
CHUNK_OVERLAP_PRIORITY = 200
CHUNK_SIZE_BODY     = 800   # 一般正文
CHUNK_OVERLAP_BODY  = 160

# ── Caption regex ─────────────────────────────────────────────────
# 匹配：行首 "Figure 3:" / "Fig. 3." / "Table 2 —" 等
# ❌ 不匹配：行中 "as shown in Figure 3"
_CAPTION_RE = re.compile(
    r"^(Figure|Fig\.|Table|FIGURE|TABLE|FIG\.)\s{0,3}"   # 元素類型
    r"(\d+[a-zA-Z]?)"                                     # 編號 (e.g. 1, 2a)
    r"[\s:.\-\u2014\u2013]+(.{3,})",                      # 分隔符 + 說明文字
    re.IGNORECASE,
)

# ── Section heading regex ─────────────────────────────────────────
_SECTION_RE = re.compile(
    r"^(?:\d+\.?\d*\.?\s+)?"
    r"(Abstract|Introduction|Related\s+Work|Background|Method(?:ology)?|"
    r"Approach|Experiment[s]?|Result[s]?|Discussion|Limitation[s]?|"
    r"Conclusion[s]?|Reference[s]?|Appendix|Acknowledge?ment[s]?|"
    r"摘要|引言|相關工作|方法|實驗|結論|結果|討論|參考文獻)"
    r"\s*$",
    re.IGNORECASE,
)

# Sections that must NOT be truncated
_PRIORITY_SECTIONS = frozenset(
    {"abstract", "conclusion", "conclusions", "摘要", "結論"}
)


# ══════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════

@dataclass
class VisualElement:
    """
    一個圖表/表格元素的完整描述包。
    A complete description bundle for one figure/table element.
    """
    element_type: str        # "IMAGE" | "TABLE"
    element_id:   str        # "Figure_1" | "Table_2"
    caption:      str        # 原始 caption 文字
    page:         int        # 1-based page number
    description:  str = ""   # Vision LLM 或 OCR 描述
    md_table:     str = ""   # 若為 TABLE 的 Markdown 格式

    def to_chunk_text(self) -> str:
        """
        序列化為結構化標籤 Chunk 文字。
        Serialize to structured-tag chunk text for precise retrieval.
        """
        parts = [
            f"[ELEMENT_TYPE: {self.element_type}]",
            f"[ID: {self.element_id}]",
            f"[PAGE: {self.page}]",
            f"[CAPTION: {self.caption}]",
        ]
        if self.md_table:
            parts.append(f"[TABLE_CONTENT:\n{self.md_table}\n]")
        if self.description:
            parts.append(f"[DESCRIPTION: {self.description}]")
        return "\n".join(parts)


@dataclass
class SemanticSection:
    """
    一個語意章節（含名稱與全文內容）。
    One semantic section with its heading name and full text content.
    """
    heading:    str
    is_priority: bool   # True → Abstract / Conclusion → large chunks
    lines:      List[str] = field(default_factory=list)

    def text(self) -> str:
        return "\n".join(self.lines)


# ══════════════════════════════════════════════════════════════════
#  STAGE 1  ─  LAYOUT-AWARE TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════════

def _detect_columns(chars: list, page_width: float) -> int:
    """
    以字元 x0 分佈判斷頁面是單欄還是雙欄。
    Returns 1 or 2.

    Algorithm:
        count chars whose x0 < page_mid  vs  x0 >= page_mid.
        If both halves hold > 20% of total chars → two-column.
    """
    if not chars:
        return 1
    page_mid = page_width / 2.0
    left  = sum(1 for c in chars if c["x0"] < page_mid)
    right = sum(1 for c in chars if c["x0"] >= page_mid)
    total = len(chars)
    is_two = (left / total > 0.20) and (right / total > 0.20)
    logger.debug(
        f"[layout] column detection: total={total} "
        f"left={left/total:.0%} right={right/total:.0%} → {'2-col' if is_two else '1-col'}"
    )
    return 2 if is_two else 1


def _extract_layout_text(plumber_page, exclude_bboxes: List[Tuple]) -> str:
    """
    版面感知文字擷取。
    Layout-aware text extraction that respects column order and
    excludes figure/table placeholder bounding boxes.

    Args:
        plumber_page:   pdfplumber Page object
        exclude_bboxes: list of (x0,top,x1,bottom) to skip
                        (visual element regions)
    Returns:
        Plain text in reading order (left-col → right-col for 2-col pages)
    """
    w = plumber_page.width
    h = plumber_page.height
    chars = plumber_page.chars

    n_cols = _detect_columns(chars, w)

    def _crop_text(bbox: Tuple) -> str:
        """Crop page and extract text, filtering out excluded bboxes."""
        region = plumber_page.within_bbox(bbox)
        if not region.chars:
            return ""
        # Filter chars inside any excluded bbox
        if exclude_bboxes:
            filtered_chars = [
                c for c in region.chars
                if not any(
                    ex[0] <= c["x0"] <= ex[2] and ex[1] <= c["top"] <= ex[3]
                    for ex in exclude_bboxes
                )
            ]
            if not filtered_chars:
                return ""
            # Re-extract text from filtered chars via extract_text
            region = region.filter(lambda c: not any(
                ex[0] <= c["x0"] <= ex[2] and ex[1] <= c["top"] <= ex[3]
                for ex in exclude_bboxes
            ))
        try:
            return region.extract_text(layout=False) or ""
        except Exception:
            return region.extract_text() or ""

    if n_cols == 1:
        text = _crop_text((0, 0, w, h))
    else:
        # Two-column: left half first, then right half
        mid = w / 2.0
        left_text  = _crop_text((0, 0, mid, h))
        right_text = _crop_text((mid, 0, w, h))
        text = left_text + "\n" + right_text

    logger.debug(f"[layout] extracted {len(text)} chars ({n_cols}-col)")
    return text.strip()


# ══════════════════════════════════════════════════════════════════
#  STAGE 2  ─  VISUAL ELEMENT EXTRACTION (圖文強綁定)
# ══════════════════════════════════════════════════════════════════

def _table_to_markdown(table: list) -> str:
    """
    將 pdfplumber 二維 List 轉換為標準 Markdown Table。
    Convert pdfplumber 2D list to standard Markdown table with | --- | row.
    """
    if not table:
        return ""
    # Clean cells: replace newlines, strip whitespace
    cleaned = [
        [str(cell or "").replace("\n", " ").strip() for cell in row]
        for row in table
    ]
    # Normalise column count (handle ragged rows)
    n_cols = max(len(r) for r in cleaned) if cleaned else 0
    if n_cols == 0:
        return ""
    padded = [r + [""] * (n_cols - len(r)) for r in cleaned]

    def fmt_row(cells: list) -> str:
        return "| " + " | ".join(cells) + " |"

    header    = padded[0]
    separator = ["---"] * n_cols
    body      = padded[1:]
    lines     = [fmt_row(header), fmt_row(separator)] + [fmt_row(r) for r in body]
    return "\n".join(lines)


def _find_image_bboxes_on_page(pdf_path: str, page_num: int) -> List[Tuple]:
    """
    用 pypdfium2 迭代頁面物件，取得所有圖片 (FPDF_PAGEOBJ_IMAGE) 的邊界框。
    Returns list of (x0, y0, x1, y1) in pdfplumber coordinate system
    (top = page_height - pdf_y).
    """
    IMAGE_TYPE = pdfium.raw.FPDF_PAGEOBJ_IMAGE
    bboxes = []
    try:
        doc  = pdfium.PdfDocument(pdf_path)
        page = doc[page_num]
        ph   = page.get_height()
        for obj in page.get_objects():
            raw_type = pdfium.raw.FPDFPageObj_GetType(obj.raw)
            if raw_type == IMAGE_TYPE:
                b = obj.get_bounds()   # left, bottom, right, top in PDF coords
                # Convert PDF coords (origin bottom-left) → pdfplumber (origin top-left)
                x0   = b.left
                x1   = b.right
                top  = ph - b.top      # pdfplumber "top"
                bot  = ph - b.bottom   # pdfplumber "bottom"
                bboxes.append((x0, top, x1, bot))
        doc.close()
    except Exception as e:
        logger.warning(f"[image_bbox] page {page_num+1} 圖片偵測失敗: {e}")
    logger.debug(f"[image_bbox] page {page_num+1}: {len(bboxes)} image object(s)")
    return bboxes


def _rasterize_region(
    pdf_path: str,
    page_num: int,
    crop_bbox: Optional[Tuple] = None,
    dpi: int = RENDER_DPI,
) -> Optional[Tuple[Image.Image, str]]:
    """
    光柵化頁面（或其局部），回傳 (PIL Image, base64 JPEG string)。
    Rasterize a page (or a sub-region) and return (PIL, base64).
    crop_bbox: (x0, top, x1, bottom) in pdfplumber coords, or None for full page.
    """
    try:
        doc  = pdfium.PdfDocument(pdf_path)
        page = doc[page_num]
        ph   = page.get_height()
        pw   = page.get_width()
        scale = dpi / 72.0
        bitmap = page.render(scale=scale)
        full_img = bitmap.to_pil()
        doc.close()

        if crop_bbox:
            # Map pdfplumber coords → pixel coords
            x0, top, x1, bot = crop_bbox
            sx = full_img.width  / pw
            sy = full_img.height / ph
            px0 = max(0, int(x0  * sx) - 5)
            py0 = max(0, int(top * sy) - 5)
            px1 = min(full_img.width,  int(x1  * sx) + 5)
            py1 = min(full_img.height, int(bot * sy) + 5)
            img = full_img.crop((px0, py0, px1, py1))
        else:
            img = full_img

        buf  = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
        b64  = base64.b64encode(buf.getvalue()).decode()
        return img, b64
    except Exception as e:
        logger.error(f"[rasterize] page {page_num+1} 光柵化失敗: {e}", exc_info=True)
        return None


def _call_vision_llm(img_b64: str, element_id: str, caption: str) -> Optional[str]:
    """
    呼叫 Gemini Vision 描述單一圖表（傳入已裁切的圖表區域）。
    Call Gemini Vision to describe a single cropped figure/chart.
    Returns description string or None on failure.
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage

        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            logger.warning("[vision] GOOGLE_API_KEY 未設定")
            return None

        # ✅ 修正：從環境變數讀取模型名稱，預設改為 gemini-2.0-flash（1.5-flash 已棄用）
        vision_model = os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.0-flash")
        llm = ChatGoogleGenerativeAI(
            model=vision_model,
            temperature=0,
            google_api_key=api_key,
            convert_system_message_to_human=True,
        )

        prompt = (
            f"你正在分析一篇學術論文中的 {element_id}，其圖說（Caption）為：\n"
            f"「{caption}」\n\n"
            "請詳細描述此圖表的視覺內容，包含：\n"
            "1. 圖表類型（折線圖、柱狀圖、架構圖、散點圖等）\n"
            "2. X 軸 / Y 軸標籤與範圍\n"
            "3. 圖例說明與各系列名稱\n"
            "4. 關鍵數值、峰值、趨勢或比較結論\n"
            "5. 任何標注文字（箭頭、標記點等）\n\n"
            "請以條列式（bullet points）中英文皆可回答。"
            "若圖片模糊無法辨識，請說明原因。\n\n"
            f"You are analyzing {element_id} from an academic paper. "
            f"Its caption is: '{caption}'. "
            "Describe the visual content in detail: chart type, axes, legends, "
            "key values, trends, and annotations. Use bullet points."
        )

        msg = HumanMessage(content=[
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": prompt},
        ])
        resp = llm.invoke([msg])
        desc = resp.content.strip()
        logger.info(f"[vision] {element_id} 描述完成（{len(desc)} 字元）")
        return desc

    except Exception as e:
        logger.warning(f"[vision] Gemini Vision 失敗 ({element_id}): {e}")
        return None


def _ocr_image(img: Image.Image, element_id: str) -> str:
    """Tesseract OCR fallback for a cropped image."""
    try:
        text = pytesseract.image_to_string(img, lang="chi_tra+chi_sim+eng").strip()
        if text:
            logger.debug(f"[ocr] {element_id} OCR: {len(text)} chars")
            return text
    except Exception as e:
        logger.warning(f"[ocr] {element_id} OCR 失敗: {e}")
    return ""


def _extract_visual_elements(
    pdf_path: str,
    page_num: int,
    plumber_page,
    use_vision: bool,
    provider: str,
) -> Tuple[List[VisualElement], List[Tuple]]:
    """
    從單一頁面提取所有視覺元素（圖 + 表）。
    Extract all visual elements (figures + tables) from one page.

    Strategy:
        A) Tables: pdfplumber find_tables() → table bbox + Markdown content
           + optional Vision description of table region
        B) Figures: detect Caption lines ("Figure X: …") in text lines
           → locate nearest image bbox above caption
           → rasterize that region → Vision LLM or OCR

    Returns:
        (elements: List[VisualElement], exclude_bboxes: List[Tuple])
        exclude_bboxes are regions to skip during body text extraction
    """
    elements: List[VisualElement] = []
    exclude_bboxes: List[Tuple] = []

    page_num_1 = page_num + 1

    # ── A: Tables via pdfplumber ──────────────────────────────────
    try:
        tables = plumber_page.find_tables()
        raw_tables = plumber_page.extract_tables()
        for t_idx, (tbl_obj, raw_tbl) in enumerate(zip(tables, raw_tables)):
            md = _table_to_markdown(raw_tbl)
            if not md.strip():
                continue
            tbl_bbox = tbl_obj.bbox   # (x0, top, x1, bottom)
            exclude_bboxes.append(tbl_bbox)

            # Try to find a caption line near the table bottom
            caption_text = _find_caption_near_bbox(
                plumber_page, tbl_bbox, search_below=True
            )
            table_id = f"Table_{t_idx+1}"
            if caption_text:
                m = _CAPTION_RE.match(caption_text.strip())
                if m:
                    table_id = f"Table_{m.group(2)}"
                    caption_text = m.group(3).strip()
            else:
                caption_text = f"Table {t_idx+1} on page {page_num_1}"

            # Optional: Vision description for complex tables
            vis_desc = ""
            if use_vision and provider == "google":
                crop_res = _rasterize_region(pdf_path, page_num, tbl_bbox)
                if crop_res:
                    _, b64 = crop_res
                    vis_desc = _call_vision_llm(b64, table_id, caption_text) or ""

            elem = VisualElement(
                element_type="TABLE",
                element_id=table_id,
                caption=caption_text,
                page=page_num_1,
                description=vis_desc,
                md_table=md,
            )
            elements.append(elem)
            logger.info(f"[visual] 擷取 {table_id}（page {page_num_1}）")
    except Exception as e:
        logger.warning(f"[visual] page {page_num_1} 表格擷取失敗: {e}")

    # ── B: Figures via Caption detection ─────────────────────────
    image_bboxes = _find_image_bboxes_on_page(pdf_path, page_num)

    try:
        text_lines = plumber_page.extract_text_lines(return_chars=True)
    except Exception:
        text_lines = []

    for line_obj in text_lines:
        line_text = line_obj.get("text", "").strip()
        m = _CAPTION_RE.match(line_text)
        if not m:
            continue

        raw_type   = m.group(1)
        elem_num   = m.group(2)
        caption_txt = m.group(3).strip()
        is_fig = raw_type.lower().startswith("fig")
        elem_type = "IMAGE" if is_fig else "TABLE_CAPTION"
        elem_id   = f"Figure_{elem_num}" if is_fig else f"Table_{elem_num}"

        # Skip if already captured as TABLE above
        if elem_type == "TABLE_CAPTION":
            continue

        # ── Find the image object closest ABOVE this caption ──────
        caption_top = line_obj.get("top", 0)
        best_bbox   = _find_nearest_image_above(image_bboxes, caption_top)
        target_bbox = best_bbox  # may be None if no image found

        # Extend exclude regions
        if target_bbox:
            exclude_bboxes.append(target_bbox)

        # ── Rasterize & describe ──────────────────────────────────
        vis_desc = ""
        # Include a few lines of context around the figure bbox for Vision
        render_bbox = target_bbox
        if target_bbox:
            # Expand bbox slightly for context
            x0, top, x1, bot = target_bbox
            render_bbox = (
                max(0, x0 - 10),
                max(0, top - 10),
                min(plumber_page.width, x1 + 10),
                min(plumber_page.height, bot + 20),
            )

        if use_vision and provider == "google":
            crop_res = _rasterize_region(pdf_path, page_num, render_bbox)
            if crop_res:
                pil_img, b64 = crop_res
                vis_desc = _call_vision_llm(b64, elem_id, caption_txt) or ""
                if not vis_desc:
                    vis_desc = _ocr_image(pil_img, elem_id)
            else:
                # Fall back to full-page rasterize
                full_res = _rasterize_region(pdf_path, page_num)
                if full_res:
                    pil_img, b64 = full_res
                    vis_desc = _call_vision_llm(b64, elem_id, caption_txt) or ""
        else:
            # OCR fallback or no-vision mode
            crop_res = _rasterize_region(pdf_path, page_num, render_bbox)
            if crop_res:
                pil_img, _ = crop_res
                vis_desc = _ocr_image(pil_img, elem_id)

        elem = VisualElement(
            element_type="IMAGE",
            element_id=elem_id,
            caption=caption_txt,
            page=page_num_1,
            description=vis_desc,
        )
        elements.append(elem)
        logger.info(
            f"[visual] 擷取 {elem_id}（page {page_num_1}）"
            f"  caption={caption_txt[:40]!r}  desc={len(vis_desc)}字"
        )

    return elements, exclude_bboxes


def _find_caption_near_bbox(plumber_page, bbox: Tuple, search_below: bool = True) -> Optional[str]:
    """
    在給定 bbox 的上方或下方 80pt 範圍內搜尋 Caption 行。
    Search for a caption line within 80pt above or below a bbox.
    """
    x0, top, x1, bottom = bbox
    h = plumber_page.height
    search_range = 80

    if search_below:
        search_bbox = (0, bottom, plumber_page.width, min(h, bottom + search_range))
    else:
        search_bbox = (0, max(0, top - search_range), plumber_page.width, top)

    try:
        region = plumber_page.within_bbox(search_bbox)
        text = region.extract_text() or ""
        for line in text.splitlines():
            if _CAPTION_RE.match(line.strip()):
                return line.strip()
    except Exception:
        pass
    return None


def _find_nearest_image_above(
    image_bboxes: List[Tuple], caption_top: float, max_distance: float = 200
) -> Optional[Tuple]:
    """
    找到 caption 行上方、距離最近（且在 max_distance pt 內）的圖片 bbox。
    Find the nearest image bbox above a caption line.
    """
    candidates = [
        bbox for bbox in image_bboxes
        if bbox[3] <= caption_top + 20   # image bottom is above or near caption top
        and (caption_top - bbox[1]) <= max_distance
    ]
    if not candidates:
        return None
    # Closest = whose bottom (bbox[3]) is nearest to caption_top
    return max(candidates, key=lambda b: b[3])


# ══════════════════════════════════════════════════════════════════
#  STAGE 4  ─  SEMANTIC / PRIORITY CHUNKING
# ══════════════════════════════════════════════════════════════════

def _split_into_sections(full_text: str) -> List[SemanticSection]:
    """
    依章節標題將全文切分為語意段落。
    Split full document text into semantic sections by heading detection.

    Rules:
        • Each heading line starts a new section.
        • Abstract / Conclusion → is_priority=True (larger chunks).
        • Lines before the first heading go to a "Preamble" section.
    """
    sections: List[SemanticSection] = []
    current = SemanticSection(heading="Preamble", is_priority=False)

    for raw_line in full_text.splitlines():
        line = raw_line.strip()
        m = _SECTION_RE.match(line)
        if m:
            # Flush current section (only if it has content)
            if current.lines:
                sections.append(current)
            sec_name = m.group(1)
            is_prio  = sec_name.lower() in _PRIORITY_SECTIONS
            current  = SemanticSection(
                heading=sec_name,
                is_priority=is_prio,
                lines=[line],   # include heading itself
            )
        else:
            current.lines.append(raw_line)

    if current.lines:
        sections.append(current)

    logger.debug(
        f"[sections] {len(sections)} sections detected: "
        + ", ".join(f"{s.heading}({'★' if s.is_priority else '-'})" for s in sections[:8])
    )
    return sections


def _chunk_sections(
    sections: List[SemanticSection],
    source: str,
    page_hint: int,
) -> list:
    """
    對每個語意段落套用對應的 chunk 策略，產生 LangChain Document 列表。
    Apply appropriate chunk strategy per section and return Document list.
    """
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs = []
    for sec in sections:
        text = sec.text().strip()
        if not text:
            continue

        if sec.is_priority:
            size    = CHUNK_SIZE_PRIORITY
            overlap = CHUNK_OVERLAP_PRIORITY
        else:
            size    = CHUNK_SIZE_BODY
            overlap = CHUNK_OVERLAP_BODY

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", ".", "！", "？", " ", ""],
        )
        chunks = splitter.split_text(text)
        for chunk in chunks:
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source":      source,
                    "page":        page_hint,
                    "section":     sec.heading,
                    "is_priority": sec.is_priority,
                    "chunk_type":  "text",
                },
            ))
    return docs


def _build_visual_documents(elements: List[VisualElement], source: str) -> list:
    """
    將視覺元素轉換為帶結構化標籤的 LangChain Documents（不再二次切塊）。
    Convert VisualElements to structured-tag Documents (no further splitting).
    """
    from langchain_core.documents import Document
    docs = []
    for elem in elements:
        docs.append(Document(
            page_content=elem.to_chunk_text(),
            metadata={
                "source":       source,
                "page":         elem.page,
                "section":      "visual_element",
                "element_type": elem.element_type,
                "element_id":   elem.element_id,
                "chunk_type":   "visual",
            },
        ))
    return docs


# ══════════════════════════════════════════════════════════════════
#  STAGE 5  ─  EMBEDDING & FAISS
# ══════════════════════════════════════════════════════════════════

def get_embedding_func(provider: str = None):
    """
    取得 Embedding 函式。
    Returns embedding function for the specified provider.

    Supported providers:
        "google"       → GoogleGenerativeAIEmbeddings (models/gemini-embedding-001)
        "groq"         → HuggingFaceEmbeddings (BAAI/bge-m3，本地，免費，支援中文)
        "huggingface"  → HuggingFaceEmbeddings (sentence-transformers，與 groq 相同)

    Note:
        "groq" 與 "huggingface" 都使用本地 HuggingFace Embedding，
        與 llm_helper.py 的 provider 命名保持一致。
    """
    p = (provider or os.getenv("LLM_PROVIDER", "google")).lower()
    logger.debug(f"[embedding] provider={p}")

    if p == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY 未設定。請在 .streamlit/secrets.toml 加入。\n"
                "GOOGLE_API_KEY not set."
            )
        # ✅ 修正：使用新版模型名稱 gemini-embedding-001（舊版 embedding-001 已棄用）
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
        )

    elif p in ("groq", "huggingface"):
        # "groq" provider 的 LLM 用 Groq API，但 embedding 只能本地 HuggingFace
        # "huggingface" 為舊版命名，兩者行為相同
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model_name = os.getenv(
            "HF_EMBED_MODEL",  # 與 llm_helper.py 保持一致
            os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-m3"),  # 向後相容舊環境變數
        )
        logger.debug(f"[embedding] HuggingFace model: {model_name}")
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    else:
        raise ValueError(
            f"不支援的 provider: {p!r}。請使用 'google'、'groq' 或 'huggingface'。\n"
            f"Unsupported provider: {p!r}. Use 'google', 'groq', or 'huggingface'."
        )


# ══════════════════════════════════════════════════════════════════
#  STAGE 5 HELPER  ─  BATCH EMBEDDING WITH RATE-LIMIT RETRY
# ══════════════════════════════════════════════════════════════════

# Google 免費額度：每分鐘 100 次請求
EMBED_BATCH_SIZE  = 20    # 每批次 chunk 數
EMBED_BATCH_DELAY = 5.0   # 批次間等待秒數
RETRY_MAX_ATTEMPTS = 5    # 最大重試次數
RETRY_BASE_DELAY   = 60.0 # 指數退避基底秒數
from langchain_community.vectorstores import FAISS

def _embed_with_batches(chunks: list, embedding_func) -> "FAISS":
    """
    分批嵌入所有 Chunk，自動處理 429 RESOURCE_EXHAUSTED 速率限制。
    Embed chunks in batches with exponential backoff on 429 errors.

    使用指數退避：第 n 次重試等待 RETRY_BASE_DELAY * 2^(n-1) 秒。
    """
    import time, re
    

    def _embed_texts_with_retry(texts: list) -> list:
        for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
            try:
                return embedding_func.embed_documents(texts)
            except Exception as e:
                err = str(e)
                is_rate_limit = "429" in err or "RESOURCE_EXHAUSTED" in err
                if is_rate_limit and attempt < RETRY_MAX_ATTEMPTS:
                    m = re.search(r"retryDelay.*?(\d+)s", err)
                    suggested = int(m.group(1)) if m else 0
                    wait = max(suggested + 5, RETRY_BASE_DELAY * (2 ** (attempt - 1)))
                    logger.warning(
                        f"[embed_batch] 第 {attempt} 次速率限制，等待 {wait:.0f}s 後重試..."
                    )
                    time.sleep(wait)
                else:
                    raise
        return []

    total = len(chunks)
    logger.info(f"[embed_batch] 共 {total} 個 Chunk，批次大小 {EMBED_BATCH_SIZE}")

    # 第一批：建立初始 FAISS 索引
    first_batch = chunks[:EMBED_BATCH_SIZE]
    texts_0 = [doc.page_content for doc in first_batch]
    metas_0 = [doc.metadata for doc in first_batch]
    embs_0  = _embed_texts_with_retry(texts_0)
    index   = FAISS.from_embeddings(list(zip(texts_0, embs_0)), embedding_func, metadatas=metas_0)

    # 後續批次：逐批合併
    import time
    for start in range(EMBED_BATCH_SIZE, total, EMBED_BATCH_SIZE):
        batch = chunks[start: start + EMBED_BATCH_SIZE]
        batch_num = start // EMBED_BATCH_SIZE + 1
        logger.debug(f"[embed_batch] 第 {batch_num} 批：Chunk {start+1}–{start+len(batch)}")
        time.sleep(EMBED_BATCH_DELAY)

        texts_b = [doc.page_content for doc in batch]
        metas_b = [doc.metadata for doc in batch]
        embs_b  = _embed_texts_with_retry(texts_b)
        sub_idx = FAISS.from_embeddings(list(zip(texts_b, embs_b)), embedding_func, metadatas=metas_b)
        index.merge_from(sub_idx)

    logger.info("[embed_batch] 所有批次嵌入完成")
    return index


# ══════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════

def embed_document(
    file_name:        str,
    file_folder:      str  = "pdf",
    embedding_folder: str  = "index",
    use_vision:       bool = True,
    provider:         str  = None,
) -> dict:
    """
    嵌入單一 PDF（版面感知 + 圖文強綁定 + 語意切塊）並儲存 FAISS 索引。
    Full pipeline: layout-aware parse → figure anchoring → semantic chunks → FAISS.

    Args:
        file_name:        PDF 檔名
        file_folder:      PDF 所在目錄（預設 "pdf"）
        embedding_folder: FAISS 索引輸出目錄（預設 "index"）
        use_vision:       是否啟用 Gemini Vision 描述圖表
        provider:         "google" | "huggingface" | None（讀環境變數 LLM_PROVIDER）

    Returns:
        dict with stats: total_pages, visual_elements, text_chunks, visual_chunks
    """
    from langchain_community.vectorstores import FAISS

    p = provider or os.getenv("LLM_PROVIDER", "google")
    pdf_path = os.path.join(file_folder, file_name)
    logger.info(
        f"[embed_document] ▶ 開始處理: {pdf_path} "
        f"provider={p}  use_vision={use_vision}"
    )

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"找不到檔案: {pdf_path}")

    all_text_docs    = []
    all_visual_docs  = []
    all_visual_elems = []

    # ── Per-page processing ────────────────────────────────────────
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        logger.info(f"[embed_document] 共 {total_pages} 頁")

        # Accumulate full document text page by page for section detection
        page_texts: List[Tuple[int, str]] = []   # (page_num_1, text)

        for i, plumber_page in enumerate(pdf.pages):
            logger.info(f"[embed_document] ─── 第 {i+1}/{total_pages} 頁 ───")

            # STAGE 2: Extract visual elements first (get exclude_bboxes)
            vis_elems, exclude_bboxes = _extract_visual_elements(
                pdf_path     = pdf_path,
                page_num     = i,
                plumber_page = plumber_page,
                use_vision   = use_vision,
                provider     = p,
            )
            all_visual_elems.extend(vis_elems)

            # STAGE 1: Layout-aware body text (exclude visual regions)
            body_text = _extract_layout_text(plumber_page, exclude_bboxes)
            page_texts.append((i + 1, body_text))

    # ── STAGE 4: Global section-aware chunking across all pages ───
    # Concatenate all pages with page markers so we can split semantically
    full_doc_text = "\n\n".join(
        f"<!-- PAGE {pg} -->\n{txt}" for pg, txt in page_texts if txt.strip()
    )
    sections = _split_into_sections(full_doc_text)

    # Chunk text sections
    # page_hint = 1 for document-level sections (page tracked in section text)
    all_text_docs = _chunk_sections(sections, source=file_name, page_hint=1)
    logger.info(f"[embed_document] 文字 Chunk 數量: {len(all_text_docs)}")

    # STAGE 3+2: Build visual element Documents (structured-tag, no re-split)
    all_visual_docs = _build_visual_documents(all_visual_elems, source=file_name)
    logger.info(f"[embed_document] 視覺元素 Chunk 數量: {len(all_visual_docs)}")

    # ── STAGE 5: Embed all chunks → FAISS ─────────────────────────
    all_chunks = all_text_docs + all_visual_docs
    if not all_chunks:
        raise ValueError(f"文件 {file_name} 未產生任何 Chunk，請確認 PDF 格式正確。")

    embedding_func = get_embedding_func(provider=p)
    logger.info(f"[embed_document] 正在向量化 {len(all_chunks)} 個 Chunk（provider={p}）...")

    # ✅ 修正：Google provider 使用分批嵌入 + 429 重試，避免免費額度速率限制
    if p == "google":
        search_index = _embed_with_batches(all_chunks, embedding_func)
    else:
        search_index = FAISS.from_documents(all_chunks, embedding_func)

    os.makedirs(embedding_folder, exist_ok=True)
    index_name = file_name + ".index"
    search_index.save_local(folder_path=embedding_folder, index_name=index_name)
    logger.info(f"[embed_document] ✅ 索引已儲存: {embedding_folder}/{index_name}")

    stats = {
        "total_pages":    total_pages,
        "visual_elements": len(all_visual_elems),
        "text_chunks":    len(all_text_docs),
        "visual_chunks":  len(all_visual_docs),
        "total_chunks":   len(all_chunks),
    }
    logger.info(f"[embed_document] 統計: {stats}")
    return stats


def embed_all_pdf_docs(
    use_vision: bool = True,
    provider:   str  = None,
) -> dict:
    """
    嵌入 pdf/ 目錄下所有 PDF。
    Embed all PDFs in the pdf/ directory.
    """
    pdf_directory = "pdf"
    logger.info(f"[embed_all_pdf_docs] 掃描目錄: {pdf_directory}")

    if not os.path.exists(pdf_directory):
        raise FileNotFoundError(f"目錄 '{pdf_directory}' 不存在。")

    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith(".pdf")]
    if not pdf_files:
        raise ValueError("pdf/ 目錄下沒有 PDF 檔案。")

    logger.info(f"[embed_all_pdf_docs] 找到 {len(pdf_files)} 個 PDF: {pdf_files}")
    results = {}
    for pdf_file in sorted(pdf_files):
        try:
            results[pdf_file] = embed_document(
                file_name   = pdf_file,
                file_folder = pdf_directory,
                use_vision  = use_vision,
                provider    = provider,
            )
        except Exception as e:
            logger.error(f"[embed_all_pdf_docs] {pdf_file} 處理失敗: {e}", exc_info=True)
            results[pdf_file] = {"error": str(e)}
    return results


def get_all_index_files() -> List[str]:
    """取得所有已建立的 FAISS 索引名稱（不含副檔名）。"""
    index_directory = "index"
    postfix = ".index.faiss"

    if not os.path.exists(index_directory):
        raise FileNotFoundError(
            f"索引目錄 '{index_directory}' 不存在，請先執行文件嵌入。"
        )
    index_files = [
        f.replace(postfix, "")
        for f in os.listdir(index_directory)
        if f.endswith(postfix)
    ]
    if not index_files:
        raise ValueError("index/ 目錄下沒有索引檔案，請先嵌入文件。")

    logger.debug(f"[get_all_index_files] 找到: {index_files}")
    return index_files


if __name__ == "__main__":
    embed_all_pdf_docs()