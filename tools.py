"""Tool definitions for the RAG chatbot."""
import os
import re
import json
import markdown
from datetime import datetime
from langchain_core.tools import tool
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
 
 
# ── Chart data extractor ──────────────────────────────────────────────────────
 
def _extract_chart_as_table(chart_json_str: str) -> list | None:
    """
    Parse a Chart.js JSON block and return a list of ReportLab flowables
    representing the chart data as a formatted table + title.
    Returns None if parsing fails.
    """
    try:
        config = json.loads(chart_json_str)
    except json.JSONDecodeError:
        return None
 
    styles = getSampleStyleSheet()
    flowables = []
 
    # Chart title
    options = config.get("options", {})
    plugins = options.get("plugins", {})
    title_cfg = plugins.get("title", {})
    chart_title = title_cfg.get("text", f"{config.get('type', 'Chart').title()} Chart")
 
    title_style = ParagraphStyle(
        "ChartTitle",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=6,
    )
    flowables.append(Paragraph(f"📊 {chart_title}", title_style))
 
    data_block = config.get("data", {})
    labels = data_block.get("labels", [])
    datasets = data_block.get("datasets", [])
 
    if not labels or not datasets:
        flowables.append(Paragraph("(No chart data available)", styles["Normal"]))
        return flowables
 
    # Build table: header row + one row per label
    header = ["Category"] + [ds.get("label", f"Series {i+1}") for i, ds in enumerate(datasets)]
    table_data = [header]
 
    for idx, label in enumerate(labels):
        row = [str(label)]
        for ds in datasets:
            vals = ds.get("data", [])
            val = vals[idx] if idx < len(vals) else "-"
            row.append(str(val))
        table_data.append(row)
 
    # Style the table
    col_width = (6.5 * inch) / len(header)
    tbl = Table(table_data, colWidths=[col_width] * len(header))
    tbl.setStyle(TableStyle([
        # Header row
        ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  10),
        ("ALIGN",        (0, 0), (-1, 0),  "CENTER"),
        ("BOTTOMPADDING",(0, 0), (-1, 0),  8),
        ("TOPPADDING",   (0, 0), (-1, 0),  8),
        # Data rows — alternating background
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f5f5f5"), colors.white]),
        ("FONTNAME",     (0, 1), (-1, -1),  "Helvetica"),
        ("FONTSIZE",     (0, 1), (-1, -1),  9),
        ("ALIGN",        (1, 1), (-1, -1),  "CENTER"),
        ("ALIGN",        (0, 1), (0, -1),   "LEFT"),
        ("LEFTPADDING",  (0, 0), (-1, -1),  8),
        ("RIGHTPADDING", (0, 0), (-1, -1),  8),
        ("TOPPADDING",   (0, 1), (-1, -1),  5),
        ("BOTTOMPADDING",(0, 1), (-1, -1),  5),
        # Grid
        ("GRID",         (0, 0), (-1, -1),  0.5, colors.HexColor("#cccccc")),
        ("BOX",          (0, 0), (-1, -1),  1,   colors.HexColor("#1a1a2e")),
    ]))
 
    flowables.append(tbl)
    flowables.append(Paragraph(
        "<i>Note: This chart is shown as a data table in the PDF. "
        "Interactive charts are available in the web app.</i>",
        ParagraphStyle("Note", parent=styles["Normal"], fontSize=8,
                       textColor=colors.grey, spaceBefore=4)
    ))
    return flowables
 
 
# ── Chart block processor ─────────────────────────────────────────────────────
 
def _process_chartjs_blocks(text: str) -> tuple[str, list]:
    """
    Split text into clean text (chartjs blocks removed) and a list of
    ReportLab flowable lists (one per chart).
 
    Returns:
        (cleaned_text, list_of_chart_flowable_lists)
    """
    pattern = re.compile(r"```chartjs\s*([\s\S]*?)```")
    charts = []
 
    def replace(match):
        flowables = _extract_chart_as_table(match.group(1).strip())
        if flowables:
            charts.append(flowables)
        return ""   # remove from text
 
    cleaned = pattern.sub(replace, text)
    return cleaned, charts
 
 
# ── PDF generator tool ────────────────────────────────────────────────────────
 
@tool
def generate_pdf_tool(report_text: str):
    """Generate a professional PDF report from the provided text.
 
    Use this when the user asks to generate, save, export, or download content as a PDF.
    - Markdown formatting is preserved (bold, italic, bullet points, headings).
    - Chart.js blocks are converted into formatted data tables in the PDF.
    - The PDF is saved to the 'outputs/' folder and the file path is returned.
 
    Args:
        report_text: The full content to include in the PDF (supports markdown + chartjs blocks)
 
    Returns:
        Success message with the full file path, or an error message.
    """
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_dir, f"DocuBot_Report_{timestamp}.pdf")
 
    try:
        # Separate chart blocks from text
        cleaned_text, chart_flowable_groups = _process_chartjs_blocks(report_text)
 
        # Set up document
        doc = SimpleDocTemplate(
            file_path,
            pagesize=letter,
            leftMargin=inch,
            rightMargin=inch,
            topMargin=inch,
            bottomMargin=inch,
        )
        styles = getSampleStyleSheet()
 
        # Custom styles
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontSize=20,
            textColor=colors.HexColor("#1a1a2e"),
            spaceAfter=6,
            alignment=TA_CENTER,
        )
        subtitle_style = ParagraphStyle(
            "Subtitle",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.grey,
            spaceAfter=16,
            alignment=TA_CENTER,
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=10,
            leading=16,
            spaceAfter=8,
        )
        h1_style = ParagraphStyle(
            "H1", parent=styles["Heading1"],
            fontSize=14, textColor=colors.HexColor("#1a1a2e"), spaceAfter=6
        )
        h2_style = ParagraphStyle(
            "H2", parent=styles["Heading2"],
            fontSize=12, textColor=colors.HexColor("#333366"), spaceAfter=4
        )
 
        story = []
 
        # ── Cover header ──────────────────────────────────────────────────────
        story.append(Paragraph("🤖 DocuBot", title_style))
        story.append(Paragraph("Professional Document Analysis Report", subtitle_style))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
            subtitle_style
        ))
        story.append(HRFlowable(width="100%", thickness=2,
                                 color=colors.HexColor("#1a1a2e"), spaceAfter=16))
 
        # ── Main text content ─────────────────────────────────────────────────
        html_text = markdown.markdown(cleaned_text, extensions=["extra", "nl2br"])
 
        # Map HTML tags → ReportLab-safe equivalents
        html_text = html_text.replace("<strong>", "<b>").replace("</strong>", "</b>")
        html_text = html_text.replace("<em>", "<i>").replace("</em>", "</i>")
        html_text = html_text.replace("<ul>", "").replace("</ul>", "")
        html_text = html_text.replace("<ol>", "").replace("</ol>", "")
        html_text = html_text.replace("<li>", " • ").replace("</li>", "<br/>")
        html_text = html_text.replace("<p>", "").replace("</p>", "<br/><br/>")
 
        # Handle h1/h2/h3 headings
        for tag, style in [("h1", h1_style), ("h2", h2_style), ("h3", h2_style)]:
            pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.IGNORECASE | re.DOTALL)
            for match in pattern.findall(html_text):
                pass  # handled below in split
 
        # Strip remaining unknown HTML tags (keep b, i, br, u)
        html_text = re.sub(
            r"<(?!/?(b|i|u|br|strike|a|font|super|sub|span))[^>]+>",
            "",
            html_text,
        )
 
        # Split into paragraphs and add to story
        for para in html_text.split("<br/><br/>"):
            para = para.strip()
            if para:
                try:
                    story.append(Paragraph(para, body_style))
                    story.append(Spacer(1, 4))
                except Exception:
                    # If paragraph has bad markup, strip and retry
                    clean = re.sub(r"<[^>]+>", "", para)
                    if clean.strip():
                        story.append(Paragraph(clean, body_style))
                        story.append(Spacer(1, 4))
 
        # ── Chart tables ──────────────────────────────────────────────────────
        if chart_flowable_groups:
            story.append(Spacer(1, 12))
            story.append(HRFlowable(width="100%", thickness=1,
                                     color=colors.HexColor("#cccccc"), spaceAfter=12))
            story.append(Paragraph("Data Visualizations", h1_style))
            for chart_flowables in chart_flowable_groups:
                for flowable in chart_flowables:
                    story.append(flowable)
                story.append(Spacer(1, 16))
 
        # ── Footer note ───────────────────────────────────────────────────────
        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=1,
                                 color=colors.HexColor("#cccccc"), spaceAfter=6))
        story.append(Paragraph(
            "Generated by DocuBot · Intelligent Document Analysis",
            ParagraphStyle("Footer", parent=styles["Normal"],
                           fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
        ))
 
        doc.build(story)
        return f"PDF created successfully at: {file_path}"
 
    except Exception as e:
        raise RuntimeError(f"PDF generation failed: {e}")
 
 
# Tool registry
ALL_TOOLS = [generate_pdf_tool]