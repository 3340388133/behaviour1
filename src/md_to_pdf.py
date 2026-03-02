from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import re

# Register Chinese font
pdfmetrics.registerFont(TTFont('SimSun', '/System/Library/Fonts/STHeiti Light.ttc'))

doc = SimpleDocTemplate("src/plan.pdf", pagesize=A4)
styles = getSampleStyleSheet()

# Custom styles with Chinese font
title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontName='SimSun', fontSize=18, spaceAfter=12)
h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontName='SimSun', fontSize=14, spaceAfter=10)
h3_style = ParagraphStyle('H3', parent=styles['Heading3'], fontName='SimSun', fontSize=12, spaceAfter=8)
normal_style = ParagraphStyle('Normal', parent=styles['Normal'], fontName='SimSun', fontSize=10, leading=14)

story = []

with open('src/plan.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

table_data = []
in_table = False

for line in lines:
    line = line.rstrip()

    if line.startswith('# '):
        story.append(Paragraph(line[2:], title_style))
        story.append(Spacer(1, 0.2*inch))
    elif line.startswith('## '):
        story.append(Paragraph(line[3:], h2_style))
        story.append(Spacer(1, 0.15*inch))
    elif line.startswith('### '):
        story.append(Paragraph(line[4:], h3_style))
        story.append(Spacer(1, 0.1*inch))
    elif line.startswith('|'):
        if not in_table:
            in_table = True
            table_data = []
        cells = [cell.strip() for cell in line.split('|')[1:-1]]
        if not all(c.startswith('-') for c in cells):
            table_data.append(cells)
    else:
        if in_table and table_data:
            t = Table(table_data)
            t.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'SimSun'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ]))
            story.append(t)
            story.append(Spacer(1, 0.2*inch))
            table_data = []
            in_table = False

        if line.strip():
            story.append(Paragraph(line, normal_style))
            story.append(Spacer(1, 0.05*inch))

doc.build(story)
print("PDF created: src/plan.pdf")
