import markdown
import os
from weasyprint import HTML, CSS

# 1. Ler o conteúdo do relatório técnico technical_report.md
with open("technical_report.md", "r", encoding="utf-8") as f:
    md_content = f.read()

# 2. Configuração de extensões Markdown para garantir interpretação das tabelas
md = markdown.Markdown(extensions=['markdown.extensions.tables', 'markdown.extensions.fenced_code'])
html_content = md.convert(md_content)

# 3. CSS de Alta Qualidade para Documentos Técnicos
# O 'pre' com 'white-space: pre' é crucial para manter o alinhamento da arte ASCII
css_style = """
    @page { size: A4; margin: 2.5cm; }
    body { font-family: 'Helvetica', sans-serif; line-height: 1.5; color: #333; }
    h1 { color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }
    table { border-collapse: collapse; width: 100%; margin: 20px 0; }
    th, td { border: 1px solid #bdc3c7; padding: 10px; text-align: left; }
    th { background-color: #f8f9fa; }
    pre { 
        background: #f4f4f4; 
        padding: 15px; 
        border-radius: 5px; 
        white-space: pre; 
        font-family: 'Courier New', monospace; 
        font-size: 0.85em;
        overflow-x: auto;
    }
    code { font-family: 'Courier New', monospace; background: #eee; padding: 2px 4px; }
    hr { border: 0; border-top: 1px solid #ddd; margin: 30px 0; }
    img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 10px auto;
    }
"""

# 4. Geração do PDF
pdf_filename = "technical_report.pdf"
HTML(string=f"<html><body>{html_content}</body></html>", base_url=os.getcwd()).write_pdf(
    pdf_filename,
    stylesheets=[CSS(string=css_style)],
    presentational_hints=True
)

print(f"Relatório gerado com sucesso: {pdf_filename}")