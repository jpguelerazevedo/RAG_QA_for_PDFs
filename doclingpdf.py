from docling.document_converter import DocumentConverter

# Definindo o caminho do arquivo PDF de origem
source = "pdfPath/pdf.pdf"

# Definindo o caminho onde o arquivo Markdown será salvo
mdPath = "mdPath/pdf2.md" # <--- ALtere este caminho para o local desejado

converter = DocumentConverter()
result = converter.convert(source)

# Salvando o conteúdo Markdown no caminho especificado por mdPath
with open(mdPath, "w", encoding="utf-8") as f:
    f.write(result.document.export_to_markdown())

print(f"O arquivo Markdown foi salvo em: {mdPath}")