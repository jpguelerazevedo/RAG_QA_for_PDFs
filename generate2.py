import os
import json
import tiktoken # Para tokenização e chunking
import re # Importar módulo para expressões regulares
import ollama

# --- Configurações ---
EMBEDDING_MODEL = "nomic-embed-text:latest"
INPUT_MD_PATH = "./mdPath/pdf2.md"
EMBEDDINGS_OUTPUT_PATH = "./embeddings/embeddings.json"
CHUNK_SIZE = 512 # Tamanho de cada pedaço de texto em tokens (para texto não-tabela)
OVERLAP_SIZE = 128 #     Sobreposição entre os pedaços (para texto não-tabela)

# --- Função para gerar embedding via Ollama ---
def generate_embedding(text: str):
    """Gera embedding usando o modelo nomic-embed-text via Ollama."""
    try:
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        return response['embedding']
    except Exception as e:
        print(f"Erro ao gerar embedding: {e}")
        return None

# --- Função para dividir texto por seções baseadas em cabeçalhos ## ---
def chunk_text_by_sections(text, max_chunk_size=2048, encoding_name="cl100k_base"):
    """
    Divide o texto em seções baseadas nos cabeçalhos ## (nível 2).
    Cada parágrafo dentro de uma seção terá seu próprio embedding.
    
    Args:
        text: Texto a ser dividido
        max_chunk_size: Tamanho máximo do chunk em tokens (para parágrafos muito grandes)
        encoding_name: Nome do encoding para tokenização
        
    Returns:
        list: Lista de dicionários com parágrafos e contexto
    """
    tokenizer = tiktoken.get_encoding(encoding_name)
    
    # Dividir o texto em linhas
    lines = text.split('\n')
    sections = []
    current_section = ""
    current_title = ""
    
    for line in lines:
        line_stripped = line.strip()
        
        # Verificar se é um cabeçalho de nível 2 (##)
        if line_stripped.startswith('## ') and not line_stripped.startswith('### '):
            # Se já temos uma seção em andamento, salvá-la
            if current_section.strip():
                sections.append({
                    "title": current_title,
                    "content": current_section.strip()
                })
            
            # Começar nova seção
            current_title = line_stripped[3:].strip()  # Remover "## "
            current_section = line + '\n'
        else:
            # Adicionar linha à seção atual
            current_section += line + '\n'
    
    # Adicionar a última seção
    if current_section.strip():
        sections.append({
            "title": current_title,
            "content": current_section.strip()
        })
    
    # Processar cada seção para criar embeddings por parágrafo
    section_chunks = []
    for section in sections:
        section_content = section["content"]
        section_title = section["title"]
        
        # Dividir a seção em parágrafos (separados por linha dupla)
        paragraphs = re.split(r'\n\s*\n', section_content)
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if paragraph:  # Ignorar parágrafos vazios
                # Verificar se o parágrafo contém apenas o cabeçalho
                lines = paragraph.split('\n')
                is_header_only = (len(lines) == 1 and lines[0].startswith('## '))
                
                # Calcular tokens do parágrafo
                paragraph_tokens = len(tokenizer.encode(paragraph))
                
                # Se é apenas cabeçalho, pular este parágrafo
                if is_header_only:
                    continue
                
                # Se o parágrafo é muito grande, dividi-lo em partes menores
                if paragraph_tokens > max_chunk_size:
                    # Dividir parágrafo em chunks menores
                    sub_chunks = chunk_text_with_overlap(paragraph, CHUNK_SIZE, OVERLAP_SIZE, encoding_name)
                    
                    for j, sub_chunk in enumerate(sub_chunks):
                        # Remover o cabeçalho se estiver presente no início do sub_chunk
                        content_without_header = sub_chunk
                        if content_without_header.startswith('## '):
                            lines_sub = content_without_header.split('\n', 1)
                            if len(lines_sub) > 1:
                                content_without_header = lines_sub[1].strip()
                            else:
                                # Se só tem cabeçalho, pular
                                continue
                        
                        contextual_content = f"[Seção: {section_title}] {content_without_header}"
                        
                        section_chunks.append({
                            "type": "text_chunk",
                            "content": sub_chunk,
                            "contextual_content": contextual_content,
                            "section_title": section_title,
                            "chunk_index": j,
                            "total_chunks": len(sub_chunks)
                        })
                else:
                    # Parágrafo pequeno o suficiente para ser um chunk único
                    # Remover o cabeçalho se estiver presente no início do parágrafo
                    content_without_header = paragraph
                    if content_without_header.startswith('## '):
                        lines_para = content_without_header.split('\n', 1)
                        if len(lines_para) > 1:
                            content_without_header = lines_para[1].strip()
                        else:
                            # Se só tem cabeçalho, pular
                            continue
                    
                    contextual_content = f"[Seção: {section_title}] {content_without_header}"
                    
                    section_chunks.append({
                        "type": "text_chunk",
                        "content": paragraph,
                        "contextual_content": contextual_content,
                        "section_title": section_title,
                        "chunk_index": 0,
                        "total_chunks": 1
                    })
    
    return section_chunks

# --- Função para dividir texto com sobreposição ---
def chunk_text_with_overlap(text, chunk_size, overlap_size, encoding_name="cl100k_base"):
    """
    Divide o texto em chunks com sobreposição entre eles.
    
    Args:
        text: Texto a ser dividido
        chunk_size: Tamanho do chunk em tokens
        overlap_size: Tamanho da sobreposição em tokens
        encoding_name: Nome do encoding para tokenização
        
    Returns:
        list: Lista de chunks de texto
    """
    tokenizer = tiktoken.get_encoding(encoding_name)
    tokens = tokenizer.encode(text)
    
    chunks = []
    start_idx = 0
    
    while start_idx < len(tokens):
        # Definir fim do chunk
        end_idx = start_idx + chunk_size
        
        # Extrair tokens do chunk
        chunk_tokens = tokens[start_idx:end_idx]
        
        # Decodificar de volta para texto
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        # Mover para o próximo chunk com sobreposição
        start_idx = end_idx - overlap_size
        
        # Se chegamos ao fim, parar
        if end_idx >= len(tokens):
            break
    
    return chunks

# --- Função para extrair tabelas do markdown ---
def extract_tables_from_markdown(text):
    """
    Extrai tabelas do markdown e retorna uma lista de dicionários com informações sobre cada tabela.
    
    Args:
        text: Texto markdown completo
        
    Returns:
        list: Lista de dicionários com informações sobre tabelas
    """
    # Regex para encontrar tabelas markdown
    table_pattern = r'(\|[^|\n]*\|[^|\n]*\|[^\n]*\n)(\|[-\s|]*\|[^\n]*\n)(((\|[^|\n]*\|[^|\n]*\|[^\n]*\n)+))'
    
    tables = []
    matches = re.finditer(table_pattern, text, re.MULTILINE)
    
    for match in matches:
        table_text = match.group(0)
        
        # Separar cabeçalho, separador e linhas
        lines = table_text.strip().split('\n')
        
        if len(lines) >= 3:  # Pelo menos cabeçalho, separador e uma linha
            header = lines[0]
            separator = lines[1]
            data_lines = lines[2:]
            
            # Extrair colunas do cabeçalho
            header_columns = [col.strip() for col in header.split('|') if col.strip()]
            
            # Processar cada linha de dados
            for i, line in enumerate(data_lines):
                if line.strip():  # Ignorar linhas vazias
                    columns = [col.strip() for col in line.split('|') if col.strip()]
                    
                    # Criar conteúdo da linha
                    row_content = ' | '.join(columns)
                    
                    # Criar conteúdo contextual sem repetir o cabeçalho
                    contextual_content = row_content
                    
                    tables.append({
                        "type": "table_row",
                        "content": row_content,
                        "contextual_content": contextual_content,
                        "table_header": header_columns,
                        "table_header_raw": header,
                        "row_number": i + 1,
                        "row_data": columns
                    })
    
    return tables

# --- Função para remover tabelas do texto ---
def remove_tables_from_text(text):
    """
    Remove tabelas do texto markdown, mantendo apenas o texto puro.
    
    Args:
        text: Texto markdown com tabelas
        
    Returns:
        str: Texto sem tabelas
    """
    # Regex para encontrar e remover tabelas
    table_pattern = r'(\|[^|\n]*\|[^|\n]*\|[^\n]*\n)(\|[-\s|]*\|[^\n]*\n)(((\|[^|\n]*\|[^|\n]*\|[^\n]*\n)+))'
    text_without_tables = re.sub(table_pattern, '', text, flags=re.MULTILINE)
    
    # Limpar linhas vazias extras
    text_without_tables = re.sub(r'\n\s*\n\s*\n', '\n\n', text_without_tables)
    
    return text_without_tables

# --- Função principal ---
def main():
    """
    Função principal que processa o arquivo markdown e gera embeddings.
    """
    # Verificar se o arquivo de entrada existe
    if not os.path.exists(INPUT_MD_PATH):
        print(f"Erro: Arquivo de entrada '{INPUT_MD_PATH}' não encontrado.")
        return
    
    # Ler o arquivo markdown
    print(f"Lendo arquivo markdown: {INPUT_MD_PATH}")
    with open(INPUT_MD_PATH, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    print(f"Arquivo lido com sucesso. Tamanho: {len(markdown_content)} caracteres")
    
    # Extrair tabelas
    print("Extraindo tabelas do markdown...")
    tables = extract_tables_from_markdown(markdown_content)
    print(f"Encontradas {len(tables)} linhas de tabela")
    
    # Remover tabelas do texto para processar texto puro
    print("Removendo tabelas do texto...")
    text_without_tables = remove_tables_from_text(markdown_content)
    
    # Dividir texto em seções
    print("Dividindo texto em seções...")
    text_chunks = chunk_text_by_sections(text_without_tables)
    print(f"Criados {len(text_chunks)} chunks de texto")
    
    # Combinar todos os itens para embedding
    all_items = text_chunks + tables
    print(f"Total de itens para embedding: {len(all_items)}")
    
    # Gerar embeddings
    print("Gerando embeddings...")
    embeddings_data = []
    
    for i, item in enumerate(all_items):
        print(f"Processando item {i+1}/{len(all_items)}: {item['type']}")
        
        # Usar conteúdo contextual para gerar embedding
        text_for_embedding = item.get('contextual_content', item['content'])
        
        # Gerar embedding
        embedding = generate_embedding(text_for_embedding)
        
        if embedding is not None:
            # Adicionar ao resultado
            embeddings_data.append({
                "id": i,
                "type": item["type"],
                "content": item["content"],
                "contextual_content": item.get("contextual_content", ""),
                "embedding": embedding,
                **{k: v for k, v in item.items() if k not in ["type", "content", "contextual_content"]}
            })
        else:
            print(f"Erro ao gerar embedding para item {i+1}. Pulando.")
    
    # Salvar embeddings
    print(f"Salvando embeddings em {EMBEDDINGS_OUTPUT_PATH}")
    os.makedirs(os.path.dirname(EMBEDDINGS_OUTPUT_PATH), exist_ok=True)
    
    with open(EMBEDDINGS_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(embeddings_data, f, ensure_ascii=False, indent=2)
    
    print(f"Embeddings salvos com sucesso!")
    print(f"Total de embeddings gerados: {len(embeddings_data)}")
    
    # Estatísticas
    text_chunks_count = len([item for item in embeddings_data if item["type"] == "text_chunk"])
    table_rows_count = len([item for item in embeddings_data if item["type"] == "table_row"])
    
    print(f"\n📊 Estatísticas:")
    print(f"   - Chunks de texto: {text_chunks_count}")
    print(f"   - Linhas de tabela: {table_rows_count}")
    print(f"   - Total: {len(embeddings_data)}")
    
    # Verificar dimensões dos embeddings
    if embeddings_data:
        embedding_dim = len(embeddings_data[0]['embedding'])
        print(f"   - Dimensão dos embeddings: {embedding_dim}")

if __name__ == "__main__":
    main()
