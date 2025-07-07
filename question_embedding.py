import os
import json
import numpy as np
import faiss
import sys # Para saída de erro mais controlada
import ollama

# --- Configurações Principais ---
# Use variáveis de ambiente ou um arquivo de configuração para opções mais flexíveis
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
EMBEDDINGS_JSON_PATH = os.getenv("EMBEDDINGS_DATA_PATH", "./embeddings/embeddings.json")
# QUERY_TEXT pode vir de uma entrada do usuário em um sistema real
# Por enquanto, mantemos aqui para teste
QUERY_TEXT = "qual é o perído de isenção da taxa ?" 
NUM_RESULTS = 15

# --- Funções Auxiliares ---
def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calcula a similaridade de cosseno entre dois vetores numpy."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0 # Evita divisão por zero para vetores nulos
        
    return dot_product / (norm_vec1 * norm_vec2)

# --- Classes/Funções de Lógica de Negócio ---

class EmbeddingGenerator:
    """Gerencia o carregamento do modelo e a geração de embeddings via Ollama."""
    def __init__(self, model_name: str = "nomic-embed-text:latest"):
        self.model_name = model_name
        self._test_model()

    def _test_model(self):
        """Testa se o modelo está disponível no Ollama."""
        try:
            print(f"Testando modelo de embedding: {self.model_name}")
            # Teste simples para verificar se o modelo está disponível
            test_response = ollama.embeddings(model=self.model_name, prompt="teste")
            print("Modelo de embedding carregado com sucesso via Ollama.")
        except Exception as e:
            print(f"Erro ao acessar o modelo via Ollama: {e}", file=sys.stderr)
            print(f"Certifique-se de que o modelo '{self.model_name}' está instalado no Ollama.", file=sys.stderr)
            print("Execute: ollama pull nomic-embed-text:latest", file=sys.stderr)
            sys.exit(1)

    def generate_embedding(self, text: str) -> np.ndarray:
        """Gera o vetor de embedding para um dado texto via Ollama."""
        try:
            response = ollama.embeddings(model=self.model_name, prompt=text)
            return np.array(response['embedding'], dtype='float32')
        except Exception as e:
            print(f"Erro ao gerar embedding para o texto: '{text[:50]}...': {e}", file=sys.stderr)
            return None

class DocumentEmbeddingsDB:
    """Gerencia o carregamento e indexação dos embeddings de documentos."""
    def __init__(self, embeddings_json_path: str):
        self.embeddings_json_path = embeddings_json_path
        self.document_data = [] # Para armazenar ID, texto e vetor raw
        self.document_vectors = None # Apenas os vetores para FAISS
        self.document_texts = []
        self.document_ids = []
        self.document_types = []  # Para armazenar o tipo de cada embedding
        self.faiss_index = None
        self._load_and_index_embeddings()

    def _load_and_index_embeddings(self):
        """Carrega os embeddings do JSON e cria o índice FAISS."""
        if not os.path.exists(self.embeddings_json_path):
            print(f"Erro: O arquivo de embeddings '{self.embeddings_json_path}' não foi encontrado.", file=sys.stderr)
            print("Certifique-se de que você já gerou os embeddings do PDF usando o script anterior.", file=sys.stderr)
            sys.exit(1)

        try:
            with open(self.embeddings_json_path, 'r', encoding='utf-8') as f:
                self.document_data = json.load(f)
            print(f"Embeddings do PDF carregados com sucesso de '{self.embeddings_json_path}'.")

            if not self.document_data:
                print("Aviso: O arquivo de embeddings está vazio. Não há dados para comparar.")
                sys.exit(0) # Sai sem erro se não há dados para processar

            # Separar os dados para uso mais eficiente
            self.document_vectors = np.array([item['embedding'] for item in self.document_data], dtype='float32')
            
            # Adaptar para a nova estrutura de dados com contextual_content
            self.document_texts = []
            self.document_ids = []
            self.document_types = []
            
            for item in self.document_data:
                self.document_ids.append(item['id'])
                self.document_types.append(item['type'])
                
                # Para embeddings de linha de tabela, usar o conteúdo contextual
                if item['type'] == 'table_row':
                    # Usar o conteúdo contextual que é mais informativo para busca
                    display_text = f"[TABELA] {item['contextual_content']}"
                    self.document_texts.append(display_text)
                else:
                    # Para chunks de texto, usar o conteúdo contextual se disponível
                    if 'contextual_content' in item and item['contextual_content']:
                        self.document_texts.append(item['contextual_content'])
                    else:
                        self.document_texts.append(item['content'])

            # Criar e adicionar ao índice FAISS
            embedding_dimension = self.document_vectors.shape[1]
            self.faiss_index = faiss.IndexFlatL2(embedding_dimension)
            print(f"Adicionando {self.document_vectors.shape[0]} vetores ao índice FAISS...")
            self.faiss_index.add(self.document_vectors)
            print("Vetores adicionados ao índice FAISS.")
            
            # Exibir estatísticas
            stats = self.get_statistics()
            print(f"\n📊 Estatísticas dos embeddings:")
            print(f"   - Total: {stats['total_embeddings']} embeddings")
            print(f"   - Chunks de texto: {stats['text_chunks']}")
            print(f"   - Linhas de tabela: {stats['table_rows']}")
            for tipo, count in stats['types'].items():
                if tipo not in ['text_chunk', 'table_row']:
                    print(f"   - {tipo}: {count}")
            print()

        except json.JSONDecodeError as e:
            print(f"Erro ao decodificar JSON do arquivo '{self.embeddings_json_path}': {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Erro ao ler ou processar o arquivo de embeddings '{self.embeddings_json_path}': {e}", file=sys.stderr)
            sys.exit(1)

    def search_similar(self, query_vector: np.ndarray, num_results: int) -> list:
        """
        Realiza a busca de similaridade no índice FAISS.
        Retorna uma lista de dicionários com 'id', 'text', 'type', 'distance_l2', 'cosine_similarity'.
        """
        if self.faiss_index is None or self.document_vectors is None:
            print("Erro: Índice FAISS ou vetores de documento não foram inicializados.", file=sys.stderr)
            return []

        # Certifica-se de que o vetor de consulta é 2D
        query_vector_2d = query_vector.reshape(1, -1) 
        
        # Buscar resultados
        search_results = min(num_results * 2, len(self.document_data))  # Buscar 2x mais para ter opções
        distances_l2, indices = self.faiss_index.search(query_vector_2d, search_results)

        results = []
        if len(indices[0]) == 0:
            return [] # Nenhuma similaridade encontrada

        for i in range(len(indices[0])): # Itera sobre os resultados reais, não num_results fixo
            doc_index = indices[0][i]
            distance_l2 = distances_l2[0][i]

            # Obter o vetor original do documento para cálculo da similaridade de cosseno
            document_vector_raw = self.document_data[doc_index]['embedding']
            cosine_similarity = calculate_cosine_similarity(query_vector, np.array(document_vector_raw, dtype='float32'))
            
            # Obter informações adicionais do documento original
            original_item = self.document_data[doc_index]
            
            result = {
                "id": self.document_ids[doc_index],
                "text": self.document_texts[doc_index],
                "type": self.document_types[doc_index],
                "distance_l2": distance_l2,
                "cosine_similarity": cosine_similarity,
                "original_content": original_item.get('content', ''),
            }
            
            # Adicionar campos específicos para linhas de tabela
            if original_item['type'] == 'table_row':
                result.update({
                    "contextual_content": original_item.get('contextual_content', ''),
                    "table_header": original_item.get('table_header_raw', ''),
                    "row_number": original_item.get('row_number', 0)
                })
            elif original_item['type'] == 'text_chunk':
                result.update({
                    "section_title": original_item.get('section_title', ''),
                    "contextual_content": original_item.get('contextual_content', '')
                })
            
            results.append(result)
        
        # Ordenar por similaridade de cosseno e retornar apenas o número solicitado
        results.sort(key=lambda x: x['cosine_similarity'], reverse=True)
        return results[:num_results]

    def get_statistics(self):
        """Retorna estatísticas sobre os embeddings carregados."""
        if not self.document_data:
            return {}
        
        stats = {
            'total_embeddings': len(self.document_data),
            'text_chunks': 0,
            'table_rows': 0,
            'types': {}
        }
        
        for item in self.document_data:
            item_type = item.get('type', 'unknown')
            stats['types'][item_type] = stats['types'].get(item_type, 0) + 1
            
            if item_type == 'text_chunk':
                stats['text_chunks'] += 1
            elif item_type == 'table_row':
                stats['table_rows'] += 1
        
        return stats

# --- Função Principal para Orquestrar ---
def main():
    # 1. Carregar o gerador de embeddings
    embedder = EmbeddingGenerator(EMBEDDING_MODEL)

    # 2. Carregar e indexar os embeddings dos documentos
    db = DocumentEmbeddingsDB(EMBEDDINGS_JSON_PATH)

    # 3. Gerar o embedding da pergunta
    print(f"\nGerando embedding para a pergunta: '{QUERY_TEXT}'...")
    query_vector = embedder.generate_embedding(QUERY_TEXT)
    if query_vector is None:
        print("Não foi possível gerar o embedding da pergunta. Encerrando.", file=sys.stderr)
        sys.exit(1)
    print("Embedding da pergunta gerado com sucesso.")

    # 4. Buscar os chunks mais similares
    similar_chunks = db.search_similar(query_vector, NUM_RESULTS)

    # 5. Imprimir os resultados
    print(f"\n--- Top {NUM_RESULTS} Chunks mais similares para a pergunta ---")
    if not similar_chunks:
        print("Nenhum resultado de similaridade encontrado.")
    else:
        for i, result in enumerate(similar_chunks):
            print(f"\n{'='*60}")
            print(f"Resultado {i+1}:")
            print(f"  Chunk ID original: {result['id']}")
            print(f"  Tipo: {result['type']}")
            print(f"  Distância Euclidiana (L2): {result['distance_l2']:.4f}")
            print(f"  Similaridade de Cosseno: {result['cosine_similarity']:.4f} (1.0 = idêntico, 0.0 = sem relação)")
            
            # Exibir informações específicas para linhas de tabela
            if result['type'] == 'table_row':
                print(f"  [LINHA DE TABELA - Linha {result['row_number']}]")
                print(f"  Conteúdo contextual: {result['contextual_content']}")
                print(f"  Conteúdo original: {result['original_content']}")
            elif result['type'] == 'text_chunk':
                print(f"  [SEÇÃO DE TEXTO]")
                if result.get('section_title'):
                    print(f"  Título da seção: {result['section_title']}")
                print(f"  Conteúdo:")
                print("  " + "-" * 50)
                # Limitar o texto para não sobrecarregar a saída
                text_display = result['text'][:500] + "..." if len(result['text']) > 500 else result['text']
                print(f"  {text_display}")
                print("  " + "-" * 50)
            else:
                print(f"  Conteúdo:")
                print("  " + "-" * 50)
                text_display = result['text'][:500] + "..." if len(result['text']) > 500 else result['text']
                print(f"  {text_display}")
                print("  " + "-" * 50)

    print("\nProcesso concluído.")

# --- Executar a função principal ---
if __name__ == "__main__":
    main()