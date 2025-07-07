# RAG_QA_for_PDFs

# FUNCIONALIDADES.
  - A aplicação foi desenvolvida para responder perguntas sobre concursos baseada no pdf escolhido

# COMO USAR:
  - Instale as dependências do arquivo requirements.txt
    
  - Instale o Ollama (https://ollama.com/).
    - Instale os modelos utilizados pelo PowerShell (altere para o modelo de sua preferência):
      - nomic-embed-text:latest (embedding generator)
      - phi3.5:latest (LLM utilizada)
        
  - Coloque o pdf de sua preferência na pasta pdfPath
    
  - Execute o arquivo doclingpdf.py
    - O script vai extrair todo o texto do PDF e armazenar na pasta mdPath (pode demorar algum tempo dependendo do tamanho do arquivo
      
  - Execute o arquivo generate2.py
    - O script vai gerar os embeddings do texto presente no mdPath e armazenar na pasta embeddings
      
  - Execute o arquivo ollama_ask2.py e teste fazendo perguntas sobre o pdf
    - O script gera o embedding da pergunta, e busca os trechos com maior similaridade presentes no arquivo embedding
    - Após a busca, o script alimenta uma inteligência artificial para gerar uma pergunta baseada nos contextos recebidos
   
# ARQUIVOS ADICIONAIS
 - O arquivo question_embedding.py foi utilizado para testas as buscas pelos embeddings
   - Dentro do arquivo há uma variavel "QUERY_TEXT" onde deve ser colocado a pergunta do usuário para a verificação da busca pelo embeddings
         
