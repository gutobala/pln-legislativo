# Classificação de Proposições de Lei por Tema (PLN no Setor Público)

Classificação automática **multirrótulo** de Projetos de Lei da Câmara dos Deputados por
**tema** (32 temas), a partir da **ementa**, com **BERTimbau**. Inclui um 2º experimento de
**domain shift** (aplicar o modelo a discursos parlamentares) e uma análise **"fala vs. faz"**.

> Projeto Final — *Deep Learning e Processamento de Linguagem Natural* · Modalidade 2 (PLN no Setor Público).

## Resultado principal
- Baseline TF-IDF + Regressão Logística: **macro-F1 (≥200) = 0,639**.
- **BERTimbau (fine-tuning): macro-F1 (≥200) = 0,700** (+0,061 sobre o baseline).
- Domain shift nos discursos: macro-F1 cai para **0,569** (com sobre-rotulação diagnosticada).

## Dados
Coletados pelo próprio grupo via **API de Dados Abertos da Câmara** (`dadosabertos.camara.leg.br`),
Projetos de Lei de **2023–2026** (19.354 proposições). Sem bases prontas. A coleta gera um
**manifesto** com `sha256`; o split usa semente fixa (42) → reprodutível.

## Como instalar
```bash
pip install -r requirements.txt
```
GPU é **opcional** (CPU roda tudo, inclusive o classificador). Para treinar em GPU, instale o
`torch` com o build CUDA da placa (ver comentários no `requirements.txt`).

## Reprodução em outra máquina (Git + Drive)
Os dois arquivos **pesados** ficam fora do Git (no Google Drive): a pasta `modelo_bertimbau/`
(~435 MB) e o `discursos_todos.csv` (~127 MB). Para reproduzir:

1. Clone o repositório: `git clone <url-do-repo>`
2. Baixe do Drive e coloque na **raiz do projeto** (mesma pasta dos notebooks):
   - pasta **`modelo_bertimbau/`** — link: `<COLE O LINK DO DRIVE AQUI>`
   - arquivo **`discursos_todos.csv`** — link: `<COLE O LINK DO DRIVE AQUI>`
3. Instale as dependências: `python -m pip install -r requirements.txt`
4. Suba o app: `python -m streamlit run app_explorer.py`

Observações:
- **Não é preciso re-treinar.** Com `modelo_bertimbau/` e `discursos_classificados.csv` presentes,
  os passos pesados (notebooks **03** e **05**) podem ser **pulados** — o resto apenas reusa.
- Sem o `discursos_todos.csv`, o app ainda abre, mas o **texto integral do discurso** (no modal)
  fica vazio; as demais páginas funcionam com o `discursos_classificados.csv` (versionado).
- RTX 5050 (Blackwell): instale o `torch` com CUDA 12.8 (ver `requirements.txt`); ou rode em CPU.

## Ordem de execução (notebooks)
| # | Notebook | O que faz | Precisa de GPU? |
|---|----------|-----------|-----------------|
| 1 | `01_coleta_dados.ipynb` | Coleta PLs + temas → `proposicoes_temas.csv` + manifesto | Não |
| 2 | `02_baseline_tfidf.ipynb` | Baseline TF-IDF + LogReg; cria o split estratificado | Não |
| 3 | `03_bertimbau.ipynb` | Fine-tuning do BERTimbau → `modelo_bertimbau/` | **Sim** |
| 4 | `04_avaliacao.ipynb` | Tabela comparativa, matriz de confusão, exemplos | Não |
| 5 | `02_coleta_proposicoes_parlamentares.ipynb` | Liga PL → deputado autor | Não |
| 6 | `02_coleta_discurso_parlamentares.ipynb` | Coleta discursos → `discursos_todos.csv` | Não |
| 7 | `05_discursos_dominio.ipynb` | Classifica discursos (chunking) + mede domain shift | Recomendado |
| 8 | `06_cruzamento_discurso_proposicao.ipynb` | Análise "fala vs. faz" + por partido | Não |

> O fine-tuning (3) e a classificação dos discursos (7) são os passos pesados — suas saídas
> já ficam salvas, então as fases seguintes **reusam** sem re-treinar.

## Aplicação web (explorador + classificador)
```bash
python -m streamlit run app_explorer.py
```
Abre em `http://localhost:8501`. Menu:
- **Classificador de ementas** — digite uma ementa, o BERTimbau prevê os temas.
- **Parlamentares** — lista filtrável por partido; clique abre a página do deputado
  (proposições com temas do CEDI + discursos com temas previstos).
- **Projetos de Lei** — filtra por parlamentar/ano/partido/tema; selecione a linha para ver
  ementa + temas + autores (com links).

## Relatório
`Relatorio_Projeto_Final.docx` — relatório completo (com a arquitetura, resultados e figuras).

## Estrutura (principais arquivos)
- Notebooks `01`–`06` (acima) · `app_explorer.py` (app) · `requirements.txt`
- Dados: `proposicoes_temas.csv`, `particao_treino_val_teste.csv`, `proposicoes_parlamentares.csv`,
  `discursos_todos.csv`, `discursos_classificados.csv`
- Resultados: `resultados_baseline.csv`, `resultados_bertimbau.csv`, `resultados_dominio.csv`,
  `tabela_comparativa.csv`, `fala_vs_faz.csv`, `figuras/*.png`
- Modelo treinado: `modelo_bertimbau/` (≈435 MB — fora do Git; compartilhar à parte)

## Notas de reprodutibilidade
- Split salvo em `particao_treino_val_teste.csv` (semente 42), reutilizado por todos os modelos.
- Limiar por tema ajustado **na validação** e aplicado aos dois modelos (comparação justa).
- O gabarito dos discursos foi gerado por LLM (`gabarito_llm.json`) — avaliação do domain shift.
