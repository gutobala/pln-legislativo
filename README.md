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
Recomendado: **Python 3.12** (faixa mais estável para `torch`/`transformers`). O projeto foi
desenvolvido em 3.14, mas use 3.12 para reproduzir sem dor de cabeça com wheels muito novos.

Use um **ambiente isolado** (venv) para não misturar versões com outros projetos:
```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```
GPU é **opcional** (CPU roda tudo, inclusive o classificador). Para treinar em GPU, instale o
`torch` com o build CUDA da placa (ver comentários no `requirements.txt`).

> **Não instale `torchvision`/`torchaudio`** — não são usados e conflitam com o build CUDA do
> torch. Se aparecer o erro `operator torchvision::nms does not exist` ao importar o
> `transformers`, remova com `pip uninstall -y torchvision torchaudio`.

### Teste rápido do classificador
Depois de instalar e ter a pasta `modelo_bertimbau/` no lugar, rode o notebook
**`07_teste_classificador.ipynb`**: ele carrega o modelo, escolhe CPU/GPU automaticamente
(com fallback seguro) e classifica algumas ementas de exemplo — serve para confirmar que o
ambiente está OK antes de subir a app.

## Reprodução em outra máquina (Git + Drive)
Os dois arquivos **pesados** ficam fora do Git (no Google Drive): a pasta `modelo_bertimbau/`
(~435 MB) e o `discursos_todos.csv` (~127 MB). Para reproduzir:

1. Clone o repositório: `git clone <url-do-repo>`
2. Baixe do Drive e coloque na **raiz do projeto** (mesma pasta dos notebooks):
   - pasta **`modelo_bertimbau/`** — link: `<COLE O LINK DO DRIVE AQUI>`
   - arquivo **`discursos_todos.csv`** — link: `<COLE O LINK DO DRIVE AQUI>`
3. Instale as dependências: `python -m pip install -r requirements.txt`
4. Suba o app: `python -m streamlit run app_explorer.py`

> **Layout da pasta do modelo (atenção!).** Os arquivos (`config.json`, `tokenizer.json`,
> `model.safetensors`, …) precisam ficar **direto** dentro de `modelo_bertimbau/`, e **não**
> aninhados em `modelo_bertimbau/modelo_bertimbau/` (erro comum ao descompactar o zip do Drive).
> Confira:
> ```powershell
> Get-ChildItem modelo_bertimbau\    # deve listar config.json, tokenizer.json, model.safetensors
> ```

Observações:
- **Não é preciso re-treinar.** Com `modelo_bertimbau/` e `discursos_classificados.csv` presentes,
  os passos pesados (notebooks **03** e **05**) podem ser **pulados** — o resto apenas reusa.
- Sem o `discursos_todos.csv`, o app ainda abre, mas o **texto integral do discurso** (no modal)
  fica vazio; as demais páginas funcionam com o `discursos_classificados.csv` (versionado).
- RTX 5050 (Blackwell): instale o `torch` com CUDA 12.8 (ver `requirements.txt`); ou rode em CPU.
  Atenção: `torch.cuda.is_available()` pode retornar `True` mas ainda assim **faltar kernel** para
  a placa (erro `no kernel image is available`). Confira se a arquitetura aparece em
  `torch.cuda.get_arch_list()` (Blackwell = `sm_120`). O notebook de teste já trata isso com
  fallback automático para CPU.

## Rodando no Google Colab
O Colab evita a parte chata da GPU (ele fornece uma GPU compatível e já vem com `torch`
instalado), mas tem armadilhas próprias:

1. **Use o `requirements.txt` flexível — NÃO fixe versões de `torch`/`numpy`/`pandas`.** Como o
   `torch` no `requirements.txt` não tem versão, o pip respeita o que o Colab já traz (com a GPU
   funcionando). Forçar versões fixas trocaria o torch do Colab por um build CPU (perde a GPU).
2. **Reinicie o runtime após instalar.** O `transformers>=4.46` sobe para a 5.x e o Colab pede
   *Runtime → Restart session*. É normal — reinicie e siga.
3. **Monte o Drive** e garanta o caminho da pasta do modelo:
   ```python
   from google.colab import drive; drive.mount('/content/drive')
   # copie/link a pasta para /content/modelo_bertimbau (sem aninhar) ou ajuste o caminho no código
   ```
4. **A app Streamlit não abre direto no Colab** (servidor em `localhost`). Para demonstrar no
   Colab, rode o classificador pelo notebook **`07_teste_classificador.ipynb`**; a app
   `app_explorer.py` é para execução **local**.

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
| 9 | `08_analise_sentimentos.ipynb` | Sentimento dos discursos (modelo pronto) × tema/partido | Recomendado |

> O fine-tuning (3) e a classificação dos discursos (7) são os passos pesados — suas saídas
> já ficam salvas, então as fases seguintes **reusam** sem re-treinar.
> O notebook **`07_teste_classificador.ipynb`** é um smoke test do classificador (não faz parte
> do pipeline) e o **`08_analise_sentimentos.ipynb`** é a análise extra de sentimentos.

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
