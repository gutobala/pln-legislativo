# Classificação de Projetos de Lei por Tema com BERTimbau
### Um estudo de *transfer learning* e *domain shift* no setor público

> Projeto Final — *Deep Learning e Processamento de Linguagem Natural* · Modalidade 2 (PLN no Setor Público)
> Integrantes: Aliendres Souto Souza · Gustavo Salvador Ferraz Ferreira · Shenia Rocha Ladeira

Este documento resume o projeto na lógica de apresentação (contexto → teoria → dados →
metodologia → resultados → **segundo experimento** → **bônus: sentimentos** → discussão →
conclusão → próximos passos), ligando cada etapa conceitual à fase técnica correspondente do
código (notebooks `01`–`08`). Para instalação e
execução, ver o [`GUIA_EXECUCAO.md`](GUIA_EXECUCAO.md). O **relatório completo** (artigo) está em
[`docs/Relatorio_Projeto_Final.docx`](docs/Relatorio_Projeto_Final.docx). Para **testar a
classificação** de uma ementa rapidamente, use o notebook
[`07_teste_classificador.ipynb`](07_teste_classificador.ipynb) (carrega o modelo e classifica
exemplos) — ou a página "Classificador de ementas" no app.

---

## 1. Contexto: qual problema real está sendo resolvido?

A Câmara dos Deputados recebe **milhares de Projetos de Lei (PLs) por ano**. Para que cidadãos,
pesquisadores e servidores naveguem nesse volume, cada PL precisa ser organizado **por tema**
(ex.: "Saúde", "Educação", "Meio Ambiente", "Direitos Humanos"). Hoje essa classificação é feita
manualmente por especialistas do **Centro de Documentação e Informação (CEDI)** — trabalho
demorado e dependente de julgamento humano.

**Pergunta do projeto:** é possível ensinar um computador a ler a **ementa** de um PL e dizer,
sozinho, a quais temas ela pertence? Uma particularidade torna o problema mais rico: um mesmo PL
quase sempre trata de **mais de um tema ao mesmo tempo** (em média **2,31 temas por proposição**).
Logo, é uma tarefa de **classificação multirrótulo** — o modelo pode marcar vários temas para a
mesma ementa.

**Diferencial.** Além de classificar PLs, aplicamos o modelo treinado a um texto bem diferente —
os **discursos parlamentares** — para investigar se um modelo treinado em textos curtos (ementas)
continua funcionando em textos longos e coloquiais (estudo de *domain shift*). Com os discursos
rotulados, cruzamos o que cada deputado **fala** com o que de fato **propõe** em lei (a análise
"fala vs. faz").

---

## 2. Base teórica: quais artigos sustentam o projeto?

**A técnica — da contagem de palavras aos Transformers** (5 referências):
- Em abordagens clássicas, os documentos são representados pelo peso estatístico das palavras —
  *bag-of-words* e **TF-IDF** (**Salton & Buckley, 1988**): boas linhas de base, mas que tratam as
  palavras de forma isolada, sem capturar o contexto.
- **Minaee et al. (2021)** revisam mais de 150 modelos de classificação textual com Deep Learning
  (CNNs, RNNs, LSTMs, Transformers), mostrando o ganho de representações contextuais sobre a
  contagem de palavras.
- A arquitetura **Transformer (Vaswani et al., 2017)**, baseada em atenção, modela dependências de
  longo alcance com paralelização.
- Sobre ela, **Devlin et al. (2019)** propuseram o **BERT** — pré-treinado em grande volume de texto
  e ajustável a tarefas específicas (*fine-tuning*); é o paradigma de ***transfer learning*** em PLN
  (também explorado por Howard & Ruder, 2018, com o ULMFiT).
- Para o português, **Souza, Nogueira & Lotufo (2020)** treinaram o **BERTimbau** — o modelo
  profundo adotado aqui.

**O domínio — PLN no legislativo/jurídico brasileiro** (5 referências):
- **Batista (2020)** classificou ~38 mil proposições da Câmara por área temática, mostrando a
  relevância da organização temática para estudos de agenda legislativa.
- **Siqueira et al. (2024)** (corpus *Ulysses Tesemõ*) evidencia a escassez de recursos de PLN para
  o domínio jurídico e governamental brasileiro.
- **Albuquerque et al. (2022)** (*UlyssesNER-Br*) construiu um corpus de reconhecimento de entidades
  em documentos legislativos da Câmara, com baselines.
- **Menezes-Neto & Clementino (2022)** mostram que 86% das decisões de tribunais federais
  ultrapassam o limite de 512 tokens do BERT — o que motiva tratar o **tamanho do texto**, relevante
  para os discursos (longos).
- **Gururangan et al. (2020)** mostram que **adaptar o modelo ao domínio** (continuar o pré-treino em
  texto do próprio domínio) melhora o desempenho — fundamento de que modelos genéricos podem não
  capturar o vocabulário jurídico-legislativo (e base para o "próximos passos").

**Fundamentos metodológicos — avaliação, validação e alternativa** (3 referências):
- **Sechidis, Tsoumakas & Vlahavas (2011)** — **estratificação de dados multirrótulo**; base do
  split iterativo 70/15/15 que preserva a proporção dos temas nas três partições.
- **Zheng et al. (2023)** — **LLM como avaliadora** (*LLM-as-a-judge*): uma LLM forte concorda em
  ~80% dos casos com avaliadores humanos, o que fundamenta o gabarito automático usado no 2º
  experimento (domain shift).
- **Rodrigues et al. (2023)** — **Albertina PT-***, Transformer mais robusto para o português,
  considerado como alternativa ao BERTimbau (ver "próximos passos").

---

## 3. Dados: como foram obtidos?

**Coleta própria, sem base pronta.** Restrição do projeto: os dados devem ser coletados pelo
grupo. Usamos a **API pública de Dados Abertos da Câmara** (`dadosabertos.camara.leg.br`) — sem
chave, sem login. O rótulo (tema do CEDI) vem **junto com a ementa** na própria listagem, sem
anotação manual nossa.

> **Fase técnica — `01_coleta_dados.ipynb`.** Baixa todos os PLs de **2023–2026** (sem teto),
> junta os temas por proposição e salva `dados/proposicoes_temas.csv` + um **manifesto** com data,
> parâmetros e a impressão digital **SHA-256** do arquivo (coleta reprodutível).

**Volume e formato.**
- **19.354 proposições**, cada uma com seus temas oficiais.
- **X (entrada):** a **ementa** (texto curto). **y (rótulo):** o conjunto de temas, dentre os **32
  temas oficiais** da Câmara. Cardinalidade média **2,31 temas/PL** → multirrótulo.

**Decisão: usar só a ementa.** Não usamos *keywords* (são indexação do próprio CEDI → risco de
**vazamento**) nem o PDF integral (caro, dilui o sinal do tema, esbarra no limite de 512 tokens).

**Limitação central — cauda longa.** Dos 32 temas, **28 têm suporte ≥ 200** e **4 são raríssimos**
(Ciências Exatas=5, Ciências Sociais=11, Processo Legislativo=75, Direito Constitucional=84) —
quase impossíveis de aprender (F1 ≈ 0). Tratado de forma transparente na métrica (dois cortes).

---

## 4. Metodologia: qual modelo de PLN foi aplicado?

**Pré-processamento.** Os rótulos viram uma matriz **multi-hot** de 32 colunas
(`MultiLabelBinarizer`): 1 nos temas presentes, 0 nos demais.

**Split 70/15/15 com estratificação iterativa.** Como o problema é multirrótulo e desbalanceado,
usamos `MultilabelStratifiedShuffleSplit` (Sechidis et al., 2011) para preservar a proporção dos
temas nas três partições. Semente **42**, salvo em `dados/particao_treino_val_teste.csv` e reusado por
todos os modelos (comparação justa). Tamanhos: **treino 13.556 · validação 2.907 · teste 2.891**.

**Modelo de referência (baseline) — `02_baseline_tfidf.ipynb`.**
- **TF-IDF** com **unigramas e bigramas** (`ngram_range=(1,2)`) e remoção de termos raros
  (`min_df=5`); sem limite de atributos.
- **Regressão Logística** *one-vs-rest* (`OneVsRestClassifier`): um classificador Sim/Não por tema,
  com `class_weight='balanced'`. Roda em CPU em 1–2 min. É a "régua".

**Modelo profundo — `03_bertimbau.ipynb`.**
- **BERTimbau** (`neuralmind/bert-base-portuguese-cased`), `problem_type="multi_label_classification"`.
- **32 saídas** com **sigmoide** (probabilidades independentes → vários temas ao mesmo tempo) e
  perda **BCEWithLogitsLoss** — o que torna a tarefa de fato multirrótulo (vs. softmax, que força
  um único rótulo).
- **`max_length=192`** (mediana das ementas = 45 tokens, p99 = 135) → cobre o p99 com folga e corta
  só **0,16%** das ementas.
- Hiperparâmetros: `batch=16`, `warmup_ratio=0,1`, `weight_decay=0,01`, até **6 épocas** com
  *early stopping* pela validação. **Sweep de learning rate** (macro-F1 ≥200 na validação):
  **2e-5 → 0,638 · 3e-5 → 0,677 · 5e-5 → 0,698 (escolhido)**. Treino em GPU (RTX 3060).

**Limiar por tema e métrica.**
- Em vez de 0,50 fixo, para cada tema escolhe-se o limiar que **maximiza o F1 na validação**
  (grade 0,05–0,95, passo 0,05; *fallback* 0,50 para temas sem F1 positivo), aplicado no teste.
  Mesma técnica nos dois modelos.
- **Métrica principal: macro-F1**, reportado em **dois cortes** — sobre os 32 temas (número "duro")
  e sobre os temas com suporte **≥ 200** (número "justo", sem o ruído da cauda longa). Também
  **micro-F1**.
- *`pos_weight`* na perda foi **considerado e não adotado** (o ajuste de limiar já trata o
  desbalanceamento; pesos altos desestabilizariam o treino) → trabalho futuro.

---

## 5. Resultados: principais métricas e achados

**Experimento principal — BERTimbau supera o baseline** (mesmo teste, limiar ajustado por tema):

| Modelo | macro-F1 (32) | macro-F1 (≥200) | micro-F1 |
|---|---|---|---|
| Baseline TF-IDF (limiar 0,50) | 0,567 | 0,620 | 0,650 |
| Baseline TF-IDF (limiar ajustado) | 0,584 | 0,639 | 0,675 |
| BERTimbau (limiar 0,50) | 0,601 | 0,687 | 0,724 |
| **BERTimbau (limiar ajustado)** | **0,632** | **0,700** | **0,723** |

No corte mais justo (≥200), o **BERTimbau atinge 0,700 contra 0,639 do baseline (+0,061)**. O ganho
é **modesto** porque a ementa é curta e cheia de palavras-chave — cenário em que o TF-IDF já é forte.
Saídas: `dados/resultados_baseline.csv`, `dados/resultados_bertimbau.csv`, `dados/tabela_comparativa.csv`, e as figuras
de matriz de confusão e suporte por tema (`04_avaliacao.ipynb`).

---

## 6. Segundo experimento: *domain shift* nos discursos

O **diferencial** do trabalho: pegar o modelo treinado nas ementas (texto curto, jurídico) e
aplicá-lo a um domínio **bem diferente** — os **discursos parlamentares** (texto longo, oratório e
coloquial) —, medindo o quanto ele perde (*domain shift*). Em seguida, cruzamos o que cada deputado
**fala** com o que **propõe** ("fala vs. faz").

**Domain shift** (`05_discursos_dominio.ipynb`). Coletamos os autores das PLs (vínculo **88,2%**) e
**46.683 discursos de 488 deputados** (`01_coleta_dados.ipynb`). Aplicamos o BERTimbau aos discursos
via **chunking** (janelas de 192 tokens, agregadas por **máximo**). Gabarito de **120 discursos**
rotulado por **LLM** (Zheng et al., 2023).
- **macro-F1 (≥200) cai de 0,700 (ementas) para 0,569 (discursos).** Queda real e mensurável.
- Mecanismo: **sobre-rotulação** — o modelo marca **5,92 temas/discurso** vs. **3,43** do gabarito
  (revocação alta, precisão baixa), por causa da agregação por máximo + limiares calibrados em
  textos curtos.

**"Fala vs. faz"** (`06_cruzamento_discurso_proposicao.ipynb`). Para cada deputado, dois perfis
temáticos normalizados (somam 100%): **agenda discursiva** (fala) × **agenda legislativa** (propõe),
restrito aos ~20 temas que transferiram bem (F1 ≥ 0,50). Exemplos: fala muito mais do que propõe
(Pastor Eurico em "Arte, Cultura e Religião"); propõe muito mais do que fala (Altineu Côrtes em
"Direito Penal"). Boa **validação de face**.

---

## 7. BÔNUS — Análise de Sentimentos nos discursos

Uma extensão analítica: além de **sobre o quê** os deputados falam (tema), medimos **em que tom**
(sentimento) — e cruzamos com **tema** e **partido**. Usamos um **modelo de sentimento pronto,
multilíngue** (`lxyuan/distilbert-base-multilingual-cased-sentiments-student`), com score contínuo
**`sent_score = P(positivo) − P(negativo)` ∈ [−1, 1]**. Notebook `08_analise_sentimentos.ipynb`.

**Escala e tom geral.** Os **46.683 discursos** se distribuem em **30.373 negativos · 16.272
positivos · 38 neutros**, com **tom médio geral −0,164** — o discurso parlamentar tende ao
**crítico** (faz sentido: tribuna usada para denúncia e cobrança).

**Por tema** (apenas os temas confiáveis do domain shift, F1 ≥ 0,50):
- Mais **negativos**: **Direito Penal e Processual Penal (−0,51)** e **Defesa e Segurança (−0,39)** —
  tom de denúncia/cobrança.
- Mais **positivos**: **Turismo (+0,29)** e **Arte, Cultura e Religião (+0,07)** — tom
  celebratório/homenagem.

**Por partido.** A oposição aparece com tom mais crítico (**PL −0,35**, **PSOL −0,26**) e os mais
positivos são **PSDB (+0,13)** e **MDB (+0,11)** — boa **validação de face** (governo × oposição).

**Entregáveis.** `dados/discursos_sentimento.csv`, `figuras/sentimento_*.png` (por tema, por
partido e heatmap tema × partido) e a página interativa **"Análise de Sentimentos"** no app
(`app_explorer.py`): filtros por partido/tema/parlamentar, agregações ao vivo e modal com o
sentimento de cada discurso.

**Ressalvas.** Análise **exploratória/descritiva, não causal**. O modelo de sentimento é **genérico**
e foi aplicado a **texto formal/retórico** (*domain mismatch*) — o `sent_score` capta um tom
aproximado (**negativo ≈ crítico/denúncia**, **positivo ≈ celebratório/homenagem**), não opinião
fina; por isso só interpretamos a fundo nos **temas confiáveis** e **no agregado**.

---

## 8. Discussão: o modelo resolveu o problema? Quais os limites?

**Utilidade.** Um macro-F1 de **0,70** nas ementas não substitui o especialista, mas serve de
**ferramenta de apoio**: sugere temas para conferência humana, organiza buscas e produz estatísticas
agregadas confiáveis nos temas bem representados.

**Limitações e vieses.**
1. **Cauda longa:** 4 temas raros com F1 ≈ 0 (daí os dois cortes de métrica).
2. **Teto de rótulo:** os rótulos do CEDI são subjetivos (anotadores podem discordar) → teto natural.
3. **Domain shift / sobre-rotulação:** rótulos de discurso individuais são ruidosos → toda a análise
   "fala vs. faz" é **agregada** e só nos temas confiáveis.
4. **Cobertura:** vínculo PL→autor cobre 88% (só deputados atuais).
5. **Gabarito por LLM:** validado por amostragem, mas herda vieses da LLM.
6. **Natureza:** o cruzamento é **exploratório/descritivo**, não causal.

**Risco.** Classificação automática sem supervisão poderia rotular leis erradas em decisões
sensíveis → manter **humano no circuito**. **LGPD:** todos os dados são públicos.

---

## 9. Conclusão: principal contribuição

Construímos um **pipeline completo e reprodutível**: coleta própria via API → baseline honesto →
modelo profundo (BERTimbau) que o supera → avaliação crítica. O número não é um fim: significa uma
**ferramenta de apoio** à organização temática do trabalho legislativo. A contribuição central está
no **2º experimento** — transferir o modelo para os discursos (*domain shift*) e cruzar "fala vs.
faz", conectando o modelo ao mundo real e exercitando, na prática, a diferença entre **transfer
learning** (a técnica) e **domain shift** (o problema de mudar de domínio).

---

## 10. Próximos passos: como aprimorar ou aplicar em escala?

- **Reduzir a sobre-rotulação:** trocar a agregação por **máximo** pela **média**, ou **recalibrar
  os limiares** nos próprios discursos.
- **Adaptação de domínio** (*domain-adaptive pretraining*, Gururangan et al., 2020): continuar o
  pré-treino do BERTimbau nos discursos, de forma não supervisionada, para reduzir o *domain shift*.
- **Modelo mais robusto:** testar o **Albertina PT-BR** (Rodrigues et al., 2023).
- **Tratar a cauda longa:** `pos_weight` na BCE (com *cap* nos temas ultra-raros) ou coleta de mais
  exemplos dos temas raros.
- **Escala/aplicação:** integrar como sugestão de temas no fluxo do CEDI (humano no circuito) e
  ampliar a análise política (séries temporais, comissões, ex-deputados).

---

### Referências (resumo)
Salton & Buckley (1988) · Batista (2020) · Vaswani et al. (2017) · Devlin et al. (2019) · Howard & Ruder (2018) ·
Souza, Nogueira & Lotufo (2020, BERTimbau) · Minaee et al. (2021) · Sechidis et al. (2011) ·
Siqueira et al. (2024, Ulysses Tesemõ) · Albuquerque et al. (2022, UlyssesNER-Br) ·
Menezes-Neto & Clementino (2022) · Zheng et al. (2023) · Gururangan et al. (2020) ·
Rodrigues et al. (2023, Albertina). Lista completa em [`docs/Relatorio_Projeto_Final.docx`](docs/Relatorio_Projeto_Final.docx).
