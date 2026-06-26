# -*- coding: utf-8 -*-
"""
Ambiente unico (Streamlit): CLASSIFICADOR DE EMENTAS + sistema de dados.

Menu (barra lateral):
  - Classificador de ementas : digite/cole uma ementa e o BERTimbau preve os temas.
  - Parlamentares            : lista filtravel por partido; clique abre a pagina do deputado
                               (proposicoes com temas do CEDI + discursos com temas previstos).
  - Projetos de Lei          : lista filtravel por parlamentar, ano e partido; clique no PL
                               abre a pagina com os temas e os autores (com links).
Tudo navegavel por links (?dep=ID e ?pl=ID).

Como rodar:
    pip install -U streamlit pandas torch transformers scikit-learn
    streamlit run app_explorer.py

Pre-requisitos na pasta: parlamentares.csv, proposicoes_temas.csv,
proposicoes_parlamentares.csv, discursos_classificados.csv, discursos_todos.csv
e (para o classificador) modelo_bertimbau/ + resultados_bertimbau.csv.
"""
import os
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Proposições e Discursos por Tema", layout="wide")

# ============================ DADOS (cache) ============================
@st.cache_data
def carregar():
    parl = pd.read_csv("parlamentares.csv", sep=";", dtype=str).rename(columns={"id": "id_deputado"})
    temas = pd.read_csv("proposicoes_temas.csv", dtype=str)[["id", "ementa", "ano", "temas"]]
    pp = pd.read_csv("proposicoes_parlamentares.csv", sep=";", dtype=str)
    props = (pp.merge(temas, left_on="proposicao_id", right_on="id", how="inner")
               .merge(parl[["id_deputado", "siglaPartido", "siglaUf"]],
                      left_on="id_autor", right_on="id_deputado", how="left"))
    dc = pd.read_csv("discursos_classificados.csv", sep=";", dtype=str)
    try:
        dt = pd.read_csv("discursos_todos.csv", sep=";", dtype=str)
        if len(dt) == len(dc):
            dc["transcricao"] = dt["transcricao"].values
    except FileNotFoundError:
        pass
    if "transcricao" not in dc.columns:
        dc["transcricao"] = ""
    dc["transcricao"] = dc["transcricao"].fillna("")
    return parl, temas, pp, props, dc

parl, temas, pp, props, dc = carregar()
nomes_temas = sorted({x for t in temas["temas"].dropna() for x in t.split("|")})
n_prop = props.groupby("id_autor")["proposicao_id"].nunique()
n_disc = dc.groupby("id_deputado").size()

# ============================ MODELO (cache, lazy) ============================
@st.cache_resource
def carregar_modelo():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("modelo_bertimbau")
    mod = AutoModelForSequenceClassification.from_pretrained("modelo_bertimbau")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mod.to(dev).eval()
    resb = pd.read_csv("resultados_bertimbau.csv")
    lim_map = dict(zip(resb["tema"], resb["limiar"]))
    limiares = np.array([float(lim_map.get(t, 0.5)) for t in nomes_temas])
    return tok, mod, dev, limiares

# ============================ HELPERS ============================
def perfil(serie):
    c = {}
    for t in serie.dropna():
        for x in str(t).split("|"):
            if x:
                c[x] = c.get(x, 0) + 1
    return pd.Series(c).sort_values(ascending=False)

def ir(menu):
    st.query_params.clear(); st.query_params["view"] = menu; st.rerun()

@st.dialog("Discurso na íntegra", width="large")
def mostrar_discurso(row):
    st.caption(f"{row.get('nome_deputado','')}  ·  {row.get('data','')}")
    texto = str(row.get("transcricao", "")).strip()
    st.write(texto if texto else "(sem texto)")
    st.divider()
    temas = str(row.get("temas_previstos", "")).strip()
    chips = " · ".join(f"`{t}`" for t in temas.split("|") if t) if temas and temas != "nan" else "(nenhum)"
    st.markdown("**Temas do discurso (previstos pelo BERTimbau):** " + chips)

@st.dialog("Projeto de Lei", width="large")
def mostrar_projeto(pl):
    row = temas[temas["id"] == pl]
    if row.empty:
        st.error("Projeto de Lei não encontrado."); return
    r = row.iloc[0]
    st.caption(f"PL id {pl}  ·  ano {r['ano']}")
    st.markdown("**Ementa**"); st.write(r["ementa"])
    chips = " · ".join(f"`{t}`" for t in str(r["temas"]).split("|") if t)
    st.markdown("**Temas (CEDI):** " + (chips or "(nenhum)"))
    st.divider(); st.markdown("**Autores**")
    aut = (pp[pp["proposicao_id"] == pl]
           .merge(parl[["id_deputado", "nome", "siglaPartido", "siglaUf"]],
                  left_on="id_autor", right_on="id_deputado", how="left"))
    if aut.empty:
        st.info("Sem autor mapeado entre os deputados atuais.")
    else:
        linhas = [f"- [{a['nome']}](?dep={a['id_autor']})  ({a['siglaPartido']}/{a['siglaUf']})"
                  for _, a in aut.iterrows() if pd.notna(a["nome"])]
        st.markdown("\n".join(linhas) if linhas else "(autores não mapeados)")

# ============================ PAGINAS ============================
def page_classificador():
    st.title("🔎 Classificador de ementas")
    st.caption("Modelo BERTimbau ajustado em ementas da Câmara — classificação multirrótulo (32 temas).")
    txt = st.text_area("Ementa da proposição", height=140,
                       placeholder="Cole ou digite a ementa de um Projeto de Lei...")
    if st.button("Classificar", type="primary"):
        if not txt.strip():
            st.warning("Digite uma ementa."); return
        try:
            import torch
            tok, mod, dev, limiares = carregar_modelo()
            enc = tok(txt.strip(), truncation=True, max_length=192, return_tensors="pt").to(dev)
            with torch.no_grad():
                probs = torch.sigmoid(mod(**enc).logits)[0].cpu().numpy()
        except Exception as e:
            st.error("O classificador precisa de modelo_bertimbau/ + torch/transformers instalados.\n\n"
                     f"Detalhe técnico: {e}"); return
        decididos = [nomes_temas[j] for j in np.argsort(-probs) if probs[j] >= limiares[j]]
        st.subheader("Temas atribuídos (decisão com limiar por tema)")
        st.success("   •   ".join(decididos) if decididos else "(nenhum tema passou do limiar)")
        st.subheader("Confiança por tema (probabilidade da sigmoide)")
        barras = pd.Series({nomes_temas[j]: probs[j] for j in np.argsort(-probs)[:10]})
        st.bar_chart(barras)

def page_parlamentares():
    st.title("👤 Parlamentares")
    partidos = sorted(parl["siglaPartido"].dropna().unique())
    f_part = st.sidebar.multiselect("Filtrar por partido", partidos)
    f_nome = st.sidebar.text_input("Buscar por nome")
    df = parl.copy()
    if f_part: df = df[df["siglaPartido"].isin(f_part)]
    if f_nome: df = df[df["nome"].str.contains(f_nome, case=False, na=False)]
    df["proposições"] = df["id_deputado"].map(n_prop).fillna(0).astype(int)
    df["discursos"] = df["id_deputado"].map(n_disc).fillna(0).astype(int)
    df["abrir"] = "?dep=" + df["id_deputado"]
    st.markdown(f"**{len(df)}** parlamentares no filtro. Clique em **abrir ▶** para ver a página.")
    st.dataframe(
        df[["nome", "siglaPartido", "siglaUf", "proposições", "discursos", "abrir"]]
          .rename(columns={"nome": "Parlamentar", "siglaPartido": "Partido", "siglaUf": "UF"}),
        use_container_width=True, hide_index=True, height=560,
        column_config={"abrir": st.column_config.LinkColumn("Página", display_text="abrir ▶")})

def page_parlamentar_detail(idd):
    linha = parl[parl["id_deputado"] == idd]
    if linha.empty:
        st.error("Parlamentar não encontrado."); return
    dep = linha.iloc[0]
    if st.button("← Voltar para a lista de parlamentares"):
        ir("parlamentares")
    c1, c2 = st.columns([1, 5])
    with c1:
        if isinstance(dep.get("urlFoto"), str) and dep["urlFoto"].startswith("http"):
            st.image(dep["urlFoto"], width=110)
    with c2:
        st.header(dep["nome"])
        st.markdown(f"**Partido:** {dep['siglaPartido']}  ·  **UF:** {dep['siglaUf']}")
    p_dep = props[props["id_autor"] == idd]
    d_dep = dc[dc["id_deputado"] == idd]
    st.markdown(f"**{p_dep['proposicao_id'].nunique()}** proposições  ·  **{len(d_dep)}** discursos com texto")

    fp, fd = perfil(p_dep["temas"]), perfil(d_dep["temas_previstos"])
    if len(fp) or len(fd):
        comp = pd.DataFrame({
            "propõe (PLs)": fp / fp.sum() if fp.sum() else fp,
            "fala (discursos)": fd / fd.sum() if fd.sum() else fd}).fillna(0.0)
        comp = comp.loc[comp.sum(axis=1).sort_values(ascending=False).index].head(8)
        if not comp.empty and float(comp.to_numpy().sum()) > 0:
            st.subheader("Comparativo de temas: o que fala × o que propõe")
            st.bar_chart(comp)

    t1, t2 = st.tabs([f"Proposições ({p_dep['proposicao_id'].nunique()})", f"Discursos ({len(d_dep)})"])
    with t1:
        pv = p_dep[["proposicao_id", "ano", "ementa", "temas"]].drop_duplicates("proposicao_id").copy()
        pv["abrir"] = "?pl=" + pv["proposicao_id"]
        st.dataframe(pv.rename(columns={"ano": "Ano", "ementa": "Ementa", "temas": "Temas (CEDI)"}),
                     use_container_width=True, hide_index=True,
                     column_config={"proposicao_id": st.column_config.TextColumn("PL (id)"),
                                    "abrir": st.column_config.LinkColumn("Abrir", display_text="ver ▶")})
    with t2:
        if len(d_dep):
            st.caption("Clique em uma linha para abrir o discurso na íntegra (modal).")
            view = d_dep.reset_index(drop=True).copy()
            view["Trecho"] = view["transcricao"].str.slice(0, 240) + "..."
            ev = st.dataframe(
                view[["data", "Trecho", "temas_previstos"]].rename(
                    columns={"data": "Data", "temas_previstos": "Temas (BERTimbau)"}),
                use_container_width=True, hide_index=True,
                on_select="rerun", selection_mode="single-row", key=f"disc_{idd}")
            linhas = ev.selection.rows
            if linhas:
                i = linhas[0]; marca = f"{idd}:{i}"
                if st.session_state.get("disc_marca") != marca:
                    st.session_state["disc_marca"] = marca
                    mostrar_discurso(view.iloc[i])
        else:
            st.info("Sem discursos com texto para este parlamentar.")

def page_projetos():
    st.title("📄 Projetos de Lei")
    partidos = sorted(props["siglaPartido"].dropna().unique())
    anos = sorted(props["ano"].dropna().unique())
    f_part = st.sidebar.multiselect("Partido", partidos)
    f_ano = st.sidebar.multiselect("Ano", anos)
    f_nome = st.sidebar.text_input("Parlamentar (nome)")
    f_tema = st.sidebar.selectbox("Tema", ["(todos)"] + nomes_temas)
    df = props
    if f_part: df = df[df["siglaPartido"].isin(f_part)]
    if f_ano:  df = df[df["ano"].isin(f_ano)]
    if f_nome: df = df[df["nome_autor"].str.contains(f_nome, case=False, na=False)]
    if f_tema != "(todos)": df = df[df["temas"].str.contains(f_tema, na=False, regex=False)]
    df = df.copy().reset_index(drop=True)
    # nome do deputado vira link p/ a pagina dele (o nome vai no fragmento #, lido pelo display_text)
    df["lnk_dep"] = "?dep=" + df["id_autor"].astype(str) + "#" + df["nome_autor"].fillna("")
    st.markdown(f"**{len(df)}** proposições (pares proposição–autor) no filtro. "
                "Clique no **nome** para abrir o parlamentar, ou **selecione a linha** (caixa à "
                "esquerda) para abrir o projeto (ementa + temas + autores).")
    ev = st.dataframe(
        df[["proposicao_id", "ano", "lnk_dep", "siglaPartido", "ementa", "temas"]]
          .rename(columns={"proposicao_id": "PL (id)", "ano": "Ano",
                           "siglaPartido": "Partido", "ementa": "Ementa", "temas": "Temas (CEDI)"}),
        use_container_width=True, hide_index=True, height=520,
        on_select="rerun", selection_mode="single-row", key="proj_table",
        column_config={"lnk_dep": st.column_config.LinkColumn("Parlamentar", display_text=r"#(.*)$")})
    sel = ev.selection.rows
    if sel:
        i = sel[0]; pl = df.iloc[i]["proposicao_id"]; marca = f"pl:{pl}:{i}"
        if st.session_state.get("proj_marca") != marca:
            st.session_state["proj_marca"] = marca
            mostrar_projeto(pl)

def page_projeto_detail(pl):
    row = temas[temas["id"] == pl]
    if row.empty:
        st.error("Projeto de Lei não encontrado."); return
    r = row.iloc[0]
    if st.button("← Voltar para Projetos de Lei"):
        ir("projetos")
    st.header(f"Projeto de Lei — id {pl}  ·  {r['ano']}")
    st.markdown("**Ementa**"); st.write(r["ementa"])
    st.markdown("**Temas (CEDI):** " + " · ".join(f"`{t}`" for t in str(r["temas"]).split("|")))
    st.subheader("Autores")
    aut = (pp[pp["proposicao_id"] == pl]
           .merge(parl[["id_deputado", "nome", "siglaPartido", "siglaUf"]],
                  left_on="id_autor", right_on="id_deputado", how="left"))
    if aut.empty:
        st.info("Sem autor mapeado entre os deputados atuais.")
    else:
        aut = aut.copy(); aut["abrir"] = "?dep=" + aut["id_autor"]
        st.dataframe(aut[["nome", "siglaPartido", "siglaUf", "abrir"]].rename(
            columns={"nome": "Parlamentar", "siglaPartido": "Partido", "siglaUf": "UF"}),
            use_container_width=True, hide_index=True,
            column_config={"abrir": st.column_config.LinkColumn("Página", display_text="abrir ▶")})

def page_temas():
    st.title("🏷️ Temas")
    st.caption("Resumo por tema: nº de **proposições** (rótulo do CEDI) e nº de **discursos** "
               "(previstos pelo BERTimbau).")
    prop_cnt = perfil(temas["temas"])              # proposicoes por tema (unicas)
    disc_cnt = perfil(dc["temas_previstos"])       # discursos por tema
    tab = pd.DataFrame({"Proposições": prop_cnt, "Discursos": disc_cnt}).fillna(0).astype(int)
    tab.index.name = "Tema"
    tab = tab.sort_values("Proposições", ascending=False).reset_index()
    st.dataframe(tab, use_container_width=True, hide_index=True, height=560)
    st.caption(f"{len(tab)} temas · {int(prop_cnt.sum())} vínculos proposição–tema · "
               f"{int(disc_cnt.sum())} vínculos discurso–tema")

def page_dashboard():
    st.title("🏠 Classificação de Proposições por Tema — Painel")
    st.caption("PLN no Setor Público · BERTimbau · dados da Câmara dos Deputados (2023–2026).")
    c = st.columns(4)
    c[0].metric("Proposições", f"{temas['id'].nunique():,}".replace(",", "."))
    c[1].metric("Parlamentares", f"{parl['id_deputado'].nunique():,}".replace(",", "."))
    c[2].metric("Discursos", f"{len(dc):,}".replace(",", "."))
    c[3].metric("Temas", len(nomes_temas))
    st.divider()
    try:
        comp = pd.read_csv("tabela_comparativa.csv")
        b = comp.loc[comp["modelo"].str.contains("baseline.*ajust", case=False, regex=True), "macro_f1_sup200"]
        m = comp.loc[comp["modelo"].str.contains("BERTimbau.*ajust", case=False, regex=True), "macro_f1_sup200"]
        if len(b) and len(m):
            cc = st.columns(2)
            cc[0].metric("macro-F1 (≥200) — Baseline", f"{float(b.iloc[0]):.3f}")
            cc[1].metric("macro-F1 (≥200) — BERTimbau", f"{float(m.iloc[0]):.3f}",
                         delta=f"+{float(m.iloc[0]) - float(b.iloc[0]):.3f}")
        st.caption("Resultado principal: o BERTimbau supera o baseline TF-IDF no mesmo conjunto de teste.")
    except FileNotFoundError:
        st.info("Rode 04_avaliacao.ipynb para ver o comparativo de desempenho.")
    st.subheader("Temas mais frequentes nas proposições")
    st.bar_chart(perfil(temas["temas"]).head(12))

def page_fala_faz():
    st.title("⚖️ Fala vs. Faz")
    st.caption("Por deputado: quanto ele FALA de um tema (discursos) × quanto PROPÕE (PLs). "
               "Perfil normalizado, apenas temas confiáveis (do estudo de domain shift).")
    try:
        ff = pd.read_csv("fala_vs_faz.csv")
    except FileNotFoundError:
        st.info("Rode 06_cruzamento_discurso_proposicao.ipynb para gerar fala_vs_faz.csv."); return
    tema = st.selectbox("Tema", sorted(ff["tema"].unique()))
    d = ff[ff["tema"] == tema].copy()
    st.scatter_chart(d, x="share_prop", y="share_disc", height=420)
    st.caption("Acima da diagonal = fala mais do que propõe; abaixo = propõe mais do que fala.")
    cols = ["nome", "siglaPartido", "siglaUf", "share_disc", "share_prop", "gap"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Fala muito mais do que propõe**")
        st.dataframe(d.sort_values("gap", ascending=False).head(10)[cols],
                     hide_index=True, use_container_width=True)
    with c2:
        st.markdown("**Propõe muito mais do que fala**")
        st.dataframe(d.sort_values("gap").head(10)[cols],
                     hide_index=True, use_container_width=True)

def page_partidos():
    st.title("🏛️ Partidos")
    p = st.selectbox("Partido", sorted(parl["siglaPartido"].dropna().unique()))
    ids = set(parl.loc[parl["siglaPartido"] == p, "id_deputado"])
    props_p = props[props["siglaPartido"] == p]
    disc_p = dc[dc["id_deputado"].isin(ids)]
    c = st.columns(3)
    c[0].metric("Deputados", len(ids))
    c[1].metric("Proposições", props_p["proposicao_id"].nunique())
    c[2].metric("Discursos", len(disc_p))
    fp, fd = perfil(props_p["temas"]), perfil(disc_p["temas_previstos"])
    if len(fp) or len(fd):
        comp = pd.DataFrame({"propõe (PLs)": fp / fp.sum() if fp.sum() else fp,
                             "fala (discursos)": fd / fd.sum() if fd.sum() else fd}).fillna(0.0)
        comp = comp.loc[comp.sum(axis=1).sort_values(ascending=False).index].head(10)
        if not comp.empty and float(comp.to_numpy().sum()) > 0:
            st.subheader(f"Perfil temático do {p}: fala × propõe")
            st.bar_chart(comp)
    st.subheader("Deputados do partido")
    dd = parl[parl["siglaPartido"] == p].copy()
    dd["lnk"] = "?dep=" + dd["id_deputado"].astype(str) + "#" + dd["nome"].fillna("")
    st.dataframe(dd[["lnk", "siglaUf"]].rename(columns={"siglaUf": "UF"}),
                 hide_index=True, use_container_width=True,
                 column_config={"lnk": st.column_config.LinkColumn("Parlamentar", display_text=r"#(.*)$")})

def page_desempenho():
    st.title("📊 Desempenho do modelo")
    try:
        st.subheader("Baseline × BERTimbau (mesmo conjunto de teste)")
        st.dataframe(pd.read_csv("tabela_comparativa.csv"), hide_index=True, use_container_width=True)
    except FileNotFoundError:
        st.info("Rode 04_avaliacao.ipynb para gerar tabela_comparativa.csv.")
    try:
        rb = pd.read_csv("resultados_bertimbau.csv")[["tema", "f1", "suporte_teste"]].rename(
            columns={"f1": "F1 BERTimbau (ementa)", "suporte_teste": "Suporte"})
        ra = pd.read_csv("resultados_baseline.csv")[["tema", "f1"]].rename(columns={"f1": "F1 baseline"})
        merged = rb.merge(ra, on="tema", how="left")
        try:
            rd = pd.read_csv("resultados_dominio.csv")[["tema", "f1_discurso"]].rename(
                columns={"f1_discurso": "F1 discurso (domain shift)"})
            merged = merged.merge(rd, on="tema", how="left")
        except FileNotFoundError:
            pass
        st.subheader("F1 por tema")
        st.dataframe(merged.sort_values("F1 BERTimbau (ementa)", ascending=False),
                     hide_index=True, use_container_width=True, height=470)
    except FileNotFoundError:
        pass
    st.subheader("Matriz de confusão e cauda longa")
    for img, cap in [("figuras/matriz_confusao_bertimbau.png", "Matriz de confusão (BERTimbau)"),
                     ("figuras/suporte_por_tema.png", "Suporte por tema (cauda longa)")]:
        if os.path.exists(img):
            st.image(img, caption=cap, use_container_width=True)

# ============================ MENU + ROTEAMENTO ============================
st.sidebar.title("Menu")
if st.sidebar.button("🏠 Início", use_container_width=True): ir("dashboard")
if st.sidebar.button("🔎 Classificador de ementas", use_container_width=True): ir("classificador")
if st.sidebar.button("🏷️ Temas", use_container_width=True): ir("temas")
if st.sidebar.button("⚖️ Fala vs. Faz", use_container_width=True): ir("falafaz")
if st.sidebar.button("👤 Parlamentares", use_container_width=True): ir("parlamentares")
if st.sidebar.button("🏛️ Partidos", use_container_width=True): ir("partidos")
if st.sidebar.button("📄 Projetos de Lei", use_container_width=True): ir("projetos")
if st.sidebar.button("📊 Desempenho do modelo", use_container_width=True): ir("desempenho")
st.sidebar.divider()

qp = st.query_params
if "pl" in qp:
    page_projeto_detail(qp["pl"])
elif "dep" in qp:
    page_parlamentar_detail(qp["dep"])
else:
    view = qp.get("view", "dashboard")
    if view == "parlamentares":   page_parlamentares()
    elif view == "projetos":      page_projetos()
    elif view == "temas":         page_temas()
    elif view == "falafaz":       page_fala_faz()
    elif view == "partidos":      page_partidos()
    elif view == "desempenho":    page_desempenho()
    elif view == "classificador": page_classificador()
    else:                         page_dashboard()
