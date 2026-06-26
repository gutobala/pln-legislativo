# -*- coding: utf-8 -*-
"""
Demonstracao web (Gradio): digite uma ementa e o BERTimbau mostra os temas.

Como rodar:
    pip install gradio
    python app_demo.py
Depois abra http://127.0.0.1:7860 no navegador.
Para gerar um LINK PUBLICO temporario (para mandar aos colegas), troque a ultima
linha para:  demo.launch(share=True)

Pre-requisitos na pasta: modelo_bertimbau/, proposicoes_temas.csv, resultados_bertimbau.csv.
Roda em CPU (nao precisa de GPU para classificar uma ementa).
"""
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
import gradio as gr

# ---------- 1) Carregar o modelo e os metadados (uma vez, ao iniciar) ----------
MODELO_DIR = "modelo_bertimbau"
tokenizer = AutoTokenizer.from_pretrained(MODELO_DIR)
modelo = AutoModelForSequenceClassification.from_pretrained(MODELO_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"
modelo.to(device).eval()

# ordem dos 32 temas = a mesma do treino (classes_ do MultiLabelBinarizer)
props = pd.read_csv("proposicoes_temas.csv", dtype=str)
nomes_temas = list(MultiLabelBinarizer().fit(props["temas"].str.split("|")).classes_)

# limiar por tema (ajustado na validacao), na ordem do modelo
resb = pd.read_csv("resultados_bertimbau.csv")
lim_map = dict(zip(resb["tema"], resb["limiar"]))
limiares = np.array([float(lim_map.get(t, 0.5)) for t in nomes_temas])

print(f"Modelo carregado em {device}. {len(nomes_temas)} temas.")

# ---------- 2) Funcao de classificacao ----------
def classificar(ementa):
    ementa = (ementa or "").strip()
    if not ementa:
        return "Digite ou cole uma ementa acima.", {}
    enc = tokenizer(ementa, truncation=True, max_length=192, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = modelo(**enc).logits
    probs = torch.sigmoid(logits)[0].cpu().numpy()           # 32 probabilidades (0..1)
    # decisao final: usa o limiar de CADA tema
    decididos = [nomes_temas[j] for j in np.argsort(-probs) if probs[j] >= limiares[j]]
    texto = "   •   ".join(decididos) if decididos else "(nenhum tema passou do limiar)"
    barras = {nomes_temas[j]: float(probs[j]) for j in range(len(nomes_temas))}  # p/ as barras
    return texto, barras

# ---------- 3) Exemplos reais (da propria base) ----------
exemplos = []
for t in ["Saúde", "Educação", "Defesa e Segurança",
          "Meio Ambiente e Desenvolvimento Sustentável", "Esporte e Lazer",
          "Direito Penal e Processual Penal"]:
    r = props[props["temas"] == t].head(1)
    if len(r):
        exemplos.append([r.iloc[0]["ementa"]])

# ---------- 4) Interface web ----------
demo = gr.Interface(
    fn=classificar,
    inputs=gr.Textbox(lines=4, label="Ementa da proposição",
                      placeholder="Cole ou digite a ementa de um Projeto de Lei..."),
    outputs=[gr.Textbox(label="Temas atribuídos (decisão final, com limiar por tema)"),
             gr.Label(num_top_classes=8, label="Confiança por tema (probabilidade da sigmoide)")],
    examples=exemplos,
    title="Classificador de Proposições de Lei por Tema — BERTimbau",
    description=("Modelo BERTimbau ajustado em ementas da Câmara dos Deputados — classificação "
                 "multirrótulo em 32 temas. Digite uma ementa e veja os temas previstos e a "
                 "confiança do modelo em cada tema. (Projeto Final — PLN no Setor Público.)"),
)

if __name__ == "__main__":
    demo.launch()   # para link publico: demo.launch(share=True)
