# ✨ Datathon PosTech — Passos Mágicos

Análise de dados educacionais e modelo preditivo de risco de defasagem para a
[Associação Passos Mágicos](https://passosmagicos.org.br/), desenvolvido como
entrega final da **Fase 5 do curso de Data Analytics da PosTech (FIAP)**.

---

## 📌 Sobre o projeto

A Associação Passos Mágicos tem 35 anos de atuação transformando a vida de
crianças e jovens em vulnerabilidade social em Embu-Guaçu (SP) por meio de
educação de qualidade, apoio psicológico e ampliação de visão de mundo.h

Este projeto analisa a base **PEDE** (Pesquisa Extensiva do Desenvolvimento
Educacional) dos anos de **2022, 2023 e 2024**, respondendo a 11 perguntas de
negócio e construindo um modelo preditivo de risco de defasagem.

---

## 🚀 App Streamlit

> 🔗 **https://datathon-postech-fiap.streamlit.app/**

O app permite:

- Visualizar os principais indicadores educacionais por ano e Pedra
- Explorar a análise de cada indicador (IDA, IEG, IPS, IPP, IAA, IPV, IAN)
- **Prever o risco de defasagem de um aluno em tempo real**

---

## 📁 Estrutura do projeto

```
Datathon-Passos-Magicos/
├── notebooks/
│   ├── 01_exploracao.ipynb          # Carga, limpeza e visão geral dos dados
│   ├── 02_analise_perguntas.ipynb   # Responde as 11 perguntas de negócio
│   └── 03_modelo_preditivo.ipynb    # Modelo ML de risco de defasagem (P9)
├── src/
│   └── *.png                        # Gráficos gerados pelos notebooks
├── app/
│   └── app.py                       # Aplicação Streamlit
├── data/                            # datasets do projeto, etc...
│   ├── df_consolidado.pkl           # gerado pelo notebook 01
│   └── modelo_risco_passos.pkl      # gerado pelo notebook 03
├── outputs/                         # resultados e gráficos adicionais
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Como rodar localmente

### 1. Clone o repositório

```bash
git clone https://github.com/lacerdax/Datathon-Passos-Magicos.git
cd Datathon-Passos-Magicos
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Execute os notebooks em ordem

```
01_exploracao.ipynb        → gera data/df_consolidado.pkl
02_analise_perguntas.ipynb → gera os gráficos em src/
03_modelo_preditivo.ipynb  → gera data/modelo_risco_passos.pkl
```

Abra cada notebook no VS Code e clique em **Run All**.

### 4. Rode o app Streamlit

```bash
python -m streamlit run app/app.py
```

---

## 📊 Indicadores analisados

| Indicador | Descrição                                                  |
| --------- | ---------------------------------------------------------- |
| **INDE**  | Índice de Desenvolvimento Educacional — nota global (0–10) |
| **IDA**   | Índice de Desempenho Acadêmico                             |
| **IEG**   | Índice de Engajamento                                      |
| **IAA**   | Índice de Autoavaliação                                    |
| **IPS**   | Índice Psicossocial                                        |
| **IPP**   | Índice Psicopedagógico                                     |
| **IPV**   | Índice do Ponto de Virada                                  |
| **IAN**   | Índice de Adequação de Nível                               |

---

## 💎 Sistema de Pedras (níveis do programa)

| Pedra       | Faixa de INDE | Perfil                                   |
| ----------- | ------------- | ---------------------------------------- |
| 🪨 Quartzo  | 0 – 6,0       | Maior defasagem, precisa de mais suporte |
| 💚 Ágata    | 6,0 – 7,0     | Em desenvolvimento                       |
| 💜 Ametista | 7,0 – 8,0     | Nível adequado, bom desempenho           |
| 🌟 Topázio  | 8,0 – 10,0    | Alta performance, candidato a destaque   |

---

## 📈 Principais resultados

| Métrica           | 2022 | 2024 | Evolução |
| ----------------- | ---- | ---- | -------- |
| INDE médio        | 7,04 | 7,40 | ▲ +0,36  |
| Alunos em Topázio | 15%  | 28%  | ▲ +13pp  |
| Defasagem severa  | 3,3% | 0,3% | ▼ -3pp   |
| Alunos em risco   | 38%  | 20%  | ▼ -18pp  |
| Nível adequado    | 30%  | 54%  | ▲ +24pp  |

### Modelo preditivo (P9)

| Métrica        | Random Forest | XGBoost   | Gradient Boosting |
| -------------- | ------------- | --------- | ----------------- |
| AUC-ROC        | 0,997         | **0,997** | 0,996             |
| Accuracy       | 96%           | **97%**   | 96%               |
| Recall (risco) | 90%           | **91%**   | 90%               |

**Modelo selecionado:** XGBoost — melhor equilíbrio entre performance e interpretabilidade.

**Features mais importantes (SHAP):**

1. Desempenho acadêmico (IDA)
2. Engajamento (IEG)
3. Defasagem atual
4. Ponto de virada (IPV)
5. Avaliação psicopedagógica (IPP)

---

## 🔍 Respostas às 11 perguntas do Datathon

| #   | Pergunta                   | Conclusão principal                        |
| --- | -------------------------- | ------------------------------------------ |
| P1  | Perfil de defasagem (IAN)  | Severa caiu de 3,3% → 0,3% em 3 anos       |
| P2  | Desempenho acadêmico (IDA) | Cresceu; Matemática é a maior lacuna       |
| P3  | Engajamento (IEG)          | Principal preditor de desempenho (r=0,71)  |
| P4  | Autoavaliação (IAA)        | Quartzo subestima a própria capacidade     |
| P5  | Psicossocial (IPS)         | Impacta engajamento e desempenho           |
| P6  | Psicopedagógico (IPP)      | Confirma parcialmente o IAN                |
| P7  | Ponto de virada (IPV)      | IDA e IEG são os maiores preditores        |
| P8  | Multidimensionalidade      | IDA+IEG+IPV explicam >75% do INDE          |
| P9  | Modelo preditivo           | XGBoost AUC-ROC = 0,997                    |
| P10 | Efetividade do programa    | INDE subiu 7,04→7,40; Topázio: 15%→28%     |
| P11 | Insights adicionais        | Permanência e gênero feminino → maior INDE |

---

## 🛠️ Tecnologias utilizadas

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.0-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![Scikit-learn](https://img.shields.io/badge/ScikitLearn-1.3-orange)

- **Análise:** Pandas, NumPy, SciPy
- **Visualização:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-learn, XGBoost, Imbalanced-learn
- **Interpretabilidade:** SHAP
- **App:** Streamlit
- **Ambiente:** VS Code + Jupyter

---

## 👤 Autor

**Lucas Campos Araujo Lacerda**
Curso de Data Analytics — PosTech FIAP
[LinkedIn](https://linkedin.com/in/lacerdaxl) · [GitHub](https://github.com/lacerdax)

---

## 📄 Licença

Este projeto foi desenvolvido para fins acadêmicos no âmbito do Datathon PosTech.
Os dados utilizados são de propriedade da Associação Passos Mágicos.
