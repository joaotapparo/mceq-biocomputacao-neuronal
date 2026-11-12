# Biocomputação neuronal: desempenho computacional e limites da hipótese de eficiência energética

Pacote de reprodutibilidade do projeto final de **MCEQ 12576 – Métodos de Computação Experimental e
Quantitativa** (PUC-Campinas, Engenharia de Software, 2º semestre de 2026).

**Equipe:** João Victor Rokemback Tapparo · Thiago Ryuji Ogawa

**Vídeo (YouTube, não listado):** _a definir_

## Pergunta de pesquisa

A acurácia de classificação de dígitos obtida a partir de sinais **reais** de culturas neuronais humanas
supera a de um **controle aleatorizado** nos dados públicos da Cortical Labs?

- **Δ** = acurácia Real − acurácia Random, por chip (média das sessões do chip)
- **H0:** μΔ ≤ 0  **H1:** μΔ > 0 (unilateral)  **α = 0,05**
- **Tipo de estudo:** reanálise **observacional e exploratória** de resultados publicados. A hipótese não
  foi pré-registrada antes da inspeção inicial da base, e não fazemos afirmações causais.
- **Desenho:** pareado (Real e Random no mesmo chip); unidade de análise = chip (n = 26)
- **Teste principal:** Wilcoxon pareado unilateral. **Sensibilidade:** t pareado.
  **Efeito:** d_z de Cohen. **IC95%** da diferença média (t de Student).

## Dados

| Item | Valor |
|---|---|
| Arquivo | `file_names.csv` |
| Origem | [Cortical-Labs/handwritten-digit-neural-cultures](https://github.com/Cortical-Labs/handwritten-digit-neural-cultures) |
| Artigo associado | LOEFFLER, A. et al. *Handwritten Digit classification with neural cultures is influenced by neural architecture, network dynamics, and decoding methods.* bioRxiv (preprint), 2026 |
| Licença | CC BY-NC 4.0 (redistribuição permitida com atribuição, para uso não comercial) |
| Acesso | 11–12/11/2026 |
| SHA-256 | `c5e762bd1f51b47762c0f4ba377698ab5525cc56b6f86ee2c881a6b9b4f52d65` |
| Tamanho | 52 sessões × 15 colunas |

Colunas usadas: `Chip ID`, `Date`, `Cell Type`, `Num Spikes`, `Real` e `Random`. Segundo o README de
origem, `Real` e `Random` são as acurácias (proporção de 0 a 1) do decodificador na tarefa MNIST
espaço-temporal com os sinais reais e com o controle aleatorizado.

**Limpeza** (feita pelo script; os dados brutos nunca são editados à mão):
1. remover os 6 controles sem neurônios (`SDK Random Spikes` ×3, `Saline` ×3);
2. remover 1 sessão com `Num Spikes < 500` (limite do artigo de origem).

Restam **45 sessões de 26 chips**, agregadas pela média por chip.

## Estrutura

```
.
├── README.md
├── requirements.txt
├── validacao.ipynb            # análise completa, executada, com explicações
├── scripts/
│   └── gerar_notebook.py      # gera o validacao.ipynb (fonte do notebook)
├── dados/
│   ├── brutos/file_names.csv  # cópia intacta da fonte original
│   └── limpos/                # gerados pelo notebook
│       ├── sessoes_limpas.csv
│       └── por_chip.csv
└── figuras/                   # gerados pelo notebook
    ├── figura1_pares.png
    └── exploratoria_lotes.png
```

## Como reexecutar

Requer Python 3.12 (o projeto foi testado com 3.12).

```bash
git clone <url-deste-repositorio>
cd computacao-experimental

# opção A: uv
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt

# opção B: venv + pip
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# executar o notebook do início ao fim (regenera dados/limpos e figuras/)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace validacao.ipynb

# ou abrir interativamente
.venv/bin/jupyter lab validacao.ipynb
```

Se `dados/brutos/file_names.csv` não existir, o notebook baixa o arquivo da fonte original e confere o
SHA-256. A última seção do notebook lista as checagens contra os números do artigo; o esperado é
**16/16**.

Para editar o texto ou o código do notebook, altere `scripts/gerar_notebook.py` e rode:

```bash
.venv/bin/python scripts/gerar_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace validacao.ipynb
```

## Resultados (n = 26 chips)

| Métrica | Valor |
|---|---|
| Acurácia Real, média (DP) | 44,07% (10,32 pp) |
| Acurácia Random, média (DP) | 11,47% (1,04 pp) |
| Diferença média [IC95%] | 32,60 pp [28,10; 37,10] |
| Wilcoxon pareado unilateral | W = 351; p = 4,14×10⁻⁶ (aproximação normal; há empates) |
| d_z de Cohen | 2,92 |
| t pareado (sensibilidade) | t(25) = 14,91; p ≈ 3×10⁻¹⁴ |
| Chips com Real > Random | 26/26 |

**Decisão:** rejeita-se H0 a α = 0,05. Nesta base, a decodificação de sinais neuronais apresentou
acurácia superior ao controle aleatorizado.

**Análises exploratórias** (fora do protocolo):
- A conclusão se mantém com mediana por chip, com sessão como unidade e mantendo a sessão de 469 spikes.
- As diferenças são **bimodais** e acompanham dois lotes de gravação: nov–dez/2024, com ~26 pp em 17
  chips, e mar/2025, com ~45 pp em 9 chips, todos PDMS. Lote, arquitetura e nível do controle variam
  juntos, o que é tratado como ameaça à validade interna.

## Limitações

- Os dados são resumos de desempenho já processados pelos autores originais. Esta reanálise não
  reproduz a aquisição dos sinais nem o treinamento do decodificador.
- O estudo é observacional, então não há inferência causal entre arquiteturas, datas ou procedimentos.
- A tarefa MNIST não representa cargas de trabalho reais de IA, e a base **não contém medidas de
  energia**. Nada aqui sustenta conclusões sobre eficiência energética.
