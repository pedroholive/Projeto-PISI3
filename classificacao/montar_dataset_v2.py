"""
HealthSync - L5: monta o dataset de classificação v2.

Junta, pelo identificador SEQN (= id_participante), variáveis de outros
módulos do NHANES 2021-2023 à base CORE:

  BMX_L     cintura (BMXWAIST) e estatura (BMXHT)
  HDL_L     colesterol HDL (LBDHDD)
  DEMO_L    raça/etnia (RIDRETH3), escolaridade (DMDEDUC2) e razão
            renda-pobreza (INDFMPIR)
  BIOPRO_L  ALT, AST, GGT, ácido úrico, creatinina, albumina, triglicerídeos
  WHQ_L     peso atual e há 1 ano, tentativa de perder peso

Códigos de resposta 7/9 (escolaridade) e >= 7000 (peso) viram valores ausentes.
Nenhuma linha é removida: o resultado tem o mesmo número de linhas da CORE.

Os arquivos .xpt devem ser baixados do site do NHANES (CDC) e colocados em
uma pasta, com os nomes originais (BMX_L.xpt, HDL_L.xpt, DEMO_L.xpt,
BIOPRO_L.xpt, WHQ_L.xpt).

Uso (na raiz do repositório):
    python classificacao/montar_dataset_v2.py <pasta_com_os_xpt>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
CSV_CORE = RAIZ / "dataset" / "healthsync_nhanes_2021_2023_core.csv"
SAIDA = RAIZ / "dataset" / "healthsync_classificacao_v2.csv"


def ler_xpt(pasta, nome, colunas):
    caminho = next(
        (p for p in pasta.iterdir() if p.name.upper().endswith(f"{nome}.XPT")),
        None)
    if caminho is None:
        raise FileNotFoundError(f"{nome}.xpt não encontrado em {pasta}")
    t = pd.read_sas(caminho, format="xport", encoding="utf-8")
    t["SEQN"] = t["SEQN"].astype(int)
    return t[["SEQN"] + colunas]


def main(pasta):
    pasta = Path(pasta)
    core = pd.read_csv(CSV_CORE, encoding="utf-8-sig")
    n0 = len(core)
    core["SEQN"] = core["id_participante"].astype(int)
    assert core["SEQN"].is_unique, "id_participante repetido"

    blocos = [
        ler_xpt(pasta, "BMX_L", ["BMXWAIST", "BMXHT"]),
        ler_xpt(pasta, "HDL_L", ["LBDHDD"]),
        ler_xpt(pasta, "DEMO_L", ["RIDRETH3", "DMDEDUC2", "INDFMPIR"]),
        ler_xpt(pasta, "BIOPRO_L", ["LBXSATSI", "LBXSASSI", "LBXSGTSI",
                                    "LBXSUA", "LBXSCR", "LBXSAL", "LBXSTR"]),
        ler_xpt(pasta, "WHQ_L", ["WHD020", "WHD050", "WHQ070"]),
    ]
    d = core
    for b in blocos:
        d = d.merge(b, on="SEQN", how="left")
    assert len(d) == n0, "a junção alterou o número de linhas"

    d = d.rename(columns={
        "BMXWAIST": "cintura_cm", "BMXHT": "altura_cm", "LBDHDD": "hdl",
        "RIDRETH3": "raca_etnia", "DMDEDUC2": "escolaridade",
        "INDFMPIR": "razao_renda_pobreza",
        "LBXSATSI": "alt", "LBXSASSI": "ast", "LBXSGTSI": "ggt",
        "LBXSUA": "acido_urico", "LBXSCR": "creatinina",
        "LBXSAL": "albumina", "LBXSTR": "trigliceridios"})

    d["escolaridade"] = d["escolaridade"].replace({7: np.nan, 9: np.nan})
    d["cintura_estatura"] = d["cintura_cm"] / d["altura_cm"]
    d["colesterol_hdl_razao"] = d["colesterol_total"] / d["hdl"]

    for c in ["WHD020", "WHD050"]:
        d[c] = d[c].where(d[c] < 7000)
    d["variacao_peso_1ano_pct"] = (d["WHD020"] - d["WHD050"]) / d["WHD050"] * 100
    d["tentou_perder_peso"] = d["WHQ070"].map({1.0: 1.0, 2.0: 0.0})
    d = d.drop(columns=["WHD020", "WHD050", "WHQ070", "SEQN"])

    d.to_csv(SAIDA, index=False, encoding="utf-8-sig")
    print(f"Salvo: {SAIDA} ({d.shape[0]} linhas x {d.shape[1]} colunas)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])