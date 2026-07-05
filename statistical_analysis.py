"""
Análise estatística para comparação de modelos.

Fluxo implementado:

1. Ler todos os arquivos parquet
2. Extrair informações dos experimentos
3. Escolher uma métrica
4. Separar por tamanho de sequência
5. Calcular Average Rank
6. Teste de Friedman
7. Teste de Nemenyi
8. Calcular Critical Difference
9. Gerar gráfico

Baseado em:

Janez Demšar (2006)
Statistical Comparisons of Classifiers over Multiple Data Sets
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd

from scipy.stats import friedmanchisquare

import scikit_posthocs as sp

import matplotlib.pyplot as plt

class StatisticalAnalysis:

    def __init__(self, results_folder="results"):

        self.results_folder = Path(results_folder)

        self.df = None

    
    def load_results(self):

        dfs = []

        files = sorted(
            self.results_folder.glob(
                "*all_experiments_results.parquet"
            )
        )

        if len(files) == 0:

            raise Exception(
                "Nenhum parquet encontrado."
            )

        for file in files:

            print(f"Lendo {file.name}")

            df = pd.read_parquet(file)

            dataset = file.stem.replace(
                "-all_experiments_results",
                ""
            )

            df["dataset"] = dataset

            dfs.append(df)

        self.df = pd.concat(
            dfs,
            ignore_index=True
        )

        print()

        print("Total de registros:")

        print(len(self.df))
    
    def parse_model_id(self):

        print()
        print("Extraindo informações...")

        # Extrai o nome do modelo
        self.df["model"] = (
            self.df["model_id"]
            .str.extract(
                r'^[^-]+-(.+?)_Layers'
            )[0]
            .str.lower()
        )

        # Normaliza os nomes dos modelos
        self.df["model"] = (
            self.df["model"]
            .str.replace("detection-", "", regex=False)
            .str.replace("detection_", "", regex=False)
        )

        self.df["layers"] = (
            self.df["model_id"]
            .str.extract(
                r'Layers-(\d+)'
            )
            .astype(int)
        )

        self.df["hidden_size"] = (
            self.df["model_id"]
            .str.extract(
                r'HiddenSize-(\d+)'
            )
            .astype(int)
        )

        self.df["sequence"] = (
            self.df["model_id"]
            .str.extract(
                r'SequenceLength-(\d+)'
            )
            .astype(int)
        )

        self.df["iteration"] = (
            self.df["model_id"]
            .str.extract(
                r'Iteration-(\d+)'
            )
            .astype(int)
        )

        print()
        print(self.df[
            [
                "dataset",
                "model",
                "sequence",
                "iteration"
            ]
        ].head())

        print("\nModelos encontrados:")
        print(sorted(self.df["model"].unique()))

    def show_models(self):

        print()

        print("Modelos encontrados:")

        for model in sorted(
            self.df["model"].unique()
        ):

            print(model)
    
    def show_sequences(self):

        print()

        print("Sequências encontradas:")

        print(

            sorted(

                self.df["sequence"].unique()

            )

        )
    
    def create_table(self, sequence, metric):

        print()
        print("=" * 60)
        print(f"SEQUÊNCIA: {sequence}")
        print(f"MÉTRICA: {metric}")
        print("=" * 60)

        # Filtra os dados
        if sequence == "TODAS":
            df_seq = self.df.copy()
            print("\nUtilizando TODAS as sequências.")
        else:
            df_seq = self.df[
                self.df["sequence"] == sequence
            ]
            print(f"\nUtilizando apenas a sequência {sequence}.")

        print(f"Total de registros: {len(df_seq)}")

        # Cria a tabela para o teste de Friedman
        if sequence == "TODAS":

            table = df_seq.pivot_table(
                index=[
                    "dataset",
                    "sequence",
                    "iteration"
                ],
                columns="model",
                values=metric
            )

        else:

            table = df_seq.pivot_table(
                index=[
                    "dataset",
                    "iteration"
                ],
                columns="model",
                values=metric
            )

        print("\nNúmero de experimentos antes do dropna:")
        print(len(table))

        print("\nValores ausentes por modelo:")
        print(table.isna().sum())

        # Remove experimentos incompletos
        table = table.dropna()

        print("\nNúmero de experimentos depois do dropna:")
        print(len(table))

        if len(table) == 0:
            raise ValueError(
                "Nenhum experimento completo foi encontrado. "
                "Verifique se todos os modelos possuem resultados."
            )

        print()
        print("Tabela criada.")

        print(f"Número de experimentos: {len(table)}")
        print(f"Número de modelos: {len(table.columns)}")

        print()
        print(table.head())

        return table

    def calculate_ranks(self, table):

        print()
        print("Calculando ranks...")

        ranks = table.rank(
            axis=1,
            ascending=False,
            method="average"
        )

        print()
        print(ranks.head())

        return ranks
    
    def average_rank(self, ranks):

        print()
        print("=" * 60)
        print("AVERAGE RANK")
        print("=" * 60)

        # Rank médio
        avg = ranks.mean()

        # Desvio padrão dos ranks
        std = ranks.std()

        # Erro padrão da média (SEM)
        sem = std / np.sqrt(len(ranks))

        # Organiza pelo menor rank
        order = avg.sort_values().index

        avg = avg.loc[order]
        sem = sem.loc[order]

        result = pd.DataFrame({
            "Average Rank": avg,
            "SEM": sem
        })

        print(result)

        return result
    
    def friedman_test(self, table):

        print()
        print("=" * 60)
        print("FRIEDMAN TEST")
        print("=" * 60)

        stat, p = friedmanchisquare(
            *[table[col] for col in table.columns]
        )

        print(f"Statistic : {stat:.4f}")
        print(f"P-value   : {p:.6f}")

        if p < 0.05:

            print()
            print("Resultado:")
            print("Existe diferença estatisticamente significativa entre os modelos.")

        else:

            print()
            print("Resultado:")
            print("Não foi encontrada diferença significativa.")

        return stat, p
    
    def nemenyi_test(self, ranks):

        print()
        print("=" * 60)
        print("NEMENYI TEST")
        print("=" * 60)

        # O scikit-posthocs espera um DataFrame simples
        ranks = ranks.reset_index(drop=True)

        nemenyi = sp.posthoc_nemenyi_friedman(ranks)

        print(nemenyi)

        return nemenyi
    
    def critical_difference(self, table):

        print()
        print("=" * 60)
        print("CRITICAL DIFFERENCE")
        print("=" * 60)

        k = len(table.columns)
        N = len(table)

        print(f"Número de modelos: {k}")
        print(f"Número de experimentos: {N}")

        # Valores críticos do teste de Nemenyi (α = 0.05)
        q_values = {
            2: 1.960,
            3: 2.344,
            4: 2.569,
            5: 2.728,
            6: 2.850,
            7: 2.949,
            8: 3.031,
            9: 3.102,
            10: 3.164
        }

        if k not in q_values:
            raise ValueError(
                f"Quantidade de modelos ({k}) não suportada."
            )

        q_alpha = q_values[k]

        cd = q_alpha * math.sqrt(
            (k * (k + 1)) /
            (6 * N)
        )

        print(f"q_alpha = {q_alpha}")
        print(f"Critical Difference = {cd:.4f}")

        return cd
    
    def save_results(

        self,

        sequence,

        metric,

        average_rank,

        nemenyi

    ):

        output = Path("outputs")

        output.mkdir(

            exist_ok=True

        )

        average_rank.to_csv(
            output /
            f"average_rank_seq_{sequence}_{metric}.csv",
            index=True
        )

        nemenyi.to_csv(

            output /

            f"nemenyi_seq_{sequence}_{metric}.csv"

        )

        print()

        print("Resultados salvos.")
    
    # def plot_average_rank(self, average_rank, cd, sequence, metric):

    #     plt.figure(figsize=(10,5))

    #     ranks = average_rank.values
    #     models = average_rank.index

    #     plt.scatter(
    #         ranks,
    #         np.arange(len(models)),
    #         s=100
    #     )

    #     for i, model in enumerate(models):

    #         plt.text(
    #             ranks[i] + 0.03,
    #             i,
    #             model,
    #             fontsize=11,
    #             va="center"
    #         )

    #     plt.yticks([])

    #     plt.xlabel("Average Rank")

    #     plt.title(
    #         f"Average Rank - Sequence {sequence} ({metric})"
    #     )

    #     plt.grid(
    #         axis="x",
    #         linestyle="--",
    #         alpha=0.4
    #     )

    #     # Barra da Critical Difference

    #     y = len(models) + 0.4

    #     x1 = ranks.min()

    #     x2 = x1 + cd

    #     plt.plot(
    #         [x1, x2],
    #         [y, y],
    #         linewidth=3
    #     )

    #     plt.text(
    #         (x1+x2)/2,
    #         y+0.1,
    #         f"CD = {cd:.3f}",
    #         ha="center"
    #     )

    #     plt.tight_layout()

    #     output = Path("outputs")

    #     output.mkdir(exist_ok=True)

    #     plt.savefig(
    #         output /
    #         f"average_rank_seq_{sequence}_{metric}.png",
    #         dpi=300
    #     )

    #     plt.show()

    def plot_average_rank(
        self,
        average_rank,
        sequence,
        metric,
        friedman_p,
        cd
    ):

        import matplotlib.pyplot as plt
        from pathlib import Path
        import numpy as np

        output = Path("outputs")
        output.mkdir(exist_ok=True)

        models = average_rank.index.tolist()

        ranks = average_rank["Average Rank"].values

        errors = average_rank["SEM"].values

        x = np.arange(len(models))

        fig, ax = plt.subplots(figsize=(6.5,5))

        # melhor modelo
        best = np.argmin(ranks)

        colors = ["royalblue"] * len(models)
        colors[best] = "red"

        # linha pontilhada
        ax.axhline(
            y=ranks[best],
            linestyle=":",
            linewidth=1.5,
            color="gray"
        )

        # pontos
        for i in range(len(models)):

            ax.errorbar(
                x=x[i],
                y=ranks[i],
                yerr=errors[i],
                fmt='o',
                markersize=8,
                color=colors[i],
                ecolor="gray",
                capsize=4,
                elinewidth=1.2
            )

        ax.set_xticks(x)

        ax.set_xticklabels(
            models,
            rotation=90
        )

        ax.set_ylabel("Average Rank")

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.3
        )

        result = (
            "Different"
            if friedman_p < 0.05
            else "Not Different"
        )

        ax.set_title(
            f"Friedman p-value: {friedman_p:.4f} | "
            f"{result} | "
            f"CritDist: {cd:.3f}"
        )

        plt.tight_layout()

        plt.savefig(
            output /
            f"average_rank_seq_{sequence}_{metric}.png",
            dpi=300
        )

        plt.show()
    

        
if __name__ == "__main__":

    analysis = StatisticalAnalysis()

    analysis.load_results()

    analysis.parse_model_id()

    analysis.show_models()

    analysis.show_sequences()

    MINHA_SEQUENCIA = "TODAS" 
    MINHA_METRICA = "auc_roc"

    # Agora passe essas variáveis para os métodos:
    table = analysis.create_table(
        sequence=MINHA_SEQUENCIA,
        metric=MINHA_METRICA
    )

    ranks = analysis.calculate_ranks(table)
    avg = analysis.average_rank(ranks)
    stat, p = analysis.friedman_test(table)
    nemenyi = analysis.nemenyi_test(ranks)
    cd = analysis.critical_difference(table)

    analysis.save_results(
        sequence=MINHA_SEQUENCIA,
        metric=MINHA_METRICA,
        average_rank=avg,
        nemenyi=nemenyi
    )

    analysis.plot_average_rank(
        average_rank=avg,
        sequence=MINHA_SEQUENCIA,
        metric=MINHA_METRICA,
        friedman_p=p,
        cd=cd
    )

