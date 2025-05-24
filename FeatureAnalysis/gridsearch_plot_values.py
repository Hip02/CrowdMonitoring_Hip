# Reimport necessary packages due to kernel reset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Full dataset again
data_full = [
    (1, 0.01, 0.001, 18.0113), (1, 0.01, 0.01, 12.3292), (1, 0.01, 0.1, 10.4937), (1, 0.01, 1, 11.2897),
    (1, 0.1, 0.001, 18.0064), (1, 0.1, 0.01, 12.3276), (1, 0.1, 0.1, 10.4980), (1, 0.1, 1, 11.2879),
    (1, 0.5, 0.001, 17.9351), (1, 0.5, 0.01, 12.3028), (1, 0.5, 0.1, 10.5097), (1, 0.5, 1, 11.2629),
    (10, 0.01, 0.001, 13.0407), (10, 0.01, 0.01, 10.5268), (10, 0.01, 0.1, 9.6842), (10, 0.01, 1, 10.4043),
    (10, 0.1, 0.001, 13.0450), (10, 0.1, 0.01, 10.5276), (10, 0.1, 0.1, 9.6915), (10, 0.1, 1, 10.3927),
    (10, 0.5, 0.001, 13.0343), (10, 0.5, 0.01, 10.5158), (10, 0.5, 0.1, 9.6747), (10, 0.5, 1, 10.3381),
    (100, 0.01, 0.001, 11.2774), (100, 0.01, 0.01, 9.9716), (100, 0.01, 0.1, 9.4885), (100, 0.01, 1, 11.5113),
    (100, 0.1, 0.001, 11.2757), (100, 0.1, 0.01, 9.9675), (100, 0.1, 0.1, 9.5027), (100, 0.1, 1, 11.4658),
    (100, 0.5, 0.001, 11.2511), (100, 0.5, 0.01, 9.9544), (100, 0.5, 0.1, 9.4867), (100, 0.5, 1, 11.3125)
]

df_full = pd.DataFrame(data_full, columns=["C", "epsilon", "gamma", "Mean MSE"])

# Plot
g = sns.FacetGrid(df_full, col="epsilon", col_wrap=3, height=4, sharex=False, sharey=False)
g.map_dataframe(
    lambda data, color: sns.heatmap(
        data.pivot(index="C", columns="gamma", values="Mean MSE"),
        annot=True, fmt=".2f", cmap="viridis", cbar=True, linewidths=0.5
    )
)
g.set_titles(col_template=r"$\epsilon$ = {col_name}")
g.set_axis_labels("gamma", "C")
plt.subplots_adjust(top=0.85)
g.fig.suptitle("Mean MSE across SVR hyperparameters (RBF)", fontsize=14)

# Save
pdf_path = "/Users/hippolytehilgers/Downloads/rbf_svr_gridsearch_heatmap.pdf"
plt.savefig(pdf_path, format="pdf")
#pdf_path