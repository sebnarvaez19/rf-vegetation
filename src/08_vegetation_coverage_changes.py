import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from functions.stats_utils import plot_corr_matrix

plt.style.use("src/style.mplstyle")

data_path = "data/processed/annual_vegetation_coverage.tif"
save_path = "images/{:02d}_{}.{}"

codes = ["Water", "Woodland", "Dessert", "Mangroves", "Thorn", "Shurbs"]
cmap = ["#0000ff", "#689d4d", "#ffff9d", "#ad256b", "#c4ff4d", "#b88454"]
cmap = ListedColormap(cmap)

handles = [Patch(color=c, label=l) for c, l in zip(cmap.colors, codes)]

def main():
    data = rioxarray.open_rasterio(data_path, decode_coords="all")

    # Mask values that are not in all years
    mask = data.count(dim="band") == 23
    data = data.where(mask == True)

    # Save animation
    fig1, ax = plt.subplots(figsize=(4, 4))
    frames = []
    for i in range(23):
        img = data.sel(band=i+1).plot(ax=ax, add_colorbar=False, cmap=cmap)
        text = ax.text(
            x=0.0225,
            y=0.0225,
            s=f"Year {i+2000}",
            ha="left",
            va="bottom",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", color="white", alpha=0.8),
        )

        frames.append([img, text])

    fig1.subplots_adjust(left=0, bottom=0, right=1, top=1)

    ax.axis("off")
    ax.set_title(None)
    ax.legend(
        handles=handles, loc=4, fontsize="xx-small", frameon=True, framealpha=0.8
    )

    writer = animation.FFMpegWriter(fps=2)

    ani = animation.ArtistAnimation(fig1, frames, interval=50, blit=True)
    ani.save(save_path.format(8, "vegetation_coverage_change", "mp4"), writer)

    # Reduce data to get coverage by year
    data = (
        data.to_dataframe(name="vc")
            .reset_index()
            .pivot(index=["x", "y"], columns="band", values="vc")
            .reset_index(drop=True)
    )

    counts = []
    for i in range(23):
        count = data.iloc[:,i].value_counts()/data.iloc[:,i].notna().sum()*100
        count = count.rename(f"{i+2000}")
        counts.append(count)

    counts = pd.DataFrame(counts).sort_index(axis=1)
    counts.index = counts.index.astype(np.int64)
    counts.columns = codes

    # Plot mean coverage by class
    fig2, ax = plt.subplots(tight_layout=True)
    ax.bar(
        x=codes,
        height=counts.describe().loc["mean",:].values,
        yerr=counts.describe().loc["std",:].values,
        alpha=0.8,
        color=cmap.colors,
    )

    ax.set_ylabel("% Coverage")
    ax.set_xticklabels(ax.get_xticklabels(), ha="right", rotation=30)

    fig2.savefig(save_path.format(9, "coverage_by_class", "svg"))

    # Standardize data
    std_codes = []

    for code in codes:
        label = code+"_std"
        counts[label] = (counts[code] - counts[code].mean())/counts[code].std()
        std_codes.append(label)

    # Plot coverage change throught time
    fig3, ax = plt.subplots(tight_layout=True, figsize=(6, 4))

    for code, color in zip(std_codes, cmap.colors):
        ax.plot(
            counts.index,
            counts[code],
            color=color,
            alpha=0.8,
            label=code[:-4],
        )

    ax.set(xlabel="Time [Y]", ylabel="Standardized change")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="xx-small")

    fig3.savefig(save_path.format(10, "standardized_coverage_ts", "svg"))

    # Plot corelation matrix
    fig4, ax = plt.subplots(tight_layout=True, figsize=(5, 4))
    plot_corr_matrix(
        data=counts,
        variables=codes,
        half=True,
        hide_insignificants=True,
        show_colorbar=True,
        ax=ax,
    )

    fig4.savefig(save_path.format(11, "coverage_correlation_matrix", "svg"))

    plt.show()

    return None


if __name__ == "__main__":
    main()
