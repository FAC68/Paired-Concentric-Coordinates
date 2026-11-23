import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pcoc_iris import (
    TAU,
    minmax,
    coc_points,
    pcoc_points,
    attribute_radii,
    draw_circles,
    draw_cases,
    circle_labels,
)

# ------------------------------------------------------------
# PAGE SETUP
# ------------------------------------------------------------

st.set_page_config(
    page_title="Iris PCoC Explorer",
    layout="wide"
)

st.title("Iris Paired Concentric Coordinates Explorer")

st.caption(
    "Compare standard Concentric Coordinates (CoC) with "
    "Paired Concentric Coordinates (PCoC) using the Fisher Iris dataset."
)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("fisher_iris.csv")

df = load_data()

# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------

feature_cols = [
    "sepal.length",
    "sepal.width",
    "petal.length",
    "petal.width"
]

pretty_names = {
    "sepal.length": "Sepal Length",
    "sepal.width": "Sepal Width",
    "petal.length": "Petal Length",
    "petal.width": "Petal Width"
}

# ------------------------------------------------------------
# NORMALIZE FULL DATASET
# ------------------------------------------------------------

X_all = minmax(df[feature_cols].values)

class_cat = df["class"].astype("category")
class_names = list(class_cat.cat.categories)
class_codes_all = class_cat.cat.codes.to_numpy()

# ------------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------------

st.sidebar.header("Filters")

selected_classes = st.sidebar.multiselect(
    "Iris Class",
    class_names,
    default=class_names
)

mask = df["class"].isin(
    selected_classes
).to_numpy(copy=True)

st.sidebar.subheader("Attribute Ranges")

for col in feature_cols:
    col_min = float(df[col].min())
    col_max = float(df[col].max())

    selected_min, selected_max = st.sidebar.slider(
        pretty_names[col],
        min_value=col_min,
        max_value=col_max,
        value=(col_min, col_max),
        step=0.1
    )

    mask &= df[col].between(
        selected_min,
        selected_max
    ).to_numpy()

# ------------------------------------------------------------
# FILTER DATA
# ------------------------------------------------------------

filtered_positions = np.flatnonzero(mask)

if len(filtered_positions) == 0:
    st.warning("No Iris records match the current filters.")
    st.stop()

filtered_df = df.iloc[filtered_positions].copy()
X_filtered = X_all[filtered_positions]
y_filtered = class_codes_all[filtered_positions]

# ------------------------------------------------------------
# DISPLAY OPTIONS
# ------------------------------------------------------------

st.sidebar.divider()
st.sidebar.header("Visualization Controls")

display_mode = st.sidebar.radio(
    "Visualization",
    [
        "Side-by-side",
        "CoC only",
        "PCoC only"
    ]
)

collocate = st.sidebar.checkbox(
    "Collocate paired circles",
    value=True,
    help="Uses shared coordinate circles in the PCoC view."
)

rotation_degrees = st.sidebar.slider(
    "Rotate coordinates",
    min_value=0,
    max_value=355,
    value=0,
    step=5,
    help="Rotates where normalized values begin around each circle."
)

start_angle = np.deg2rad(rotation_degrees)

reverse_coordinates = st.sidebar.checkbox(
    "Reverse coordinate order",
    value=False,
    help="Reverses the Iris attribute order before drawing CoC and PCoC."
)

reverse_direction = st.sidebar.checkbox(
    "Reverse angular direction",
    value=False,
    help="Draws increasing values clockwise instead of counter-clockwise."
)

span = st.sidebar.slider(
    "Angular Span",
    min_value=4.0,
    max_value=float(TAU),
    value=5.5,
    step=0.1
)

if reverse_direction:
    span = -span

plot_feature_cols = (
    list(reversed(feature_cols))
    if reverse_coordinates
    else list(feature_cols)
)

plot_indices = [
    feature_cols.index(col)
    for col in plot_feature_cols
]

X_plot = X_filtered[:, plot_indices]

# ------------------------------------------------------------
# LINE HIGHLIGHTING
# ------------------------------------------------------------

st.sidebar.divider()
st.sidebar.subheader("Line Highlighting")

enable_highlight = st.sidebar.checkbox(
    "Highlight selected Iris record",
    value=True
)

selected_id = st.sidebar.selectbox(
    "Highlight Iris Row ID",
    filtered_df["id"].astype(int).tolist()
)

highlight_matches = np.flatnonzero(
    filtered_df["id"].to_numpy() == selected_id
)

highlight_local = int(highlight_matches[0])

highlight_arg = (
    highlight_local
    if enable_highlight
    else None
)

selected_record = filtered_df.iloc[highlight_local]

# ------------------------------------------------------------
# PAGE LAYOUT
# ------------------------------------------------------------

left_column, right_column = st.columns([3, 1])

# ------------------------------------------------------------
# SELECTED FLOWER INFO
# ------------------------------------------------------------

with right_column:
    st.subheader("Selected Iris")

    st.metric(
        "Row ID",
        int(selected_record["id"])
    )

    st.write(
        f"**Class:** {selected_record['class']}"
    )

    st.write(
        f"**Sepal Length:** {selected_record['sepal.length']}"
    )

    st.write(
        f"**Sepal Width:** {selected_record['sepal.width']}"
    )

    st.write(
        f"**Petal Length:** {selected_record['petal.length']}"
    )

    st.write(
        f"**Petal Width:** {selected_record['petal.width']}"
    )

    st.write(
        f"**Visible Records:** {len(filtered_df)} / {len(df)}"
    )

    if rotation_degrees != 0:
        st.info(
            f"Coordinate mapping rotated {rotation_degrees}°."
        )

    if reverse_coordinates:
        st.info(
            "Coordinate order is reversed."
        )

    if reverse_direction:
        st.info(
            "Angular direction is reversed."
        )

# ------------------------------------------------------------
# COC VISUALIZATION
# ------------------------------------------------------------

def make_coc_axis(ax):
    number_of_features = X_plot.shape[1]

    radii = attribute_radii(
        number_of_features,
        collocate=False
    )

    points = coc_points(
        X_plot,
        radii,
        span,
        start=start_angle
    )

    labels = [
        pretty_names[col]
        for col in plot_feature_cols
    ]

    draw_circles(
        ax,
        sorted(set(radii)),
        circle_labels(
            labels,
            False
        )
    )

    draw_cases(
        ax,
        points,
        y_filtered,
        class_names,
        highlight=highlight_arg
    )

    ax.set_title(
        "Concentric Coordinates (CoC)\n"
        f"{number_of_features} points • "
        f"{number_of_features - 1} segments • "
        f"{len(set(radii))} circles per case"
    )

# ------------------------------------------------------------
# PCOC VISUALIZATION
# ------------------------------------------------------------

def make_pcoc_axis(ax):
    number_of_features = X_plot.shape[1]

    radii = attribute_radii(
        number_of_features,
        collocate=collocate
    )

    points, flags = pcoc_points(
        X_plot,
        radii,
        span,
        start=start_angle
    )

    labels = [
        pretty_names[col]
        for col in plot_feature_cols
    ]

    draw_circles(
        ax,
        sorted(set(radii)),
        circle_labels(
            labels,
            collocate
        )
    )

    draw_cases(
        ax,
        points,
        y_filtered,
        class_names,
        highlight=highlight_arg,
        arrows=True
    )

    number_of_points = points.shape[1]

    ax.set_title(
        "Paired Concentric Coordinates (PCoC)\n"
        f"{number_of_points} points • "
        f"{max(number_of_points - 1, 0)} segment • "
        f"{len(set(radii))} circles per case"
    )

# ------------------------------------------------------------
# DRAW VISUALIZATION
# ------------------------------------------------------------

with left_column:

    if display_mode == "Side-by-side":

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(14, 7)
        )

        make_coc_axis(
            axes[0]
        )

        make_pcoc_axis(
            axes[1]
        )

        handles, labels = (
            axes[1]
            .get_legend_handles_labels()
        )

        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=len(class_names),
                frameon=False
            )

        fig.tight_layout(
            rect=(0, 0.06, 1, 1)
        )

    elif display_mode == "CoC only":

        fig, ax = plt.subplots(
            figsize=(8, 8)
        )

        make_coc_axis(
            ax
        )

        handles, labels = (
            ax.get_legend_handles_labels()
        )

        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=len(class_names),
                frameon=False
            )

        fig.tight_layout(
            rect=(0, 0.06, 1, 1)
        )

    else:

        fig, ax = plt.subplots(
            figsize=(8, 8)
        )

        make_pcoc_axis(
            ax
        )

        handles, labels = (
            ax.get_legend_handles_labels()
        )

        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=len(class_names),
                frameon=False
            )

        fig.tight_layout(
            rect=(0, 0.06, 1, 1)
        )

    st.pyplot(
        fig,
        width="stretch"
    )

    plt.close(fig)

# ------------------------------------------------------------
# PCOC COMPARISON INFORMATION
# ------------------------------------------------------------

st.subheader("Why PCoC?")

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "CoC Points Per Iris Record",
        4
    )

with metric2:
    st.metric(
        "PCoC Points Per Iris Record",
        2
    )

with metric3:
    st.metric(
        "Segments Reduced",
        "3 → 1"
    )

st.write(
    "For the four Iris attributes, standard Concentric "
    "Coordinates use four plotted points and three connecting "
    "segments for every Iris record. Paired Concentric Coordinates "
    "combine consecutive attribute pairs, reducing each record to "
    "two paired points connected by one segment."
)

st.caption(
    "Use the sidebar controls to rotate the coordinate mapping, "
    "reverse the coordinate order or angular direction, filter the "
    "dataset, and highlight an individual Iris record."
)

# ------------------------------------------------------------
# SHOW FILTERED DATA
# ------------------------------------------------------------

with st.expander("Show Filtered Iris Data"):

    st.dataframe(
        filtered_df[
            [
                "id",
                "class",
                "sepal.length",
                "sepal.width",
                "petal.length",
                "petal.width"
            ]
        ],
        width="stretch",
        hide_index=True
    )