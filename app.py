
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

plt.style.use("fivethirtyeight")

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Decision Tree Classifier",
    page_icon="🌳",
    layout="wide"
)

st.title("Decision Tree Classifier_Visualizer")
st.write(
    "Visualize decision boundaries, model accuracy "
    "and the complete decision tree."
)
st.subheader(
  "Designed by Ayush Raj"
)

# ---------------- Sidebar ----------------
st.sidebar.header("Model Hyperparameters")

dataset = st.sidebar.selectbox(
    "Select Dataset",
    ["Binary", "Multiclass"]
)

criterion = st.sidebar.selectbox(
    "Criterion",
    ["gini", "entropy", "log_loss"]
)

splitter = st.sidebar.selectbox(
    "Splitter",
    ["best", "random"]
)

max_depth = st.sidebar.selectbox(
    "Maximum Depth",
    [None, 1, 2, 3, 4, 5, 7, 10, 15, 20],
    index=3
)

min_samples_split = st.sidebar.slider(
    "Minimum Samples Split",
    2, 20, 2
)

min_samples_leaf = st.sidebar.slider(
    "Minimum Samples Leaf",
    1, 10, 1
)

max_features = st.sidebar.selectbox(
    "Maximum Features",
    [None, "sqrt", "log2", 1, 2]
)

max_leaf_nodes = st.sidebar.selectbox(
    "Maximum Leaf Nodes",
    [None, 2, 3, 5, 10, 15, 20, 30]
)

min_impurity_decrease = st.sidebar.selectbox(
    "Minimum Impurity Decrease",
    [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
)

test_size = st.sidebar.slider(
    "Test Size",
    min_value=0.1,
    max_value=0.5,
    value=0.25,
    step=0.05
)

# ---------------- Dataset ----------------
@st.cache_data
def load_dataset(dataset):
    if dataset == "Binary":
        return make_blobs(
            n_samples=300,
            n_features=2,
            centers=2,
            cluster_std=2.0,
            random_state=6
        )

    return make_blobs(
        n_samples=300,
        n_features=2,
        centers=3,
        cluster_std=2.0,
        random_state=2
    )


X, y = load_dataset(dataset)

# ---------------- Original Dataset Graph ----------------
st.subheader("1. Original Dataset")

fig1, ax1 = plt.subplots(figsize=(8, 5))

ax1.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap="rainbow",
    edgecolors="black"
)

ax1.set_xlabel("Col1")
ax1.set_ylabel("Col2")
ax1.set_title("Original Dataset")

st.pyplot(fig1)
plt.close(fig1)

# ---------------- Run Algorithm ----------------
if st.sidebar.button("Run Algorithm", type="primary"):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    clf = DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        random_state=42
    )

    clf.fit(X_train, y_train)

    # ---------------- Predictions ----------------
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    error_rate = 1 - accuracy

    # ---------------- Accuracy Metrics ----------------
    st.subheader("2. Model Accuracy")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
    col2.metric("Error Rate", f"{error_rate * 100:.2f}%")
    col3.metric("Tree Depth", clf.get_depth())

    # Accuracy and error graph
    fig2, ax2 = plt.subplots(figsize=(7, 4))

    bars = ax2.bar(
        ["Accuracy", "Error Rate"],
        [accuracy * 100, error_rate * 100],
        color=["royalblue", "tomato"]
    )

    ax2.set_ylabel("Percentage (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("Model Performance")

    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.2f}%",
            ha="center"
        )

    st.pyplot(fig2)
    plt.close(fig2)

    # ---------------- Decision Boundary Graph ----------------
    st.subheader("3. Decision Boundary Graph")

    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1
    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1

    XX, YY = np.meshgrid(
        np.arange(x_min, x_max, 0.05),
        np.arange(y_min, y_max, 0.05)
    )

    input_array = np.c_[XX.ravel(), YY.ravel()]
    labels = clf.predict(input_array).reshape(XX.shape)

    fig3, ax3 = plt.subplots(figsize=(9, 6))

    ax3.contourf(
        XX,
        YY,
        labels,
        alpha=0.35,
        cmap="rainbow"
    )

    # Training data
    ax3.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        cmap="rainbow",
        edgecolors="black",
        s=35,
        label="Training Data"
    )

    # Test data
    ax3.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap="rainbow",
        edgecolors="black",
        marker="x",
        s=60,
        label="Test Data"
    )

    ax3.set_xlabel("Col1")
    ax3.set_ylabel("Col2")
    ax3.set_title("Decision Tree Classification Regions")
    ax3.legend()

    st.pyplot(fig3)
    plt.close(fig3)

    # ---------------- Actual Decision Tree ----------------
    st.subheader("4. Actual Decision Tree Structure")

    # Wider figure for a more readable tree
    tree_depth = clf.get_depth()
    figure_width = min(24, max(12, 3 * (2 ** min(tree_depth, 3))))
    figure_height = min(16, max(7, 2.5 * (tree_depth + 1)))

    fig4, ax4 = plt.subplots(
        figsize=(figure_width, figure_height)
    )

    plot_tree(
        clf,
        feature_names=["Col1", "Col2"],
        class_names=[str(c) for c in clf.classes_],
        filled=True,
        rounded=True,
        proportion=False,
        precision=2,
        fontsize=9,
        ax=ax4
    )

    ax4.set_title("Decision Tree Structure")
    plt.tight_layout()

    st.pyplot(fig4)
    plt.close(fig4)

    # ---------------- Hyperparameter Summary ----------------
    st.subheader("5. Selected Hyperparameters")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Criterion:** {criterion}")
        st.write(f"**Splitter:** {splitter}")
        st.write(f"**Maximum Depth:** {max_depth}")
        st.write(f"**Minimum Samples Split:** {min_samples_split}")

    with col2:
        st.write(f"**Minimum Samples Leaf:** {min_samples_leaf}")
        st.write(f"**Maximum Features:** {max_features}")
        st.write(f"**Maximum Leaf Nodes:** {max_leaf_nodes}")
        st.write(
            f"**Minimum Impurity Decrease:** "
            f"{min_impurity_decrease}"
        )

    st.write(f"**Number of Leaf Nodes:** {clf.get_n_leaves()}")
    st.write(f"**Training Samples:** {len(X_train)}")
    st.write(f"**Testing Samples:** {len(X_test)}")

    st.success("Decision Tree algorithm executed successfully!")
