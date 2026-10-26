import fnmatch
import io
import re
from collections.abc import Callable

import numpy as np
import torch
from jaxtyping import Float
from matplotlib import pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from torch import Tensor

from spd.configs import SamplingType
from spd.models.component_model import CIOutputs, ComponentModel
from spd.models.components import Components
from spd.utils.general_utils import get_obj_device
from spd.utils.target_ci_solutions import permute_to_dense, permute_to_identity


def _render_figure(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    buf.close()
    return img


_LAYER_NAME_PATTERN = re.compile(r"^.+?\.(\d+)\.(.+)$")


def _parse_layer_grid(
    module_names: list[str],
) -> tuple[list[str], list[int], dict[str, tuple[int, int]]] | None:
    """Parse module names into a 2D grid of (matrix_type, layer_index).

    Returns (matrix_types, layer_indices, name_to_pos) where name_to_pos maps
    module name -> (row, col), or None if names don't follow the layer pattern.
    """
    parsed: list[tuple[str, int, str]] = []  # (name, layer_idx, matrix_type)
    for name in module_names:
        m = _LAYER_NAME_PATTERN.match(name)
        if m is None:
            return None
        parsed.append((name, int(m.group(1)), m.group(2)))

    matrix_types: list[str] = list(dict.fromkeys(mt for _, _, mt in parsed))
    layer_indices: list[int] = sorted(set(li for _, li, _ in parsed))

    mt_to_row = {mt: i for i, mt in enumerate(matrix_types)}
    li_to_col = {li: i for i, li in enumerate(layer_indices)}

    name_to_pos = {name: (mt_to_row[mt], li_to_col[li]) for name, li, mt in parsed}
    return matrix_types, layer_indices, name_to_pos


def _setup_layer_grid_labels(
    axs: np.ndarray,
    matrix_types: list[str],
    layer_indices: list[int],
) -> None:
    """Add row/column labels to a layer grid and remove individual subplot titles."""
    for col, layer_idx in enumerate(layer_indices):
        axs[0, col].set_title(f"Layer {layer_idx}", fontsize=10)
    for row, matrix_type in enumerate(matrix_types):
        axs[row, 0].set_ylabel(matrix_type, fontsize=10)


def _plot_causal_importances_figure(
    ci_vals: dict[str, Float[Tensor, "... C"]],
    title_prefix: str,
    colormap: str,
    input_magnitude: float,
    has_pos_dim: bool,
    title_formatter: Callable[[str], str] | None = None,
) -> Image.Image:
    """Plot causal importances for components stacked vertically.

    Args:
        ci_vals: Dictionary of causal importances (or causal importances upper leaky relu) to plot
        title_prefix: String to prepend to the title (e.g., "causal importances" or
            "causal importances upper leaky relu")
        colormap: Matplotlib colormap name
        input_magnitude: Input magnitude value for the title
        has_pos_dim: Whether the masks have a position dimension
        title_formatter: Optional callable to format subplot titles. Takes mask_name as input.

    Returns:
        The matplotlib figure
    """
    figsize = (5, 5 * len(ci_vals))
    fig, axs = plt.subplots(
        len(ci_vals),
        1,
        figsize=figsize,
        constrained_layout=True,
        squeeze=False,
        dpi=300,
    )
    axs = np.array(axs)

    images = []
    for j, (mask_name, mask) in enumerate(ci_vals.items()):
        # mask has shape (batch, C) or (batch, pos, C)
        mask_data = mask.detach().cpu().numpy()
        if has_pos_dim:
            assert mask_data.ndim == 3
            mask_data = mask_data[:, 0, :]
        ax = axs[j, 0]
        im = ax.matshow(mask_data, aspect="auto", cmap=colormap)
        images.append(im)

        # Move x-axis ticks to bottom
        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position("bottom")
        ax.set_xlabel("Subcomponent index")
        ax.set_ylabel("Input feature index")

        # Apply custom title formatting if provided
        title = title_formatter(mask_name) if title_formatter is not None else mask_name
        ax.set_title(title)

    # Add unified colorbar
    norm = plt.Normalize(
        vmin=min(mask.min().item() for mask in ci_vals.values()),
        vmax=max(mask.max().item() for mask in ci_vals.values()),
    )
    for im in images:
        im.set_norm(norm)
    fig.colorbar(images[0], ax=axs.ravel().tolist())

    # Capitalize first letter of title prefix for the figure title
    fig.suptitle(f"{title_prefix.capitalize()} - Input magnitude: {input_magnitude}")

    img = _render_figure(fig)
    plt.close(fig)

    return img


def plot_mean_component_cis_both_scales(
    mean_component_cis: dict[str, Float[Tensor, " C"]],
) -> tuple[Image.Image, Image.Image]:
    """
    Efficiently plot mean CI per component with both linear and log scales.

    This function optimizes the plotting by pre-processing data once and
    reusing it for both plots.

    Args:
        mean_component_cis: Dictionary mapping module names to mean CI tensors

    Returns:
        Tuple of (linear_scale_image, log_scale_image)
    """
    n_modules = len(mean_component_cis)
    grid = _parse_layer_grid(list(mean_component_cis.keys()))

    if grid is not None:
        n_rows, n_cols = len(grid[0]), len(grid[1])
        name_to_pos = grid[2]
    else:
        max_rows = 6
        n_cols = (n_modules + max_rows - 1) // max_rows
        n_rows = min(n_modules, max_rows)
        name_to_pos = None

    fig_width = 8 * n_cols
    fig_height = 3 * n_rows

    processed_data = []
    for module_name, mean_component_ci in mean_component_cis.items():
        sorted_components = torch.sort(mean_component_ci, descending=True)[0]
        processed_data.append((module_name, sorted_components.detach().cpu().numpy()))

    images = []
    for log_y in [False, True]:
        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fig_width, fig_height),
            dpi=200,
            squeeze=False,
        )
        axs = np.array(axs)
        if axs.ndim == 1:
            axs = axs.reshape(n_rows, n_cols)

        for i, (module_name, sorted_components_np) in enumerate(processed_data):
            if name_to_pos is not None:
                row, col = name_to_pos[module_name]
            else:
                row, col = i % n_rows, i // n_rows
            ax = axs[row, col]

            if log_y:
                ax.set_yscale("log")

            ax.scatter(
                range(len(sorted_components_np)),
                sorted_components_np,
                marker="x",
                s=10,
            )

            if row == n_rows - 1:
                ax.set_xlabel("Component")
            if name_to_pos is None:
                ax.set_ylabel("mean CI")
                ax.set_title(module_name, fontsize=10)

        if grid is not None:
            _setup_layer_grid_labels(axs, grid[0], grid[1])

        fig.tight_layout()
        img = _render_figure(fig)
        plt.close(fig)
        images.append(img)

    return images[0], images[1]


def get_single_feature_causal_importances(
    model: ComponentModel,
    batch_shape: tuple[int, ...],
    input_magnitude: float,
    sampling: SamplingType,
) -> CIOutputs:
    """Compute causal importance arrays for single active features.

    Args:
        model: The ComponentModel
        batch_shape: Shape of the batch
        input_magnitude: Magnitude of input features

    Returns:
        Tuple of (ci_raw, ci_upper_leaky_raw) dictionaries of causal importance arrays (2D tensors)
    """
    device = get_obj_device(model)
    # Create a batch of inputs with single active features
    has_pos_dim = len(batch_shape) == 3
    n_features = batch_shape[-1]
    batch = torch.eye(n_features, device=device) * input_magnitude
    if has_pos_dim:
        # NOTE: For now, we only use the first pos dim
        batch = batch.unsqueeze(1)

    pre_weight_acts = model(batch, cache_type="input").cache

    return model.calc_causal_importances(
        pre_weight_acts=pre_weight_acts,
        detach_inputs=False,
        sampling=sampling,
    )


def plot_causal_importance_vals(
    model: ComponentModel,
    batch_shape: tuple[int, ...],
    input_magnitude: float,
    sampling: SamplingType,
    identity_patterns: list[str] | None = None,
    dense_patterns: list[str] | None = None,
    plot_raw_cis: bool = True,
    title_formatter: Callable[[str], str] | None = None,
) -> tuple[dict[str, Image.Image], dict[str, Float[Tensor, " C"]]]:
    """Plot the values of the causal importances for a batch of inputs with single active features.

    Args:
        model: The ComponentModel
        batch_shape: Shape of the batch
        input_magnitude: Magnitude of input features
        sampling: Sampling method to use
        plot_raw_cis: Whether to plot the raw causal importances (blue plots)
        title_formatter: Optional callable to format subplot titles. Takes mask_name as input.
        identity_patterns: List of patterns to match for identity permutation
        dense_patterns: List of patterns to match for dense permutation
        plot_raw_cis: Whether to plot the raw causal importances (blue plots)
        title_formatter: Optional callable to format subplot titles. Takes mask_name as input.

    Returns:
        Tuple of:
            - Dictionary of figures with keys 'causal_importances' (if plot_raw_cis=True) and 'causal_importances_upper_leaky'
            - Dictionary of permutation indices for causal importances
    """
    # Get the causal importance arrays
    ci_output = get_single_feature_causal_importances(
        model=model,
        batch_shape=batch_shape,
        input_magnitude=input_magnitude,
        sampling=sampling,
    )

    ci: dict[str, Float[Tensor, "... C"]] = {}
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]] = {}
    all_perm_indices: dict[str, Float[Tensor, " C"]] = {}
    for k in ci_output.lower_leaky:
        # Determine permutation strategy based on patterns
        if identity_patterns and any(fnmatch.fnmatch(k, pattern) for pattern in identity_patterns):
            ci[k], _ = permute_to_identity(ci_vals=ci_output.lower_leaky[k])
            ci_upper_leaky[k], all_perm_indices[k] = permute_to_identity(
                ci_vals=ci_output.upper_leaky[k]
            )
        elif dense_patterns and any(fnmatch.fnmatch(k, pattern) for pattern in dense_patterns):
            ci[k], _ = permute_to_dense(ci_vals=ci_output.lower_leaky[k])
            ci_upper_leaky[k], all_perm_indices[k] = permute_to_dense(
                ci_vals=ci_output.upper_leaky[k]
            )
        else:
            # Default: identity permutation
            ci[k], _ = permute_to_identity(ci_vals=ci_output.lower_leaky[k])
            ci_upper_leaky[k], all_perm_indices[k] = permute_to_identity(
                ci_vals=ci_output.upper_leaky[k]
            )

    # Create figures dictionary
    figures: dict[str, Image.Image] = {}

    # TODO: Need to handle this differently for e.g. convolutional tasks
    has_pos_dim = len(batch_shape) == 3
    if plot_raw_cis:
        ci_fig = _plot_causal_importances_figure(
            ci_vals=ci,
            title_prefix="importance values lower leaky relu",
            colormap="Blues",
            input_magnitude=input_magnitude,
            has_pos_dim=has_pos_dim,
            title_formatter=title_formatter,
        )
        figures["causal_importances"] = ci_fig

    ci_upper_leaky_fig = _plot_causal_importances_figure(
        ci_vals=ci_upper_leaky,
        title_prefix="importance values",
        colormap="Reds",
        input_magnitude=input_magnitude,
        has_pos_dim=has_pos_dim,
        title_formatter=title_formatter,
    )
    figures["causal_importances_upper_leaky"] = ci_upper_leaky_fig

    return figures, all_perm_indices


def plot_UV_matrices(
    components: dict[str, Components],
    all_perm_indices: dict[str, Float[Tensor, " C"]] | None = None,
) -> Image.Image:
    """Plot V and U matrices for each instance, grouped by layer."""
    n_layers = len(components)

    # Create figure for plotting - 2 rows per layer (V and U)
    fig, axs = plt.subplots(
        n_layers,
        2,  # U, V
        figsize=(5 * 2, 5 * n_layers),
        constrained_layout=True,
        squeeze=False,
    )
    axs = np.array(axs)

    images = []

    # Plot V and U matrices for each layer
    for j, (name, component) in enumerate(sorted(components.items())):
        # Plot V matrix
        V = component.V if all_perm_indices is None else component.V[:, all_perm_indices[name]]
        V_np = V.detach().cpu().numpy()
        im = axs[j, 0].matshow(V_np, aspect="auto", cmap="coolwarm")
        axs[j, 0].set_ylabel("d_in index")
        axs[j, 0].set_xlabel("Component index")
        axs[j, 0].set_title(f"{name} (V matrix)")
        images.append(im)

        # Plot U matrix
        U = component.U if all_perm_indices is None else component.U[all_perm_indices[name], :]
        U_np = U.detach().cpu().numpy()
        im = axs[j, 1].matshow(U_np, aspect="auto", cmap="coolwarm")
        axs[j, 1].set_ylabel("Component index")
        axs[j, 1].set_xlabel("d_out index")
        axs[j, 1].set_title(f"{name} (U matrix)")
        images.append(im)

    # Add unified colorbar
    all_matrices = [c.V for c in components.values()] + [c.U for c in components.values()]
    norm = plt.Normalize(
        vmin=min(m.min().item() for m in all_matrices),
        vmax=max(m.max().item() for m in all_matrices),
    )
    for im in images:
        im.set_norm(norm)
    fig.colorbar(images[0], ax=axs.ravel().tolist())

    fig_img = _render_figure(fig)
    plt.close(fig)

    return fig_img


def plot_component_activation_density(
    component_activation_density: dict[str, Float[Tensor, " C"]],
    bins: int = 100,
) -> Image.Image:
    """Plot the activation density of each component as a histogram in a grid layout."""

    n_modules = len(component_activation_density)
    grid = _parse_layer_grid(list(component_activation_density.keys()))

    if grid is not None:
        n_rows, n_cols = len(grid[0]), len(grid[1])
        name_to_pos = grid[2]
    else:
        max_rows = 6
        n_cols = (n_modules + max_rows - 1) // max_rows
        n_rows = min(n_modules, max_rows)
        name_to_pos = None

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 5 * n_rows),
        squeeze=False,
    )
    axs = np.array(axs)
    if axs.ndim == 1:
        axs = axs.reshape(n_rows, n_cols)

    for i, (module_name, density) in enumerate(component_activation_density.items()):
        if name_to_pos is not None:
            row, col = name_to_pos[module_name]
        else:
            row, col = i % n_rows, i // n_rows
        ax = axs[row, col]

        data = density.detach().cpu().numpy()
        ax.hist(data, bins=bins)
        ax.set_yscale("log")
        if name_to_pos is None:
            ax.set_title(module_name)

        if row == n_rows - 1:
            ax.set_xlabel("Activation density")
        if name_to_pos is None:
            ax.set_ylabel("Frequency")

    if grid is not None:
        _setup_layer_grid_labels(axs, grid[0], grid[1])

    fig.tight_layout()

    fig_img = _render_figure(fig)
    plt.close(fig)

    return fig_img


def plot_targeted_ci_heatmaps(
    target_cis: dict[str, Float[Tensor, "... C"]],
    nontarget_cis: dict[str, Float[Tensor, "... C"]],
    n_nontarget_examples: int,
    target_labels: list[str] | None = None,
    nontarget_labels: list[str] | None = None,
) -> Image.Image:
    """Plot CI heatmaps comparing target vs nontarget data.

    Layout: 2 rows (target/nontarget) x N columns (modules).
    Each subplot shows a heatmap with X=subcomponents, Y=inputs.
    For LMs, batch * seq_pos is flattened into the Y-axis.
    """
    module_names = list(target_cis.keys())
    n_modules = len(module_names)

    first_module = module_names[0]
    n_target_rows = (
        target_cis[first_module].reshape(-1, target_cis[first_module].shape[-1]).shape[0]
    )
    n_nontarget_rows = min(
        nontarget_cis[first_module].reshape(-1, nontarget_cis[first_module].shape[-1]).shape[0],
        n_nontarget_examples,
    )
    row_height = max(0.15, min(0.5, 8 / (n_target_rows + n_nontarget_rows)))
    fig_height = max(6, (n_target_rows + n_nontarget_rows) * row_height + 2)

    fig, axs = plt.subplots(
        2,
        n_modules,
        figsize=(4 * n_modules, fig_height),
        constrained_layout=True,
        squeeze=False,
    )

    all_vals = [target_cis[n] for n in module_names] + [nontarget_cis[n] for n in module_names]
    vmin = min(v.min().item() for v in all_vals)
    vmax = max(v.max().item() for v in all_vals)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    nontarget_sample_indices: torch.Tensor | None = None

    images = []
    for col_idx, module_name in enumerate(module_names):
        target_2d = target_cis[module_name].reshape(-1, target_cis[module_name].shape[-1])
        nontarget_2d = nontarget_cis[module_name].reshape(-1, nontarget_cis[module_name].shape[-1])

        if nontarget_2d.shape[0] > n_nontarget_examples:
            if nontarget_sample_indices is None:
                nontarget_sample_indices = torch.arange(n_nontarget_examples)
            nontarget_2d = nontarget_2d[nontarget_sample_indices]

        # Target row
        im = axs[0, col_idx].imshow(target_2d.numpy(), aspect="auto", cmap="viridis", norm=norm)
        images.append(im)
        axs[0, col_idx].set_title(f"{module_name}\n(Target)")
        axs[0, col_idx].set_xlabel("Subcomponent")
        if col_idx == 0:
            axs[0, col_idx].set_ylabel("Input")
            if target_labels is not None and len(target_labels) == target_2d.shape[0]:
                axs[0, col_idx].set_yticks(range(len(target_labels)))
                axs[0, col_idx].set_yticklabels(
                    target_labels,
                    fontsize=max(6, 10 - len(target_labels) // 10),
                    parse_math=False,
                )

        # Nontarget row
        im = axs[1, col_idx].imshow(nontarget_2d.numpy(), aspect="auto", cmap="viridis", norm=norm)
        images.append(im)
        axs[1, col_idx].set_title(f"{module_name}\n(Nontarget)")
        axs[1, col_idx].set_xlabel("Subcomponent")
        if col_idx == 0:
            axs[1, col_idx].set_ylabel("Input")
            sampled_nontarget_labels = nontarget_labels
            if nontarget_labels is not None and nontarget_sample_indices is not None:
                sampled_nontarget_labels = [
                    nontarget_labels[i] for i in nontarget_sample_indices.tolist()
                ]
            if (
                sampled_nontarget_labels is not None
                and len(sampled_nontarget_labels) == nontarget_2d.shape[0]
            ):
                axs[1, col_idx].set_yticks(range(len(sampled_nontarget_labels)))
                axs[1, col_idx].set_yticklabels(
                    sampled_nontarget_labels,
                    fontsize=max(6, 10 - len(sampled_nontarget_labels) // 10),
                    parse_math=False,
                )

    fig.colorbar(images[0], ax=axs.ravel().tolist(), shrink=0.8)
    fig.suptitle("Causal Importances: Target vs Nontarget")

    img = _render_figure(fig)
    plt.close(fig)
    return img


def plot_ci_values_histograms(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    bins: int = 100,
) -> Image.Image:
    """Plot histograms of mask values for all layers in a grid layout.

    Args:
        causal_importances: Dictionary of causal importances for each component.
        bins: Number of bins for the histogram.

    Returns:
        Single figure with subplots for each layer.
    """
    assert len(causal_importances) > 0, "No causal importances to plot"
    n_layers = len(causal_importances)
    grid = _parse_layer_grid(list(causal_importances.keys()))

    if grid is not None:
        n_rows, n_cols = len(grid[0]), len(grid[1])
        name_to_pos = grid[2]
    else:
        max_rows = 6
        n_cols = (n_layers + max_rows - 1) // max_rows
        n_rows = min(n_layers, max_rows)
        name_to_pos = None

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 5 * n_rows),
        squeeze=False,
    )
    axs = np.array(axs)
    if axs.ndim == 1:
        axs = axs.reshape(n_rows, n_cols)

    for i, (layer_name_raw, layer_ci) in enumerate(causal_importances.items()):
        if name_to_pos is not None:
            row, col = name_to_pos[layer_name_raw]
        else:
            row, col = i % n_rows, i // n_rows
        ax = axs[row, col]

        data = layer_ci.flatten().cpu().numpy()
        ax.hist(data, bins=bins)
        ax.set_yscale("log")
        if name_to_pos is None:
            layer_name = layer_name_raw.replace(".", "_")
            ax.set_title(f"Causal importances for {layer_name}")

        if row == n_rows - 1:
            ax.set_xlabel("Causal importance value")
        if name_to_pos is None:
            ax.set_ylabel("Frequency")

    if grid is not None:
        _setup_layer_grid_labels(axs, grid[0], grid[1])

    fig.tight_layout()

    fig_img = _render_figure(fig)
    plt.close(fig)

    return fig_img


def plot_weight_magnitude(
    weight_magnitudes: dict[str, Tensor],
    max_cis: dict[str, Tensor],
    mean_cis: dict[str, Tensor],
) -> Image.Image:
    """Plot weight magnitude per component, sorted by mean CI and colored by max CI.

    Args:
        weight_magnitudes: ||V|| * ||U|| per component, keyed by layer name (y-axis).
        max_cis: Max CI over target inputs per component, keyed by layer name (color).
        mean_cis: Mean CI over target inputs per component, keyed by layer name (sorting).
    """
    assert len(weight_magnitudes) > 0, "No weight magnitude data to plot"
    n_modules = len(weight_magnitudes)
    grid = _parse_layer_grid(list(weight_magnitudes.keys()))

    if grid is not None:
        n_rows, n_cols = len(grid[0]), len(grid[1])
        name_to_pos = grid[2]
    else:
        max_rows = 6
        n_cols = (n_modules + max_rows - 1) // max_rows
        n_rows = min(n_modules, max_rows)
        name_to_pos = None

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8 * n_cols, 3 * n_rows),
        squeeze=False,
        layout="constrained",
    )
    axs = np.array(axs)
    if axs.ndim == 1:
        axs = axs.reshape(n_rows, n_cols)

    all_max_cis = torch.cat(list(max_cis.values()))
    vmin, vmax = all_max_cis.min().item(), all_max_cis.max().item()

    scatter_plots = []
    for i, layer_name in enumerate(weight_magnitudes.keys()):
        if name_to_pos is not None:
            row, col = name_to_pos[layer_name]
        else:
            row, col = i % n_rows, i // n_rows
        ax = axs[row, col]

        sort_indices = torch.argsort(mean_cis[layer_name], descending=True)
        sorted_weight_mags = weight_magnitudes[layer_name][sort_indices].cpu().numpy()
        sorted_max_cis = max_cis[layer_name][sort_indices].cpu().numpy()

        scatter = ax.scatter(
            range(len(sorted_weight_mags)),
            sorted_weight_mags,
            c=sorted_max_cis,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=20,
        )
        scatter_plots.append(scatter)

        if name_to_pos is None:
            ax.set_title(layer_name, fontsize=10)
        if row == n_rows - 1:
            ax.set_xlabel("Component")
        if name_to_pos is None:
            ax.set_ylabel("Weight magnitude")

    if grid is not None:
        _setup_layer_grid_labels(axs, grid[0], grid[1])

    fig.colorbar(scatter_plots[0], ax=axs.ravel().tolist(), shrink=0.8, label="Max CI")

    fig_img = _render_figure(fig)
    plt.close(fig)

    return fig_img
