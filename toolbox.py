import numpy as np
import matplotlib.pyplot as plt
import os
import inspect
import csv
from ur5e_analytic import inverse_kinematics
import numpy as np
import ast

class LowPassFilter:
    def __init__(self, cutoff_freq, dt, dim):
        self.dt = dt
        self.fc = cutoff_freq
        self.alpha = (2 * np.pi * self.fc * dt) / (1 + 2 * np.pi * self.fc * dt)
        self.y = np.zeros(dim)

    def update(self, x):
        self.y += self.alpha * (x - self.y)
        return self.y.copy()

def slice_time_window(time_vector, dt, t_start=0.5, t_end=2.5):
    """
    Returns indices and slice corresponding to [t_start, t_end].

    Parameters
    ----------
    time_vector : (N,) array
    dt : float
    t_start : float
    t_end : float

    Returns
    -------
    idx_start : int
    idx_end : int
    sl : slice
    """
    idx_start = int(np.floor(t_start / dt))
    idx_end   = int(np.ceil(t_end / dt))

    # clamp to valid bounds
    idx_start = max(idx_start, 0)
    idx_end   = min(idx_end, len(time_vector))

    return idx_start, idx_end, slice(idx_start, idx_end)

def render_manipulability_geometry(
    viewer,
    T_EE_in_WF,
    J_WF,
    linear_scalar=1.0,
    angular_scalar=1.0,
    renderer=None,
    renderer_ngeom=None,
):
    """
    Renders manipulability ellipsoid (linear) and principal axes (angular).

    Parameters
    ----------
    viewer : mujoco viewer
    T_EE_in_WF : (4,4) ndarray
        End-effector pose in world frame
    J_WF : (6,n) ndarray
        Spatial Jacobian in world frame
    linear_scalar : float
    angular_scalar : float
    renderer : mujoco.Renderer or None
        If provided, also renders into offscreen renderer
    renderer_ngeom : int or None
        Starting geom index for renderer scene (must be managed outside)
    """

    def add_geoms(scene, alpha):
        pos = T_EE_in_WF[0:3, -1]
        Jv = J_WF[0:3, :]
        Jw = J_WF[3:6, :]

        # --- Linear manipulability ellipsoid ---
        Uv, Sv, _ = np.linalg.svd(Jv)
        if np.linalg.det(Uv) < 0:
            Uv[:, -1] *= -1

        g = scene.geoms[scene.ngeom]
        scene.ngeom += 1

        g.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
        g.pos[:] = pos
        g.mat[:] = Uv
        g.size[:] = Sv * linear_scalar / (np.max(Sv) + 1e-6)
        g.rgba[:] = np.array([0, 0, 1, alpha])

        # --- Angular manipulability axes ---
        Uw, Sw, _ = np.linalg.svd(Jw)
        if np.linalg.det(Uw) < 0:
            Uw[:, -1] *= -1

        for k in range(3):
            g = scene.geoms[scene.ngeom]
            scene.ngeom += 1

            g.type = mujoco.mjtGeom.mjGEOM_ARROW
            g.pos[:] = pos
            g.size[0] = 0.005
            g.size[1] = 0.01
            g.size[2] = Sw[k] * angular_scalar / (np.max(Sw) + 1e-6)
            g.rgba[:] = np.array([1, 0, 0, alpha])

            z_axis = Uw[:, k]

            # Build orthonormal frame
            x_temp = np.array([1, 0, 0])
            if abs(np.dot(x_temp, z_axis)) > 0.9:
                x_temp = np.array([0, 1, 0])

            y_axis = np.cross(z_axis, x_temp)
            y_axis /= np.linalg.norm(y_axis)

            x_axis = np.cross(y_axis, z_axis)

            g.mat[:] = np.column_stack((x_axis, y_axis, z_axis))

    # --- Viewer ---
    viewer.user_scn.ngeom = 0
    add_geoms(viewer.user_scn, alpha=0.2)

    # --- Renderer (optional) ---
    if renderer is not None and renderer_ngeom is not None:
        scene = renderer.scene
        scene.ngeom = renderer_ngeom
        add_geoms(scene, alpha=0.9)

def sample_params(n_samples=1, seed=None):
    rng = np.random.default_rng(seed)

    samples = []

    for _ in range(n_samples):
        # --- gains ---
        kp = rng.uniform(200, 700)          # stiffness
        ki = rng.uniform(0, 50)            # integral (often small or zero)
        linear_d = rng.uniform(5, 30)     # damping

        # --- scaling ---
        ratio = rng.uniform(0.5, 2)
        final_link_ratio = rng.uniform(0.5, 2)

        # --- filtering ---
        filter_vref = rng.choice([True, False])
        filter_cutoff = rng.choice([1, 5, 10, 50])  # Hz

        # --- solref ---
        solref = np.array([
            rng.uniform(0.005, 0.03),   # time constant (smaller = stiffer)
            rng.uniform(0.5, 2.0)      # damping ratio
        ])

        # --- solimp ---
        imp0 = rng.uniform(0.01, 0.9)
        imp1 = rng.uniform(0.901, 0.99)  # ensure imp1 > imp0

        solimp = np.array([
            imp0,
            imp1,
            rng.uniform(1e-4, 1e-2),   # width
            rng.uniform(0.01, 0.9),     # midpoint
            rng.choice([1, 2])      # power
        ])

        samples.append({
            "kp": kp,
            "ki": ki,
            "linear_d": linear_d,
            "ratio": ratio,
            "final_link_ratio": final_link_ratio,
            "filter_vref": filter_vref,
            "filter_cutoff": filter_cutoff,
            "solref": solref,
            "solimp": solimp
        })

    return samples

def plot_control_trajectories(u, t=None, sharex=True, title="", save_path = None):
    """
    Plot control trajectories in stacked subplots.

    Parameters
    ----------
    u : (n_steps, ndim) array_like
        Control trajectories
    t : (n_steps,) array_like or None
        Time vector. If None, uses indices.
    sharex : bool
        Share x-axis across subplots
    """
    u = np.asarray(u)
    n_steps, ndim = u.shape

    xlab = ""
    if t is None:
        t = np.arange(n_steps)
        xlab = "(s)"

    fig, axes = plt.subplots(ndim, 1, sharex=sharex, figsize=(8, 2.5 * ndim))

    if ndim == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t, u[:, i])
        ax.set_ylabel(f"u[{i}]")
        ax.grid(True)

    axes[-1].set_xlabel(f"time {xlab}")
    plt.suptitle(title)

    plt.tight_layout()
    if save_path is not None: plt.savefig(save_path)
    else: plt.show(block=False)

def load_settings():
    import json
    import numpy as np

    with open("config.json", "r") as f:
        config = json.load(f)

    config["solref"] = np.array(config["solref"])
    config["solimp"] = np.array(config["solimp"])
    config["wrench_applied"] = np.array(config["wrench_applied"])

    globals().update(config)

def plot_force_torque_dict(
    vec_dict,
    time=None,
    quantity_name="Force",
    component_labels=("Fx", "Fy", "Fz"),
    units="N",
    yscale=None,
    fig_title=None,
    save_path=None,
    show=False,
):
    """
    Compare multiple force/torque vectors stored in a dictionary.

    Layout:
        n x 1 subplots, where
        n = 3 components + 1 magnitude.
        Each subplot overlays all dictionary entries with a legend.

    Parameters
    ----------
    vec_dict : dict[str, (T, 3) array_like]
        Dictionary mapping condition name -> (T, 3) vector.
    time : (T,) array_like or None
        Time vector. If None, sample indices are used.
    quantity_name : str
        Name of the quantity (e.g., "Force", "Torque").
    component_labels : tuple of str
        Labels for the x, y, z components.
    units : str
        Units for labeling axes.
    fig_title : str or None
        Figure title.
    save_path : str or None
        If provided, path to save the figure.
    """

    # -------------------------
    # Validate inputs
    # -------------------------
    assert isinstance(vec_dict, dict) and len(vec_dict) > 0, "vec_dict must be a non-empty dict"

    names = list(vec_dict.keys())
    vecs = {k: np.asarray(v) for k, v in vec_dict.items()}

    T = next(iter(vecs.values())).shape[0]
    for v in vecs.values():
        assert v.ndim == 2 and v.shape[1] == 3, "Each vec must be (T, 3)"
        assert v.shape[0] == T, "All vecs must have the same length"

    if time is None:
        time = np.arange(T)
    else:
        time = np.asarray(time)
        assert time.shape[0] == T, "time must match vec length"

    # -------------------------
    # Figure: n x 1 subplots
    # -------------------------
    n_rows = 4  # Fx, Fy, Fz, |F|
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(10, 2.5 * n_rows),
        sharex=True,
    )

    if fig_title is None:
        fig_title = f"{quantity_name} Component Comparison"
    fig.suptitle(fig_title, y=0.98)

    # -------------------------
    # Plot components
    # -------------------------
    for i, label in enumerate(component_labels):
        ax = axes[i]
        for name in names:
            ax.plot(time, vecs[name][:, i], label=name)
        ax.set_ylabel(f"{label} [{units}]")
        ax.grid(True)
        if yscale is not None: ax.set_yscale(yscale)

    # -------------------------
    # Plot magnitude
    # -------------------------
    ax_mag = axes[-1]
    for name in names:
        mag = np.linalg.norm(vecs[name], axis=1)
        ax_mag.plot(time, mag, label=name)

    ax_mag.set_ylabel(f"|{quantity_name}| [{units}]")
    ax_mag.set_xlabel("Time")
    ax_mag.grid(True)

    # -------------------------
    # Legend (single, shared)
    # -------------------------
    axes[0].legend(loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        plt.savefig(save_path)

    if show: plt.show(block=False)

def save_dict_of_vectors(data: dict, out_dir: str):
    """
    Save dict[str, list[np.ndarray]] to CSV files (one per key).

    Each row is one 1D vector. CSVs are readable without Python.
    """

    for key, vectors in data.items():
        if len(vectors) == 0:
            continue

        # Ensure all are 1D numpy arrays
        vectors = [np.asarray(v).ravel() for v in vectors]

        max_len = max(len(v) for v in vectors)
        filepath = os.path.join(out_dir, f"{key}.csv")

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"dim_{i}" for i in range(max_len)])

            for v in vectors:
                row = list(v) + [""] * (max_len - len(v))
                writer.writerow(row)

def load_dict_of_vectors(in_dir: str):
    """
    Load dict[str, list[np.ndarray]] from CSV files produced above.
    """
    data = {}

    for fname in os.listdir(in_dir):
        if not fname.endswith(".csv"):
            continue

        key = fname[:-4]
        filepath = os.path.join(in_dir, fname)

        vectors = []
        with open(filepath, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)  # discard header

            for row in reader:
                # Remove empty padding cells
                vals = [float(x) for x in row if x != ""]
                vectors.append(np.array(vals))

        data[key] = vectors

    return data 

def get_value_from_dict(in_dir: str,key: str):
    """
    Load dict[str, list[np.ndarray]] from CSV files produced above.
    """
    data = load_dict_of_vectors(in_dir=in_dir)
    return np.array(data[key])
  
def dump_globals_to_txt(filename="globals_dump.txt"):
    """
    Write all global variables from the caller's global workspace to a text file.
    """
    # Get caller's globals
    caller_globals = inspect.currentframe().f_back.f_globals

    with open(filename, "w") as f:
        for name, value in sorted(caller_globals.items()):
            # Skip private names
            if name.startswith("__"):
                continue

            # Skip modules and functions
            if inspect.ismodule(value) or inspect.isfunction(value) or inspect.isclass(value):
                continue

            f.write(f"{name} = ")

            try:
                f.write(repr(value))
            except Exception as e:
                f.write(f"<unrepresentable: {e}>")

            f.write("\n\n")

def read_python_variable(file_path, key):
    """
    Read the value of a top-level variable assignment from a Python file
    without executing the file.

    Parameters
    ----------
    file_path : str
        Path to the Python file.
    key : str
        Variable name to search for.

    Returns
    -------
    Any
        Parsed Python value.

    Raises
    ------
    KeyError
        If the variable is not found.
    ValueError
        If the value cannot be safely parsed.
    """

    with open(file_path, "r") as f:
        source = f.read()

    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == key:
                    try:
                        return ast.literal_eval(node.value)
                    except Exception as e:
                        raise ValueError(
                            f"Could not safely evaluate variable '{key}'"
                        ) from e

    raise KeyError(f"Variable '{key}' not found")