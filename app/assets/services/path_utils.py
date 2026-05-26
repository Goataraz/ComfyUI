import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import folder_paths
from app.assets.helpers import normalize_tags


# These names are bootstrapped into folder_names_and_paths by core but are not
# model folders (matching /api/experiment/models' exclusion). Intentionally
# duplicated here so the assets layer stays decoupled from the legacy
# model-manager code it will eventually replace.
_NON_MODEL_FOLDER_NAMES = frozenset({"configs", "custom_nodes"})


@dataclass(frozen=True)
class AssetPathInfo:
    asset_type: Literal["input", "output", "temp", "model"]
    model_folder: str | None


@dataclass(frozen=True)
class AssetResponsePathInfo(AssetPathInfo):
    file_path: str
    display_name: str | None


@dataclass(frozen=True)
class AssetPathContext(AssetPathInfo):
    base_path: str
    relative_path: str


def get_comfy_models_folders() -> list[tuple[str, list[str]]]:
    """Build list of (folder_name, base_paths[]) for all model locations.

    Includes every category registered in folder_names_and_paths,
    regardless of whether its paths are under the main models_dir,
    but excludes non-model entries like configs and custom_nodes.
    """
    targets: list[tuple[str, list[str]]] = []
    for name, values in folder_paths.folder_names_and_paths.items():
        if name in _NON_MODEL_FOLDER_NAMES:
            continue
        paths, _exts = values[0], values[1]
        if paths:
            targets.append((name, paths))
    return targets


def resolve_destination_from_tags(tags: list[str]) -> tuple[str, list[str]]:
    """Validates and maps tags -> (base_dir, subdirs_for_fs)"""
    if not tags:
        raise ValueError("tags must not be empty")
    root = tags[0].lower()
    if root == "models":
        if len(tags) < 2:
            raise ValueError("at least two tags required for model asset")
        try:
            bases = folder_paths.folder_names_and_paths[tags[1]][0]
        except KeyError:
            raise ValueError(f"unknown model category '{tags[1]}'")
        if not bases:
            raise ValueError(f"no base path configured for category '{tags[1]}'")
        base_dir = os.path.abspath(bases[0])
        raw_subdirs = tags[2:]
    elif root == "input":
        base_dir = os.path.abspath(folder_paths.get_input_directory())
        raw_subdirs = tags[1:]
    elif root == "output":
        base_dir = os.path.abspath(folder_paths.get_output_directory())
        raw_subdirs = tags[1:]
    else:
        raise ValueError(f"unknown root tag '{tags[0]}'; expected 'models', 'input', or 'output'")
    _sep_chars = frozenset(("/", "\\", os.sep))
    for i in raw_subdirs:
        if i in (".", "..") or _sep_chars & set(i):
            raise ValueError("invalid path component in tags")

    return base_dir, raw_subdirs if raw_subdirs else []


def validate_path_within_base(candidate: str, base: str) -> None:
    cand_abs = Path(os.path.abspath(candidate))
    base_abs = Path(os.path.abspath(base))
    if not cand_abs.is_relative_to(base_abs):
        raise ValueError("destination escapes base directory")


def compute_relative_filename(file_path: str) -> str | None:
    """
    Return the path relative to the matched asset root or model folder, using
    forward slashes, eg:
      /.../models/checkpoints/flux/123/flux.safetensors -> "flux/123/flux.safetensors"
      /.../models/text_encoders/clip_g.safetensors -> "clip_g.safetensors"
      /.../input/sub/image.png -> "sub/image.png"

    For unknown paths, returns None.
    """
    try:
        context = resolve_asset_path_context(file_path)
    except ValueError:
        return None

    return _normalize_relative_path(context.relative_path)


def _normalize_relative_path(relative_path: str) -> str | None:
    parts = [
        seg
        for seg in Path(relative_path).parts
        if seg not in (".", "..", Path(relative_path).anchor)
    ]
    if not parts:
        return None

    return "/".join(parts)


def resolve_asset_path_context(file_path: str) -> AssetPathContext:
    """Resolve a path against Core's asset roots and model-folder registration.

    This is the source of truth for path-derived asset classification. For
    model assets, ``model_folder`` is the exact registered folder name whose
    base path contains the file, and ``relative_path`` is relative to that
    matched base path. When multiple registered bases contain the file, the
    deepest base wins.
    """
    fp_abs = os.path.abspath(file_path)

    def _check_is_within(child: str, parent: str) -> bool:
        return Path(child).is_relative_to(parent)

    def _compute_relative(child: str, parent: str) -> str:
        # Normalize relative path, stripping any leading ".." components
        # by anchoring to root (os.sep) then computing relpath back from it.
        return os.path.relpath(
            os.path.join(os.sep, os.path.relpath(child, parent)), os.sep
        )

    best: tuple[int, str, str, str] | None = None
    for model_folder, bases in get_comfy_models_folders():
        for base in bases:
            base_abs = os.path.abspath(base)
            if not _check_is_within(fp_abs, base_abs):
                continue
            cand = (
                len(base_abs),
                model_folder,
                base_abs,
                _compute_relative(fp_abs, base_abs),
            )
            if best is None or cand[0] > best[0]:
                best = cand

    if best is not None:
        _, model_folder, base_path, relative_path = best
        return AssetPathContext(
            asset_type="model",
            model_folder=model_folder,
            base_path=base_path,
            relative_path=relative_path,
        )

    input_base = os.path.abspath(folder_paths.get_input_directory())
    if _check_is_within(fp_abs, input_base):
        return AssetPathContext(
            asset_type="input",
            model_folder=None,
            base_path=input_base,
            relative_path=_compute_relative(fp_abs, input_base),
        )

    output_base = os.path.abspath(folder_paths.get_output_directory())
    if _check_is_within(fp_abs, output_base):
        return AssetPathContext(
            asset_type="output",
            model_folder=None,
            base_path=output_base,
            relative_path=_compute_relative(fp_abs, output_base),
        )

    temp_base = os.path.abspath(folder_paths.get_temp_directory())
    if _check_is_within(fp_abs, temp_base):
        return AssetPathContext(
            asset_type="temp",
            model_folder=None,
            base_path=temp_base,
            relative_path=_compute_relative(fp_abs, temp_base),
        )

    raise ValueError(
        f"Path is not within input, output, temp, or configured model bases: {file_path}"
    )


def get_asset_category_and_relative_path(
    file_path: str,
) -> tuple[Literal["input", "output", "temp", "models"], str]:
    """Determine which root category a file path belongs to.

    Categories:
      - 'input': under folder_paths.get_input_directory()
      - 'output': under folder_paths.get_output_directory()
      - 'temp': under folder_paths.get_temp_directory()
      - 'models': under any base path from get_comfy_models_folders()

    Returns:
        (root_category, relative_path_inside_that_root)

    Raises:
        ValueError: path does not belong to any known root.
    """
    fp_abs = os.path.abspath(file_path)

    def _check_is_within(child: str, parent: str) -> bool:
        return Path(child).is_relative_to(parent)

    def _compute_relative(child: str, parent: str) -> str:
        # Normalize relative path, stripping any leading ".." components
        # by anchoring to root (os.sep) then computing relpath back from it.
        return os.path.relpath(
            os.path.join(os.sep, os.path.relpath(child, parent)), os.sep
        )

    input_base = os.path.abspath(folder_paths.get_input_directory())
    if _check_is_within(fp_abs, input_base):
        return "input", _compute_relative(fp_abs, input_base)

    output_base = os.path.abspath(folder_paths.get_output_directory())
    if _check_is_within(fp_abs, output_base):
        return "output", _compute_relative(fp_abs, output_base)

    temp_base = os.path.abspath(folder_paths.get_temp_directory())
    if _check_is_within(fp_abs, temp_base):
        return "temp", _compute_relative(fp_abs, temp_base)

    best: tuple[int, str, str] | None = None
    for model_folder, bases in get_comfy_models_folders():
        for base in bases:
            base_abs = os.path.abspath(base)
            if not _check_is_within(fp_abs, base_abs):
                continue
            relative_path = _compute_relative(fp_abs, base_abs)
            combined = os.path.join(model_folder, relative_path)
            cand = (len(base_abs), base_abs, combined)
            if best is None or cand[0] > best[0]:
                best = cand

    if best is not None:
        return "models", os.path.relpath(os.path.join(os.sep, best[2]), os.sep)

    raise ValueError(
        f"Path is not within input, output, temp, or configured model bases: {file_path}"
    )


def get_asset_path_info(file_path: str) -> AssetPathInfo:
    """Return typed asset classification derived from the actual filesystem path.

    This intentionally reads the ComfyUI model folder registration from
    ``folder_paths.folder_names_and_paths`` instead of inferring it from tags.
    For model files, ``model_folder`` is the registered folder name whose base
    path contains ``file_path``.

    Raises:
        ValueError: path does not belong to any known root.
    """
    context = resolve_asset_path_context(file_path)
    return AssetPathInfo(
        asset_type=context.asset_type,
        model_folder=context.model_folder,
    )


def get_asset_response_path_info(file_path: str) -> AssetResponsePathInfo:
    """Return API-facing path fields derived from the actual filesystem path.

    ``file_path`` is a logical namespace key: ``models/<model_folder>/<relative>``
    for model assets and ``<asset_type>/<relative>`` for input/output/temp assets.
    ``display_name`` is the path below the matched root or model folder.

    Raises:
        ValueError: path does not belong to any known root.
    """
    context = resolve_asset_path_context(file_path)
    display_name = _normalize_relative_path(context.relative_path)

    if context.asset_type == "model":
        logical_file_path = (
            f"models/{context.model_folder}/{display_name}"
            if display_name
            else f"models/{context.model_folder}"
        )
    else:
        logical_file_path = (
            f"{context.asset_type}/{display_name}"
            if display_name
            else context.asset_type
        )

    return AssetResponsePathInfo(
        asset_type=context.asset_type,
        model_folder=context.model_folder,
        file_path=logical_file_path,
        display_name=display_name,
    )


def get_stored_asset_response_path_info(
    file_path: str,
    asset_type: str | None,
    model_folder: str | None,
) -> AssetResponsePathInfo:
    """Return API-facing path fields from persisted classification.

    ``asset_type`` and ``model_folder`` are written at ingest time and are the
    classification source of truth for API responses. The physical ``file_path``
    is still used to compute the display path below the stored root.
    """
    if asset_type not in {"input", "output", "temp", "model"}:
        raise ValueError(f"unknown persisted asset_type: {asset_type}")

    fp_abs = os.path.abspath(file_path)

    def _check_is_within(child: str, parent: str) -> bool:
        return Path(child).is_relative_to(parent)

    def _compute_relative(child: str, parent: str) -> str:
        return os.path.relpath(
            os.path.join(os.sep, os.path.relpath(child, parent)), os.sep
        )

    if asset_type == "model":
        if not model_folder:
            raise ValueError("model asset is missing persisted model_folder")
        best: tuple[int, str] | None = None
        for folder_name, bases in get_comfy_models_folders():
            if folder_name != model_folder:
                continue
            for base in bases:
                base_abs = os.path.abspath(base)
                if not _check_is_within(fp_abs, base_abs):
                    continue
                relative_path = _compute_relative(fp_abs, base_abs)
                cand = (len(base_abs), relative_path)
                if best is None or cand[0] > best[0]:
                    best = cand
        if best is None:
            raise ValueError(
                f"Path is not within persisted model folder roots: {file_path}"
            )
        display_name = _normalize_relative_path(best[1])
        logical_file_path = (
            f"models/{model_folder}/{display_name}"
            if display_name
            else f"models/{model_folder}"
        )
        return AssetResponsePathInfo(
            asset_type="model",
            model_folder=model_folder,
            file_path=logical_file_path,
            display_name=display_name,
        )

    root_by_type = {
        "input": folder_paths.get_input_directory,
        "output": folder_paths.get_output_directory,
        "temp": folder_paths.get_temp_directory,
    }
    root = os.path.abspath(root_by_type[asset_type]())
    if not _check_is_within(fp_abs, root):
        raise ValueError(f"Path is not within persisted asset root: {file_path}")
    display_name = _normalize_relative_path(_compute_relative(fp_abs, root))
    logical_file_path = f"{asset_type}/{display_name}" if display_name else asset_type
    return AssetResponsePathInfo(
        asset_type=asset_type,
        model_folder=None,
        file_path=logical_file_path,
        display_name=display_name,
    )


def get_name_and_tags_from_asset_path(file_path: str) -> tuple[str, list[str]]:
    """Return (name, tags) derived from a filesystem path.

    - name: base filename with extension
    - tags: [root_category] + parent folder names in order

    Raises:
        ValueError: path does not belong to any known root.
    """
    root_category, some_path = get_asset_category_and_relative_path(file_path)
    p = Path(some_path)
    parent_parts = [
        part for part in p.parent.parts if part not in (".", "..", p.anchor)
    ]
    return p.name, list(dict.fromkeys(normalize_tags([root_category, *parent_parts])))
