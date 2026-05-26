"""Tests for path_utils – asset category resolution."""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app.assets.services.path_utils import (
    compute_relative_filename,
    get_comfy_models_folders,
    get_asset_category_and_relative_path,
    get_asset_path_info,
    get_asset_response_path_info,
    resolve_asset_path_context,
)


@pytest.fixture
def fake_dirs():
    """Create temporary input, output, and temp directories."""
    with tempfile.TemporaryDirectory() as root:
        root_path = Path(root)
        input_dir = root_path / "input"
        output_dir = root_path / "output"
        temp_dir = root_path / "temp"
        models_dir = root_path / "models" / "checkpoints"
        for d in (input_dir, output_dir, temp_dir, models_dir):
            d.mkdir(parents=True)

        with patch("app.assets.services.path_utils.folder_paths") as mock_fp:
            mock_fp.get_input_directory.return_value = str(input_dir)
            mock_fp.get_output_directory.return_value = str(output_dir)
            mock_fp.get_temp_directory.return_value = str(temp_dir)

            with patch(
                "app.assets.services.path_utils.get_comfy_models_folders",
                return_value=[("checkpoints", [str(models_dir)])],
            ):
                yield {
                    "input": input_dir,
                    "output": output_dir,
                    "temp": temp_dir,
                    "models": models_dir,
                }


class TestGetAssetCategoryAndRelativePath:
    def test_input_file(self, fake_dirs):
        f = fake_dirs["input"] / "photo.png"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "input"
        assert rel == "photo.png"

    def test_output_file(self, fake_dirs):
        f = fake_dirs["output"] / "result.png"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "output"
        assert rel == "result.png"

    def test_temp_file(self, fake_dirs):
        """Regression: temp files must be categorised, not raise ValueError."""
        f = fake_dirs["temp"] / "GLSLShader_output_00004_.png"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "temp"
        assert rel == "GLSLShader_output_00004_.png"

    def test_temp_file_in_subfolder(self, fake_dirs):
        sub = fake_dirs["temp"] / "sub"
        sub.mkdir()
        f = sub / "ComfyUI_temp_tczip_00004_.png"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "temp"
        assert os.path.normpath(rel) == os.path.normpath("sub/ComfyUI_temp_tczip_00004_.png")

    def test_model_file(self, fake_dirs):
        f = fake_dirs["models"] / "model.safetensors"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "models"

    def test_unknown_path_raises(self, fake_dirs):
        with pytest.raises(ValueError, match="not within"):
            get_asset_category_and_relative_path("/some/random/path.png")


class TestGetAssetPathInfo:
    def test_get_comfy_models_folders_excludes_core_infrastructure(self, tmp_path: Path):
        controlnet_dir = tmp_path / "models" / "controlnet"
        configs_dir = tmp_path / "models" / "configs"
        custom_nodes_dir = tmp_path / "custom_nodes"
        for directory in (controlnet_dir, configs_dir, custom_nodes_dir):
            directory.mkdir(parents=True)

        with patch("app.assets.services.path_utils.folder_paths") as mock_fp:
            mock_fp.folder_names_and_paths = {
                "controlnet": ([str(controlnet_dir)], {".safetensors"}),
                "configs": ([str(configs_dir)], {".yaml"}),
                "custom_nodes": ([str(custom_nodes_dir)], set()),
            }

            folders = get_comfy_models_folders()

        assert folders == [("controlnet", [str(controlnet_dir)])]

    def test_model_file_uses_registered_model_folder(self, fake_dirs):
        f = fake_dirs["models"] / "subdir" / "model.safetensors"
        f.parent.mkdir()
        f.touch()

        info = get_asset_path_info(str(f))

        assert info.asset_type == "model"
        assert info.model_folder == "checkpoints"

        response_info = get_asset_response_path_info(str(f))
        assert response_info.file_path == "models/checkpoints/subdir/model.safetensors"
        assert response_info.display_name == "subdir/model.safetensors"

    def test_arbitrary_registered_folder_is_model_folder(self, fake_dirs):
        controlnet_dir = fake_dirs["models"].parent / "controlnet"
        controlnet_dir.mkdir()
        f = controlnet_dir / "pose.safetensors"
        f.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[("controlnet", [str(controlnet_dir)])],
        ):
            response_info = get_asset_response_path_info(str(f))

        assert response_info.asset_type == "model"
        assert response_info.model_folder == "controlnet"
        assert response_info.file_path == "models/controlnet/pose.safetensors"
        assert response_info.display_name == "pose.safetensors"

    def test_multiple_physical_roots_for_same_model_folder(self, fake_dirs):
        root_a = fake_dirs["models"]
        root_b = fake_dirs["output"] / "checkpoints"
        root_b.mkdir()
        file_a = root_a / "subdir" / "model_a.safetensors"
        file_b = root_b / "subdir" / "model_b.safetensors"
        file_a.parent.mkdir()
        file_b.parent.mkdir()
        file_a.touch()
        file_b.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[("checkpoints", [str(root_a), str(root_b)])],
        ):
            response_a = get_asset_response_path_info(str(file_a))
            response_b = get_asset_response_path_info(str(file_b))

        assert response_a.asset_type == response_b.asset_type == "model"
        assert response_a.model_folder == response_b.model_folder == "checkpoints"
        assert response_a.file_path == "models/checkpoints/subdir/model_a.safetensors"
        assert response_b.file_path == "models/checkpoints/subdir/model_b.safetensors"
        assert response_a.display_name == "subdir/model_a.safetensors"
        assert response_b.display_name == "subdir/model_b.safetensors"

    def test_same_named_files_under_multiple_roots_share_logical_file_path(self, fake_dirs):
        root_a = fake_dirs["models"]
        root_b = fake_dirs["output"] / "checkpoints"
        root_b.mkdir()
        file_a = root_a / "duplicate.safetensors"
        file_b = root_b / "duplicate.safetensors"
        file_a.touch()
        file_b.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[("checkpoints", [str(root_a), str(root_b)])],
        ):
            response_a = get_asset_response_path_info(str(file_a))
            response_b = get_asset_response_path_info(str(file_b))

        assert response_a.file_path == response_b.file_path
        assert response_a.file_path == "models/checkpoints/duplicate.safetensors"
        assert response_a.display_name == response_b.display_name == "duplicate.safetensors"

    def test_input_file_has_no_model_folder(self, fake_dirs):
        f = fake_dirs["input"] / "subdir" / "photo.png"
        f.parent.mkdir()
        f.touch()

        info = get_asset_path_info(str(f))

        assert info.asset_type == "input"
        assert info.model_folder is None

        response_info = get_asset_response_path_info(str(f))
        assert response_info.file_path == "input/subdir/photo.png"
        assert response_info.display_name == "subdir/photo.png"

    def test_output_backed_registered_model_folder_is_model(self, fake_dirs):
        output_checkpoints_dir = fake_dirs["output"] / "checkpoints"
        output_checkpoints_dir.mkdir()
        f = output_checkpoints_dir / "saved.safetensors"
        f.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[("checkpoints", [str(output_checkpoints_dir)])],
        ):
            context = resolve_asset_path_context(str(f))
            response_info = get_asset_response_path_info(str(f))

        assert context.asset_type == "model"
        assert context.model_folder == "checkpoints"
        assert context.relative_path == "saved.safetensors"

        assert response_info.file_path == "models/checkpoints/saved.safetensors"
        assert response_info.display_name == "saved.safetensors"

    def test_registered_model_folder_can_contain_slash(self, fake_dirs):
        nested_model_dir = fake_dirs["models"].parent / "text_encoders" / "clip"
        nested_model_dir.mkdir(parents=True)
        f = nested_model_dir / "clip.safetensors"
        f.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[("text_encoders/clip", [str(nested_model_dir)])],
        ):
            info = get_asset_path_info(str(f))
            response_info = get_asset_response_path_info(str(f))

        assert info.asset_type == "model"
        assert info.model_folder == "text_encoders/clip"

        assert response_info.file_path == "models/text_encoders/clip/clip.safetensors"
        assert response_info.display_name == "clip.safetensors"

    def test_slash_model_folder_relative_filename_uses_registered_base(self, fake_dirs):
        nested_model_dir = fake_dirs["models"].parent / "text_encoders" / "clip"
        nested_model_dir.mkdir(parents=True)
        f = nested_model_dir / "subdir" / "clip.safetensors"
        f.parent.mkdir()
        f.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[("text_encoders/clip", [str(nested_model_dir)])],
        ):
            assert compute_relative_filename(str(f)) == "subdir/clip.safetensors"

    def test_deepest_registered_model_base_wins(self, fake_dirs):
        parent_dir = fake_dirs["models"].parent / "text_encoders"
        nested_model_dir = parent_dir / "clip"
        nested_model_dir.mkdir(parents=True)
        f = nested_model_dir / "clip.safetensors"
        f.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[
                ("text_encoders", [str(parent_dir)]),
                ("text_encoders/clip", [str(nested_model_dir)]),
            ],
        ):
            context = resolve_asset_path_context(str(f))

        assert context.asset_type == "model"
        assert context.model_folder == "text_encoders/clip"
        assert context.relative_path == "clip.safetensors"

    def test_deepest_registered_model_base_wins_independent_of_registration_order(
        self, fake_dirs
    ):
        parent_dir = fake_dirs["models"].parent / "text_encoders"
        nested_model_dir = parent_dir / "clip"
        nested_model_dir.mkdir(parents=True)
        f = nested_model_dir / "clip.safetensors"
        f.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[
                ("text_encoders/clip", [str(nested_model_dir)]),
                ("text_encoders", [str(parent_dir)]),
            ],
        ):
            context = resolve_asset_path_context(str(f))

        assert context.asset_type == "model"
        assert context.model_folder == "text_encoders/clip"
        assert context.relative_path == "clip.safetensors"

    def test_path_under_unregistered_models_folder_is_unknown(self, fake_dirs):
        unregistered_dir = fake_dirs["models"].parent / "unregistered"
        unregistered_dir.mkdir()
        f = unregistered_dir / "model.safetensors"
        f.touch()

        with pytest.raises(ValueError, match="not within"):
            resolve_asset_path_context(str(f))

    def test_registered_model_folder_prefix_boundary(self, fake_dirs):
        checkpoints_extra_dir = fake_dirs["models"].parent / "checkpoints_extra"
        checkpoints_extra_dir.mkdir()
        f = checkpoints_extra_dir / "model.safetensors"
        f.touch()

        with pytest.raises(ValueError, match="not within"):
            resolve_asset_path_context(str(f))
