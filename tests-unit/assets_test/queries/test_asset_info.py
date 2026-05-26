import time
import uuid
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.assets.api import schemas_in
from app.assets.api.routes import _build_asset_response
from app.assets.database.models import Asset, AssetReference, AssetReferenceMeta
from app.assets.database.queries import (
    reference_exists_for_asset_id,
    get_reference_by_id,
    insert_reference,
    get_or_create_reference,
    update_reference_timestamps,
    list_references_page,
    fetch_reference_asset_and_tags,
    fetch_reference_and_asset,
    update_reference_access_time,
    set_reference_metadata,
    delete_reference_by_id,
    set_reference_preview,
    bulk_insert_references_ignore_conflicts,
    get_reference_ids_by_ids,
    ensure_tags_exist,
    add_tags_to_reference,
)
from app.assets.helpers import get_utc_now
from app.assets.services import schemas


class TestListAssetsQueryPathFilters:
    def test_model_folder_requires_model_asset_type(self):
        with pytest.raises(ValueError, match="model_folder can only be used"):
            schemas_in.ListAssetsQuery.model_validate({"model_folder": "checkpoints"})

    def test_model_folder_accepts_explicit_model_asset_type(self):
        query = schemas_in.ListAssetsQuery.model_validate(
            {"asset_type": "model", "model_folder": "checkpoints"}
        )

        assert query.asset_type == "model"
        assert query.model_folder == "checkpoints"

    def test_model_folder_rejects_non_model_asset_type(self):
        with pytest.raises(ValueError, match="model_folder can only be used"):
            schemas_in.ListAssetsQuery.model_validate(
                {"asset_type": "input", "model_folder": "checkpoints"}
            )

    def test_query_layer_rejects_model_folder_without_model_asset_type(
        self, session: Session
    ):
        with pytest.raises(ValueError, match="model_folder can only be used"):
            list_references_page(session, model_folder="checkpoints")

    def test_upload_tags_preserve_model_folder_case_for_destination(self):
        spec = schemas_in.UploadAssetSpec.model_validate(
            {"tags": ['["models", "LLM", "SubDir"]']}
        )

        assert spec.tags == ["models", "LLM", "SubDir"]


def _make_asset(session: Session, hash_val: str | None = None, size: int = 1024) -> Asset:
    asset = Asset(hash=hash_val, size_bytes=size, mime_type="application/octet-stream")
    session.add(asset)
    session.flush()
    return asset


def _make_reference(
    session: Session,
    asset: Asset,
    name: str = "test",
    owner_id: str = "",
    file_path: str | None = None,
    asset_type: str | None = None,
    model_folder: str | None = None,
) -> AssetReference:
    now = get_utc_now()
    ref = AssetReference(
        owner_id=owner_id,
        name=name,
        asset_id=asset.id,
        file_path=file_path,
        asset_type=asset_type,
        model_folder=model_folder,
        created_at=now,
        updated_at=now,
        last_access_time=now,
    )
    session.add(ref)
    session.flush()
    return ref


def _reference_data(
    *,
    name: str,
    file_path: str | None,
    asset_type: str | None = None,
    model_folder: str | None = None,
) -> schemas.ReferenceData:
    now = get_utc_now()
    return schemas.ReferenceData(
        id=str(uuid.uuid4()),
        name=name,
        file_path=file_path,
        asset_type=asset_type,
        model_folder=model_folder,
        user_metadata={},
        preview_id=None,
        created_at=now,
        updated_at=now,
        last_access_time=now,
    )


def _asset_detail_result(ref: schemas.ReferenceData) -> schemas.AssetDetailResult:
    return schemas.AssetDetailResult(
        ref=ref,
        asset=schemas.AssetData(
            hash="blake3:" + "a" * 64,
            size_bytes=123,
            mime_type="application/octet-stream",
        ),
        tags=[],
    )


class TestBuildAssetResponsePathFields:
    def test_model_response_fields_use_persisted_classification(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        checkpoint_dir = tmp_path / "models" / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        model_path = checkpoint_dir / "sub" / "model.safetensors"
        model_path.parent.mkdir()
        model_path.write_text("data")
        monkeypatch.setattr(
            "app.assets.services.path_utils.get_comfy_models_folders",
            lambda: [("checkpoints", [str(checkpoint_dir)])],
        )

        asset = _build_asset_response(
            _asset_detail_result(
                _reference_data(
                    name="model.safetensors",
                    file_path=str(model_path),
                    asset_type="model",
                    model_folder="checkpoints",
                )
            )
        )

        assert asset.asset_type == "model"
        assert asset.model_folder == "checkpoints"
        assert asset.display_name == "sub/model.safetensors"
        assert asset.file_path == "models/checkpoints/sub/model.safetensors"

    def test_input_output_response_fields_use_persisted_classification(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        input_path = input_dir / "sub" / "image.png"
        output_path = output_dir / "result.png"
        input_path.parent.mkdir()
        input_path.write_text("input")
        output_path.write_text("output")
        monkeypatch.setattr(
            "app.assets.services.path_utils.folder_paths.get_input_directory",
            lambda: str(input_dir),
        )
        monkeypatch.setattr(
            "app.assets.services.path_utils.folder_paths.get_output_directory",
            lambda: str(output_dir),
        )

        input_asset = _build_asset_response(
            _asset_detail_result(
                _reference_data(
                    name="image.png",
                    file_path=str(input_path),
                    asset_type="input",
                )
            )
        )
        output_asset = _build_asset_response(
            _asset_detail_result(
                _reference_data(
                    name="result.png",
                    file_path=str(output_path),
                    asset_type="output",
                )
            )
        )

        assert input_asset.asset_type == "input"
        assert input_asset.model_folder is None
        assert input_asset.display_name == "sub/image.png"
        assert input_asset.file_path == "input/sub/image.png"
        assert output_asset.asset_type == "output"
        assert output_asset.model_folder is None
        assert output_asset.display_name == "result.png"
        assert output_asset.file_path == "output/result.png"

    def test_pathless_response_omits_typed_path_fields(self):
        asset = _build_asset_response(
            _asset_detail_result(_reference_data(name="manual", file_path=None))
        )

        assert asset.asset_type is None
        assert asset.model_folder is None
        assert asset.display_name is None
        assert asset.file_path is None


class TestReferenceExistsForAssetId:
    def test_returns_false_when_no_reference(self, session: Session):
        asset = _make_asset(session, "hash1")
        assert reference_exists_for_asset_id(session, asset_id=asset.id) is False

    def test_returns_true_when_reference_exists(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_reference(session, asset)
        assert reference_exists_for_asset_id(session, asset_id=asset.id) is True


class TestGetReferenceById:
    def test_returns_none_for_nonexistent(self, session: Session):
        assert get_reference_by_id(session, reference_id="nonexistent") is None

    def test_returns_reference(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset, name="myfile.txt")

        result = get_reference_by_id(session, reference_id=ref.id)
        assert result is not None
        assert result.name == "myfile.txt"


class TestListReferencesPage:
    def test_empty_db(self, session: Session):
        refs, tag_map, total = list_references_page(session)
        assert refs == []
        assert tag_map == {}
        assert total == 0

    def test_returns_references_with_tags(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset, name="test.bin")
        ensure_tags_exist(session, ["alpha", "beta"])
        add_tags_to_reference(session, reference_id=ref.id, tags=["alpha", "beta"])
        session.commit()

        refs, tag_map, total = list_references_page(session)
        assert len(refs) == 1
        assert refs[0].id == ref.id
        assert set(tag_map[ref.id]) == {"alpha", "beta"}
        assert total == 1

    def test_name_contains_filter(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_reference(session, asset, name="model_v1.safetensors")
        _make_reference(session, asset, name="config.json")
        session.commit()

        refs, _, total = list_references_page(session, name_contains="model")
        assert total == 1
        assert refs[0].name == "model_v1.safetensors"

    def test_owner_visibility(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_reference(session, asset, name="public", owner_id="")
        _make_reference(session, asset, name="private", owner_id="user1")
        session.commit()

        # Empty owner sees only public
        refs, _, total = list_references_page(session, owner_id="")
        assert total == 1
        assert refs[0].name == "public"

        # Owner sees both
        refs, _, total = list_references_page(session, owner_id="user1")
        assert total == 2

    def test_include_tags_filter(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref1 = _make_reference(session, asset, name="tagged")
        _make_reference(session, asset, name="untagged")
        ensure_tags_exist(session, ["wanted"])
        add_tags_to_reference(session, reference_id=ref1.id, tags=["wanted"])
        session.commit()

        refs, _, total = list_references_page(session, include_tags=["wanted"])
        assert total == 1
        assert refs[0].name == "tagged"

    def test_exclude_tags_filter(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_reference(session, asset, name="keep")
        ref_exclude = _make_reference(session, asset, name="exclude")
        ensure_tags_exist(session, ["bad"])
        add_tags_to_reference(session, reference_id=ref_exclude.id, tags=["bad"])
        session.commit()

        refs, _, total = list_references_page(session, exclude_tags=["bad"])
        assert total == 1
        assert refs[0].name == "keep"

    def test_model_folder_filter_uses_registered_paths(self, session: Session, tmp_path: Path):
        checkpoints_dir = tmp_path / "models" / "checkpoints"
        loras_dir = tmp_path / "models" / "loras"
        checkpoints_dir.mkdir(parents=True)
        loras_dir.mkdir(parents=True)

        asset = _make_asset(session, "hash1")
        checkpoint = _make_reference(
            session,
            asset,
            name="checkpoint",
            file_path=str(checkpoints_dir / "model.safetensors"),
            asset_type="model",
            model_folder="checkpoints",
        )
        _make_reference(
            session,
            asset,
            name="lora",
            file_path=str(loras_dir / "model.safetensors"),
            asset_type="model",
            model_folder="loras",
        )
        session.commit()

        refs, _, total = list_references_page(
            session, asset_type="model", model_folder="checkpoints"
        )

        assert total == 1
        assert refs[0].id == checkpoint.id

    def test_model_folder_filter_includes_all_registered_roots_for_folder(
        self, session: Session, tmp_path: Path
    ):
        checkpoints_a = tmp_path / "root_a" / "checkpoints"
        checkpoints_b = tmp_path / "root_b" / "checkpoints"
        checkpoints_a.mkdir(parents=True)
        checkpoints_b.mkdir(parents=True)

        asset = _make_asset(session, "hash1")
        ref_a = _make_reference(
            session,
            asset,
            name="checkpoint-a",
            file_path=str(checkpoints_a / "a.safetensors"),
            asset_type="model",
            model_folder="checkpoints",
        )
        ref_b = _make_reference(
            session,
            asset,
            name="checkpoint-b",
            file_path=str(checkpoints_b / "b.safetensors"),
            asset_type="model",
            model_folder="checkpoints",
        )
        session.commit()

        refs, _, total = list_references_page(
            session, asset_type="model", model_folder="checkpoints"
        )

        assert total == 2
        assert {ref.id for ref in refs} == {ref_a.id, ref_b.id}

    def test_same_named_files_under_multiple_roots_both_return_in_model_folder_filter(
        self, session: Session, tmp_path: Path
    ):
        checkpoints_a = tmp_path / "root_a" / "checkpoints"
        checkpoints_b = tmp_path / "root_b" / "checkpoints"
        checkpoints_a.mkdir(parents=True)
        checkpoints_b.mkdir(parents=True)

        asset = _make_asset(session, "hash1")
        ref_a = _make_reference(
            session,
            asset,
            name="checkpoint-a",
            file_path=str(checkpoints_a / "duplicate.safetensors"),
            asset_type="model",
            model_folder="checkpoints",
        )
        ref_b = _make_reference(
            session,
            asset,
            name="checkpoint-b",
            file_path=str(checkpoints_b / "duplicate.safetensors"),
            asset_type="model",
            model_folder="checkpoints",
        )
        session.commit()

        refs, _, total = list_references_page(
            session, asset_type="model", model_folder="checkpoints"
        )

        assert total == 2
        assert {ref.id for ref in refs} == {ref_a.id, ref_b.id}

    def test_arbitrary_registered_folder_filter_works(
        self, session: Session, tmp_path: Path
    ):
        controlnet_dir = tmp_path / "models" / "controlnet"
        controlnet_dir.mkdir(parents=True)

        asset = _make_asset(session, "hash1")
        ref = _make_reference(
            session,
            asset,
            name="controlnet",
            file_path=str(controlnet_dir / "pose.safetensors"),
            asset_type="model",
            model_folder="controlnet",
        )
        session.commit()

        refs, _, total = list_references_page(
            session, asset_type="model", model_folder="controlnet"
        )

        assert total == 1
        assert refs[0].id == ref.id

    def test_unknown_model_folder_filter_returns_none_when_other_models_exist(
        self, session: Session, tmp_path: Path
    ):
        checkpoints_dir = tmp_path / "models" / "checkpoints"
        checkpoints_dir.mkdir(parents=True)

        asset = _make_asset(session, "hash1")
        _make_reference(
            session,
            asset,
            name="checkpoint",
            file_path=str(checkpoints_dir / "model.safetensors"),
            asset_type="model",
            model_folder="checkpoints",
        )
        session.commit()

        refs, _, total = list_references_page(
            session, asset_type="model", model_folder="controlnet"
        )

        assert total == 0
        assert refs == []

    def test_model_folder_filter_excludes_deeper_registered_model_folder(
        self, session: Session, tmp_path: Path
    ):
        text_encoders_dir = tmp_path / "models" / "text_encoders"
        clip_dir = text_encoders_dir / "clip"
        clip_dir.mkdir(parents=True)

        asset = _make_asset(session, "hash1")
        text_encoder = _make_reference(
            session,
            asset,
            name="text_encoder",
            file_path=str(text_encoders_dir / "t5xxl.safetensors"),
            asset_type="model",
            model_folder="text_encoders",
        )
        _make_reference(
            session,
            asset,
            name="clip",
            file_path=str(clip_dir / "clip_l.safetensors"),
            asset_type="model",
            model_folder="text_encoders/clip",
        )
        session.commit()

        refs, _, total = list_references_page(
            session, asset_type="model", model_folder="text_encoders"
        )

        assert total == 1
        assert refs[0].id == text_encoder.id

    def test_child_model_folder_filter_returns_only_child(
        self, session: Session, tmp_path: Path
    ):
        text_encoders_dir = tmp_path / "models" / "text_encoders"
        clip_dir = text_encoders_dir / "clip"
        clip_dir.mkdir(parents=True)

        asset = _make_asset(session, "hash1")
        _make_reference(
            session,
            asset,
            name="text_encoder",
            file_path=str(text_encoders_dir / "t5xxl.safetensors"),
            asset_type="model",
            model_folder="text_encoders",
        )
        clip = _make_reference(
            session,
            asset,
            name="clip",
            file_path=str(clip_dir / "clip_l.safetensors"),
            asset_type="model",
            model_folder="text_encoders/clip",
        )
        session.commit()

        refs, _, total = list_references_page(
            session, asset_type="model", model_folder="text_encoders/clip"
        )

        assert total == 1
        assert refs[0].id == clip.id

    def test_model_asset_type_filter_includes_parent_and_child_registered_roots(
        self, session: Session, tmp_path: Path
    ):
        text_encoders_dir = tmp_path / "models" / "text_encoders"
        clip_dir = text_encoders_dir / "clip"
        clip_dir.mkdir(parents=True)

        asset = _make_asset(session, "hash1")
        text_encoder = _make_reference(
            session,
            asset,
            name="text_encoder",
            file_path=str(text_encoders_dir / "t5xxl.safetensors"),
            asset_type="model",
            model_folder="text_encoders",
        )
        clip = _make_reference(
            session,
            asset,
            name="clip",
            file_path=str(clip_dir / "clip_l.safetensors"),
            asset_type="model",
            model_folder="text_encoders/clip",
        )
        session.commit()

        refs, _, total = list_references_page(session, asset_type="model")

        assert total == 2
        assert {ref.id for ref in refs} == {text_encoder.id, clip.id}

    def test_model_asset_type_filter_with_no_registered_paths_returns_none(
        self, session: Session, tmp_path: Path
    ):
        asset = _make_asset(session, "hash1")
        _make_reference(
            session,
            asset,
            name="orphan",
            file_path=str(tmp_path / "models" / "checkpoints" / "model.safetensors"),
        )
        session.commit()

        refs, _, total = list_references_page(session, asset_type="model")

        assert total == 0
        assert refs == []

    def test_model_asset_type_filter_excludes_unregistered_models_folder(
        self, session: Session, tmp_path: Path
    ):
        checkpoints_dir = tmp_path / "models" / "checkpoints"
        unregistered_dir = tmp_path / "models" / "unregistered"
        checkpoints_dir.mkdir(parents=True)
        unregistered_dir.mkdir(parents=True)

        asset = _make_asset(session, "hash1")
        checkpoint = _make_reference(
            session,
            asset,
            name="checkpoint",
            file_path=str(checkpoints_dir / "model.safetensors"),
            asset_type="model",
            model_folder="checkpoints",
        )
        _make_reference(
            session,
            asset,
            name="unregistered",
            file_path=str(unregistered_dir / "model.safetensors"),
        )
        session.commit()

        refs, _, total = list_references_page(session, asset_type="model")

        assert total == 1
        assert refs[0].id == checkpoint.id

    def test_model_asset_type_filter_respects_prefix_boundaries(
        self, session: Session, tmp_path: Path
    ):
        checkpoints_dir = tmp_path / "models" / "checkpoints"
        checkpoints_extra_dir = tmp_path / "models" / "checkpoints_extra"
        checkpoints_dir.mkdir(parents=True)
        checkpoints_extra_dir.mkdir(parents=True)

        asset = _make_asset(session, "hash1")
        checkpoint = _make_reference(
            session,
            asset,
            name="checkpoint",
            file_path=str(checkpoints_dir / "model.safetensors"),
            asset_type="model",
            model_folder="checkpoints",
        )
        _make_reference(
            session,
            asset,
            name="checkpoints_extra",
            file_path=str(checkpoints_extra_dir / "model.safetensors"),
        )
        session.commit()

        refs, _, total = list_references_page(session, asset_type="model")

        assert total == 1
        assert refs[0].id == checkpoint.id

    def test_model_asset_type_filter_is_case_exact(
        self, session: Session, tmp_path: Path
    ):
        registered_dir = tmp_path / "models" / "checkpoints"
        case_sibling_dir = tmp_path / "MODELS" / "checkpoints"
        registered_dir.mkdir(parents=True)
        case_sibling_dir.mkdir(parents=True)

        asset = _make_asset(session, "hash1")
        checkpoint = _make_reference(
            session,
            asset,
            name="checkpoint",
            file_path=str(registered_dir / "model.safetensors"),
            asset_type="model",
            model_folder="checkpoints",
        )
        _make_reference(
            session,
            asset,
            name="case_sibling",
            file_path=str(case_sibling_dir / "model.safetensors"),
        )
        session.commit()

        refs, _, total = list_references_page(session, asset_type="model")

        assert total == 1
        assert refs[0].id == checkpoint.id

    def test_model_asset_type_filter_includes_output_backed_model_folder(
        self, session: Session, tmp_path: Path
    ):
        output_checkpoints_dir = tmp_path / "output" / "checkpoints"
        output_checkpoints_dir.mkdir(parents=True)

        asset = _make_asset(session, "hash1")
        checkpoint_ref = _make_reference(
            session,
            asset,
            name="checkpoint",
            file_path=str(output_checkpoints_dir / "saved.safetensors"),
            asset_type="model",
            model_folder="checkpoints",
        )
        session.commit()

        refs, _, total = list_references_page(session, asset_type="model")

        assert total == 1
        assert refs[0].id == checkpoint_ref.id

    def test_asset_type_filter_uses_root_paths(self, session: Session, tmp_path: Path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        temp_dir = tmp_path / "temp"
        for directory in (input_dir, output_dir, temp_dir):
            directory.mkdir()

        asset = _make_asset(session, "hash1")
        input_ref = _make_reference(
            session,
            asset,
            name="input",
            file_path=str(input_dir / "image.png"),
            asset_type="input",
        )
        _make_reference(
            session,
            asset,
            name="output",
            file_path=str(output_dir / "image.png"),
            asset_type="output",
        )
        session.commit()

        refs, _, total = list_references_page(session, asset_type="input")

        assert total == 1
        assert refs[0].id == input_ref.id

    def test_output_asset_type_filter_excludes_output_backed_model_folders(
        self, session: Session, tmp_path: Path
    ):
        output_dir = tmp_path / "output"
        output_checkpoints_dir = output_dir / "checkpoints"
        output_checkpoints_dir.mkdir(parents=True)

        asset = _make_asset(session, "hash1")
        output_ref = _make_reference(
            session,
            asset,
            name="output",
            file_path=str(output_dir / "image.png"),
            asset_type="output",
        )
        _make_reference(
            session,
            asset,
            name="checkpoint",
            file_path=str(output_checkpoints_dir / "saved.safetensors"),
            asset_type="model",
            model_folder="checkpoints",
        )
        session.commit()

        refs, _, total = list_references_page(session, asset_type="output")

        assert total == 1
        assert refs[0].id == output_ref.id

    def test_sorting(self, session: Session):
        asset = _make_asset(session, "hash1", size=100)
        asset2 = _make_asset(session, "hash2", size=500)
        _make_reference(session, asset, name="small")
        _make_reference(session, asset2, name="large")
        session.commit()

        refs, _, _ = list_references_page(session, sort="size", order="desc")
        assert refs[0].name == "large"

        refs, _, _ = list_references_page(session, sort="name", order="asc")
        assert refs[0].name == "large"


class TestFetchReferenceAssetAndTags:
    def test_returns_none_for_nonexistent(self, session: Session):
        result = fetch_reference_asset_and_tags(session, "nonexistent")
        assert result is None

    def test_returns_tuple(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset, name="test.bin")
        ensure_tags_exist(session, ["tag1"])
        add_tags_to_reference(session, reference_id=ref.id, tags=["tag1"])
        session.commit()

        result = fetch_reference_asset_and_tags(session, ref.id)
        assert result is not None
        ret_ref, ret_asset, ret_tags = result
        assert ret_ref.id == ref.id
        assert ret_asset.id == asset.id
        assert ret_tags == ["tag1"]


class TestFetchReferenceAndAsset:
    def test_returns_none_for_nonexistent(self, session: Session):
        result = fetch_reference_and_asset(session, reference_id="nonexistent")
        assert result is None

    def test_returns_tuple(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset)
        session.commit()

        result = fetch_reference_and_asset(session, reference_id=ref.id)
        assert result is not None
        ret_ref, ret_asset = result
        assert ret_ref.id == ref.id
        assert ret_asset.id == asset.id


class TestUpdateReferenceAccessTime:
    def test_updates_last_access_time(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset)
        original_time = ref.last_access_time
        session.commit()

        import time
        time.sleep(0.01)

        update_reference_access_time(session, reference_id=ref.id)
        session.commit()

        session.refresh(ref)
        assert ref.last_access_time > original_time


class TestDeleteReferenceById:
    def test_deletes_existing(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset)
        session.commit()

        result = delete_reference_by_id(session, reference_id=ref.id, owner_id="")
        assert result is True
        assert get_reference_by_id(session, reference_id=ref.id) is None

    def test_returns_false_for_nonexistent(self, session: Session):
        result = delete_reference_by_id(session, reference_id="nonexistent", owner_id="")
        assert result is False

    def test_respects_owner_visibility(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset, owner_id="user1")
        session.commit()

        result = delete_reference_by_id(session, reference_id=ref.id, owner_id="user2")
        assert result is False
        assert get_reference_by_id(session, reference_id=ref.id) is not None


class TestSetReferencePreview:
    def test_sets_preview(self, session: Session):
        asset = _make_asset(session, "hash1")
        preview_asset = _make_asset(session, "preview_hash")
        ref = _make_reference(session, asset)
        preview_ref = _make_reference(session, preview_asset, name="preview.png")
        session.commit()

        set_reference_preview(session, reference_id=ref.id, preview_reference_id=preview_ref.id)
        session.commit()

        session.refresh(ref)
        assert ref.preview_id == preview_ref.id

    def test_clears_preview(self, session: Session):
        asset = _make_asset(session, "hash1")
        preview_asset = _make_asset(session, "preview_hash")
        ref = _make_reference(session, asset)
        preview_ref = _make_reference(session, preview_asset, name="preview.png")
        ref.preview_id = preview_ref.id
        session.commit()

        set_reference_preview(session, reference_id=ref.id, preview_reference_id=None)
        session.commit()

        session.refresh(ref)
        assert ref.preview_id is None

    def test_raises_for_nonexistent_reference(self, session: Session):
        with pytest.raises(ValueError, match="not found"):
            set_reference_preview(session, reference_id="nonexistent", preview_reference_id=None)

    def test_raises_for_nonexistent_preview(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset)
        session.commit()

        with pytest.raises(ValueError, match="Preview AssetReference"):
            set_reference_preview(session, reference_id=ref.id, preview_reference_id="nonexistent")


class TestInsertReference:
    def test_creates_new_reference(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = insert_reference(
            session, asset_id=asset.id, owner_id="user1", name="test.bin"
        )
        session.commit()

        assert ref is not None
        assert ref.name == "test.bin"
        assert ref.owner_id == "user1"

    def test_allows_duplicate_names(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref1 = insert_reference(session, asset_id=asset.id, owner_id="user1", name="dup.bin")
        session.commit()

        # Duplicate names are now allowed
        ref2 = insert_reference(
            session, asset_id=asset.id, owner_id="user1", name="dup.bin"
        )
        session.commit()

        assert ref1 is not None
        assert ref2 is not None
        assert ref1.id != ref2.id


class TestGetOrCreateReference:
    def test_creates_new_reference(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref, created = get_or_create_reference(
            session, asset_id=asset.id, owner_id="user1", name="new.bin"
        )
        session.commit()

        assert created is True
        assert ref.name == "new.bin"

    def test_always_creates_new_reference(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref1, created1 = get_or_create_reference(
            session, asset_id=asset.id, owner_id="user1", name="existing.bin"
        )
        session.commit()

        # Duplicate names are allowed, so always creates new
        ref2, created2 = get_or_create_reference(
            session, asset_id=asset.id, owner_id="user1", name="existing.bin"
        )
        session.commit()

        assert created1 is True
        assert created2 is True
        assert ref1.id != ref2.id


class TestUpdateReferenceTimestamps:
    def test_updates_timestamps(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset)
        original_updated_at = ref.updated_at
        session.commit()

        time.sleep(0.01)
        update_reference_timestamps(session, ref)
        session.commit()

        session.refresh(ref)
        assert ref.updated_at > original_updated_at

    def test_updates_preview_id(self, session: Session):
        asset = _make_asset(session, "hash1")
        preview_asset = _make_asset(session, "preview_hash")
        ref = _make_reference(session, asset)
        preview_ref = _make_reference(session, preview_asset, name="preview.png")
        session.commit()

        update_reference_timestamps(session, ref, preview_id=preview_ref.id)
        session.commit()

        session.refresh(ref)
        assert ref.preview_id == preview_ref.id


class TestSetReferenceMetadata:
    def test_sets_metadata(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset)
        session.commit()

        set_reference_metadata(
            session, reference_id=ref.id, user_metadata={"key": "value"}
        )
        session.commit()

        session.refresh(ref)
        assert ref.user_metadata == {"key": "value"}
        # Check metadata table
        meta = session.query(AssetReferenceMeta).filter_by(asset_reference_id=ref.id).all()
        assert len(meta) == 1
        assert meta[0].key == "key"
        assert meta[0].val_str == "value"

    def test_replaces_existing_metadata(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset)
        session.commit()

        set_reference_metadata(
            session, reference_id=ref.id, user_metadata={"old": "data"}
        )
        session.commit()

        set_reference_metadata(
            session, reference_id=ref.id, user_metadata={"new": "data"}
        )
        session.commit()

        meta = session.query(AssetReferenceMeta).filter_by(asset_reference_id=ref.id).all()
        assert len(meta) == 1
        assert meta[0].key == "new"

    def test_clears_metadata_with_empty_dict(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset)
        session.commit()

        set_reference_metadata(
            session, reference_id=ref.id, user_metadata={"key": "value"}
        )
        session.commit()

        set_reference_metadata(
            session, reference_id=ref.id, user_metadata={}
        )
        session.commit()

        session.refresh(ref)
        assert ref.user_metadata == {}
        meta = session.query(AssetReferenceMeta).filter_by(asset_reference_id=ref.id).all()
        assert len(meta) == 0

    def test_raises_for_nonexistent(self, session: Session):
        with pytest.raises(ValueError, match="not found"):
            set_reference_metadata(
                session, reference_id="nonexistent", user_metadata={"key": "value"}
            )


class TestBulkInsertReferencesIgnoreConflicts:
    def test_inserts_multiple_references(self, session: Session):
        asset = _make_asset(session, "hash1")
        now = get_utc_now()
        rows = [
            {
                "id": str(uuid.uuid4()),
                "owner_id": "",
                "name": "bulk1.bin",
                "asset_id": asset.id,
                "preview_id": None,
                "user_metadata": {},
                "created_at": now,
                "updated_at": now,
                "last_access_time": now,
            },
            {
                "id": str(uuid.uuid4()),
                "owner_id": "",
                "name": "bulk2.bin",
                "asset_id": asset.id,
                "preview_id": None,
                "user_metadata": {},
                "created_at": now,
                "updated_at": now,
                "last_access_time": now,
            },
        ]
        bulk_insert_references_ignore_conflicts(session, rows)
        session.commit()

        refs = session.query(AssetReference).all()
        assert len(refs) == 2

    def test_allows_duplicate_names(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_reference(session, asset, name="existing.bin", owner_id="")
        session.commit()

        now = get_utc_now()
        rows = [
            {
                "id": str(uuid.uuid4()),
                "owner_id": "",
                "name": "existing.bin",
                "asset_id": asset.id,
                "preview_id": None,
                "user_metadata": {},
                "created_at": now,
                "updated_at": now,
                "last_access_time": now,
            },
            {
                "id": str(uuid.uuid4()),
                "owner_id": "",
                "name": "new.bin",
                "asset_id": asset.id,
                "preview_id": None,
                "user_metadata": {},
                "created_at": now,
                "updated_at": now,
                "last_access_time": now,
            },
        ]
        bulk_insert_references_ignore_conflicts(session, rows)
        session.commit()

        # Duplicate names allowed, so all 3 rows exist
        refs = session.query(AssetReference).all()
        assert len(refs) == 3

    def test_empty_list_is_noop(self, session: Session):
        bulk_insert_references_ignore_conflicts(session, [])
        assert session.query(AssetReference).count() == 0


class TestGetReferenceIdsByIds:
    def test_returns_existing_ids(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref1 = _make_reference(session, asset, name="a.bin")
        ref2 = _make_reference(session, asset, name="b.bin")
        session.commit()

        found = get_reference_ids_by_ids(session, [ref1.id, ref2.id, "nonexistent"])

        assert found == {ref1.id, ref2.id}

    def test_empty_list_returns_empty(self, session: Session):
        found = get_reference_ids_by_ids(session, [])
        assert found == set()
