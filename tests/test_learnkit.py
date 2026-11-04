"""learnkit.py のテスト．

実行方法（ai-learn-kit のルートで）:
    python -m pytest tests/
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import learnkit  # noqa: E402


def make_curriculum(root):
    """3ステージの教材を作成する．"""
    learnkit.command_init(root, "VAE 入門", "VAEを説明・実装できる", "owner/vae_training", "main")
    learnkit.command_new_stage(root, "mnist_nn", "NN による分類", "学習ループ", "講座の基礎", "表現を学習していない")
    learnkit.command_new_stage(root, "mnist_ae", "Autoencoder", "Encoder/Decoder", "教師信号が入力画像", "補間が破綻する")
    learnkit.command_new_stage(root, "mnist_vae", "VAE", "KL", "潜在変数が分布", kind="到達")
    return learnkit.load_manifest(root)


def cells_text(notebook):
    return ["".join(cell["source"]) for cell in notebook["cells"]]


def test_init_creates_files(tmp_path):
    (tmp_path / ".gitignore").write_text("tokens.json\n")
    learnkit.command_init(tmp_path, "T", "G", "owner/repo", "main")
    for name in ["curriculum.json", "README.md", "curriculum_plan.md", ".gitignore", ".gitignore.orig"]:
        assert (tmp_path / name).exists()
    assert (tmp_path / ".gitignore").read_text().startswith("# 教材リポジトリ")
    with pytest.raises(learnkit.LearnKitError):
        learnkit.command_init(tmp_path, "T", "G", "owner/repo", "main")
    with pytest.raises(learnkit.LearnKitError):
        learnkit.command_init(tmp_path / "x", "T", "G", "not-a-repo", "main")


def test_template_notebook_is_valid_python(tmp_path):
    manifest = make_curriculum(tmp_path)
    notebook = learnkit.load_notebook(tmp_path / "ex001_mnist_nn.ipynb")
    assert notebook["nbformat"] == 4
    code = "\n\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    compile(code, "ex001", "exec")                       # 雛形のコードが構文として正しい
    assert 'EXPERIMENT_NAME = "ex001_mnist_nn"' in code
    assert "{{" not in json.dumps(notebook, ensure_ascii=False)
    assert learnkit.colab_url(manifest, "ex001_mnist_nn") in cells_text(notebook)[0]


def test_new_stage_derives_from_previous(tmp_path):
    make_curriculum(tmp_path)
    nb1 = learnkit.load_notebook(tmp_path / "ex001_mnist_nn.ipynb")
    # ステージ1のモデルを変更すると，ステージ2を作り直したときに引き継がれる
    for cell in nb1["cells"]:
        text = "".join(cell["source"])
        if "class MNISTClassifier" in text:
            cell["source"] = learnkit.split_source(text + "\n# custom change in stage 1\n")
    learnkit.save_notebook(tmp_path / "ex001_mnist_nn.ipynb", nb1)
    learnkit.command_new_stage(tmp_path, "mnist_ae", "Autoencoder", "c", "d", "b", stage_id=2, force=True)

    nb2 = learnkit.load_notebook(tmp_path / "ex002_mnist_ae.ipynb")
    texts = cells_text(nb2)
    assert any("# custom change in stage 1" in t for t in texts)
    assert any('EXPERIMENT_NAME = "ex002_mnist_ae"' in t for t in texts)
    assert not any("ex001_mnist_nn" in t for t in texts if "EXPERIMENT_NAME" in t)
    assert "**本実験**: Autoencoder" in texts[0]
    assert all(c.get("outputs", []) == [] for c in nb2["cells"] if c["cell_type"] == "code")


def test_sync_updates_lists_and_is_idempotent(tmp_path):
    make_curriculum(tmp_path)
    readme = (tmp_path / "README.md").read_text()
    assert "[VAE](ex003_mnist_vae.md)" in readme and "flowchart LR" in readme
    nav = (tmp_path / "ex002_mnist_ae.md").read_text()
    assert "(ex001_mnist_nn.md)" in nav and "(ex003_mnist_vae.md)" in nav
    assert learnkit.command_sync(tmp_path) == []

    manifest = learnkit.load_manifest(tmp_path)
    manifest["repo"] = "owner/renamed"
    manifest["stages"][1]["title"] = "AE（改題）"
    learnkit.save_manifest(tmp_path, manifest)
    changed = learnkit.command_sync(tmp_path)
    assert len(changed) == 7                             # ノートブック3，解説3，README
    assert "owner/renamed" in (tmp_path / "ex003_mnist_vae.md").read_text()
    assert "AE（改題）" in "".join(learnkit.load_notebook(tmp_path / "ex003_mnist_vae.ipynb")["cells"][0]["source"])


def test_check_reports_problems(tmp_path):
    make_curriculum(tmp_path)
    issues = learnkit.collect_issues(tmp_path)
    flat = "\n".join(m for messages in issues.values() for m in messages)
    assert "TODO(AI)" in flat
    assert "未実行のセル" in flat
    assert "到達目標の振り返り" in flat                   # 最後のステージの節
    assert "figures/ex001_mnist_nn/loss.png" in flat


def test_check_passes_for_completed_curriculum(tmp_path):
    make_curriculum(tmp_path)
    fill_curriculum(tmp_path)
    issues = learnkit.collect_issues(tmp_path)
    assert issues == {}, issues


def test_check_detects_local_import_and_errors(tmp_path):
    make_curriculum(tmp_path)
    fill_curriculum(tmp_path)
    (tmp_path / "helpers.py").write_text("x = 1\n")
    path = tmp_path / "ex002_mnist_ae.ipynb"
    notebook = learnkit.load_notebook(path)
    code_cell = next(c for c in notebook["cells"] if c["cell_type"] == "code")
    code_cell["source"] = learnkit.split_source("".join(code_cell["source"]) + "\nimport helpers\n")
    code_cell["outputs"] = [{"output_type": "error", "ename": "ValueError", "evalue": "boom", "traceback": []}]
    learnkit.save_notebook(path, notebook)

    messages = learnkit.collect_issues(tmp_path)["ex002_mnist_ae.ipynb"]
    assert any("helpers" in m for m in messages)
    assert any("ValueError" in m for m in messages)


def fill_curriculum(root):
    """TODO(AI) を埋め，実行済みの状態を模擬する．"""
    for path in [root / "README.md", root / "curriculum_plan.md"] + list(root.glob("ex*.md")):
        text = path.read_text().replace("TODO(AI)", "記入済み")
        if path.name == "ex003_mnist_vae.md":
            text = text.replace("## 8. 次のステージへ", "## 8. 到達目標の振り返り")
        path.write_text(text)
        for target in __import__("re").findall(r"\]\((figures/[^)]+)\)", text):
            (root / target).parent.mkdir(parents=True, exist_ok=True)
            (root / target).write_bytes(b"png")

    for path in root.glob("ex*.ipynb"):
        notebook = learnkit.load_notebook(path)
        count = 0
        for cell in notebook["cells"]:
            cell["source"] = learnkit.split_source("".join(cell["source"]).replace("TODO(AI)", "記入済み"))
            if cell["cell_type"] == "code" and "".join(cell["source"]).strip():
                count += 1
                cell["execution_count"] = count
        learnkit.save_notebook(path, notebook)

    manifest = learnkit.load_manifest(root)
    for stage in manifest["stages"]:
        for field in ("concept", "diff", "bridge"):
            stage[field] = stage.get(field) or ("" if stage["id"] == 3 and field == "bridge" else "記入済み")
    learnkit.save_manifest(root, manifest)
    learnkit.command_sync(root, quiet=True)
