#!/usr/bin/env python3
"""段階的な教材リポジトリを作成・検証するためのCLI．

教材リポジトリのルート（プロジェクトルート）で実行する．標準ライブラリのみで動作する（`run` のみ jupyter が必要）．

    python .ai/ai-learn-kit/scripts/learnkit.py init --title "VAE 入門" --goal "..." --repo owner/repo
    python .ai/ai-learn-kit/scripts/learnkit.py new-stage mnist_nn --title "全結合 NN による MNIST 分類" \
        --concept "学習ループ" --diff "講座の基礎" --bridge "潜在表現を学習していない"
    python .ai/ai-learn-kit/scripts/learnkit.py sync
    python .ai/ai-learn-kit/scripts/learnkit.py run [--only ex001_mnist_nn] [--smoke]
    python .ai/ai-learn-kit/scripts/learnkit.py check [--allow-todo] [--allow-unexecuted]

ステージの一覧は `curriculum.json` で管理し，各ノートブックの冒頭のステージ一覧，各解説のナビゲーション，
`README.md` のステージの表と流れの図は `sync` が `curriculum.json` から生成する．
"""

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

KIT_DIR = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = KIT_DIR / "templates"
MANIFEST_NAME = "curriculum.json"
TODO_MARK = "TODO(AI)"
STAGE_KINDS = ["基礎", "本線", "可視化", "到達", "発展"]

MARKERS = {
    "stages": ("<!-- learnkit:stages:start -->", "<!-- learnkit:stages:end -->"),
    "nav": ("<!-- learnkit:nav:start -->", "<!-- learnkit:nav:end -->"),
    "table": ("<!-- learnkit:table:start -->", "<!-- learnkit:table:end -->"),
    "flow": ("<!-- learnkit:flow:start -->", "<!-- learnkit:flow:end -->"),
}

# 解説 .md に必須の節（見出しに含まれる語）と，この順序で現れる必要がある
REQUIRED_SECTIONS = ["位置づけ", "目的", "手法", "実験条件", "実験前に考える", "実験結果",
                     "実験後に考える", "考察", "まとめ", "発展課題"]
LAST_SECTION_MIDDLE = "次のステージへ"
LAST_SECTION_FINAL = "到達目標の振り返り"
FOLDED_SECTIONS = ["実験結果", "考察", "発展課題"]   # <details> を含む必要がある節

COLAB_PATTERN = re.compile(r"https://colab\.research\.google\.com/github/[^\s)\"']+?\.ipynb")
IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)\s]+)[^)]*\)|<img[^>]+src=[\"']([^\"']+)[\"']")
IMPORT_PATTERN = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)


class LearnKitError(Exception):
    """利用者に表示するエラー．"""


# =====================================================================================
# マニフェスト
# =====================================================================================

def stem_of(stage):
    """ステージのファイル名（拡張子なし）を返す．

    Args:
        stage (dict): curriculum.json のステージ．"id" と "name" を含む．

    Returns:
        str: "ex{id:03d}_{name}"．
    """
    return f"ex{stage['id']:03d}_{stage['name']}"


def colab_url(manifest, stem):
    """ノートブックを Colab で開くURLを返す．

    Args:
        manifest (dict): curriculum.json の内容．
        stem (str): ノートブックのファイル名（拡張子なし）．

    Returns:
        str: Colab のURL．
    """
    return f"https://colab.research.google.com/github/{manifest['repo']}/blob/{manifest['branch']}/{stem}.ipynb"


def load_manifest(root):
    """curriculum.json を読み込む．

    Args:
        root (Path): 教材リポジトリのルート．

    Returns:
        dict: マニフェスト．

    Raises:
        LearnKitError: curriculum.json が無い場合．
    """
    path = root / MANIFEST_NAME
    if not path.exists():
        raise LearnKitError(f"{MANIFEST_NAME} がありません．先に `learnkit.py init` を実行してください．")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.setdefault("stages", [])
    manifest["stages"].sort(key=lambda stage: stage["id"])
    return manifest


def save_manifest(root, manifest):
    """curriculum.json を保存する．

    Args:
        root (Path): 教材リポジトリのルート．
        manifest (dict): マニフェスト．

    Returns:
        None
    """
    manifest["stages"].sort(key=lambda stage: stage["id"])
    write_text(root / MANIFEST_NAME, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def write_text(path, text):
    """テキストファイルを書き込む（改行はLFに統一する）．

    Args:
        path (Path): 保存先．
        text (str): 内容．

    Returns:
        None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def render(template_text, values):
    """テンプレートの {{KEY}} を値で置換する．

    Args:
        template_text (str): テンプレート．
        values (dict[str, str]): 置換する値．

    Returns:
        str: 置換後の文字列．
    """
    for key, value in values.items():
        template_text = template_text.replace("{{" + key + "}}", str(value))
    return template_text


def stage_values(manifest, stage):
    """ステージのテンプレートに埋め込む値を返す．

    Args:
        manifest (dict): マニフェスト．
        stage (dict): ステージ．

    Returns:
        dict[str, str]: テンプレートの置換に用いる値．
    """
    stem = stem_of(stage)
    return {
        "TITLE": manifest["title"],
        "GOAL": manifest["goal"],
        "EXPERIMENT_NAME": stem,
        "STAGE_NUMBER": f"{stage['id']:03d}",
        "STAGE_TITLE": stage["title"],
        "CONCEPT": stage.get("concept") or TODO_MARK,
        "DIFF": stage.get("diff") or TODO_MARK,
        "BRIDGE": stage.get("bridge") or TODO_MARK,
        "COLAB_URL": colab_url(manifest, stem),
    }


# =====================================================================================
# ノートブック
# =====================================================================================

def new_cell(cell_type, source):
    """nbformat 4.5 のセルを作成する．

    Args:
        cell_type (str): "markdown" または "code"．
        source (str): セルの内容．

    Returns:
        dict: セル．
    """
    cell = {"cell_type": cell_type, "id": uuid.uuid4().hex[:8], "metadata": {}, "source": split_source(source)}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def split_source(text):
    """セルの内容を nbformat の行のリストに変換する．

    Args:
        text (str): セルの内容．

    Returns:
        list[str]: 各行（最終行以外は改行を含む）．
    """
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def cell_text(cell):
    """セルの内容を1つの文字列として返す．

    Args:
        cell (dict): セル．

    Returns:
        str: 内容．
    """
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def parse_percent_template(text):
    """jupytext の percent 形式のテンプレートをセルのリストに変換する．

    Args:
        text (str): テンプレートの内容．

    Returns:
        list[dict]: セルのリスト．
    """
    cells = []
    cell_type = None
    buffer = []

    def flush():
        if cell_type is None:
            return
        while buffer and not buffer[-1].strip():
            buffer.pop()
        while buffer and not buffer[0].strip():
            buffer.pop(0)
        if cell_type == "markdown":
            lines = [line[2:] if line.startswith("# ") else line.lstrip("#") for line in buffer]
        else:
            lines = buffer
        cells.append(new_cell(cell_type, "\n".join(lines)))

    for line in text.split("\n"):
        if line.startswith("# %%"):
            flush()
            cell_type = "markdown" if "[markdown]" in line else "code"
            buffer = []
        elif cell_type is not None:
            buffer.append(line)
    flush()
    return cells


def notebook_from_cells(cells):
    """セルのリストからノートブックを作成する．

    Args:
        cells (list[dict]): セル．

    Returns:
        dict: ノートブック．
    """
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def load_notebook(path):
    """ノートブックを読み込む．

    Args:
        path (Path): ノートブックのパス．

    Returns:
        dict: ノートブック．
    """
    return json.loads(path.read_text(encoding="utf-8"))


def save_notebook(path, notebook):
    """ノートブックを保存する．

    Args:
        path (Path): 保存先．
        notebook (dict): ノートブック．

    Returns:
        None
    """
    write_text(path, json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")


def template_cells(values):
    """共通骨格のテンプレートからセルを作成する．

    Args:
        values (dict[str, str]): テンプレートの置換に用いる値．

    Returns:
        list[dict]: セル．
    """
    text = (TEMPLATE_DIR / "notebook_template.py").read_text(encoding="utf-8")
    return parse_percent_template(render(text, values))


def derive_notebook(previous, previous_stem, values):
    """前のステージのノートブックを複製し，新しいステージ用に書き換える．

    冒頭のセルと「次のステージへ」のセルはテンプレートのものに置き換え，`EXPERIMENT_NAME` と
    マークダウン中の前のステージのファイル名を新しいものに置換し，出力を消去する．

    Args:
        previous (dict): 前のステージのノートブック．
        previous_stem (str): 前のステージのファイル名（拡張子なし）．
        values (dict[str, str]): 新しいステージのテンプレートの値．

    Returns:
        dict: 新しいステージのノートブック．
    """
    fresh = template_cells(values)
    header = fresh[0]
    bridge = next((cell for cell in fresh if "次のステージへ" in cell_text(cell).split("\n")[0]), None)

    notebook = copy.deepcopy(previous)
    stem = values["EXPERIMENT_NAME"]
    cells = []
    for index, cell in enumerate(notebook["cells"]):
        text = cell_text(cell)
        first_line = text.split("\n")[0]
        if index == 0 and cell["cell_type"] == "markdown":
            cells.append(copy.deepcopy(header))
            continue
        if bridge is not None and cell["cell_type"] == "markdown" and "次のステージへ" in first_line:
            cells.append(copy.deepcopy(bridge))
            continue
        if cell["cell_type"] == "code":
            text = re.sub(r'^EXPERIMENT_NAME\s*=\s*["\'][^"\']*["\']', f'EXPERIMENT_NAME = "{stem}"', text, flags=re.MULTILINE)
            cell["outputs"] = []
            cell["execution_count"] = None
        else:
            text = text.replace(previous_stem, stem)
        cell["source"] = split_source(text)
        cell["id"] = uuid.uuid4().hex[:8]
        cells.append(cell)
    notebook["cells"] = cells
    return notebook


# =====================================================================================
# 同期（sync）
# =====================================================================================

def replace_block(text, marker_key, body):
    """マーカーで囲まれた範囲を置換する．

    Args:
        text (str): 対象の文字列．
        marker_key (str): MARKERS のキー．
        body (str): マーカーの間に入れる内容．

    Returns:
        str: 置換後の文字列．マーカーが無い場合は元の文字列．
    """
    start, end = MARKERS[marker_key]
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        return text
    return pattern.sub(lambda _: f"{start}\n{body}\n{end}" if body else f"{start}\n{end}", text, count=1)


def escape_cell(text):
    """表のセルに入れる文字列の "|" をエスケープする．

    Args:
        text (str): 文字列．

    Returns:
        str: エスケープ後の文字列．
    """
    return (text or TODO_MARK).replace("|", "\\|").replace("\n", " ")


def stage_list_block(manifest, current):
    """ノートブック冒頭のステージ一覧を作成する．

    Args:
        manifest (dict): マニフェスト．
        current (dict): 現在のステージ．

    Returns:
        str: 番号付きリスト．
    """
    lines = []
    for index, stage in enumerate(manifest["stages"], start=1):
        if stage["id"] == current["id"]:
            lines.append(f"{index}. **本実験**: {stage['title']}")
        else:
            lines.append(f"{index}. {stage['title']}")
    return "\n".join(lines)


def nav_block(manifest, current):
    """解説 .md のナビゲーションを作成する．

    Args:
        manifest (dict): マニフェスト．
        current (dict): 現在のステージ．

    Returns:
        str: 前後のステージと目次へのリンク．
    """
    stages = manifest["stages"]
    index = next(i for i, stage in enumerate(stages) if stage["id"] == current["id"])
    links = []
    if index > 0:
        prev = stages[index - 1]
        links.append(f"[← 前: 実験{prev['id']:03d} {prev['title']}]({stem_of(prev)}.md)")
    links.append("[目次](README.md)")
    if index < len(stages) - 1:
        nxt = stages[index + 1]
        links.append(f"[次: 実験{nxt['id']:03d} {nxt['title']} →]({stem_of(nxt)}.md)")
    return " ｜ ".join(links)


def table_block(manifest):
    """README のステージの表を作成する．

    Args:
        manifest (dict): マニフェスト．

    Returns:
        str: マークダウンの表．
    """
    lines = [
        "| # | 種類 | 実験 | 新しく学ぶこと | 前実験との差分 | 限界（次の動機） | ノートブック |",
        "|:---:|:---:|:---|:---|:---|:---|:---:|",
    ]
    for index, stage in enumerate(manifest["stages"]):
        stem = stem_of(stage)
        is_last = index == len(manifest["stages"]) - 1
        bridge = stage.get("bridge") or ("−" if is_last else "")
        lines.append(
            f"| {stage['id']:03d} | {escape_cell(stage.get('kind'))} | [{escape_cell(stage['title'])}]({stem}.md) "
            f"| {escape_cell(stage.get('concept'))} | {escape_cell(stage.get('diff'))} "
            f"| {escape_cell(bridge)} | [Colab]({colab_url(manifest, stem)}) |")
    return "\n".join(lines)


def flow_block(manifest):
    """README のステージの流れの図（mermaid）を作成する．

    Args:
        manifest (dict): マニフェスト．

    Returns:
        str: mermaid のコードブロック．ステージが無い場合は空文字列．
    """
    stages = manifest["stages"]
    if not stages:
        return ""
    lines = ["```mermaid", "flowchart LR"]
    for stage in stages:
        title = stage["title"].replace('"', "'")
        lines.append(f'    s{stage["id"]}["{stage["id"]:03d}<br/>{title}"]')
    for prev, nxt in zip(stages, stages[1:]):
        arrow = "-.->" if nxt.get("kind") == "発展" else "-->"
        lines.append(f"    s{prev['id']} {arrow} s{nxt['id']}")
    goal_ids = [f"s{stage['id']}" for stage in stages if stage.get("kind") == "到達"]
    if goal_ids:
        lines.append("    classDef goal fill:#fef3c7,stroke:#d97706,stroke-width:2px")
        lines.append(f"    class {','.join(goal_ids)} goal")
    lines.append("```")
    return "\n".join(lines)


def fix_colab_urls(text, manifest, stem):
    """文字列中の Colab のURLを現在の repo・branch に合わせる．

    Args:
        text (str): 対象の文字列．
        manifest (dict): マニフェスト．
        stem (str): ノートブックのファイル名（拡張子なし）．

    Returns:
        str: 置換後の文字列．
    """
    expected = colab_url(manifest, stem)
    return COLAB_PATTERN.sub(lambda m: expected if m.group(0).endswith(f"/{stem}.ipynb") else m.group(0), text)


def synced_outputs(root, manifest):
    """sync を実行した場合の各ファイルの内容を計算する（ファイルは書き換えない）．

    Args:
        root (Path): 教材リポジトリのルート．
        manifest (dict): マニフェスト．

    Returns:
        dict[Path, str]: 内容が変わるファイルと，その新しい内容．
    """
    changes = {}
    for stage in manifest["stages"]:
        stem = stem_of(stage)
        notebook_path = root / f"{stem}.ipynb"
        if notebook_path.exists():
            original = notebook_path.read_text(encoding="utf-8")
            notebook = json.loads(original)
            for cell in notebook["cells"]:
                if cell["cell_type"] != "markdown":
                    continue
                text = cell_text(cell)
                new_text = fix_colab_urls(replace_block(text, "stages", stage_list_block(manifest, stage)), manifest, stem)
                if new_text != text:
                    cell["source"] = split_source(new_text)
            updated = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
            if json.loads(updated) != json.loads(original):
                changes[notebook_path] = updated

        md_path = root / f"{stem}.md"
        if md_path.exists():
            original = md_path.read_text(encoding="utf-8")
            updated = fix_colab_urls(replace_block(original, "nav", nav_block(manifest, stage)), manifest, stem)
            if updated != original:
                changes[md_path] = updated

    readme_path = root / "README.md"
    if readme_path.exists():
        original = readme_path.read_text(encoding="utf-8")
        updated = replace_block(original, "table", table_block(manifest))
        updated = replace_block(updated, "flow", flow_block(manifest))
        for stage in manifest["stages"]:
            updated = fix_colab_urls(updated, manifest, stem_of(stage))
        if updated != original:
            changes[readme_path] = updated
    return changes


def command_sync(root, quiet=False):
    """ステージ一覧・ナビゲーション・README の表と図を curriculum.json に同期する．

    Args:
        root (Path): 教材リポジトリのルート．
        quiet (bool): True の場合，変更が無いときは何も表示しない．

    Returns:
        list[Path]: 更新したファイル．
    """
    manifest = load_manifest(root)
    changes = synced_outputs(root, manifest)
    for path, text in changes.items():
        write_text(path, text)
        print(f"[sync] 更新: {path.relative_to(root)}")
    if not changes and not quiet:
        print("[sync] 変更はありません．")
    return list(changes)


# =====================================================================================
# init / new-stage
# =====================================================================================

def command_init(root, title, goal, repo, branch, force=False):
    """教材リポジトリを初期化する．

    curriculum.json，README.md，.gitignore，curriculum_plan.md，figures/ を作成する．
    既存の .gitignore は .gitignore.orig に退避してから，公開用の .gitignore に置き換える．

    Args:
        root (Path): 教材リポジトリのルート．
        title (str): 教材のタイトル．
        goal (str): 到達目標．
        repo (str): 公開先の "owner/repo"．
        branch (str): 公開先のブランチ．
        force (bool): True の場合，既存の curriculum.json，README.md を上書きする．

    Returns:
        None
    """
    manifest_path = root / MANIFEST_NAME
    if manifest_path.exists() and not force:
        raise LearnKitError(f"{MANIFEST_NAME} は既に存在します．上書きする場合は --force を指定してください．")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise LearnKitError(f"--repo は owner/repo の形式で指定してください: {repo}")

    manifest = {"title": title, "goal": goal, "repo": repo, "branch": branch, "stages": []}
    save_manifest(root, manifest)
    print(f"[init] 作成: {MANIFEST_NAME}")

    values = {"TITLE": title, "GOAL": goal}
    files = {
        "README.md": TEMPLATE_DIR / "README.md",
        "curriculum_plan.md": TEMPLATE_DIR / "curriculum_plan.md",
    }
    for name, template in files.items():
        path = root / name
        if path.exists() and not force:
            print(f"[init] {name} は既に存在するため作成しません．")
            continue
        write_text(path, render(template.read_text(encoding="utf-8"), values))
        print(f"[init] 作成: {name}")

    gitignore = root / ".gitignore"
    template_text = (TEMPLATE_DIR / "gitignore").read_text(encoding="utf-8")
    if gitignore.exists() and gitignore.read_text(encoding="utf-8") != template_text:
        shutil.copy(gitignore, root / ".gitignore.orig")
        print("[init] 既存の .gitignore を .gitignore.orig に退避しました．")
    write_text(gitignore, template_text)
    print("[init] 作成: .gitignore（公開するファイルのみを許可）")

    (root / "figures").mkdir(exist_ok=True)
    command_sync(root, quiet=True)


def command_new_stage(root, name, title, concept="", diff="", bridge="", kind=None,
                      stage_id=None, from_stem=None, blank=False, force=False):
    """ステージを追加し，ノートブックと解説の雛形を作成する．

    ノートブックは，前のステージのノートブックがあればそれを複製して作成し（差分を最小にするため），
    無ければ共通骨格のテンプレートから作成する．

    Args:
        root (Path): 教材リポジトリのルート．
        name (str): ステージ名（英小文字・数字・"_"．例: "mnist_vae"）．
        title (str): ステージのタイトル．
        concept (str): 新しく学ぶこと．
        diff (str): 前のステージとの差分．
        bridge (str): このステージの限界（次のステージの動機）．
        kind (str | None): ステージの種類（基礎・本線・可視化・到達・発展）．
        stage_id (int | None): ステージ番号．None の場合は最大の番号+1．
        from_stem (str | None): 複製元のノートブック（拡張子なし）．None の場合は直前のステージ．
        blank (bool): True の場合，前のステージを複製せずテンプレートから作成する．
        force (bool): True の場合，同じ番号のステージや既存のファイルを上書きする．

    Returns:
        dict: 追加したステージ．
    """
    if not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", name):
        raise LearnKitError(f"ステージ名は英小文字・数字・'_' で指定してください（例: mnist_vae）: {name}")
    if kind is not None and kind not in STAGE_KINDS:
        raise LearnKitError(f"--kind は {STAGE_KINDS} のいずれかを指定してください: {kind}")

    manifest = load_manifest(root)
    stages = manifest["stages"]
    if stage_id is None:
        stage_id = max((stage["id"] for stage in stages), default=0) + 1
    existing = [stage for stage in stages if stage["id"] == stage_id]
    if existing and not force:
        raise LearnKitError(f"ステージ {stage_id:03d} は既に存在します（{stem_of(existing[0])}）．上書きする場合は --force．")

    if kind is None:
        kind = "基礎" if not stages else "本線"
    stage = {"id": stage_id, "name": name, "title": title, "kind": kind,
             "concept": concept, "diff": diff, "bridge": bridge}
    stem = stem_of(stage)
    notebook_path = root / f"{stem}.ipynb"
    md_path = root / f"{stem}.md"
    for path in (notebook_path, md_path):
        if path.exists() and not force:
            raise LearnKitError(f"{path.name} は既に存在します．上書きする場合は --force を指定してください．")

    previous = [s for s in stages if s["id"] < stage_id]
    if from_stem is None and previous and not blank:
        from_stem = stem_of(previous[-1])

    manifest["stages"] = [s for s in stages if s["id"] != stage_id] + [stage]
    manifest["stages"].sort(key=lambda s: s["id"])
    values = stage_values(manifest, stage)

    source_path = root / f"{from_stem}.ipynb" if from_stem else None
    if source_path is not None and source_path.exists() and not blank:
        notebook = derive_notebook(load_notebook(source_path), from_stem, values)
        origin = f"{from_stem}.ipynb を複製"
    else:
        if from_stem and not blank:
            print(f"[new-stage] {from_stem}.ipynb が無いため，テンプレートから作成します．")
        notebook = notebook_from_cells(template_cells(values))
        origin = "テンプレートから作成"

    save_notebook(notebook_path, notebook)
    write_text(md_path, render((TEMPLATE_DIR / "stage.md").read_text(encoding="utf-8"), values))
    save_manifest(root, manifest)
    print(f"[new-stage] 作成: {notebook_path.name}（{origin}）")
    print(f"[new-stage] 作成: {md_path.name}")
    command_sync(root, quiet=True)
    return stage


# =====================================================================================
# check
# =====================================================================================

def check_markdown_sections(text, is_last):
    """解説 .md の必須の節が順に現れるかを検証する．

    Args:
        text (str): 解説の内容．
        is_last (bool): 最後のステージの場合 True．

    Returns:
        list[str]: 問題の説明のリスト．
    """
    headings = [line for line in text.split("\n") if re.match(r"^#{2,3}\s", line)]
    required = REQUIRED_SECTIONS + [LAST_SECTION_FINAL if is_last else LAST_SECTION_MIDDLE]
    issues = []
    position = 0
    for keyword in required:
        found = next((i for i in range(position, len(headings)) if keyword in headings[i]), None)
        if found is None:
            if any(keyword in heading for heading in headings):
                issues.append(f"節「{keyword}」の順序が規約と異なります．")
            else:
                issues.append(f"節「{keyword}」がありません．")
            continue
        position = found + 1

    if is_last and any(LAST_SECTION_MIDDLE in heading for heading in headings):
        issues.append(f"最後のステージでは「{LAST_SECTION_MIDDLE}」ではなく「{LAST_SECTION_FINAL}」の節にしてください．")
    return issues


def section_body(text, keyword):
    """見出しに keyword を含む節の本文を返す（同じ深さ以上の次の見出しまで）．

    Args:
        text (str): 解説の内容．
        keyword (str): 見出しに含まれる語．

    Returns:
        str | None: 節の本文．節が無い場合は None．
    """
    lines = text.split("\n")
    for index, line in enumerate(lines):
        match = re.match(r"^(#{2,3})\s", line)
        if match and keyword in line:
            level = len(match.group(1))
            body = []
            for following in lines[index + 1:]:
                next_match = re.match(r"^(#{1,6})\s", following)
                if next_match and len(next_match.group(1)) <= level and not body_inside_details(body):
                    break
                body.append(following)
            return "\n".join(body)
    return None


def body_inside_details(lines):
    """行のリストの末尾が <details> の内側かどうかを判定する．

    Args:
        lines (list[str]): 行．

    Returns:
        bool: <details> が閉じていない場合 True．
    """
    text = "\n".join(lines)
    return text.count("<details") > text.count("</details>")


def check_markdown(root, path, is_last):
    """解説 .md を検証する．

    Args:
        root (Path): 教材リポジトリのルート．
        path (Path): 解説のパス．
        is_last (bool): 最後のステージの場合 True．

    Returns:
        list[str]: 問題の説明のリスト．
    """
    text = path.read_text(encoding="utf-8")
    issues = check_markdown_sections(text, is_last)

    if text.count("<details") != text.count("</details>"):
        issues.append("<details> と </details> の数が一致しません．")
    for keyword in FOLDED_SECTIONS:
        body = section_body(text, keyword)
        if body is not None and "<details" not in body:
            issues.append(f"節「{keyword}」の内容（結果・解説・回答例）を <details> で折りたたんでください．")

    for match in IMAGE_PATTERN.finditer(text):
        target = match.group(1) or match.group(2)
        if re.match(r"^(https?:|data:)", target):
            continue
        if not (root / target).exists():
            issues.append(f"参照している図がありません: {target}")
    return issues


def local_module_names(root):
    """教材リポジトリ内で import されてはならないモジュール名を返す．

    Args:
        root (Path): 教材リポジトリのルート．

    Returns:
        set[str]: モジュール名．
    """
    names = {"machine_learning_utils", "programs"}
    names.update(path.stem for path in root.glob("*.py"))
    names.update(path.parent.name for path in root.glob("*/__init__.py"))
    return names


def check_notebook(root, manifest, stage, allow_unexecuted):
    """ノートブックを検証する．

    Args:
        root (Path): 教材リポジトリのルート．
        manifest (dict): マニフェスト．
        stage (dict): ステージ．
        allow_unexecuted (bool): True の場合，未実行のセルを許容する．

    Returns:
        list[str]: 問題の説明のリスト．
    """
    stem = stem_of(stage)
    try:
        notebook = load_notebook(root / f"{stem}.ipynb")
    except json.JSONDecodeError as error:
        return [f"ノートブックのJSONが壊れています: {error}"]

    issues = []
    cells = notebook.get("cells", [])
    if not cells or cells[0]["cell_type"] != "markdown" or colab_url(manifest, stem) not in cell_text(cells[0]):
        issues.append("冒頭のセルに正しい Colab バッジがありません（`learnkit.py sync` で修正できます）．")

    code_cells = [cell for cell in cells if cell["cell_type"] == "code" and cell_text(cell).strip()]
    code = "\n".join(cell_text(cell) for cell in code_cells)
    if not re.search(rf'^EXPERIMENT_NAME\s*=\s*["\']{re.escape(stem)}["\']', code, re.MULTILINE):
        issues.append(f'EXPERIMENT_NAME = "{stem}" がありません．')

    forbidden = local_module_names(root)
    imported = {name for name in IMPORT_PATTERN.findall(code) if name in forbidden}
    if imported:
        issues.append(f"教材リポジトリ内のモジュールを import しています（ノートブックは単体で実行できる必要があります）: {sorted(imported)}")

    issues.extend(check_execution(code_cells, allow_unexecuted))
    return issues


def check_execution(code_cells, allow_unexecuted):
    """コードセルの実行状態を検証する．

    Args:
        code_cells (list[dict]): 空でないコードセル．
        allow_unexecuted (bool): True の場合，未実行のセルを許容する．

    Returns:
        list[str]: 問題の説明のリスト．
    """
    issues = []
    counts = [cell.get("execution_count") for cell in code_cells]
    if not allow_unexecuted:
        if any(count is None for count in counts):
            issues.append("未実行のセルがあります（`learnkit.py run --only ...` で実行してください）．")
        elif counts != sorted(counts) or len(set(counts)) != len(counts):
            issues.append("セルが上から順に1回で実行されていません（`learnkit.py run` で再実行してください）．")

    for index, cell in enumerate(code_cells):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                issues.append(f"コードセル{index + 1}の出力にエラーがあります: {output.get('ename')}: {output.get('evalue')}")
            text = "".join(output.get("text", [])) if isinstance(output.get("text"), list) else str(output.get("text", ""))
            if "LEARNKIT_SMOKE" in text:
                issues.append("動作確認（--smoke）の出力が残っています．通常の設定で再実行してください．")
                break
    return issues


def find_todos(path):
    """ファイル中の TODO(AI) の位置を返す．

    Args:
        path (Path): 対象のファイル．

    Returns:
        list[str]: 位置の説明（"12行目" または "セル3"）．
    """
    if path.suffix == ".ipynb":
        notebook = load_notebook(path)
        return [f"セル{index + 1}" for index, cell in enumerate(notebook.get("cells", [])) if TODO_MARK in cell_text(cell)]
    lines = path.read_text(encoding="utf-8").split("\n")
    return [f"{index + 1}行目" for index, line in enumerate(lines) if TODO_MARK in line]


def collect_issues(root, allow_todo=False, allow_unexecuted=False):
    """教材リポジトリ全体を検証し，問題を収集する．

    Args:
        root (Path): 教材リポジトリのルート．
        allow_todo (bool): True の場合，TODO(AI) の残りを許容する．
        allow_unexecuted (bool): True の場合，未実行のノートブックを許容する．

    Returns:
        dict[str, list[str]]: ファイル名をキー，問題の説明のリストを値とする辞書．
    """
    manifest = load_manifest(root)
    issues = {}

    def add(name, messages):
        if messages:
            issues.setdefault(name, []).extend(messages)

    if not manifest["stages"]:
        add(MANIFEST_NAME, ["ステージがありません（`learnkit.py new-stage` で追加してください）．"])
    ids = [stage["id"] for stage in manifest["stages"]]
    if len(ids) != len(set(ids)):
        add(MANIFEST_NAME, ["ステージ番号が重複しています．"])

    stems = set()
    for index, stage in enumerate(manifest["stages"]):
        stem = stem_of(stage)
        stems.add(stem)
        is_last = index == len(manifest["stages"]) - 1
        for field in ("concept", "diff", "bridge"):
            if not stage.get(field) and not (field == "bridge" and is_last):
                add(MANIFEST_NAME, [f"{stem} の {field} が未記入です．"])
        for suffix in (".ipynb", ".md"):
            if not (root / f"{stem}{suffix}").exists():
                add(f"{stem}{suffix}", ["ファイルがありません．"])
        if (root / f"{stem}.md").exists():
            add(f"{stem}.md", check_markdown(root, root / f"{stem}.md", is_last))
        if (root / f"{stem}.ipynb").exists():
            add(f"{stem}.ipynb", check_notebook(root, manifest, stage, allow_unexecuted))

    for path in sorted(root.glob("ex*.ipynb")) + sorted(root.glob("ex*.md")):
        if re.match(r"ex\d{3}_", path.name) and path.stem not in stems:
            add(path.name, [f"{MANIFEST_NAME} に登録されていないステージのファイルです．"])

    if not allow_todo:
        targets = [root / "README.md", root / "curriculum_plan.md"]
        targets += [root / f"{stem}{suffix}" for stem in sorted(stems) for suffix in (".ipynb", ".md")]
        for path in targets:
            if path.exists():
                positions = find_todos(path)
                if positions:
                    add(path.name, [f"{TODO_MARK} が残っています: {', '.join(positions[:10])}{' ...' if len(positions) > 10 else ''}"])

    for path in synced_outputs(root, manifest):
        add(path.name, ["ステージ一覧・ナビゲーション等が curriculum.json と一致しません（`learnkit.py sync` を実行してください）．"])
    return issues


def command_check(root, allow_todo=False, allow_unexecuted=False):
    """教材リポジトリを検証し，結果を表示する．

    Args:
        root (Path): 教材リポジトリのルート．
        allow_todo (bool): True の場合，TODO(AI) の残りを許容する．
        allow_unexecuted (bool): True の場合，未実行のノートブックを許容する．

    Returns:
        int: 問題の件数．
    """
    issues = collect_issues(root, allow_todo, allow_unexecuted)
    total = sum(len(messages) for messages in issues.values())
    for name, messages in issues.items():
        print(f"■ {name}")
        for message in messages:
            print(f"  - {message}")
    if total:
        print(f"\n[check] {total} 件の問題があります．")
    else:
        print("[check] 問題はありません．")
    return total


# =====================================================================================
# run
# =====================================================================================

def command_run(root, only=None, smoke=False, timeout=3600, kernel=None, python=None):
    """ノートブックをステージの順に実行し，出力を保存する．

    通常の実行では，ノートブック自身に出力を上書き保存する．
    `smoke=True` の場合は，環境変数 LEARNKIT_SMOKE=1 で短時間の動作確認を行い，
    出力は `outputs/_smoke/` に保存する（教材のノートブックは変更しない）．

    Args:
        root (Path): 教材リポジトリのルート．
        only (list[str] | None): 実行するノートブック（拡張子なし）．None の場合は全ステージ．
        smoke (bool): True の場合，動作確認として実行する．
        timeout (int): 1セルあたりのタイムアウト（秒）．
        kernel (str | None): 使用する Jupyter カーネル名．
        python (str | None): jupyter を実行する Python．None の場合はこのスクリプトを実行している Python．

    Returns:
        None
    """
    manifest = load_manifest(root)
    stems = [stem_of(stage) for stage in manifest["stages"]]
    if only:
        unknown = [stem for stem in only if stem not in stems]
        if unknown:
            raise LearnKitError(f"curriculum.json に登録されていないノートブックです: {unknown}")
        stems = [stem for stem in stems if stem in only]

    env = dict(os.environ)
    if smoke:
        env["LEARNKIT_SMOKE"] = "1"
    smoke_dir = root / "outputs" / "_smoke"

    for stem in stems:
        command = [python or sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                   f"--ExecutePreprocessor.timeout={timeout}"]
        if kernel:
            command.append(f"--ExecutePreprocessor.kernel_name={kernel}")
        if smoke:
            command += ["--output-dir", str(smoke_dir), "--output", f"{stem}.ipynb"]
        else:
            command.append("--inplace")
        command.append(str(root / f"{stem}.ipynb"))

        print(f"[run] {'動作確認' if smoke else '実行'}: {stem}.ipynb")
        result = subprocess.run(command, cwd=root, env=env)
        if result.returncode != 0:
            raise LearnKitError(f"{stem}.ipynb の実行に失敗しました．エラーを修正して再実行してください．")
    print("[run] 完了しました．")


# =====================================================================================
# CLI
# =====================================================================================

def build_parser():
    """コマンドライン引数のパーサを作成する．

    Returns:
        argparse.ArgumentParser: パーサ．
    """
    parser = argparse.ArgumentParser(description="段階的な教材リポジトリを作成・検証するCLI")
    parser.add_argument("--root", default=".", help="教材リポジトリのルート（既定: カレントディレクトリ）")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="教材リポジトリを初期化する")
    p.add_argument("--title", required=True, help="教材のタイトル")
    p.add_argument("--goal", required=True, help="到達目標（「〜を説明・実装できる」の形）")
    p.add_argument("--repo", required=True, help="公開先の owner/repo")
    p.add_argument("--branch", default="main", help="公開先のブランチ（既定: main）")
    p.add_argument("--force", action="store_true", help="既存の curriculum.json，README.md 等を上書きする")

    p = sub.add_parser("new-stage", help="ステージを追加し，ノートブックと解説の雛形を作成する")
    p.add_argument("name", help="ステージ名（例: mnist_vae）．ファイル名は ex{NNN}_{name} になる")
    p.add_argument("--title", required=True, help="ステージのタイトル")
    p.add_argument("--concept", default="", help="新しく学ぶこと")
    p.add_argument("--diff", default="", help="前のステージとの差分")
    p.add_argument("--bridge", default="", help="このステージの限界（次のステージの動機）")
    p.add_argument("--kind", choices=STAGE_KINDS, help="ステージの種類")
    p.add_argument("--id", type=int, dest="stage_id", help="ステージ番号（既定: 最大の番号+1）")
    p.add_argument("--from", dest="from_stem", help="複製元のノートブック（例: ex002_mnist_ae．既定: 直前のステージ）")
    p.add_argument("--blank", action="store_true", help="前のステージを複製せず，テンプレートから作成する")
    p.add_argument("--force", action="store_true", help="既存のステージ・ファイルを上書きする")

    sub.add_parser("sync", help="ステージ一覧・ナビゲーション・README の表と図を curriculum.json に同期する")

    p = sub.add_parser("run", help="ノートブックを順に実行する")
    p.add_argument("--only", nargs="+", help="実行するノートブック（例: ex001_mnist_nn）")
    p.add_argument("--smoke", action="store_true", help="1エポック・少量のデータで動作確認する（出力は outputs/_smoke/）")
    p.add_argument("--timeout", type=int, default=3600, help="1セルあたりのタイムアウト（秒）")
    p.add_argument("--kernel", help="使用する Jupyter カーネル名")
    p.add_argument("--python", help="jupyter を実行する Python（例: .venv_pytorch/bin/python）")

    p = sub.add_parser("check", help="教材リポジトリの構成・整合性を検証する")
    p.add_argument("--allow-todo", action="store_true", help="TODO(AI) の残りを許容する（作成途中の確認用）")
    p.add_argument("--allow-unexecuted", action="store_true", help="未実行のノートブックを許容する（作成途中の確認用）")
    return parser


def main(argv=None):
    """CLI のエントリポイント．

    Args:
        argv (list[str] | None): コマンドライン引数．

    Returns:
        int: 終了コード．
    """
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    try:
        if args.command == "init":
            command_init(root, args.title, args.goal, args.repo, args.branch, args.force)
        elif args.command == "new-stage":
            command_new_stage(root, args.name, args.title, args.concept, args.diff, args.bridge, args.kind,
                              args.stage_id, args.from_stem, args.blank, args.force)
        elif args.command == "sync":
            command_sync(root)
        elif args.command == "run":
            command_run(root, args.only, args.smoke, args.timeout, args.kernel, args.python)
        elif args.command == "check":
            return 1 if command_check(root, args.allow_todo, args.allow_unexecuted) else 0
    except LearnKitError as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
