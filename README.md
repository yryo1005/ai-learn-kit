# ai-learn-kit
`ai-learn-kit` は，**到達目標を決め，そこまでを段階的に学べる教材リポジトリ**をAIエディタ（Claude Code等）に作成させるための規約・テンプレート・CLIである．
例えば「VAEを理解する」という目標に対して，NN → AE → VAE のように前提知識を逆算してステージに刻み，各ステージを「1つのノートブック（`.ipynb`）と1つの解説（`.md`）」として作成する．

完成形のサンプル: [Deep_Variation_Information_Bottleneck_training](https://github.com/yryo1005/Deep_Variation_Information_Bottleneck_training)（NN → AE → VAE → Deep VIB → 可視化・発展の8ステージ）

```mermaid
flowchart LR
    A(["order<br/>到達目標・学習者の前提"]) --> B["curriculum_plan.md<br/>前提知識の逆算・ステージ設計"]
    B --> C["learnkit.py new-stage<br/>共通骨格から雛形を生成"]
    C --> D["ex{NNN}_*.ipynb<br/>差分だけ実装して実行"]
    D --> E["figures/ex{NNN}_*/<br/>学習曲線・可視化"]
    E --> F["ex{NNN}_*.md<br/>実測値で解説を記述"]
    F --> G["learnkit.py check<br/>構成・整合性の検証"]
    G --> H(["教材リポジトリ<br/>README.md + 全ステージ"])
    F -. 次のステージ .-> C
```

---

# 教材の思想
1. **到達目標から逆算する**：目標の概念に必要な前提を洗い出し，依存関係の順に並べる．
2. **1ステージ1新概念**：前のステージとの差分を最小にし，何が結果を変えたのかを明確にする．
3. **限界が次の動機になる**：各ステージの最後に，そのモデルや手法の限界を実験で示し，次のステージが必要な理由にする．
4. **同じ骨格**：`iteration` / `epoch` / `train` の学習ループ，関数名，共通ハイパーパラメータを全ステージで揃える．
5. **問い → 予想 → 実験 → 結果 → 考察 → 発展**：結果・解説・回答例は折りたたみにし，学習者が先に予想を書けるようにする．
6. **実測値だけを書く**：解説に記載する数値と図は，AIエディタが実際にノートブックを実行して得たものに限る．

---

# ファイル構成
AIエディタは，教材の作成を依頼された場合，本ファイルに加えて `curriculum.md` と `pedagogy.md` を必ず読み込む．

| ファイル | 内容 | 読むタイミング |
| :--- | :--- | :--- |
| `README.md` | 本ファイル．概要と使い方 | 最初 |
| `curriculum.md` | **教材作成の手順と規約（必読）** | 教材作成の全工程 |
| `pedagogy.md` | **ステージの刻み方，問いと発展課題の設計の原則（必読）** | ステージ設計時 |
| `templates/curriculum_plan.md` | 教材の設計書のテンプレート | ステージ設計時 |
| `templates/stage.md` | 各ステージの解説 `.md` のテンプレート | 解説の執筆時 |
| `templates/notebook_template.py` | 各ステージのノートブックの共通骨格（jupytextのpercent形式） | `new-stage` が使用 |
| `templates/README.md` | 教材リポジトリの `README.md` のテンプレート | `init` が使用 |
| `templates/gitignore` | 教材リポジトリの `.gitignore`（公開するファイルのみを許可） | `init` が使用 |
| `scripts/learnkit.py` | 雛形の生成・同期・検証・一括実行を行うCLI（標準ライブラリのみで動作） | 全工程 |
| `examples/order_example.md` | 教材作成を依頼するorderの例 | 依頼時 |
| `CLAUDE.md`，`AGENTS.md` | ai-dev-kit を使わずに単独で使う場合のエントリポイント | 単独で使う場合 |
| `tests/test_learnkit.py` | `learnkit.py` のテスト | kitの開発時 |

---

# 導入
## ai-dev-kit のプロジェクトで使う場合（推奨）
ai-dev-kit の `init_project.sh` を実行すると，`.ai/ai-learn-kit` にサブモジュールとして自動で追加される．
`@.ai/ai-dev-kit/root_prompt.md` の「読み込むドキュメント」の表から本kitの規約が参照されるため，orderに本ファイルの参照を書く必要は無い．

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/yryo1005/ai-dev-kit/master/init_project.sh) --tokens
```

既存のプロジェクトに追加する場合は，以下を実行する．

```bash
git submodule add https://github.com/yryo1005/ai-learn-kit.git .ai/ai-learn-kit
```

## 単独で使う場合
ai-dev-kit を使わない場合は，`CLAUDE.md`・`AGENTS.md` をプロジェクトルートにコピーする．

```bash
git submodule add https://github.com/yryo1005/ai-learn-kit.git .ai/ai-learn-kit
cp .ai/ai-learn-kit/CLAUDE.md .ai/ai-learn-kit/AGENTS.md ./
```

---

# 使い方
1. orderに到達目標と学習者の前提を書く（`examples/order_example.md` を参照）．記載の無い項目は `curriculum.md` の Phase 0 の既定値が用いられる．

```text
VAEを理解するための教材を作成してください．
* 学習者: Pythonは書ける．PyTorchは初めて．線形代数と確率の基礎は既習
* 実行環境: Google Colab（GPU）
* 公開先: https://github.com/yryo1005/vae_training
```

2. AIエディタは `curriculum.md` の手順（Phase 0〜6）に従い，設計書 `curriculum_plan.md` → 各ステージのノートブックと解説 → `README.md` の順に作成する．
   設計だけを先に確認したい場合は，orderに「計画のみ」と書くと `curriculum_plan.md` の作成で止まる．
3. 作成後，教材リポジトリを公開する．

## 公開に関する注意
`learnkit.py init` が作成する `.gitignore` は，`README.md`，`ex*.ipynb`，`ex*.md`，`figures/`，`report.pdf` 等の公開用のファイルのみを許可する（既存の `.gitignore` は `.gitignore.orig` に退避される）．
そのため，`.orders/`，`.reports/`，`curriculum_plan.md`，`outputs/` 等は公開されない．

ただし，`.gitignore` は既にGitで追跡しているファイルには効かない．`init_project.sh` の直後に初期状態をコミットした場合は，公開前に以下で追跡を外す．

```bash
git rm -r --cached --ignore-unmatch CLAUDE.md AGENTS.md .vscode .orders .reports .logs
git commit -m "Untrack private files"
```

---

# `learnkit.py` のコマンド
プロジェクトルート（教材リポジトリのルート）で実行する．標準ライブラリのみで動作する（`run` のみ `jupyter nbconvert` が必要）．

| コマンド | 内容 |
| :--- | :--- |
| `init` | 教材リポジトリを初期化する（`curriculum.json`，`README.md`，`.gitignore`，`curriculum_plan.md`，`figures/` を作成） |
| `new-stage` | ステージを追加し，直前のステージ（または `--from` で指定したステージ）を複製してノートブックと解説の雛形を作成する |
| `sync` | 各ノートブックの冒頭，各解説のナビゲーション，`README.md` のステージの表と図を `curriculum.json` に同期する |
| `run` | ノートブックを順に実行し，出力をノートブックに保存する |
| `check` | 構成・整合性を検証する（未記入の `TODO(AI)`，未実行のセル，存在しない図の参照，必須の節の欠落，ローカルの `.py` の import 等） |

```bash
L=.ai/ai-learn-kit/scripts/learnkit.py

# 教材リポジトリの初期化
python $L init --title "VAE 入門" --goal "VAEの目的関数と潜在空間の性質を説明・実装できる" --repo yryo1005/vae_training

# ステージの追加（ex001_mnist_nn.ipynb と ex001_mnist_nn.md を作成）
python $L new-stage mnist_nn --title "全結合 NN による MNIST 分類" --concept "学習ループ，交差エントロピー" --diff "講座の基礎" --bridge "潜在表現を学習していない"

# ステージ一覧を同期
python $L sync

# 動作確認（1エポック・データの2%．出力は outputs/_smoke/ に保存し，教材のノートブックは変更しない）
python $L run --only ex001_mnist_nn --smoke

# 全ノートブックを順に実行（--only で個別に実行，--python で仮想環境を指定）
python $L run --python .venv_pytorch/bin/python

# 構成・整合性の検証（公開前にエラーを0件にする）
python $L check

# 作成途中の検証（TODO(AI) と未実行のセルを許容する）
python $L check --allow-todo --allow-unexecuted
```

主なオプションは `python $L {コマンド} -h` で確認できる．

---

# 他のkitとの関係
| kit | 教材作成での役割 |
| :--- | :--- |
| ai-dev-kit | order・report，進捗の報告，Docstring，文章の表記の規約に従う．教材のノートブックはColabで単体実行するため，`machine_learning.md` の一部の規約（`programs/` の分割，`machine_learning_utils.py` の import 等）は `curriculum.md` の「6. ai-dev-kit の規約との関係」に従って置き換える |
| ai-survey-kit | 発展ステージの題材や，解説で紹介する原論文を選ぶ際に，調査結果（`survey/`）を参照する．文献を引用する場合は ai-survey-kit の規約に従う |
| ai-tex-kit | 教材全体をまとめたPDF（`report.pdf`）を作成する場合に使用する |
| ai-pptx-kit | 講義スライドを作成する場合に使用する |
| ai-agent | `.reports/` に記載した未解決の論点をもとに，追加するステージ等の次のorderを検討する |

---

# kitの開発
```bash
python -m pytest -q tests
```
