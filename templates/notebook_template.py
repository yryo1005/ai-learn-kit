# 教材ノートブックの共通骨格（jupytext の percent 形式）．
# `learnkit.py new-stage` がこのファイルからノートブックを生成する．
# "# %% [markdown]" で始まるセルはマークダウン（各行の先頭の "# " を除去する），"# %%" で始まるセルはコードである．
# {{...}} はステージごとの値に置換される．この説明の行（最初のセルより前）はノートブックに含まれない．
# 雛形のままで「全結合NNによるMNIST分類」として実行できる．ステージの主題に合わせて変更する．

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({{COLAB_URL}})
#
# > **実行する前に，ランタイムを GPU に変更すること．**
# > Google Colab ではメニューの **ランタイム → ランタイムのタイプを変更** を開き，ハードウェア アクセラレータを **GPU** に設定して保存する．CPU のままだと学習に時間がかかる．
#
# # 実験{{STAGE_NUMBER}}: {{STAGE_TITLE}}
#
# この講座では，次の順でプログラムを積み上げる．
#
# <!-- learnkit:stages:start -->
# <!-- learnkit:stages:end -->
#
# TODO(AI): 本ステージで何を学習するかと，前のステージからの変更点（{{DIFF}}）を1〜3文で書く．詳しい解説は `{{EXPERIMENT_NAME}}.md` にある．

# %% [markdown]
# ## 1. ライブラリ
#
# Colab に標準で入っているライブラリのみを用いる．

# %%
import json
import os
import random
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm

# %% [markdown]
# ## 2. 再現性とログ
#
# `set_seed` で乱数を固定し，同じコードから同じ結果が得られるようにする．`ResultLogger` はエポックごとの指標を記録し，`log.json` として保存する．

# %%
def set_seed(seed=0):
    """実験の再現性を担保するため，乱数シードを固定する．

    Args:
        seed (int): 固定するシード値．

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)


class ResultLogger:
    """学習中の指標（Loss，Accuracy など）を記録し，JSON として保存・読み込みするクラス．

    Args:
        target_path (str | None): 既存のログ JSON のパス．指定した場合は初期化時に読み込む．
    """

    def __init__(self, target_path=None):
        self.names = None
        self.history = {}
        if target_path:
            self.load(target_path)

    def set_names(self, *names):
        """記録する指標名を登録する．

        Args:
            *names (str): 指標名．例: "train_loss", "val_accuracy"．

        Returns:
            None
        """
        if self.names:
            raise RuntimeError("指標名はすでに登録されている．")
        self.names = list(names)
        for name in self.names:
            if name not in self.history:
                self.history[name] = []

    def __call__(self, *values):
        """1 エポック分の指標の値を履歴へ追加する．

        Args:
            *values (float): set_names で登録した順の値．

        Returns:
            None
        """
        if self.names is None:
            raise RuntimeError("先に set_names で指標名を登録すること．")
        if len(values) != len(self.names):
            raise RuntimeError("値の数が登録済みの指標名の数と一致しない．")
        for name, value in zip(self.names, values):
            self.history[name].append(value)

    def save(self, target_path):
        """履歴を JSON ファイルとして保存する．

        Args:
            target_path (str): 保存先のパス．

        Returns:
            None
        """
        with open(target_path, "w") as f:
            json.dump(self.history, f, indent=4)

    def load(self, target_path):
        """JSON ファイルから履歴を読み込む．

        Args:
            target_path (str): 読み込む JSON のパス．

        Returns:
            None
        """
        with open(target_path, "r") as f:
            self.history = json.load(f)
        self.names = list(self.history.keys())

    def __getitem__(self, key):
        """指定した指標の履歴を取得する．

        Args:
            key (str): 指標名．

        Returns:
            history (list): 指標の履歴．存在しない場合は空のリスト．
        """
        return self.history.get(key, [])

# %% [markdown]
# ## 3. 実験設定
#
# 結果は `outputs/{{EXPERIMENT_NAME}}/{最適化手法}/{学習率}_{バッチサイズ}/{seed}/` に，解説で用いる図は `figures/{{EXPERIMENT_NAME}}/` に保存する．
# `MONITOR` は最良モデルを選ぶ基準の指標である．

# %%
EXPERIMENT_NAME = "{{EXPERIMENT_NAME}}"

PROJECT_ROOT = Path.cwd().resolve()
DATASET_DIR = PROJECT_ROOT / "datasets" / EXPERIMENT_NAME
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME
FIGURE_DIR = PROJECT_ROOT / "figures" / EXPERIMENT_NAME

EPOCHS = 10
SEEDS = [0]
OPTIMIZER_NAMES = ["Adam"]
LEARNING_RATES = [0.001]
BATCH_SIZES = [128]
MONITOR = "val_loss"   # 最良モデルの基準
MONITOR_MODE = "min"   # "min": 小さいほど良い，"max": 大きいほど良い
DATA_FRACTION = 1.0    # 学習に用いるデータの割合

# 動作確認用の設定（教材の作成時に `learnkit.py run --smoke` から実行した場合のみ有効．学習者は変更不要）
if os.environ.get("LEARNKIT_SMOKE") == "1":
    EPOCHS = 1
    DATA_FRACTION = 0.02
    OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "_smoke" / EXPERIMENT_NAME
    FIGURE_DIR = OUTPUT_ROOT / "figures"
    print("LEARNKIT_SMOKE: 動作確認モードで実行している（1エポック・データの2%）．")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}")

# %% [markdown]
# ## 4. データセットと DataLoader
#
# MNIST は学習用 60,000 枚，検証用 10,000 枚の手書き数字画像である．画素値は `ToTensor()` により $[0, 1]$ に変換する．DataLoader のシャッフルは `torch.Generator` で決定論的にする．

# %%
def load_dataloader(seed=0, batch_size=128):
    """MNIST の学習用・検証用 DataLoader を作成する．

    Args:
        seed (int): データの並びを固定する乱数シード．
        batch_size (int): ミニバッチサイズ．

    Returns:
        train_dataloader (torch.utils.data.DataLoader): 学習用 DataLoader．
            各バッチは画像 (N, 1, 28, 28) とラベル (N,) のタプル．
        test_dataloader (torch.utils.data.DataLoader): 検証用 DataLoader．各バッチの形状は学習用と同じ．
    """
    set_seed(seed)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=str(DATASET_DIR), train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=str(DATASET_DIR), train=False, download=True, transform=transform)

    if DATA_FRACTION < 1.0:
        train_dataset = Subset(train_dataset, range(int(len(train_dataset) * DATA_FRACTION)))
        test_dataset = Subset(test_dataset, range(int(len(test_dataset) * DATA_FRACTION)))

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_dataloader, test_dataloader

# %% [markdown]
# ### データの確認
#
# 学習用 DataLoader から 1 バッチを取り出し，形状とラベルの例を表示する．

# %%
preview_loader, _ = load_dataloader(seed=0, batch_size=8)
preview_images, preview_labels = next(iter(preview_loader))
print(f"images shape: {tuple(preview_images.shape)}")
print(f"labels shape: {tuple(preview_labels.shape)}")
print(f"labels      : {preview_labels.tolist()}")

fig, axes = plt.subplots(1, 8, figsize=(16, 2.5))
for i, ax in enumerate(axes):
    ax.imshow(preview_images[i, 0], cmap="gray")
    ax.set_title(f"label={preview_labels[i].item()}")
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. モデル
#
# 28×28 の画像を 784 次元のベクトルに平坦化し，全結合層 784-256-128-10 でクラスごとのスコア（ロジット）を出力する．

# %%
class MNISTClassifier(nn.Module):
    """MNIST を 10 クラスに分類する全結合ニューラルネットワーク．

    Args:
        hidden_dims (tuple[int, int]): 中間層のニューロン数．
    """

    def __init__(self, hidden_dims=(256, 128)):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 10)

    def forward(self, x):
        """順伝播を計算する．

        Args:
            x (torch.Tensor): 入力画像．形状は (N, 1, 28, 28)．

        Returns:
            logits (torch.Tensor): クラスごとのロジット．形状は (N, 10)．
        """
        h = x.flatten(start_dim=1)   # (N, 784)
        h = F.relu(self.fc1(h))      # (N, 256)
        h = F.relu(self.fc2(h))      # (N, 128)
        logits = self.fc3(h)         # (N, 10)
        return logits


def load_model(ModelClass, weight_path=None, seed=0, **model_kwargs):
    """モデルを生成し，必要であれば学習済みの重みを読み込む．

    Args:
        ModelClass (type): 生成するモデルのクラス．
        weight_path (str | None): 重みのパス．None の場合は初期値のまま返す．
        seed (int): パラメータの初期値を固定する乱数シード．
        **model_kwargs: ModelClass に渡す引数．

    Returns:
        model (torch.nn.Module): 生成したモデル．
    """
    set_seed(seed)
    model = ModelClass(**model_kwargs)
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    return model


MODEL_CLASS = MNISTClassifier
MODEL_KWARGS = {}

# %% [markdown]
# ## 6. 誤差関数と評価指標
#
# 誤差関数は交差エントロピーである．ミニバッチサイズを $N$，サンプル $n$ の正解クラスを $y_{n}$，クラス $k$ のロジットを $z_{n,k}$ として，
#
# $$
# \mathcal{L} = -\frac{1}{N}\sum_{n=1}^{N}\left(z_{n,y_{n}} - \log\sum_{k=1}^{10}\exp(z_{n,k})\right)
# $$
#
# である．評価指標は Accuracy である．`make_teacher_signals` は，ミニバッチから教師信号を作る関数である（分類ではラベルそのもの）．

# %%
def make_teacher_signals(inputs, labels):
    """ミニバッチから教師信号を作成する．

    Args:
        inputs (torch.Tensor): 入力画像．形状は (N, 1, 28, 28)．
        labels (torch.Tensor): ラベル．形状は (N,)．

    Returns:
        teacher_signals (torch.Tensor): 教師信号．分類ではラベル (N,)．
    """
    return labels


def loss_func(outputs, teacher_signals):
    """交差エントロピー誤差を計算する．

    Args:
        outputs (torch.Tensor): モデルの出力（ロジット）．形状は (N, 10)．
        teacher_signals (torch.Tensor): 正解ラベル．形状は (N,)．

    Returns:
        loss (torch.Tensor): 1 データあたりの平均誤差．形状は ()．
    """
    return F.cross_entropy(outputs, teacher_signals)


def metrics_func(outputs, teacher_signals):
    """評価指標を計算する．

    Args:
        outputs (torch.Tensor): モデルの出力（ロジット）．形状は (N, 10)．
        teacher_signals (torch.Tensor): 正解ラベル．形状は (N,)．

    Returns:
        metrics_to_value (dict[str, float]): 指標名をキー，1 データあたりの平均値を値とする辞書．
    """
    accuracy = (outputs.argmax(dim=1) == teacher_signals).float().mean().item()
    return {"accuracy": accuracy}

# %% [markdown]
# ## 7. 学習ループ
#
# 学習は次の 3 段に分ける．
#
# 1. `iteration`: 1 ミニバッチの順伝播・誤差の計算・（学習時のみ）パラメータの更新
# 2. `epoch`: DataLoader 全件を処理し，1 データあたりの平均指標を返す
# 3. `train`: エポックを繰り返し，最良モデルとログを保存する
#
# `optimizer` が `None` のときは検証として動作する（`model.eval()`，`torch.no_grad()`）．

# %%
def iteration(model, inputs, teacher_signals, optimizer=None):
    """1 ミニバッチを学習または検証する．

    Args:
        model (torch.nn.Module): 学習 / 検証するモデル．
        inputs (torch.Tensor): 入力．形状は (N, ...)．
        teacher_signals (torch.Tensor): 教師信号．形状は (N, ...)．
        optimizer (torch.optim.Optimizer | None): 最適化手法．None なら検証．

    Returns:
        metrics_to_value (dict[str, float]): "loss" と評価指標の 1 データあたりの平均値の辞書．
    """
    if optimizer is None:
        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_func(outputs, teacher_signals)
    else:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, teacher_signals)
        loss.backward()
        optimizer.step()

    metrics_to_value = {"loss": loss.item()}
    with torch.no_grad():
        metrics_to_value.update(metrics_func(outputs, teacher_signals))
    return metrics_to_value


def epoch(model, dataloader, optimizer=None):
    """DataLoader 全件を 1 周し，1 データあたりの平均指標を返す．

    Args:
        model (torch.nn.Module): 学習 / 検証するモデル．
        dataloader (torch.utils.data.DataLoader): (入力, ラベル) を返す DataLoader．
        optimizer (torch.optim.Optimizer | None): 最適化手法．None なら検証．

    Returns:
        metrics_to_value (dict[str, float]): エポック全体の 1 データあたりの平均指標．
    """
    model.train(optimizer is not None)
    device = next(model.parameters()).device
    sum_metrics = {}
    n_total = 0

    progress = tqdm(dataloader, leave=False)
    for inputs, labels in progress:
        inputs = inputs.to(device)
        labels = labels.to(device)
        teacher_signals = make_teacher_signals(inputs, labels)

        batch_metrics = iteration(model, inputs, teacher_signals, optimizer)
        batch_size = inputs.size(0)
        n_total += batch_size
        for name, value in batch_metrics.items():
            sum_metrics[name] = sum_metrics.get(name, 0.0) + value * batch_size

        progress.set_postfix({name: f"{total / n_total:.4f}" for name, total in sum_metrics.items()})

    return {name: total / n_total for name, total in sum_metrics.items()}


def build_optimizer(model, optimizer_name, lr):
    """最適化手法の名前から Optimizer を生成する．

    Args:
        model (torch.nn.Module): パラメータを更新するモデル．
        optimizer_name (str): 最適化手法の名前．"Adam" または "SGD"．
        lr (float): 学習率．

    Returns:
        optimizer (torch.optim.Optimizer): 生成した Optimizer．
    """
    if optimizer_name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer_name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    raise ValueError(f"未対応の最適化手法である: {optimizer_name}")


def record(logger, epoch_index, epochs, train_metrics, val_metrics):
    """1 エポック分の指標を記録し，表示する．

    Args:
        logger (ResultLogger): 記録先のロガー．
        epoch_index (int): エポック番号．
        epochs (int): 総エポック数．
        train_metrics (dict[str, float]): 学習用データの指標．
        val_metrics (dict[str, float]): 検証用データの指標．

    Returns:
        metrics (dict[str, float]): "train_{指標}"，"val_{指標}" をキーとする辞書．
    """
    metrics = {f"train_{name}": value for name, value in train_metrics.items()}
    metrics.update({f"val_{name}": value for name, value in val_metrics.items()})
    if logger.names is None:
        logger.set_names(*metrics.keys())
    logger(*[metrics[name] for name in logger.names])

    text = " ".join(f"{name}={value:.4f}" for name, value in metrics.items())
    print(f"[epoch {epoch_index}/{epochs}] {text}")
    return metrics


def is_better(value, best_value):
    """MONITOR の値が過去の最良値より良いかを判定する．

    Args:
        value (float): 今回の値．
        best_value (float): 過去の最良値．

    Returns:
        better (bool): 良い場合 True．
    """
    if MONITOR_MODE == "min":
        return value < best_value
    return value > best_value


def train(target_dir, ModelClass, epochs, batch_size, seed=0, lr=0.001, optimizer_name="Adam", model_kwargs=None):
    """モデルを学習し，最良の重みとログを保存する．

    Args:
        target_dir (str): 結果の保存先ディレクトリ．
        ModelClass (type): 学習するモデルのクラス．
        epochs (int): 学習するエポック数．
        batch_size (int): ミニバッチサイズ．
        seed (int): 乱数シード．
        lr (float): 学習率．
        optimizer_name (str): 最適化手法の名前．
        model_kwargs (dict | None): ModelClass に渡す引数．

    Returns:
        None
    """
    os.makedirs(target_dir, exist_ok=True)
    train_dataloader, test_dataloader = load_dataloader(seed=seed, batch_size=batch_size)
    model = load_model(ModelClass, seed=seed, **(model_kwargs or {})).to(DEVICE)
    optimizer = build_optimizer(model, optimizer_name, lr)
    logger = ResultLogger()

    # 0 エポック目: 学習を行わずに初期値で評価する
    train_metrics = epoch(model, train_dataloader, optimizer=None)
    val_metrics = epoch(model, test_dataloader, optimizer=None)
    metrics = record(logger, 0, epochs, train_metrics, val_metrics)
    best_value = metrics[MONITOR]
    torch.save(model.state_dict(), os.path.join(target_dir, "best_model.pth"))

    for epoch_index in range(1, epochs + 1):
        train_metrics = epoch(model, train_dataloader, optimizer=optimizer)
        val_metrics = epoch(model, test_dataloader, optimizer=None)
        metrics = record(logger, epoch_index, epochs, train_metrics, val_metrics)

        if is_better(metrics[MONITOR], best_value):
            best_value = metrics[MONITOR]
            torch.save(model.state_dict(), os.path.join(target_dir, "best_model.pth"))

    logger.save(os.path.join(target_dir, "log.json"))

# %% [markdown]
# ## 8. 学習の実行
#
# 全てのハイパーパラメータの組み合わせについて学習する．`log.json` と `best_model.pth` が既にある条件はスキップする．

# %%
for optimizer_name, lr, batch_size, seed in product(OPTIMIZER_NAMES, LEARNING_RATES, BATCH_SIZES, SEEDS):
    target_dir = OUTPUT_ROOT / optimizer_name / f"{lr}_{batch_size}" / str(seed)
    if (target_dir / "log.json").exists() and (target_dir / "best_model.pth").exists():
        print(f"skip: {target_dir}")
        continue

    print(f"train: {target_dir}")
    train(
        target_dir=str(target_dir),
        ModelClass=MODEL_CLASS,
        epochs=EPOCHS,
        batch_size=batch_size,
        seed=seed,
        lr=lr,
        optimizer_name=optimizer_name,
        model_kwargs=MODEL_KWARGS,
    )

# %% [markdown]
# ## 9. 学習曲線
#
# 保存したログから，指標ごとに学習用と検証用の推移を描画し，`figures/{{EXPERIMENT_NAME}}/` に保存する．

# %%
def plot_history(logger, save_dir):
    """指標ごとに学習用・検証用の推移を描画し，画像として保存する．

    Args:
        logger (ResultLogger): 学習のログ．
        save_dir (pathlib.Path): 図の保存先ディレクトリ．"{指標名}.png" として保存する．

    Returns:
        None
    """
    metric_names = [name[len("train_"):] for name in logger.names if name.startswith("train_")]
    for metric in metric_names:
        epochs_axis = list(range(len(logger[f"train_{metric}"])))
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(epochs_axis, logger[f"train_{metric}"], marker="o", label="train")
        ax.plot(epochs_axis, logger[f"val_{metric}"], marker="o", label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_dir / f"{metric}.png", dpi=150)
        plt.show()


BASE_DIR = OUTPUT_ROOT / OPTIMIZER_NAMES[0] / f"{LEARNING_RATES[0]}_{BATCH_SIZES[0]}" / str(SEEDS[0])
logger = ResultLogger(str(BASE_DIR / "log.json"))
plot_history(logger, FIGURE_DIR)

best_index = int(np.argmin(logger[MONITOR]) if MONITOR_MODE == "min" else np.argmax(logger[MONITOR]))
for name in logger.names:
    print(f"{name:20s} epoch0={logger[name][0]:.4f}  final={logger[name][-1]:.4f}  best_epoch({best_index})={logger[name][best_index]:.4f}")

# %% [markdown]
# ## 10. 結果の可視化
#
# TODO(AI): 本ステージの主題が観察できる可視化に置き換える（例: 再構成画像，潜在空間の散布図）．雛形では最良モデルによる検証画像の予測を表示する．

# %%
_, test_loader = load_dataloader(seed=SEEDS[0], batch_size=8)
images, labels = next(iter(test_loader))

model = load_model(MODEL_CLASS, weight_path=str(BASE_DIR / "best_model.pth"), seed=SEEDS[0], **MODEL_KWARGS).to(DEVICE)
model.eval()
with torch.no_grad():
    predictions = model(images.to(DEVICE)).argmax(dim=1).cpu()

fig, axes = plt.subplots(1, 8, figsize=(16, 2.5))
for i, ax in enumerate(axes):
    ax.imshow(images[i, 0], cmap="gray")
    ax.set_title(f"y={labels[i].item()} / pred={predictions[i].item()}")
    ax.axis("off")
plt.tight_layout()
fig.savefig(FIGURE_DIR / "predictions.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 11. 次のステージへ
#
# TODO(AI): 本ステージの限界（{{BRIDGE}}）を，上の結果に基づいて1〜3文で書く．詳しくは `{{EXPERIMENT_NAME}}.md` の「8. 次のステージへ」を参照するよう書く．
