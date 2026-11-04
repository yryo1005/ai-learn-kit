<!-- learnkit:nav:start -->
<!-- learnkit:nav:end -->

# 実験{{STAGE_NUMBER}}: {{STAGE_TITLE}}

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({{COLAB_URL}})

## 0. 位置づけ

TODO(AI): 前のステージとの差分（{{DIFF}}）と，本ステージで新しく学ぶこと（{{CONCEPT}}）を2〜4文で書く．講座全体の中での役割にも触れる．

## 1. 目的

TODO(AI): 本ステージの目的を1段落で書く．「〜を実装し，〜を確認すること」の形とする．

## 2. 手法

### 2.1 モデル

TODO(AI): 入力・出力の形状と，各層の写像を数式で書く．前ステージと同じ部分は「実験{前のステージ番号}と同型である」と書いて省略してよい．

### 2.2 誤差関数

TODO(AI): 目的関数を数式で書き，式中のすべての記号を説明する．前ステージの目的関数との違いを明記する．

### 2.3 評価指標

TODO(AI): 記録する評価指標と，最良モデルの選択基準を書く．

## 3. 実験条件

| 項目 | 設定 |
|:---|:---|
| データセット | TODO(AI) |
| 前処理 | TODO(AI) |
| モデル | TODO(AI) |
| 最適化手法 | TODO(AI) |
| 学習率 | TODO(AI) |
| バッチサイズ | TODO(AI) |
| エポック数 | TODO(AI)（0 エポック目は初期値評価） |
| シード | TODO(AI) |
| 最良モデルの基準 | TODO(AI) |

結果の保存先は `outputs/{{EXPERIMENT_NAME}}/{最適化手法}/{学習率}_{バッチサイズ}/{seed}/` である．

### 実験前に考える

ノートブックを実行する前に，次の問いに自分の言葉で答えよ．

1. TODO(AI): 実験で答え合わせができる問い（pedagogy.md の 4.1 を参照）
2. TODO(AI)

## 4. 実験結果

<details>
<summary>実験結果を見る（先に予想してから開く）</summary>

TODO(AI): 実測値の要約を書く（0 エポック目の値，最良のエポックと値）．

| epoch | train Loss | val Loss | TODO(AI): 指標を追加 |
|:---:|:---:|:---:|:---:|
| 0 | TODO(AI) | TODO(AI) | TODO(AI) |

### 4.1 学習曲線

![Loss](figures/{{EXPERIMENT_NAME}}/loss.png)

TODO(AI): 他の指標の図を追加し，読み取れることを書く．

### 4.2 TODO(AI): 本ステージ固有の可視化

TODO(AI): `figures/{{EXPERIMENT_NAME}}/` に保存した図を参照し，読み取れることを書く．

</details>

### 実験後に考える

1. TODO(AI): 実験前の予想と結果を照合させる問い
2. TODO(AI): 結果の理由を考えさせる問い
3. TODO(AI): 次のステージにつながる問い

## 5. 考察

<details>
<summary>解説を見る</summary>

TODO(AI): 「実験後に考える」の各問いへの解説と，結果の解釈を書く．

</details>

## 6. まとめ

- TODO(AI): 箇条書き3〜5項目（実装したこと，得られた数値，分かったこと，保存先）

## 7. 発展課題

### 課題

TODO(AI): 課題の内容を1文で書く．

1. **何を変更するか**: TODO(AI)
2. **何を比較するか**: TODO(AI)
3. **予想**: TODO(AI)

<details>
<summary>回答例を見る</summary>

```python
# TODO(AI): 変更するコード
```

条件ごとに保存先が重ならないよう，`EXPERIMENT_NAME` を `{{EXPERIMENT_NAME}}_xxx` のように変更して実行する．

### 実験方法

TODO(AI)

### 期待される結果

TODO(AI)

### 結果から分かること

TODO(AI)

</details>

## 8. 次のステージへ

TODO(AI): 本ステージの限界（{{BRIDGE}}）を，実験結果に基づいて説明し，次のステージで何を変えるとそれが解決するかを書く．最後のステージでは見出しを「8. 到達目標の振り返り」に変更し，教材全体で学んだことを到達目標と対応づけてまとめる．
