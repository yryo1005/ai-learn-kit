# 教材作成を依頼する order の例

`.orders/order_{n:03}.md` に記述する例である．項目が無い場合は `curriculum.md` の Phase 0 の既定値が用いられる．

## 例1: 到達目標のみを指定する

```text
@.ai/ai-learn-kit/README.md に従い，VAEを理解するための教材を作成してください．
公開先: https://github.com/yryo1005/vae_training
```

## 例2: 学習者と構成を指定する

```text
@.ai/ai-learn-kit/README.md に従い，Deep Variational Information Bottleneck（Deep VIB）を理解するための教材を作成してください．
* 学習者: PyTorch の初学者．Python は書ける．確率の基礎は既習
* 実行環境: Google Colab（GPU）
* 題材: MNIST
* 構成: NN → AE → VAE → Deep VIB を本線とし，その後に潜在空間の2次元可視化，敵対的学習，損失関数の比較を発展ステージとして追加する
* 公開先: https://github.com/yryo1005/Deep_Variation_Information_Bottleneck_training
```

## 例3: 設計だけを先に確認する

```text
@.ai/ai-learn-kit/README.md に従い，Transformer の Self-Attention を理解するための教材の計画のみを作成してください．
* 学習者: CNN までは理解している
* curriculum_plan.md を作成した時点で止め，ステージの一覧を報告してください
```

## 例4: 既存の教材にステージを追加する

```text
既存の教材に，Deep VIB の β を変えたときの情報量と精度のトレードオフを観察するステージを ex005 の後に発展ステージとして追加してください．
```
