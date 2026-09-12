# 導入・利用ガイド（Getting Started）

TabulaShogi をダウンロード・ビルドし、将棋GUI（ShogiHome や 将棋所）に登録して対局を始めるための手順書です。

---

## 1. 動作環境

- **OS**: Windows 10 / 11 (x86_64), Linux (x86_64)
- **ビルド環境**: Rust 1.98 以上 (Rust 2024 Edition / Cargo)
- **将棋GUI**: [ShogiHome](https://sunfish-shogi.github.io/shogihome/)、[将棋所](http://shogidokoro.starfree.jp/) 等の USI 準拠 GUI

---

## 2. ビルド手順

リポジトリをクローンまたはダウンロードし、以下のコマンドでリリースビルドを行います：

```bash
cargo build --release
```

ビルドが完了すると、以下のパスに実行ファイルが生成されます：

- **Windows**: `target/release/tabula-shogi.exe`
- **Linux**: `target/release/tabula-shogi`

---

## 3. 将棋GUIへの登録方法

### ShogiHome の場合

1. **ShogiHome** を起動します。
2. 画面右上の **「設定」** ➜ **「エンジン設定」** を開きます。
3. **「エンジンを追加」** をクリックします。
4. ビルドされた `tabula-shogi.exe` の絶対パスを指定します。
5. エンジン名が自動的に **TabulaShogi <version>**（例: `TabulaShogi 0.1.0`）、作者が **Takumi Kusumoto** と認識されます。
6. 設定を保存すれば、通常の対局や検討、棋譜解析で選択可能になります。

### 将棋所（Shogidokoro）の場合

1. **将棋所** を起動します。
2. メニューの **「対局」** ➜ **「エンジン管理」** を開きます。
3. **「追加」** ボタンを押し、 `tabula-shogi.exe` を選択します。
4. エンジン一覧に TabulaShogi が追加されたら完了です。

---

## 4. USI主要オプションの設定

対局GUIのエンジン設定画面から以下のパラメータを調整できます：

| オプション名 |  型  | デフォルト値 |   調整範囲   | 説明                                                                                                          |
| :----------- | :--: | :----------: | :----------: | :------------------------------------------------------------------------------------------------------------ |
| **USI_Hash** | spin |      64      | 1 〜 8192 MB | 置換表（探索結果のメモリキャッシュ）のサイズ。PCの搭載メモリに応じて 256 や 1024 に増やすと読みが安定します。 |
| **Threads**  | spin |      1       |   1 〜 64    | 探索に使用する並列スレッド数。対戦ベンチマーク時は 1、本気で強くしたいときは 4 や 8 に設定します。            |

---

## 5. コマンドラインからの直接実行テスト

ターミナルやコマンドプロンプトから対話的に起動することも可能です：

```bash
.\target\release\tabula-shogi.exe
usi
# -> id name TabulaShogi 0.1.0
# -> id author Takumi Kusumoto
# -> usiok
isready
# -> readyok
quit
```
