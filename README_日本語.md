# 引用付き RAG エージェント（評価機能付き）

## プロジェクト概要

本プロジェクトは、要件定義からデプロイ・運用まで単独で開発した、クラウドネイティブな適応型（Adaptive）RAG エージェントシステムです。

エージェントアーキテクチャ：LangGraph を基盤に構築され、自律的なクエリルーティング、マルチホップ推理、および検索結果の自己修正（Self-correction）機能を備え、根拠が追跡可能な引用付き回答を生成します。

AWS データ管理基盤：S3 によるデータレイクへの取り込みから、AWS Glue を用いた自動化 ETL パイプライン（解析・チャンク分割・埋め込みベクトルの次元削減）、そして pgvector（HNSWインデックス）を備えた Amazon RDS PostgreSQL への永続化まで、堅牢なデータインフラを構築しました。

## 技術スタックの概要
LangGraph 上に構築された適応型 Retrieval-Augmented Generation（RAG）システムで、
マルチホップな質問に対して追跡可能な出典引用付きで回答します。本プロジェクトは
クラウドネイティブなデータパイプラインを一通り実装しています。生のドキュメントは
Amazon S3 に取り込まれ、AWS Glue の ETL ジョブがそれらを解析・チャンク分割し、
埋め込みベクトルに変換して pgvector を備えた Amazon RDS PostgreSQL に格納します。
そして、セマンティックキャッシュを備えた FastAPI サービスが、インタラクティブな
Streamlit ダッシュボードを通じて回答を提供します。さらに、付属の評価モジュールは
ナレッジグラフのノードトラッキングによって合成テストセットを生成し、
チェーンカバレッジおよびステップカバレッジの指標を測定します。

---

## 目次

- [アーキテクチャ概要](#アーキテクチャ概要)
- [プロジェクト構成](#プロジェクト構成)
- [ETL パイプライン](#etl-パイプライン)
- [主な機能](#主な機能)
- [セットアップ](#セットアップ)
  - [前提条件](#前提条件)
  - [インストール](#インストール)
  - [設定](#設定)
- [使い方](#使い方)
  - [ローカルでの ETL 実行](#ローカルでの-etl-実行)
  - [AWS Glue での ETL 実行](#aws-glue-での-etl-実行)
  - [FastAPI サービス](#fastapi-サービス)
  - [Streamlit ダッシュボード](#streamlit-ダッシュボード)
- [データモデル](#データモデル)
- [ベクトルストアのバックエンド](#ベクトルストアのバックエンド)
- [セマンティックキャッシュ](#セマンティックキャッシュ)
- [評価フレームワーク](#評価フレームワーク)
- [AWS セットアップスクリプト](#aws-セットアップスクリプト)
- [AWS デプロイ (CDK)](#aws-デプロイ-cdk)
- [API リファレンス](#api-リファレンス)

---

## アーキテクチャ概要

![アーキテクチャ概要](assets/archi_overview.png)

1. **Extract（抽出）** -- 生のドキュメント（PDF、HTML、プレーンテキスト）を、
   データレイクとして機能する S3 バケットにアップロードします。
2. **Transform（変換）** -- AWS Glue の Python Shell ジョブが各ファイルを
   ダウンロードして解析し、テキストをオーバーラップ付きの文字数ベースの
   チャンクに分割し、Google Generative AI の埋め込み API
   （`gemini-embedding-001`）を呼び出して 768 次元のベクトルを生成します
   （`output_dimensionality` により 3072 次元から削減）。
3. **Load（格納）** -- チャンクをその埋め込みベクトルおよび JSONB メタデータと
   ともに、pgvector による近似最近傍探索のための HNSW インデックスを備えた
   RDS PostgreSQL テーブルに upsert します。
4. **Serve（提供）** -- LangGraph を基盤とする FastAPI サービスが、適応型の
   検索、関連性グレーディング、クエリのリライト、引用に基づく回答生成を行います。
   埋め込みベースのセマンティックキャッシュが類似クエリを横取りし、
   レイテンシを削減します。
5. **Dashboard（ダッシュボード）** -- Streamlit アプリが、展開可能な引用パネルと
   キャッシュ管理機能を備えたチャットインターフェースを提供します。

---

## プロジェクト構成

```
.
├── rag_agent/                  # コア RAG エンジン（Python パッケージ）
│   ├── api.py                  #   セマンティックキャッシュ付き FastAPI サービス
│   ├── cache.py                #   埋め込みベースのセマンティックキャッシュ
│   ├── config.py               #   モデル名、チャンク分割パラメータ、URL ローダー
│   ├── graph_builder.py        #   LangGraph ステートマシンの構築
│   ├── loader.py               #   ドキュメントローダー（PDF、HTML、ローカルファイル）
│   ├── models.py               #   LLM および埋め込みモデルの初期化
│   ├── nodes.py                #   ノード関数（ルーター、グレーダー、リライター、ジェネレーター）
│   ├── tools.py                #   引用フォーマット付きリトリーバーツール
│   └── vectorstore.py          #   ベクトルストアファクトリ（pgvector / インメモリ）
│
├── etl/                        # ETL パイプライン
│   ├── glue_etl_job.py         #   AWS Glue Python Shell ジョブスクリプト
│   ├── local_runner.py         #   ローカルテスト用エントリーポイント
│   └── schema.sql              #   PostgreSQL DDL（pgvector テーブル + インデックス）
│
├── evaluation/                 # 評価および合成データ生成
│   ├── evaluation_schemas.py   #   Pydantic スキーマ（テストケース、結果）
│   ├── evaluation_metrics.py   #   チェーンカバレッジおよびステップカバレッジ指標
│   ├── evaluator.py            #   カスタム指標 + Ragas 指標を統合するオーケストレーター
│   └── kg_testset_generator.py #   KG を考慮した合成テストセットジェネレーター
│
├── streamlit_app/              # インタラクティブダッシュボード
│   └── app.py                  #   Streamlit UI（チャット、引用、キャッシュ管理）
│
├── infra/                      # AWS CDK によるインフラ・アズ・コード
│   ├── app.py                  #   CDK アプリのエントリーポイント
│   ├── cdk.json                #   CDK 設定
│   ├── requirements.txt        #   CDK Python 依存パッケージ
│   └── stacks/
│       └── rag_service_stack.py#   S3, Glue, RDS/pgvector, ECS Fargate, ALB
│
├── scripts/                       # AWS プロビジョニングおよびヘルパースクリプト
│   ├── aws_setup.py               #   boto3 で S3, RDS, Glue をプロビジョニング
│   ├── s3_upload.py               #   ローカルファイルを S3 データレイクにアップロード
│   ├── run_glue_job.py            #   Glue ETL ジョブのトリガーと監視
│   └── check_rds.py               #   RDS データの検証（pgvector、行数）
│
├── sample_docs/                   # テスト用サンプルドキュメント
│
├── Dockerfile                  #   FastAPI サービス用コンテナイメージ
├── requirements.txt            #   Python 依存パッケージ
├── urls.txt                    #   ドキュメントソースの URL（開発モード）
└── README.md
```

---

## ETL パイプライン

ETL パイプラインは `etl/` 配下にあり、**AWS Glue Python Shell** ジョブとして
実行されるよう設計されています。アプリケーション全体を持ち込まずに Glue
ランタイム内で実行できるよう、意図的に自己完結型（`rag_agent` パッケージから
import しない）になっています。

### パイプラインのステップ

| ステージ | コンポーネント | 詳細 |
|---|---|---|
| Extract | `list_s3_objects` / `download_s3_bytes` | S3 データレイクからファイルを一覧・ダウンロード |
| Parse | `parse_pdf`, `parse_html`, `parse_text` | 生のバイト列をメタデータ付きの構造化テキストに変換 |
| Chunk | `chunk_text` / `split_documents` | 安定したハッシュ ID 付きの、文字数ベースのオーバーラップチャンク分割 |
| Embed | `generate_embeddings` | Google `gemini-embedding-001` へのバッチ呼び出し（`output_dimensionality` により 768 次元） |
| Load | `ensure_schema` / `upsert_chunks` | PostgreSQL に upsert。`chunk_id` で重複をスキップ |

### Glue ジョブのパラメータ

| パラメータ | 必須 | デフォルト | 説明 |
|---|---|---|---|
| `--S3_BUCKET` | はい | -- | ソースとなる S3 バケット |
| `--S3_PREFIX` | いいえ | `raw/` | スキャン対象のキープレフィックス |
| `--DB_HOST` | はい* | -- | RDS エンドポイント |
| `--DB_NAME` | いいえ | `ragdb` | データベース名 |
| `--DB_USER` | はい* | -- | データベースユーザー |
| `--DB_PASSWORD` | はい* | -- | データベースパスワード |
| `--DB_SECRET_ARN` | はい* | -- | Secrets Manager ARN（ユーザー／パスワードを上書き） |
| `--GOOGLE_API_KEY` | はい | -- | Google AI Studio の API キー |
| `--CHUNK_SIZE` | いいえ | `1024` | チャンクあたりの文字数 |
| `--CHUNK_OVERLAP` | いいえ | `50` | オーバーラップ文字数 |

*`DB_SECRET_ARN`、または `DB_HOST` / `DB_USER` / `DB_PASSWORD` の 3 点セットの
いずれかを指定してください。

### Glue の追加 Python モジュール

```
pypdf,beautifulsoup4,google-generativeai,pg8000
```

すべてのモジュールは純粋な Python であり、Glue Python Shell ランタイム
（C コンパイル不可）との互換性を確保しています。`pg8000` は `psycopg2-binary` の
代わりに使用され、SSL 経由で RDS に接続します。

---

## 主な機能

- **クラウドネイティブな ETL** -- S3 データレイク、AWS Glue による変換、RDS
  pgvector ウェアハウスによる、標準的な Extract-Transform-Load パイプライン。
  インフラはすべてコードで管理されます。
- **適応型ルーティング** -- LLM がクエリの意図に基づき、直接生成と
  検索拡張生成（RAG）のどちらを行うかを自律的に判断します。
- **自己修正型の検索** -- グレーディングステップが検索されたチャンクを評価し、
  コンテキストの品質が低い場合にクエリのリライトを発動します。
- **引用によるグラウンディング** -- すべての回答に、リトリーバー出力から決定論的に
  抽出された構造化された出典情報（URL、ページ、チャンク ID、テキストスニペット）が
  埋め込まれます。
- **セマンティックキャッシュ** -- 埋め込み類似度に基づくキャッシュが、類似クエリに
  対して計算済みの回答を返し、レイテンシとコストを削減します。
- **2 種類のベクトルストアバックエンド** -- 本番向けの pgvector（Glue ETL が投入）と、
  開発向けのインメモリ（起動時に `urls.txt` から読み込み）。
- **KG を考慮した評価** -- ナレッジグラフのノードトラッキングを用いた合成テストセット
  生成、ならびにチェーンカバレッジおよびステップカバレッジ指標。

---

## セットアップ

### 前提条件

- Python 3.10 以降
- Google AI Studio の API キー
  （[取得はこちら](https://aistudio.google.com/apikey)）
- `pgvector` 拡張を備えた PostgreSQL 16（ETL および本番利用向け）
- AWS CLI および CDK CLI（クラウドデプロイ時のみ）

### インストール

```bash
git clone https://github.com/<your-username>/doc-Q-A-RAG-Agent.git
cd doc-Q-A-RAG-Agent

# 仮想環境
conda create -n ragent python=3.10 && conda activate ragent
# または: python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

# ETL ローカルランナー向け（任意）
pip install google-generativeai psycopg2-binary
```

### 設定

| 変数 | 必須 | 説明 |
|---|---|---|
| `GOOGLE_API_KEY` | はい | Google AI Studio の API キー |
| `PGVECTOR_CONNECTION_STRING` | いいえ | PostgreSQL URI。省略するとインメモリストアを使用 |
| `PGVECTOR_COLLECTION` | いいえ | pgvector のテーブル名（デフォルト: `documents`） |

```bash
export GOOGLE_API_KEY="your_key"

# 本番
export PGVECTOR_CONNECTION_STRING="postgresql://user:pass@host:5432/ragdb"
```

Windows PowerShell:

```powershell
$env:GOOGLE_API_KEY="your_key"
```

---

## 使い方

### ローカルでの ETL 実行

サンプルドキュメントを格納したディレクトリを作成し、ローカルランナーを実行します。

```bash
mkdir sample_docs
# PDF / HTML / TXT ファイルを sample_docs/ に配置

# ドライラン（解析 + チャンク分割のみ。埋め込みや DB 書き込みは行わない）
python -m etl.local_runner --input_dir ./sample_docs --dry_run

# フル実行（PostgreSQL と GOOGLE_API_KEY が必要）
python -m etl.local_runner \
    --input_dir ./sample_docs \
    --db_host localhost \
    --db_name ragdb \
    --db_user postgres \
    --db_password secret \
    --google_api_key $GOOGLE_API_KEY
```

### AWS Glue での ETL 実行

1. 生のドキュメントを S3 バケットの `raw/` プレフィックス配下にアップロードします。
2. `etl/glue_etl_job.py` を `s3://<bucket>/glue-scripts/` にアップロードします。
3. 上記のパラメータを指定し、そのスクリプトを指す Glue Python Shell ジョブを
   作成します。
4. ジョブを手動で実行するか、EventBridge トリガーを設定します。

または CDK 経由でデプロイすれば（[AWS デプロイ](#aws-デプロイ-cdk) を参照）、
ジョブが自動的にプロビジョニングされます。

### FastAPI サービス

```bash
uvicorn rag_agent.api:app --host 0.0.0.0 --port 8000
```

### Streamlit ダッシュボード

まず FastAPI サービスを起動し、その後に以下を実行します。

```bash
streamlit run streamlit_app/app.py
```

---

## データモデル

ETL パイプラインは、以下の PostgreSQL テーブルに書き込みます。

```sql
CREATE TABLE documents (
    id          SERIAL       PRIMARY KEY,
    chunk_id    TEXT         UNIQUE NOT NULL,
    content     TEXT         NOT NULL,
    metadata    JSONB        DEFAULT '{}'::jsonb,
    embedding   vector(768),
    created_at  TIMESTAMPTZ  DEFAULT now()
);
```

`metadata` カラムには以下を格納します。

```json
{
  "source": "s3://bucket/raw/report.pdf",
  "page": 3,
  "title": "Annual Report",
  "chunk_index": 0
}
```

インデックス:
- 近似コサイン類似度検索のための `embedding` 上の **HNSW**。
- 重複排除およびソースフィルタリングのための `metadata->>'source'` 上の **B-tree**。

完全な DDL は `etl/schema.sql` にあります。

---

## ベクトルストアのバックエンド

| バックエンド | 利用条件 | データソース | 永続化 |
|---|---|---|---|
| **pgvector** | `PGVECTOR_CONNECTION_STRING` が設定されている | Glue ETL が書き込み、API が読み込み | PostgreSQL |
| **InMemory** | 変数が未設定 | 起動時に `urls.txt` から読み込み | なし |

---

## セマンティックキャッシュ

FastAPI サービスは、埋め込みベースのキャッシュ（`rag_agent/cache.py`）で
LangGraph エージェントをラップします。

1. 受信したクエリを埋め込みベクトルに変換し、キャッシュ済みエントリと
   （コサイン類似度で）比較します。
2. 類似度がしきい値（デフォルト 0.92）を超える場合、キャッシュ済みの回答を
   即座に返します。
3. それ以外の場合はエージェントのパイプライン全体を実行し、結果を保存します。

実行時管理用のエンドポイント:
- `GET  /v1/cache/stats` -- 現在のエントリ数
- `DELETE /v1/cache` -- フラッシュ

---

## 評価フレームワーク

`evaluation/` パッケージは、マルチホップ検索の品質を測定します。

| 指標 | 定義 |
|---|---|
| チェーンカバレッジ | 検索されたゴールド標準チャンクの割合 |
| ステップカバレッジ | 必要なチャンクを少なくとも 1 つ検索できた推論ステップの割合 |
| 回答の正確性 | Ragas のセマンティック類似度（システム回答 vs ゴールド回答） |
| 忠実性（Faithfulness） | Ragas のグラウンディングスコア |
| コンテキスト適合率／再現率 | Ragas のコンテキストレベル指標 |

`kg_testset_generator.py` は Ragas を拡張し、生成された各質問・回答ペアを
どのナレッジグラフノードが裏付けているかを追跡します。これにより、合成データに
対するチェーンカバレッジおよびステップカバレッジの自動計算が可能になります。

---

## AWS セットアップスクリプト

`scripts/` ディレクトリは、CDK を使わずに素早くプロビジョニングするための
boto3 ベースのヘルパーを提供します。

| スクリプト | 目的 |
|---|---|
| `aws_setup.py` | S3 バケット、RDS PostgreSQL（pgvector）、IAM ロール、Glue ジョブを作成 |
| `s3_upload.py` | ローカルディレクトリを S3 データレイクの `raw/` 配下にアップロード |
| `run_glue_job.py` | Glue ETL ジョブをトリガーし、完了をポーリング |
| `check_rds.py` | RDS に接続し、pgvector 拡張、テーブル、行数を検証 |

```bash
# 1. すべての AWS リソースをプロビジョニング（RDS は約 5 分）
python scripts/aws_setup.py

# 2. ドキュメントのアップロード
python scripts/s3_upload.py ./sample_docs

# 3. ETL の実行
python scripts/run_glue_job.py --wait

# 4. 検証
python scripts/check_rds.py
```

---

## AWS デプロイ (CDK)

`infra/` ディレクトリには、以下をプロビジョニングする AWS CDK スタックが
含まれています。

| リソース | サービス | 詳細 |
|---|---|---|
| ネットワーク | VPC | 2 AZ、パブリック + プライベートサブネット、NAT |
| データレイク | S3 | 生のドキュメント用バケット |
| ETL | Glue Python Shell | 解析、チャンク分割、埋め込み、格納 |
| データベース | RDS PostgreSQL 16 | pgvector 拡張、db.t4g.micro |
| API | ECS Fargate | ALB 配下の FastAPI コンテナ |
| シークレット | Secrets Manager | Google API キー、DB 認証情報 |

```bash
cd infra
pip install -r requirements.txt
cdk bootstrap   # 初回のみ
cdk deploy
```

デプロイ後:
1. ドキュメントを S3 バケットの `raw/` 配下にアップロードします。
2. AWS コンソールまたは CLI から Glue ジョブを実行します。
3. 出力された ALB の DNS 名経由で API にアクセスします。

---

## レート制限

`/v1/chat` エンドポイントは [slowapi](https://github.com/laurentS/slowapi) により
**IP あたり 30 リクエスト/分** にレート制限されています。上限を超えると
HTTP 429 を返します。

```json
{"detail": "Rate limit exceeded: 30 per 1 minute"}
```

実装: `rag_agent/api.py` が `Limiter(key_func=get_remote_address)` を作成し、
チャットエンドポイントを `@limiter.limit("30/minute")` でデコレートします。

---

## API リファレンス

### POST /v1/chat

リクエスト:

```json
{
  "message": "How does Money Forward use AI?",
  "timeout_s": 60
}
```

レスポンス:

```json
{
  "trace_id": "a1b2c3d4-...",
  "answer": "Money Forward uses AI to ...",
  "citations": [
    {
      "source": "s3://bucket/raw/report.pdf",
      "title": "AI Strategy",
      "page": 3,
      "chunk": "12",
      "snippet": "Money Forward leverages ..."
    }
  ],
  "cached": false
}
```

### GET /healthz

`{"status": "ok"}` を返します。

### GET /v1/cache/stats

`{"entries": 42}` を返します。

### DELETE /v1/cache

セマンティックキャッシュをフラッシュします。`{"status": "cleared"}` を返します。

---
