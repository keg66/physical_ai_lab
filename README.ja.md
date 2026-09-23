[English](README.md) | 日本語

# Physical AI Lab

VLMによる知覚とgrounding、LLMによるskill planning、Robot Runtime、robot learning、System 1 recovery、安全境界を、小さなPython実験で接続する学習用リポジトリです。

- 教材: [`doc/textbook.ja.md`](doc/textbook.ja.md)
- 最終デモ: [`main.py`](main.py)

## 制作について

教材本文は、リポジトリ所有者がChatGPTとともに行った学習・実験の履歴をもとに、ChatGPTによって生成され、公開に向けて確認・編集されています。コードはAIによる補完や単純なロジック生成の支援を一部利用しつつ、ほぼすべてリポジトリ所有者が作成し、内容を一通り確認しています。詳しくは[教材の「制作について」](doc/textbook.ja.md#制作について)を参照してください。

英語版はChatGPTの支援を受けて作成・確認した要約翻訳で、Day 1〜5の全構成と節番号を維持しています。逐語訳ではなく、日本語版を原文として扱います。

## 重要な注意

このリポジトリは教育・研究目的の実験的なソフトウェアです。安全認証を受けた制御システムではなく、実機ロボットやその他の安全上重要な用途への適合性を保証しません。実機へ接続する場合、利用者は独立した検証、速度・力・衝突制限、非常停止などの安全対策、および適用される法令・規格への対応に責任を負います。

現在の`RobotRuntime`はfake実装です。`SafetySupervisor`もrequest-levelの最小検査であり、runtime/control safetyを実装するものではありません。

## 学習内容

| Day | テーマ | 主なコード |
|---|---|---|
| 1 | VLM、structured output、grounding | `day1/`, `physical_ai/world_state.py`, `physical_ai/grounder.py` |
| 2 | System 2、skill planning、runtime、timing | `physical_ai/agent.py`, `day2/rt_mini_lab.py` |
| 3 | Behavior Cloning、VLA、action chunking、ACT | `day3/` |
| 4 | Skill routing、System 1 recovery、failure injection | `physical_ai/system1_recovery.py`, `day4/` |
| 5 | Request-level safety、evaluation、Sim2Real | `physical_ai/safety_supervisor.py` |

## 実装範囲

| 状態 | 内容 |
|---|---|
| 実装済み | schema、task parsing、grounding、skill planning、plan validation、recovery routing |
| Fake実装 | `RobotRuntime`によるPICK/PLACEとfailure injection |
| 部分実装 | request-level safety、staticな`SkillRouter` |
| 未実装 | 実際のreposition、ROS 2/Gazebo/実機制御、runtime/control safety |
| 外部サービス依存 | Geminiによる知覚・解析・計画、JevによるSystem 1 recovery |

## セットアップ

検証時の環境はPython 3.12です。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

LeRobot関連は大きな依存やデータセットのダウンロードを伴います。Day 1、2、4だけ試す場合は、必要なパッケージに絞ってインストールしても構いません。

### APIキー

外部APIを使う例を実行する場合だけ設定します。

```bash
cp .env.example .env
# .envを編集する
source source_env.bash
```

Google GenAI SDKは`GEMINI_API_KEY`または`GOOGLE_API_KEY`を使用します。Jevのhosted APIを使う例には`TYPESAFE_API_KEY`が必要です。`.env`はGit管理対象外です。APIキーをソースコード、ログ、issueへ貼らないでください。

## 最初に試す

すべてのコマンドはリポジトリのルートで実行してください。

### 1. 再現用画像を生成する

`output/`は生成物としてGit管理対象外です。次のコマンドで、既存の例が参照する画像を再現できます。

```bash
python -m physical_ai.scene_generator --output output/scene_1.png --seed 1
python -m physical_ai.scene_generator --output output/scene_4.png --seed 4
```

同じseedから同じ画像とground-truth JSONが生成されます。seedを省略するとランダムなsceneになります。

### 2. 外部APIなしのmini-lab

```bash
python day2/rt_mini_lab.py
python day3/bc_mini_lab.py
python day3/bc_mini_lab_action_chunking.py
```

BCの例は最後にMatplotlibのウィンドウを表示します。

### 3. Geminiを使うDay 1

画像生成後、GeminiのAPIキーを設定して実行します。

```bash
python day1/vlm_world_state.py
python day1/vlm_task_spec.py
python day1/vlm_grounding.py
```

モデルの利用可能性はアカウント、地域、実行時期によって変わります。コード中のモデル名が利用できない場合は、利用中のGemini APIで提供されているモデルを確認してください。

### 4. LeRobot / ACT

```bash
python day3/lerobot_dataset_lab.py
python day3/lerobot_act_lab.py
```

これらはHugging FaceからPushT datasetや関連ファイルを取得します。`lerobot_act_lab.py`はCUDAを使用するため、対応するGPUとPyTorch環境が必要です。

### 5. JevによるSystem 1 recovery

```bash
python system1/jev_mini_lab.py
python day4/jev_recovery_benchmark.py
```

Jev APIへの外部通信が発生します。最終デモもGeminiとJevの両方を呼び出します。

```bash
python main.py
```

出力はmodel、network、実行時期によって変わり得ます。教材に記載したlatencyや判断結果は、特定時点の小規模な観測例であり、性能保証ではありません。

## 外部サービスとデータ

一部の例は次の外部サービスを使用します。

- [Google Gemini API](https://ai.google.dev/gemini-api/docs): perception、task parsing、grounding、skill planning
- [TypeSafe AI Jev](https://www.typesafeai.org/tools): System 1 recovery decision
- [Hugging Face LeRobot](https://github.com/huggingface/lerobot): robot learning datasetとpolicy

実行時には画像、prompt、structured world stateなどが外部サービスへ送信される場合があります。機密情報、個人情報、安全上重要なデータを送信する前に、各providerの最新の利用規約、料金、利用可能地域、データ取扱方針を確認してください。

これらのサービスやデータは本リポジトリのMIT Licenseには含まれず、それぞれの利用条件が適用されます。本プロジェクトは各providerとの提携や、各providerによる推奨を示すものではありません。

## Third-party dependencies

Python依存パッケージは[`requirements.txt`](requirements.txt)に記載しています。各パッケージには、それぞれの著作権とライセンスが適用されます。本リポジトリのMIT Licenseが依存パッケージ、外部モデル、dataset、APIへ適用されるわけではありません。

依存物を同梱したbinary、container、配布用環境を作る場合は、各ライセンスを改めて確認し、必要なcopyright noticeとlicense textを配布物へ含めてください。

## License

このリポジトリ自身のコードとドキュメントは、個別に別の表示があるものを除き、[MIT License](LICENSE)で提供します。
