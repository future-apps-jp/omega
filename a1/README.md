「A1 (Axiom 1)」 —— 宇宙の最小公理を記述するための、最小の言語。

この哲学に基づき、余計な装飾を削ぎ落とした**Scheme方言「A1」**の実装計画を策定します。Python上で動作するトランスパイラとして実装し、最短で論文用データを生成することを目指します。

Project A1: Implementation Plan
1. ディレクトリ構成
シンプルさを徹底します。

Plaintext

omega/
├── a1/
│   ├── __init__.py
│   ├── core.py         # パーサーとインタプリタの核（A1クラス）
│   ├── gates.py        # 量子ゲート定義 (H, CNOT, etc.)
│   └── metrics.py      # 記述長(Complexity)計測エンジン
├── experiments/
│   ├── hello_world.a1  # ベル状態生成のデモコード
│   ├── run_local.py    # ローカルシミュレーション実行用
│   ├── run_aws.py      # AWS Braket実行用
│   └── compare.py      # A1 vs 古典(NumPy) 記述長比較スクリプト
└── requirements.txt
2. フェーズ 1: コア・エンジンの実装 (The A1 Kernel)
A1の文法は極限まで単純化されたS式（S-expression）です。

目標: 以下のA1コードをパースし、AWS Braketの回路オブジェクトに変換する。

Scheme

; bell-state.a1
(DEFINE make-bell
  (LAMBDA (q0 q1)
    (CNOT (H q0) q1)))

(make-bell 0 1)
Step 1.1: パーサー (S式 → Pythonリスト) Python標準の機能だけで実装し、依存関係を減らします。

入力: "(CNOT (H 0) 1)"

出力: ['CNOT', ['H', 0], 1]

Step 1.2: インタプリタ (Evaluator) AST（抽象構文木）を再帰的に評価します。A1特有の挙動として、**「ゲート関数は、作用した量子ビットのインデックスを返す」**ように設計します。これにより、Lispらしいネスト構造が可能になります。

(H 0) -> 量子ビット0にHゲートを適用し、0 を返す。

(CNOT (H 0) 1) -> (H 0)が0になるので、実質 (CNOT 0 1) として評価される。

これにより、「Hゲートをかけてから、それを制御ビットとしてCNOTする」というフローが1行で書けます。

Step 1.3: 環境 (Environment) 変数と関数を保持する辞書 env を実装します。

DEFINE: env に値を登録。

LAMBDA: 引数と本体（body）を持つクロージャを作成。

3. フェーズ 2: 複雑性計測 (The A1 Metric)
論文の主張「量子基質上では記述長が最小になる」を証明するためのモジュールです。

Step 2.1: A1 Complexity Counter

A1のソースコードをトークン化し、その数をカウントします。

(H 0) -> 2トークン（括弧除く）。

定義済み関数（プリミティブ）はコスト1とみなします。

Step 2.2: Classical Complexity Counter (比較用)

同じベル状態生成を、**Python + NumPy（行列演算）**で実装したコードを用意します。

複素数行列の定義、行列積（np.dot）、テンソル積（np.kron）などの文字数/トークン数をカウントします。

期待される結果: A1は数トークン、古典コードは数百トークンとなり、数桁のオーダー差が出ることを示します。

4. フェーズ 3: クラウド実行 (Proof on Substrate)
AWS Braket SDKと連携させます。

Step 3.1: トランスパイラ出力 インタプリタが最終的に braket.circuits.Circuit オブジェクトを返すようにします。

Step 3.2: 実行ランナー

LocalSimulator: 開発用（無料・高速）。

AwsDevice: 本番データ取得用（SV1, IonQ, Rigetti）。

開発の第一歩： core.py のプロトタイプ
まずはこのコードが動くようにします。これがA1の心臓部です。

Python

# omega/a1/core.py (Concept)
import shlex
from braket.circuits import Circuit

class A1:
    def __init__(self):
        self.circuit = Circuit()
        self.env = {
            'H': self._gate_h,
            'CNOT': self._gate_cnot,
            # 他のゲートもここで定義
        }

    def _gate_h(self, q):
        self.circuit.h(int(q))
        return q # チェイン用にqubit indexを返す

    def _gate_cnot(self, c, t):
        self.circuit.cnot(int(c), int(t))
        return t

    def parse(self, code):
        # 簡易S式パーサー
        code = code.replace('(', ' ( ').replace(')', ' ) ')
        tokens = shlex.split(code)
        return self._read_from_tokens(tokens)

    def _read_from_tokens(self, tokens):
        if len(tokens) == 0: raise SyntaxError('unexpected EOF')
        token = tokens.pop(0)
        if token == '(':
            L = []
            while tokens[0] != ')':
                L.append(self._read_from_tokens(tokens))
            tokens.pop(0) # pop ')'
            return L
        elif token == ')':
            raise SyntaxError('unexpected )')
        else:
            return self._atom(token)

    def _atom(self, token):
        try: return int(token)
        except ValueError: return str(token)

    def eval(self, x, env=None):
        if env is None: env = self.env
        
        if isinstance(x, str):      # 変数参照
            return env[x]
        elif not isinstance(x, list): # 定数 (数値)
            return x
        
        op_name = x[0]
        
        # 特殊形式 (DEFINE, LAMBDA)
        if op_name == 'DEFINE':
            (_, var, exp) = x
            env[var] = self.eval(exp, env)
            return None
        elif op_name == 'LAMBDA':
            (_, vars, body) = x
            return lambda *args: self.eval(body, {**env, **dict(zip(vars, args))})
        
        # 関数適用
        proc = self.eval(op_name, env)
        args = [self.eval(arg, env) for arg in x[1:]]
        return proc(*args)

    def run(self, code):
        ast_list = self.parse(f"({code})") # 全体をリストでラップして処理
        # 実際にはトップレベルで複数の式を順次評価するロジックが必要
        for exp in ast_list:
             self.eval(exp)
        return self.circuit