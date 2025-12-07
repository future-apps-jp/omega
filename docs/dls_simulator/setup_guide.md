CPU用セットアップ手順（WSL/Ubuntu）
先ほどのGPU用の手順の代わりに、以下のCPU最適化版の手順を実行してください。これで「Genesis-docker」を起動できます。

1. WSLの更新と基本ツールのインストール

Bash

sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-venv build-essential
2. Python環境の作成

Bash

python3 -m venv genesis-env
source genesis-env/bin/activate
3. JAX (CPU版) のインストール GPU版ではなく、CPU版をインストールします。

Bash

pip install --upgrade pip
pip install "jax[cpu]"
4. 動作確認 (Canvas Check) 以下のコードを check_cpu.py として保存し、実行してください。 Generated A1 Code (Bell State) が一瞬で表示されれば、実験開始の準備完了です。

Python

import jax
import jax.numpy as jnp
import time

def check_cpu_performance():
    # JAXがCPUを使っているか確認
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    
    # 擬似的なA1コード実行（ベル状態生成のシミュレーション）
    # 行列サイズ: 2量子ビット (4x4行列) - 実験初期はこの規模で十分
    n_qubits = 2
    dim = 2**n_qubits
    
    print(f"\nSimulating 'Genesis-docker' container (Qubits: {n_qubits})...")
    
    # アダマールゲート (H) と CNOT の行列定義（簡易版）
    # 実際はDSLインタプリタがこれを動的に生成します
    H = jnp.array([[1, 1], [1, -1]]) / jnp.sqrt(2)
    I = jnp.eye(2)
    H_gate = jnp.kron(H, I) # H on q0
    
    CNOT = jnp.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
    
    # 初期状態 |00>
    state = jnp.array([1.0, 0, 0, 0])
    
    start_time = time.time()
    
    # 進化エンジンが発見すべきターゲット（A1コード実行）
    # (CNOT (H q0) q1)
    state = jnp.dot(H_gate, state)
    state = jnp.dot(CNOT, state)
    
    # 計算実行
    state.block_until_ready()
    duration = (time.time() - start_time) * 1000
    
    print(f"Result State Vector: {state}")
    print(f"Execution Time: {duration:.4f} ms")
    print("-" * 30)
    print("Environment is READY for Phase I experiments.")

if __name__ == "__main__":
    check_cpu_performance()
次のステップ
上記の check_cpu.py がエラーなく動けば、Surface上で「Genesis-docker」の開発に着手できます。

まずは研究計画書の [Phase I] プロトタイプ実装 - 最小構成のPython進化エンジンの開発  から始めましょう。