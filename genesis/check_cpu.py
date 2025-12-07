"""
Genesis-Matrix CPU環境の動作確認スクリプト
"""
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
    print("-" * 50)
    print("✓ Environment is READY for Phase I experiments.")
    print("-" * 50)
    
    # 追加テスト: より大きな行列
    print("\n[Additional Test] Larger matrix operations...")
    for n in [4, 6, 8]:
        dim = 2**n
        M = jnp.eye(dim)
        start = time.time()
        result = jnp.dot(M, M)
        result.block_until_ready()
        dur = (time.time() - start) * 1000
        print(f"  {n} qubits ({dim}x{dim} matrix): {dur:.2f} ms")

if __name__ == "__main__":
    check_cpu_performance()

