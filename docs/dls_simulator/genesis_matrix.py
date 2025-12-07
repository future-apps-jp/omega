import numpy as np
import random
import copy
import time

# --- 1. 宇宙の環境設定 (The Landscape) ---
# タスク: 5ノードのグラフで、スタート(0)からゴール(4)へ 3ステップで着く経路数を求めよ
# 正解は、隣接行列 A の 3乗 (A^3) の [0, 4] 成分
NODES = 5
STEPS = 3
# グラフの隣接行列 (迷路の地図)
ADJ_MATRIX = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 1, 1, 0, 1],
    [0, 0, 0, 1, 0]
])
# ターゲットデータ (量子ウォークの結果)
# 実際に正解を計算しておく (これが自然界の真理となる)
TARGET_VAL = np.linalg.matrix_power(ADJ_MATRIX, STEPS)[0, 4]

print(f"--- Genesis Simulation: The Quantum Dawn ---")
print(f"Target Task: Find path count from Node 0 to 4 in {STEPS} steps.")
print(f"Physical Truth (Target Value): {TARGET_VAL}")

# --- 2. DSLの定義 (The Species) ---

class Gene:
    """DSLの構成要素（遺伝子）"""
    def evaluate(self, context): pass
    def size(self): return 1
    def __str__(self): return ""

# --- 種族 A: 古典的スカラー演算のみ (The Classical Beings) ---
# 彼らは1つずつ足し合わせるしかない
class ScalarOp(Gene):
    def __init__(self, op, left, right):
        self.op = op; self.left = left; self.right = right
    def evaluate(self, ctx):
        l, r = self.left.evaluate(ctx), self.right.evaluate(ctx)
        if self.op == '+': return l + r
        if self.op == '*': return l * r
        if self.op == 'IF': return l if r > 0 else 0
        return 0
    def size(self): return 1 + self.left.size() + self.right.size()
    def __str__(self): return f"({self.op} {self.left} {self.right})"

class ScalarVal(Gene):
    def __init__(self, val): self.val = val
    def evaluate(self, ctx): return self.val
    def size(self): return 1
    def __str__(self): return str(self.val)

# --- 種族 B: 行列演算を持つ (The Quantum/A1 Mutants) ---
# 彼らは「重ね合わせ（行列）」を一気に計算できる
class MatrixOp(Gene):
    def __init__(self, mat): self.mat = mat
    def evaluate(self, ctx): return self.mat # 行列そのものを返す
    def size(self): return 1 # 「行列」という概念自体は1単位とする(A1の強み)
    def __str__(self): return "M"

class MatMulOp(Gene):
    """行列の掛け算 (ユニタリ進化/並列探索に相当)"""
    def __init__(self, left, right):
        self.left = left; self.right = right
    def evaluate(self, ctx):
        l, r = self.left.evaluate(ctx), self.right.evaluate(ctx)
        try:
            return np.dot(l, r)
        except:
            return 0 # 型エラー
    def size(self): return 1 + self.left.size() + self.right.size()
    def __str__(self): return f"(DOT {self.left} {self.right})"

class GetElementOp(Gene):
    """観測 (Measurement)"""
    def __init__(self, matrix, row, col):
        self.matrix = matrix; self.row = row; self.col = col
    def evaluate(self, ctx):
        m = self.matrix.evaluate(ctx)
        r = self.row.evaluate(ctx); c = self.col.evaluate(ctx)
        try:
            return m[r, c]
        except:
            return 0
    def size(self): return 1 + self.matrix.size() + self.row.size() + self.col.size()
    def __str__(self): return f"(GET {self.matrix} {self.row} {self.col})"

# --- 3. 進化エンジン (The Engine) ---

def generate_scalar_individual(depth=4):
    """スカラー演算だけで解こうとする個体を生成"""
    if depth == 0: return ScalarVal(random.randint(0, 5))
    return ScalarOp(random.choice(['+', '*', 'IF']), 
                    generate_scalar_individual(depth-1), 
                    generate_scalar_individual(depth-1))

def generate_matrix_individual(depth=3):
    """行列演算を使って解こうとする個体を生成"""
    # 構造: (GET (DOT (DOT M M) M) 0 4) のような形を目指す
    if depth == 0:
        if random.random() < 0.4: return MatrixOp(ADJ_MATRIX)
        return ScalarVal(random.randint(0, 5))
    
    op_type = random.random()
    if op_type < 0.5: # 行列積の発見確率
        return MatMulOp(generate_matrix_individual(depth-1), generate_matrix_individual(depth-1))
    elif op_type < 0.8: # 観測の発見確率
        return GetElementOp(generate_matrix_individual(depth-1), 
                            ScalarVal(0), ScalarVal(4)) 
    else:
        return ScalarOp('+', generate_matrix_individual(depth-1), ScalarVal(1)) # ノイズ

def fitness(program):
    """
    適応度関数: インフレーション速度に相当
    Fitness = 精度 / (1 + 記述長K * ペナルティ係数)
    """
    try:
        val = program.evaluate({})
        # 行列がそのまま返ってきたら観測していないのでペナルティ
        if isinstance(val, np.ndarray): return 0.1
        
        error = abs(TARGET_VAL - val)
        
        # ホログラフィック制約: Kが長いと生存確率は激減する
        k = program.size()
        k_penalty = 0.5 
        
        score = 1000 / (1 + error * 100 + k * k_penalty)
        return score
    except:
        return 0

# --- 実験実行 ---
def run_experiment():
    population_size = 50
    generations = 30
    
    print("\n[Phase 1] Classical Era: Only Scalar DSL exists.")
    # 最初はスカラー種族しかいない
    population = [generate_scalar_individual() for _ in range(population_size)]
    
    for g in range(1, 11): # 10世代回す
        population.sort(key=fitness, reverse=True)
        best = population[0]
        print(f"Gen {g}: Best Fitness {fitness(best):.2f} (K={best.size()}) | Code: {str(best)[:50]}...")

    print("\n[Phase 2] Mutation Occurred: Matrix Ops Introduced.")
    print(">>> A new DSL (Mutation) allowing 'Matrix Operations' has appeared in the gene pool.")
    
    # 行列演算を使える「変異種」を投入 (全体の20%程度)
    pop_matrix = [generate_matrix_individual() for _ in range(10)]
    population = population[:40] + pop_matrix 
    
    for g in range(11, generations+1):
        population.sort(key=fitness, reverse=True)
        best = population[0]
        f = fitness(best)
        
        # 種族の判定
        species = "Quantum" if "DOT" in str(best) or "M" in str(best) else "Classical"
        
        print(f"Gen {g}: Best Fitness {f:.2f} (K={best.size()}) [{species}]")
        print(f"  Code: {best}")
            
        if f > 200: # 収束条件
            print(f"\n!!! EVOLUTION CONVERGED at Gen {g} !!!")
            print(f"Winning Law: {best}")
            print(f"Description Length K: {best.size()}")
            print(f"Execution Result: {best.evaluate({})}")
            
            if species == "Quantum":
                print("\n[Conclusion] The Matrix-based DSL (A1) dominated the universe due to minimal K.")
            break
            
        # 淘汰と増殖 (エリート戦略 + ランダムコピー)
        survivors = population[:15]
        population = survivors[:]
        while len(population) < population_size:
            parent = random.choice(survivors)
            # 簡易的な複製（本来はここで交叉・変異が入る）
            population.append(copy.deepcopy(parent))

if __name__ == "__main__":
    run_experiment()