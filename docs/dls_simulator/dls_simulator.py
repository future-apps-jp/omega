import random
import copy
import math
import uuid

# --- 1. DSLの基本定義 ---
# 基本演算セット: 宇宙の最もプリミティブな命令
PRIMITIVE_OPS = ['ADD', 'SUB', 'MUL', 'CONST', 'VAR']

class Node:
    """DSLの構文木（AST）ノード。プログラムの設計図。"""
    def __init__(self, op, children=None, value=None):
        self.op = op
        self.children = children if children is not None else []
        self.value = value
        self.id = uuid.uuid4() # 識別子

    def evaluate(self, x):
        """プログラムを実行する (The Execution)"""
        if self.op == 'CONST': return self.value
        if self.op == 'VAR': return x
        
        # プリミティブ演算
        if self.op == 'ADD': return self.children[0].evaluate(x) + self.children[1].evaluate(x)
        if self.op == 'SUB': return self.children[0].evaluate(x) - self.children[1].evaluate(x)
        if self.op == 'MUL': 
            try:
                val = self.children[0].evaluate(x) * self.children[1].evaluate(x)
                return max(min(val, 1e12), -1e12) # オーバーフロー防止
            except OverflowError:
                return 1e12
        
        # 創発的演算（新しいBuilt-in Feature）の実行
        if self.op.startswith('NEW_OP_'):
            return self.evaluate_new_op(x)
        
        return 0

    def evaluate_new_op(self, x):
        """創発された新しい命令の実行（新しいDSLのコア）"""
        # ここで、創発された命令の定義（e.g., POWER, SQRTなど）を適用する
        # このプロトタイプでは、NEW_OPを強制的にべき乗として扱う (理想のA1への誘導)
        if len(self.children) == 1:
            return self.children[0].evaluate(x) ** 2 # (x^2) を発明したと見なす
        return 0

    def size(self):
        """記述長 K (Kolmogorov Complexity) を計測"""
        count = 1
        for child in self.children:
            count += child.size()
        
        # 新しい命令は、プリミティブな命令よりも記述長が短い（情報圧縮）と仮定する
        if self.op.startswith('NEW_OP_'):
            return 1 # 新しい原子的な命令は常にK=1
        
        return count

    def __str__(self):
        """コードとして表示"""
        if self.op == 'CONST': return str(self.value)
        if self.op == 'VAR': return "x"
        
        children_str = " ".join(str(c) for c in self.children)
        return f"({self.op} {children_str})"

# --- 2. DSLの創発と変異の仕組み ---

def create_random_tree(current_depth=0, max_depth=3):
    """初期のランダムなDSLコードを生成"""
    if current_depth >= max_depth or (current_depth > 0 and random.random() < 0.3):
        return Node('VAR') if random.random() < 0.5 else Node('CONST', value=random.randint(1, 5))
    
    op = random.choice(['ADD', 'SUB', 'MUL'])
    return Node(op, [create_random_tree(current_depth + 1, max_depth), create_random_tree(current_depth + 1, max_depth)])

def get_all_subtrees(node, subtrees):
    """変異のために全てのサブツリーを収集"""
    subtrees.append(node)
    for child in node.children:
        get_all_subtrees(child, subtrees)

def mutate(parent_program):
    """ASTの構造的な変異（遺伝的プログラミングの正攻法）"""
    program = copy.deepcopy(parent_program)
    subtrees = []
    get_all_subtrees(program, subtrees)
    
    if not subtrees: return create_random_tree()

    # ランダムなノードを選択し、新しいランダムなサブツリーで置き換える (変異の注入)
    target_node = random.choice(subtrees)
    target_node.__dict__.update(create_random_tree(max_depth=2).__dict__) 
    
    return program

def invent_new_op(best_program, existing_ops):
    """頻出する構造を新しい原子命令（Built-in Feature）として抽象化する"""
    # 実際には頻度分析が必要だが、ここでは「ある種の複雑な構造」が生まれたら発明と見なす
    
    # ターゲット: 最小のDSLで x^2 のような複雑な操作を可能にする (A1の本質)
    # K=5 (MUL x x) のような構造を K=1 (NEW_OP) に圧縮する
    if best_program.size() >= 5 and random.random() < 0.2:
        
        new_op_name = f'NEW_OP_{len(existing_ops)}'
        
        # 創発された命令を既存の命令セットに追加（これはDSLそのものの進化）
        # ここでは便宜上、新しい命令は常に一引数(x)のべき乗として定義されていると仮定する
        print(f"\n[DSL INVENTION] New Atomic Feature Created: {new_op_name} (K=1)")
        existing_ops.add(new_op_name)
        
        # ベストプログラムを新しい命令で置き換える
        # これは、既存の冗長なコードを新しい短い命令で「リファクタリング」するプロセス
        return Node(new_op_name, [Node('VAR')]) 

    return best_program

# --- 3. 宇宙の評価と淘汰 ---

# ターゲット法則: f(x) = x^2 + x + 1 のような、プリミティブ演算では冗長になる法則
TARGET_FUNCTION = lambda x: x**2 + x + 1
TARGET_DATA = [(x, TARGET_FUNCTION(x)) for x in range(-3, 4)]


def calculate_fitness(program):
    """適応度関数: 誤差の少なさ + 記述長Kの短さ (淘汰圧)"""
    error = 0.0
    try:
        for x, target_y in TARGET_DATA:
            pred_y = program.evaluate(x)
            error += abs(target_y - pred_y)
    except:
        error = 1e15 # 計算崩壊は即座に淘汰

    # Kペナルティ: 誤差が同じなら、短いコードを優遇する
    # これはホログラフィック・スクリーンの究極の淘汰圧となる
    k_penalty = program.size() * 1.0 # Kペナルティを非常に高く設定
    
    return error + k_penalty

# --- メインループ ---
def run_genesis():
    population_size = 100
    generations = 100
    
    # 初期DSLセット (プリミティブなもののみ)
    evolved_ops = set(PRIMITIVE_OPS)
    
    # 初期個体群 (ランダムな宇宙)
    population = [create_random_tree() for _ in range(population_size)]
    
    print(f"--- Genesis Started: Target Law is x^2 + x + 1 ---")
    print(f"Initial K: Avg {sum(p.size() for p in population)/population_size:.1f}")
    
    for gen in range(1, generations + 1):
        # 評価
        scored_pop = [(calculate_fitness(p), p) for p in population]
        scored_pop.sort(key=lambda x: x[0])
        
        best_fitness, best_program = scored_pop[0]
        
        # 淘汰と繁殖
        survivors = [p for _, p in scored_pop[:population_size//5]] # 上位20%が生き残る
        
        # DSLの進化 (新しい原子命令の発明)
        new_best = invent_new_op(best_program, evolved_ops)
        
        if new_best != best_program:
            survivors.append(new_best) # 新発明を生存者に加える

        # 次世代の生成
        next_gen = survivors[:] 
        
        while len(next_gen) < population_size:
            parent = random.choice(survivors)
            child = mutate(parent)
            next_gen.append(child)
        
        population = next_gen

        # ログ出力
        if gen % 5 == 0 or best_fitness < 0.1:
            print(f"\n[GEN {gen:02d}] Fitness: {best_fitness:.2f} | Best K: {best_program.size()}")
            print(f"  Code: {best_program}")
            if best_fitness < 0.1:
                print("\n--- OPTIMIZATION CONVERGED (A1 Reached) ---")
                break

if __name__ == "__main__":
    run_genesis()