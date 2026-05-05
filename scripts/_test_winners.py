"""Teste rapido: vencedor por horizonte."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enso.eval.compare import load_metrics, unified_score, winner_per_horizon

m = load_metrics(Path("runs/train_full_v1"))
print(f"metrics: {len(m)} linhas, horizontes: {sorted(m['horizon'].unique().tolist())}")

w = winner_per_horizon(m)
print("\nVencedores por horizonte:")
print(w[["horizon", "model", "score", "rmse", "r2", "acc"]].to_string(index=False))

print("\nScore unificado completo:")
s = unified_score(m).sort_values(["horizon", "score"], ascending=[True, False])
print(s[["horizon", "model", "score", "rmse", "r2", "acc"]].to_string(index=False))
