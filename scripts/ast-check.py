"""Confere que a limpeza de comentários dos simulation.py não mudou código (v1.2.0, etapa 10).

Compara a AST de cada arquivo com a versão em um commit de referência: comentários
não entram na AST, então qualquer diferença significa que algo além de comentário mudou.
Docstrings fazem parte da AST e também têm de ficar iguais.

Uso: python3 scripts/ast-check.py <commit-de-referência>   (ex.: HEAD)
Sai com código 1 se algum arquivo divergir.
"""
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ref = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
files = sorted(p for p in ROOT.glob("simulations/**/*.py") if "extra" not in p.parts)
bad = 0
for p in files:
    rel = p.relative_to(ROOT).as_posix()
    try:
        old = subprocess.check_output(["git", "show", f"{ref}:{rel}"], cwd=ROOT, text=True)
    except subprocess.CalledProcessError:
        print(f"?  {rel}: não existe em {ref}")
        continue
    new = p.read_text()
    same = ast.dump(ast.parse(old)) == ast.dump(ast.parse(new))
    removed = len(old.splitlines()) - len(new.splitlines())
    print(f"{'ok' if same else 'XX'} {rel}: {removed:+d} linhas" if same else f"XX {rel}: AST DIFERENTE")
    bad += not same
sys.exit(1 if bad else 0)
