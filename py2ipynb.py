import jupytext
import glob


py_files = glob.glob("*.py")
py_files.remove("py2ipynb.py")
py_files.remove("ipynb2py.py")

for py_file in py_files:
    nb = jupytext.read(py_file)
    jupytext.write(nb, py_file.replace(".py", ".ipynb"), fmt="py:percent")
    print(f"Converted {py_file} to {py_file.replace('.py', '.ipynb')}")

print("Conversion complete!")
