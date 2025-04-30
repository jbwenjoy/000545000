import jupytext
import glob

# Get all ipynb files in current directory
ipynb_files = glob.glob("*.ipynb")

for ipynb_file in ipynb_files:
    # Read the notebook
    nb = jupytext.read(ipynb_file)
    # Write as py file using percent format
    py_file = ipynb_file.replace(".ipynb", ".py")
    jupytext.write(nb, py_file, fmt="py:percent")
    print(f"Converted {ipynb_file} to {py_file}")

print("Conversion complete!")
