# 000545000

Since using Git with .ipynb is very unpleasant, I recommend installing jupytext

```bash
conda install jupytext -c conda-forge
```

Then, you can convert between .ipynb and .py files by running `ipynb2py.py` or `py2ipynb.py` in the root directory of the repository.

Always remember to convert ipynb into py and clean the notebook outputs when commiting:

1. Colab commit to your own branch on GitHub
2. Run `ipynb2py.py` to convert the notebook into a python script
3. Run `py2ipynb.py` to convert the python script back into a CLEAN notebook
4. Commit the notebook and the python script to your own branch on GitHub
5. Create a pull request to the main branch of the repository
6. Review and merge
