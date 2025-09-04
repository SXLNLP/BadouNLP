# 环境检查代码
# 环境运行在conda中py312之下

python --version
python -c "import torch; print(f'\nPyTorch版本: {torch.__version__}\nCUDA是否可用: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'\nTransformers版本: {transformers.__version__}')"
python -c "import peft; print(f'\nPEFT版本: {peft.__version__}')"
python -c "import sklearn; print(f'\nScikit-learn版本: {sklearn.__version__}')"
python -c "import numpy; print(f'\nNumPy版本: {numpy.__version__}')"
python -c "import pandas; print(f'\nPandas版本: {pandas.__version__}')"

# 运行结果如下：

'''(py312) C:\Windows\System32>python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
2.6.0+cu118 True

(py312) C:\Windows\System32>python -c "import transformers; print(transformers.__version__)"
4.51.0

(py312) C:\Windows\System32>python -c "import peft; print(peft.__version__)"
0.15.0

(py312) C:\Windows\System32>python -c "import sklearn; print(sklearn.__version__)"
1.5.1

(py312) C:\Windows\System32>python -c "import numpy; print(f'\nNumPy版本: {numpy.__version__}')"

NumPy版本: 1.26.4

(py312) C:\Windows\System32>python -c "import pandas; print(f'\nPandas版本: {pandas.__version__}')"

Pandas版本: 2.2.2'''

