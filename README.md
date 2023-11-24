# TOSM-Two-and-a-half-Order-Score-based-Model-for-Solving-3D-Ill-posed-Inverse-Problems

## install: requirements.txt

## train: run main_ImageNet.py 
  parser.add_argument('--runner', type=str, default='AapmRunnerdata_10C', help='The runner to execute')
  parser.add_argument('--test', action='store_true', help='Whether to test the model', default=False)
## reconstruction: run main_ImageNet.py    
  parser.add_argument('--runner', type=str, default='Aapm_Runner_CTtest_10_noconv', help='The runner to execute')
  parser.add_argument('--test', action='store_true', help='Whether to test the model', default=True)
