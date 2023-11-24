# TOSM-Two-and-a-half-Order-Score-based-Model-for-Solving-3D-Ill-posed-Inverse-Problems

## Install: conda install requirements.txt

## Datasets: Download AAPM datasets for CT Reconstruction.

## Train: run main_ImageNet.py 
  parser.add_argument('--runner', type=str, default='AapmRunnerdata_10C', help='The runner to execute')
  //
  parser.add_argument('--test', action='store_true', help='Whether to test the model', default=False)
## Reconstruction: run main_ImageNet.py    
  parser.add_argument('--runner', type=str, default='Aapm_Runner_CTtest_10_noconv', help='The runner to execute')
  //
  parser.add_argument('--test', action='store_true', help='Whether to test the model', default=True)
