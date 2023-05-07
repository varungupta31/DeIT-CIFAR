# DeIT CIFAR
Data Effiecient Image Transformers Implemented on a smaller Scale, using the CIFAR100 dataset and downsampled Imagenet ImageNet32.

The following is an attempt to implement the [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) Paper

<h2> Model Zoo of the Various Models Trained </h2>

| S.No | Model                        | Link (Present in [OneDrive](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/gupta_varun_students_iiit_ac_in/EnIPg91IiAZEjVky2XSnb8ABbYKCGu50zLCNB7_ssRFFMA?e=2LQX0A))        |
| -------------------- | ---------------------------- | --------------------------------- |
| 1                    | Teacher - RegNetY_16GF Model | regnet_y_16gf_32_3                |
| 2                    | DeIT-B Scratch CIFAR         | VIT_B_cifar_scratch_38            |
| 3                    | DeIT-B Imagenet32 Scratch    | VIT_B_imagenet_scratch_7          |
| 4                    | DeIT-B Hard Distillation     | vit_b_reg16gf_hard_dist_53        |
| 5                    | DeIT-B⚗︎                     | vit_b_reg16gf_hard_dist_token_28  |
| 6                    | DeIT-S Scratch CIFAR         | VIT_S_cifar_scratch_50            |
| 7                    | DeIT-S Hard Distillation     | vit_s_hard_dist_no_token_87       |
| 8                    | DeIT-S⚗︎                     | vit_S_reg16gf_hard_dist_token_49  |
| 9                    | DeIT-Ti Scratch CIFAR        | VIT_Ti_CIFAR_SCRATCH_43           |
| 10                   | DeIT-Ti Hard Distillation    | vit_ti_hard_dist_no_token_89      |
| 11                   | DeIT-Ti⚗︎                    | vit_Ti_reg16gf_hard_dist_token_65 |

<img src="/media/results.png" alt="Results">

To install the Conda environmnet, simply run the following command

```
conda env create -n ENVNAME --file environment.yml
```

Colab Inference Demo


For this demo to be functional, kindly download the DeIT-B⚗︎ model from the above link and update the path in the notebook.
![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10LXtebYncHbuwuqd9NNWEf46UVYDZfEx?usp=sharing


Trained Models Available [Here](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/gupta_varun_students_iiit_ac_in/EnIPg91IiAZEjVky2XSnb8ABbYKCGu50zLCNB7_ssRFFMA?e=2LQX0A)

ImageNet32 Dataset can be downloaded from the official [ImageNet Downloads Page](https://www.image-net.org/index.php).

Note: Models with the extension `.pth` are completely contained and simply require the call
`model = torch.load(model_path)`, instead of the usual, `load_state_dict` call.