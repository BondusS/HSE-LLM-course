> Обучал модель на ноутбуке с **MPS M4**, поэтому в коде ставил соответствующие настройки

* Код - https://github.com/BondusS/HSE-LLM-course/blob/main/HomeWork1/hw1_solution.py
* Полный лог обучения - https://github.com/BondusS/HSE-LLM-course/blob/main/HomeWork1/train_log.txt
* Краткий лог обучения:

```
/opt/anaconda3/bin/python /Users/aleksey/Desktop/HSE/LLM/hw1_solution.py
Training samples: 1940063
Validation samples: 5000
Model pad token id: 2
Total params: 960,881,664
[transformers] The speedups for torchdynamo mostly come with GPU Ampere or higher and which is not detected here.
  0%|          | 1/485016 [00:16<2132:23:55, 15.83s/it]{'loss': '11', 'grad_norm': '7.062', 'learning_rate': '0', 'epoch': '2.062e-06'}
  0%|          | 2/485016 [00:20<1230:40:52,  9.13s/it]{'loss': '11.14', 'grad_norm': '9.688', 'learning_rate': '2.5e-07', 'epoch': '4.124e-06'}
  ...
  0%|          | 99/485016 [12:55<1003:08:55,  7.45s/it]{'loss': '10.64', 'grad_norm': '6.438', 'learning_rate': '2.45e-05', 'epoch': '0.0002041'}
  0%|          | 100/485016 [13:03<996:28:34,  7.40s/it]{'loss': '10.59', 'grad_norm': '6.688', 'learning_rate': '2.475e-05', 'epoch': '0.0002062'}

  0%|          | 0/625 [00:00<?, ?it/s]
  0%|          | 2/625 [00:05<27:46,  2.67s/it]
  ...
100%|█████████▉| 623/625 [51:37<00:10,  5.26s/it]
100%|█████████▉| 624/625 [51:42<00:05,  5.29s/it]

  0%|          | 100/485016 [1:04:57<996:28:34,  7.40s/it]
100%|██████████| 625/625 [51:51<00:00,  5.33s/it]

{'eval_loss': '10.39', 'eval_runtime': '3114', 'eval_samples_per_second': '1.606', 'eval_steps_per_second': '0.201', 'epoch': '0.0002062'}

Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]
Writing model shards: 100%|██████████| 1/1 [00:02<00:00,  2.30s/it]
  0%|          | 101/485016 [1:05:10<127069:39:03, 943.36s/it]Training stopped after 3910.24 seconds
  0%|          | 101/485016 [1:05:10<127069:39:03, 943.36s/it]{'loss': '10.43', 'grad_norm': '11.44', 'learning_rate': '2.5e-05', 'epoch': '0.0002082'}

  0%|          | 0/625 [00:00<?, ?it/s]
  0%|          | 2/625 [00:05<30:04,  2.90s/it]
  ...
100%|█████████▉| 623/625 [57:07<00:11,  5.63s/it]
100%|█████████▉| 624/625 [57:13<00:05,  5.58s/it]

{'eval_loss': '10.36', 'eval_runtime': '3444', 'eval_samples_per_second': '1.452', 'eval_steps_per_second': '0.181', 'epoch': '0.0002082'}

  0%|          | 101/485016 [2:02:34<127069:39:03, 943.36s/it]
100%|██████████| 625/625 [57:22<00:00,  5.57s/it]

Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]
Writing model shards: 100%|██████████| 1/1 [00:01<00:00,  1.32s/it]

{'train_runtime': '7361', 'train_samples_per_second': '263.6', 'train_steps_per_second': '65.89', 'train_loss': '10.96', 'epoch': '0.0002082'}

  0%|          | 101/485016 [2:02:42<9818:37:40, 72.89s/it]
100%|██████████| 625/625 [56:55<00:00,  5.47s/it]

Final evaluation results: {'eval_loss': 10.362265586853027, 'eval_runtime': 3417.2403, 'eval_samples_per_second': 1.463, 'eval_steps_per_second': 0.183, 'epoch': 0.00020824055288897687}

Process finished with exit code 0
```
