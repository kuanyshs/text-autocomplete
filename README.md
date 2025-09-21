# Text Autocomplete (LSTM + GPT?2 baseline)

## Коротко
- Задача: автодополнение текста (next-token prediction) на коротких сообщениях.
- Реализовано: обучение LSTM с нуля, оценка по ROUGE-1/ROUGE-2, базовая оценка предобученного трансформера DistilGPT-2.

## Структура проекта
- data/
  - raw_dataset.txt — сырые строки (одна строка = один текст)
  - dataset_processed.txt — очищенные строки
  - train.txt, val.txt, test.txt — сплиты
- src/
  - __init__.py
  - data_utils.py — препроцессинг и сплиты (CLI)
  - next_token_dataset.py — PyTorch Dataset/DataLoader/коллатер
  - lstm_model.py — LSTM-модель и генерация
  - eval_lstm.py — оценка LSTM по ROUGE
  - lstm_train.py — обучение LSTM (CLI)
  - eval_transformer_pipeline.py — оценка DistilGPT?2 (CLI)
  - utils_print.py — печать/отбор/сохранение примеров (опционально)
- configs/
  - default.yaml — конфиг
- models/ — сохранённые отчёты/веса
- solution.ipynb — ноутбук
- requirements.txt — зависимости

## Установка и окружение
- Рекомендуемый Python: 3.10–3.11.
- Создать окружение (пример с conda):
  - conda create -n taenv python=3.11 -y
  - conda activate taenv
  - pip install -r requirements.txt
- Проверить интерпретатор:
  - where python
  - python -c "import sys; print(sys.executable)"
- GPU (опционально):
  - conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
  - python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"

## Основные команды
- Подготовка датасета:
  - python -m src.data_utils --input data/raw_dataset.txt --output_dir data --val_size 0.1 --test_size 0.1 --seed 123
- Обучение LSTM и оценка на валидации (использует configs/default.yaml):
  - python -m src.lstm_train --config configs/default.yaml
- Оценка DistilGPT-2 на валидации и тесте:
  - python -m src.eval_transformer_pipeline --config configs/default.yaml

## Подготовка данных
- Исходные строки положите в data/raw_dataset.txt.
- Команда из data_utils.py:
  - чистит строки (URL > <url>, @user > <user>, нормализация пробелов),
  - сохраняет data/dataset_processed.txt,
  - делит на train/val/test.

## Конфигурация (configs/default.yaml)
- data: пути к train/val/test и max_len
- vocab: max_size, min_freq
- training: batch_size, lr, weight_decay, num_epochs, clip_grad_norm, seed, ckpt_dir
- model: emb_dim, hidden_dim, num_layers, dropout
- eval: quarter (доля хвоста), do_sample, top_p, top_k, temperature, max_new_tokens
- device: auto | cpu | cuda | cuda:0

## Обучение LSTM
- Запуск: python -m src.lstm_train --config configs/default.yaml
- Что делает:
  - строит словарь по train (с [PAD], [BOS], [EOS], [UNK]),
  - создаёт Dataset/DataLoader, сортированный по длинам батч-сэмплер,
  - LSTMAutocomplete: weight tying, pack_padded_sequence,
  - CrossEntropyLoss(ignore_index=[PAD]), клиппинг градиента, Adam,
  - после эпох: оценка ROUGE на val, сохранение лучшего чекпоинта по val loss в models/lstm_autocomplete_best.pt,
  - опционально сохраняет сводку примеров и метрик.

## Оценка DistilGPT-2
- Запуск: python -m src.eval_transformer_pipeline --config configs/default.yaml
- Что делает:
  - берёт тексты из val.txt и test.txt,
  - обрезает контекст до max_length и генерирует хвост,
  - считает ROUGE?1/ROUGE?2 и собирает примеры.

## Печать и сохранение выбранных примеров
- Печать: используйте print_selected_examples(examples, indices).
- Чтобы сохранить те же примеры в файл (TXT), используйте save_summary(best_results, best_examples, out_dir, filename_prefix, top_k).

## Заметки по метрикам
- По умолчанию можно считать ROUGE через пакет evaluate, но на Windows это иногда тянет несовместимые зависимости (transformers > accelerate > boto3 > urllib3).

