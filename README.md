# ObjectRecognitionInfoSearch

System for object detection and find information about detected object

## Установка и запуск
### Установка

При установке данных требований, запуск возможен только на cpu
Для запуска на gpu нужно переустановить torch с актуальной версией

```bash
pip install -r requirements.txt # Установка зависимостей
python main.py # запуск
```

Для запуска требуется .env файл с настройками
Нужно дописать настройки ключа для поиска google и кода поисковой системы

```env
api_key=''
cx_code=''
```

### Запуск

Для работы системы нужно запустить скрипт `main.py`
При запуске система установит модель yolo. По умолчанию `yolov5x6u.pt`