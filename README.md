# ObjectRecognitionInfoSearch

System for object detection and find information about detected object

## Установка и запуск

При установке данных требований, запуск возможен только на cpu
Для запуска на gpu нужно переустановить torch с актуальной версией
При запуске система установит модель yolo. По умолчанию `yolov5x6u.pt`

```bash
pip install -r requirements.txt # Установка зависимостей
python main.py # запуск
```

Для запуска требуется .env файл с настройками
Нужно дописать настройки ключа для поиска google, кода поисковой системы и название модели yolo

```env
api_key=''
cx_code=''
model_name='yolov5x6u'
```
