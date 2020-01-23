import tensorflow as tf
import pathlib
import random

# Константы для формирования датасета
DATA_DIRECTORY = "D:/Learning/Texzrenie_laba31"
BATCH_SIZE = 128
IMAGE_SIZE = 24
CHANNELS = 3
CLASS_NUM = 4
NUM_IMAGE_PER_EPOCH_FOR_TRAIN = 49250
NUM_IMAGE_PER_EPOCH_FOR_TEST = 10000


# Функция получения путей к файлам и их лейблов
# Аргументы: путь к датасету
# Возвращает списки путей для тренировочных, тестовых данных
# и их лейблов
def get_path_to_images_and_labels(directory=DATA_DIRECTORY):
    # Получение пути в папку с датасетом
    data_root = pathlib.Path(directory)
    print(data_root)

    # Создание списка путей до файлов
    all_image_paths = list(data_root.glob("*/*"))
    all_image_paths = [str(path) for path in all_image_paths]

    # Перемешивание списка для разделения
    random.shuffle(all_image_paths)

    # Количество изображений
    count = len(all_image_paths)
    print(count)

    # Разделение на тестовые и тренировочные данные
    path_for_train = []
    path_for_test = []

    for i in range(count):
        if i < NUM_IMAGE_PER_EPOCH_FOR_TRAIN:
            path_for_train.append(all_image_paths[i])
        else:
            path_for_test.append(all_image_paths[i])

    # Получение лейблов классов
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    print(label_names)
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    print(label_to_index)

    # Получение лейблов для изображений
    train_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in path_for_train]
    print("First 10 labels indices: ", train_labels[:10])

    test_labels = [label_to_index[pathlib.Path(path).parent.name]
                   for path in path_for_test]
    print("First 10 labels indices: ", test_labels[:10])

    return path_for_train, train_labels, path_for_test, test_labels


# Функция считывания и препроцессинга данных
# Аргументы: список путей к файлам
# Возвращает тензор изображений
def load_and_preprocess_image(path_to_images):
    # Считывание файлов
    image = tf.read_file(path_to_images)

    # Получение тензора
    image = tf.image.decode_jpeg(image, channels=CHANNELS)

    image = tf.image.resize_images(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0
    print(image)
    return image


# Функция генерации батчей тренировочных и тестовых данных
# Аргументы: датасет для тренировки,
#            датасет для тестирования,
#            число изображений для тренировке,
#            число изображений для теста
# Возвращает батчи с тренировочными и тестовыми данными
def create_train_and_test_batches(image_label_ds_for_train, image_label_ds_for_test, count1, count2):
    # Перемешивание данных
    ds_for_train = image_label_ds_for_train.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=count1))
    ds_for_test = image_label_ds_for_test.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=count2))

    # Формирование батчей
    ds_for_train = ds_for_train.batch(BATCH_SIZE)
    ds_for_train = ds_for_train.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    ds_for_test = ds_for_test.batch(BATCH_SIZE)
    ds_for_test = ds_for_test.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    # Создание итераторов
    iterator1 = ds_for_train.make_one_shot_iterator()
    iterator2 = ds_for_test.make_one_shot_iterator()

    # Получение батчей
    train_image, train_label = iterator1.get_next()
    test_image, test_label = iterator2.get_next()

    train_image = tf.reshape(train_image, [1, 24, 24, 3])
    test_image = tf.reshape(train_image, [1, 24, 24, 3])
    train_label.set_shape([1])
    test_label.set_shape([1])

    print(train_image.shape)
    print(train_label)
    print(test_image)
    print(test_label)

    return train_image, train_label, test_image, test_label


# Функция формирования датасета
def create_dataset():
    # Получение путей к файлам и их лейблов
    path_for_train, train_labels, path_for_test, test_labels = get_path_to_images_and_labels()

    # Получение количества изображений для тестирования и тренировки
    count1 = len(path_for_train)
    count2 = len(path_for_test)
    print(count1)
    print(count2)

    # Формирование датасета для тренировки
    path_ds_for_train = tf.data.Dataset.from_tensor_slices(path_for_train)
    image_ds_for_train = path_ds_for_train.map(load_and_preprocess_image, num_parallel_calls=tf.contrib.data.AUTOTUNE)
    label_ds_for_train = tf.data.Dataset.from_tensor_slices(tf.cast(train_labels, tf.int32))
    image_label_ds_for_train = tf.data.Dataset.zip((image_ds_for_train, label_ds_for_train))

    # Формирование датасета для теста
    path_ds_for_test = tf.data.Dataset.from_tensor_slices(path_for_test)
    image_ds_for_test = path_ds_for_test.map(load_and_preprocess_image, num_parallel_calls=tf.contrib.data.AUTOTUNE)
    label_ds_for_test = tf.data.Dataset.from_tensor_slices(tf.cast(test_labels, tf.int32))
    image_label_ds_for_test = tf.data.Dataset.zip((image_ds_for_test, label_ds_for_test))

    return create_train_and_test_batches(image_label_ds_for_train, image_label_ds_for_test, count1, count2)
