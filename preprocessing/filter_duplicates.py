def filter_duplicates(dataset):
    # ID 제외 중복 행 제거
    dataset = dataset.drop_duplicates(subset=dataset.columns.difference(["id"]), keep="first")
    # 중복 행 중 label값이 다른 애들 제거
    dataset = dataset.drop([6749, 8364, 22258, 277, 25094])
    return dataset
