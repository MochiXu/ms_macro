import string

import h5py


def get_hdf5_info(file_path: str):
    print(f"HDF5 INFO ----- {file_path.split('/')[-1]}")
    with h5py.File(file_path, 'r') as train_hdf5:
        print(f"attrs keys:{train_hdf5.attrs.keys()}")
        print(f"datasets:{train_hdf5.keys()}")
        for k in train_hdf5.keys():
            print(f"key:{k}\tsize:{len(train_hdf5[k])}\tdtype:{train_hdf5[k].dtype}\tshape:{train_hdf5[k].shape}")


def get_query_and_answer(file_path: str, range_left: int = 10, range_right: int = 20):
    print(f"HDF5 INFO ----- {file_path.split('/')[-1]}, reader range: {range_left}-{range_right}")
    translator = str.maketrans('', '', string.punctuation)
    with h5py.File(file_path, 'r') as train_hdf5:
        query_text = [raw_text.decode('utf-8').translate(translator) for raw_text in
                      train_hdf5['query_text'][range_left:range_right].tolist()]
        query_vector = train_hdf5['test'][range_left:range_right].tolist()
        query_answer = train_hdf5['neighbors'][range_left:range_right].tolist()

        for i in range(range_right-range_left):
            print(f"Query - {range_left+i}")
            print(f"> text - '{query_text[i]}'")
            print(f"> vector - {query_vector[i]}")
            print(f"> answer - {query_answer[i]}\n")


directory_prefix = "/Volumes/970EVO/vector-db-benchmark-datasets/downloaded/ms-macro2-768-full-cosine"
# get_hdf5_info(f"{directory_prefix}/ms-macro2-768-full-cosine.hdf5")
# get_hdf5_info(f"{directory_prefix}/ms-macro2-768-full-cosine-dev-query.hdf5")
get_query_and_answer(f"{directory_prefix}/ms-macro2-768-full-cosine-dev-query.hdf5",20,25)
