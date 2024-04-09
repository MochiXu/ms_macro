import h5py


def get_hdf5_info(file_path: str):
    print(f"HDF5 INFO ----- {file_path.split('/')[-1]}")
    with h5py.File(file_path, 'r') as train_hdf5:
        print(f"attrs keys:{train_hdf5.attrs.keys()}")
        print(f"datasets:{train_hdf5.keys()}")
        for k in train_hdf5.keys():
            print(f"key:{k}\tsize:{len(train_hdf5[k])}\tdtype:{train_hdf5[k].dtype}\tshape:{train_hdf5[k].shape}")


directory_prefix = "/Volumes/970EVO/vector-db-benchmark-datasets/downloaded/ms-macro2-768-full-cosine"
get_hdf5_info(f"{directory_prefix}/ms-macro2-768-full-cosine.hdf5")
get_hdf5_info(f"{directory_prefix}/ms-macro2-768-full-cosine-dev-query.hdf5")
