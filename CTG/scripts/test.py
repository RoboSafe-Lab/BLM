import h5py

with h5py.File('/home/visier/safety-critical/safety_critical_trained_models/ppo_latent_dm/data.hdf5', 'r') as f:
    # 1) 查看顶层 group 名称：
    print("Top-level groups:", list(f.keys()))

    # 2) 针对每个 group，查看其内部数据集及形状：
    for gname in f.keys():
        print(f"\n检查 group {gname}:")
        for dname in f[gname].keys():
            data = f[gname][dname]
            print(f"数据集 {dname} 的形状: {data.shape}")
            
            # 如果数据集本身包含子数据集
            if isinstance(data, h5py.Group):
                print(f"  {dname} 是一个 group,包含以下数据集:")
                for sub_dname in data.keys():
                    sub_data = data[sub_dname]
                    print(f"    - {sub_dname} 的形状: {sub_data.shape}")
