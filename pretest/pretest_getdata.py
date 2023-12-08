# Script for extracting features from pdb-bind complexes.import
# Mahmudulla Hassan
# Last modified: 09/10/2018

# 更新优化脚本程序
# 常怡彬
# 最后修改时间：2023年8月31日

import moleculekit.molecule as ht
import moleculekit.tools.voxeldescriptors as vd
import moleculekit.smallmol.smallmol as sm
from tqdm import *
from oddt import datasets
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
from moleculekit.tools.preparation import systemPrepare
from sklearn.model_selection import train_test_split
import numpy as np
import os
import h5py

# 设置路径
data_dir = ".\\data"
pdbbind_dir = os.path.join(data_dir, "refined-set-2016\\")
pdbbind_2016 = datasets.pdbbind(home=pdbbind_dir, default_set='refined', version=2016)
pdbbind_dir = os.path.join(data_dir, "refined-set-2020\\")
pdbbind_2020 = datasets.pdbbind(home=pdbbind_dir, default_set='refined', version=2020)

def get_pdb_complex_feature(pro_file, lig_file):
    """ 返回组合了protein和ligand的element feature体素化结果 """

    # 分别读取蛋白质分子的 .pdbqt 和小分子的 .mol2 文件
    pro_mol = ht.Molecule(pro_file)
    lig_mol = sm.SmallMol(lig_file)

    # 定义盒子大小和体素化中心（小分子的欧几里得中心）
    n = [24, 24, 24]
    center = list(sm.SmallMol.getCenter(lig_mol))

    # 体素化
    # 蛋白质准备
    _ = pro_mol.remove("element H")
    _ = pro_mol.filter('protein')

    # 采用两种方法进行蛋白处理，包括特殊氨基酸结构区分处理。
    try:
        pro_mol = prepareProteinForAtomtyping(pro_mol, verbose=False)
    except Exception as e:
        if 'ignore_ns_errors=True' in str(e):
            pro_mol = systemPrepare(pro_mol, ignore_ns_errors=True, verbose=False)
            pro_mol = prepareProteinForAtomtyping(pro_mol,protonate=False, verbose=False)
    
    # 蛋白质体素化过程
    pro_vox, pro_centers, pro_N = vd.getVoxelDescriptors(pro_mol, boxsize=n, center=center)
    p_features = pro_vox.reshape(pro_N[0], pro_N[1], pro_N[2], pro_vox.shape[1])
    # 小分子体素化过程
    lig_vox, lig_centers, lig_N = vd.getVoxelDescriptors(lig_mol, boxsize=n, center=center)
    l_features = lig_vox.reshape(lig_N[0], lig_N[1], lig_N[2], lig_vox.shape[1])

    # 组合两个体素化过后的张量
    features = np.concatenate((p_features, l_features), axis=3)
    
    return features

def get_pdb_features(ids, sets="refined"):
    """ 根据PDBid 进行体素化 并返回体素化结果x 和标签y """
    pdb_ids = []
    pdb_features = []

    # 逐一进行体素化，并使用tqdm生成进度条。
    for pdbid in tqdm(ids):
        print('\n', 'PDB: ', pdbid, ' ', 'has began')

        # 根据PDBid获得文件路径
        protein_file = os.path.join(pdbbind_dir, pdbid, pdbid + "_protein.pdbqt")
        ligand_file = os.path.join(pdbbind_dir, pdbid, pdbid + "_ligand.mol2")

        # 若文件不存在则跳过，并记录在错误报告中。
        if not os.path.isfile(protein_file) or not os.path.isfile(ligand_file):
            with open('.\\error.log', 'a') as f:
                f.write('MISSING in ' + protein_file + ' or ' + ligand_file + '\n')
            continue

        # 若文件不能体素化则输出错误，并记录后跳过。
        try:
            features = get_pdb_complex_feature(protein_file, ligand_file)
        except Exception as e:
            print("ERROR in ", pdbid , " ", str(e))
            with open('.\\error.log', 'a') as f:
                f.write("ERROR in " + pdbid + " " + str(e)+ '\n')
            continue
        
        # 对应获得PDBid的列表和element feature的数组
        pdb_ids.append(pdbid)
        pdb_features.append(features)
    
    # 根据PDBid返回其对应的标签值（Ki, Kd, IC50）
    data_x = np.array(pdb_features, dtype=np.float32)
    data_y = np.array([pdbbind_2020.sets[sets][_id] for _id in pdb_ids], dtype=np.float32)

    return data_x, data_y

def get_features():
    """ 返回数据库中的体素化结果x 并对应标签y值 """ 

    # 核心数据集选择
    core_ids = list(pdbbind_2016.sets['core'].keys())
    # 精炼数据集选择
    refined_ids = list(pdbbind_2020.sets['refined'].keys()) 
    # 将精炼数据集和核心数据集独立
    refined_ids = [i for i in refined_ids if i not in core_ids]
    
    # 执行批量体素化 
    print("Extracting features for the core set")
    core_x, core_y = get_pdb_features(core_ids)
    print("Extracting features for the refined set")
    refined_x, refined_y = get_pdb_features(refined_ids)    
    
    return core_x, core_y, refined_x, refined_y

def main():
    # 核心数据集作为测试集
    test_x, test_y, train_x, train_y = get_features()
    # 划分训练集、验证集
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=1)
    print("Shapes in the training, test and the validation set: ", train_x.shape, test_x.shape, valid_x.shape)

    # 切片蛋白体素和小分子体素并分别保存
    train_xp, train_xl = train_x[..., 0:8], train_x[..., 8:]
    valid_xp, valid_xl = valid_x[..., 0:8], valid_x[..., 8:]
    test_xp, test_xl = test_x[..., 0:8], test_x[..., 8:]

    # 保存
    print("Saving the data in PDBbind2020.h5")
    h5f = h5py.File(os.path.join(data_dir, "PDBbind2020.h5"), 'w')
    h5f.create_dataset('train_xp', data=train_xp)
    h5f.create_dataset('train_xl', data=train_xl)
    h5f.create_dataset('train_y', data=train_y)
    h5f.create_dataset('valid_xp', data=valid_xp)
    h5f.create_dataset('valid_xl', data=valid_xl)
    h5f.create_dataset('valid_y', data=valid_y)
    h5f.create_dataset('test_xp', data=test_xp)
    h5f.create_dataset('test_xl', data=test_xl)
    h5f.create_dataset('test_y', data=test_y)
    h5f.close()

if __name__=="__main__":main()

