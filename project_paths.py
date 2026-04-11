import os
import os.path as osp


PROJECT_ROOT = osp.dirname(osp.abspath(__file__))


def project_path(*parts):
    return osp.join(PROJECT_ROOT, *parts)


def data_pyg_root():
    return project_path('data_pyg')


def attack_data_root():
    return project_path('data_pyg', 'Attack_data')


def attack_unit_test_data_root():
    return project_path('data_pyg', 'Attack_unit_test_data')


def data_attack_fewshot_root():
    return project_path('data_attack_fewshot')
