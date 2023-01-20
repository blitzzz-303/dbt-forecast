import json
import re
import os

def create_func_from_frame():
    PATH_DBT_PROJECT = '/dbt/python'
    MACRO_DBT_PROJECT = '/dbt/macros'
    sf_func, sf_regis, sf_lib, sf_frame, python_func = ['sf_func.txt', 'sf_regis.txt', 'sf_lib.txt',
                                                        'sf_frame.txt', 'snow_prophet.py']

    sf_func_content = get_file_content(PATH_DBT_PROJECT, sf_func)
    sf_regis_content = get_file_content(PATH_DBT_PROJECT, sf_regis)
    sf_lib_content = get_file_content(PATH_DBT_PROJECT, sf_lib)
    sf_frame_content = get_file_content(PATH_DBT_PROJECT, sf_frame)
    python_func_content = get_file_content(PATH_DBT_PROJECT, python_func).split('# python function')[1]

    sf_full = (sf_frame_content.replace(sf_func, sf_func_content)
                                .replace(sf_regis, sf_regis_content)
                                .replace(sf_lib, sf_lib_content)
                                .replace(python_func, python_func_content))
    macro_path = os.path.join(MACRO_DBT_PROJECT, 'store_procedure_gen.sql')
    print('created macro file at ', macro_path)

    with open(macro_path, 'w') as f:
        f.write(sf_full)

def get_file_content(*args):
    with open(os.path.join(args[0], args[1]), 'r') as f:
        return f.read()
    
if __name__ == "__main__":
    create_func_from_frame()