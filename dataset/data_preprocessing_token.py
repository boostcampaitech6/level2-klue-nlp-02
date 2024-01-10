import pandas as pd
import numpy as np
from ast import literal_eval

def str_to_dict(dataset):
    '''
    string type인 데이터셋 칼럼을 dictionary type으로 변경
    '''
    def func(obj):
        List = literal_eval(obj)
        return List
    
    out = dataset.copy()
    out['subject_entity'] = dataset['subject_entity'].apply(func)
    out['object_entity'] = dataset['object_entity'].apply(func)
    
    return out

def use_ent_token_1(dataset):
    '''
    entity 전과 후에 [ENT] token 삽입
    '''
    out_dataset = str_to_dict(dataset)
    sens =[]
    for i in list(out_dataset['id']):
        subject_ent = out_dataset['subject_entity'][i]['word']
        object_ent = out_dataset['object_entity'][i]['word']
        sub_type = out_dataset['subject_entity'][i]['type']
        obj_type = out_dataset['object_entity'][i]['type']
        sen = out_dataset['sentence'][i]
        sen = sen.replace(subject_ent, f'[ENT]{subject_ent}[/ENT]')
        sen = sen.replace(object_ent, f'[ENT]{object_ent}[/ENT]')
        sens.append(sen)

    dataset['sentence'] = sens
        
    return dataset

def use_entity_mask_2(dataset):
    '''
    entity 전과 후로 해당 subj, obj entity와 단어로 해당 단어 대체
    '''
    out_dataset = str_to_dict(dataset)
    sens = []
    for i in list(out_dataset['id']):
        subject_ent = out_dataset['subject_entity'][i]['word']
        object_ent = out_dataset['object_entity'][i]['word']
        sub_type = out_dataset['subject_entity'][i]['type']
        obj_type = out_dataset['object_entity'][i]['type']
        sen = out_dataset['sentence'][i]
        sen = sen.replace(subject_ent, f'[SUBJ-{sub_type}]')
        sen = sen.replace(object_ent, f'[OBJ-{obj_type}]')
        sens.append(sen)

    dataset['sentence'] = sens
        
    return dataset

def use_entity_mark_3(dataset):
    '''
    entity 전과 후에 [E1] 또는 [E2] token 삽입
    '''
    out_dataset = str_to_dict(dataset)
    sens =[]
    for i in list(out_dataset['id']):
        subject_ent = out_dataset['subject_entity'][i]['word']
        object_ent = out_dataset['object_entity'][i]['word']
        sub_type = out_dataset['subject_entity'][i]['type']
        obj_type = out_dataset['object_entity'][i]['type']
        sen = out_dataset['sentence'][i]
        sen = sen.replace(subject_ent, f'[E1]{subject_ent}[/E1]')
        sen = sen.replace(object_ent, f'[E2]{object_ent}[/E2]')
        sens.append(sen)

    dataset['sentence'] = sens
        
    return dataset


def use_entity_mark2_4(dataset):
    '''
    entity 전과 후에 해당하는 entity라벨값 삽입 
    '''
    out_dataset = str_to_dict(dataset)
    sens = []
    for i in list(out_dataset['id']):
        subject_ent = out_dataset['subject_entity'][i]['word']
        object_ent = out_dataset['object_entity'][i]['word']
        sub_type = out_dataset['subject_entity'][i]['type']
        obj_type = out_dataset['object_entity'][i]['type']
        sen = out_dataset['sentence'][i]
        sen = sen.replace(subject_ent, f'[{sub_type}]{subject_ent}[/{sub_type}]')
        sen = sen.replace(object_ent, f'[{obj_type}]{object_ent}[/{obj_type}]')
        sens.append(sen)

    dataset['sentence'] = sens
        
    return dataset

def use_entity_mark_punt_5(dataset):
    '''
    entity 전과 후에 특수기호 # 또는 @ 삽입 
    '''
    out_dataset = str_to_dict(dataset)
    sens = []
    for i in list(out_dataset['id']):
        subject_ent = out_dataset['subject_entity'][i]['word']
        object_ent = out_dataset['object_entity'][i]['word']
        sub_type = out_dataset['subject_entity'][i]['type']
        obj_type = out_dataset['object_entity'][i]['type']
        sen = out_dataset['sentence'][i]
        sen = sen.replace(subject_ent, f' @ {subject_ent} @ ')
        sen = sen.replace(object_ent, f' # {object_ent} # ')
        sens.append(sen)

    dataset['sentence'] = sens
        
    return dataset

def use_typed_entity_mark_6(dataset):
    '''
    entity 전과 후에 <O:entity>, <S:entity> 삽입 
    '''
    out_dataset = str_to_dict(dataset)
    sens = []
    for i in list(out_dataset['id']):
        subject_ent = out_dataset['subject_entity'][i]['word']
        object_ent = out_dataset['object_entity'][i]['word']
        sub_type = out_dataset['subject_entity'][i]['type']
        obj_type = out_dataset['object_entity'][i]['type']
        sen = out_dataset['sentence'][i]
        sen = sen.replace(subject_ent, f'<S:{sub_type}> {subject_ent} </S:{sub_type}>')
        sen = sen.replace(object_ent, f'<O:{obj_type}> {object_ent} </O:{obj_type}>')
        sens.append(sen)

    dataset['sentence'] = sens
        
    return dataset


def use_typed_entity_mark_punc_7(dataset):
    '''
    entity 전과 후에 @ * {sub_type} * {subject_ent} @ , # ^ {obj_type} ^ {object_ent} # 삽입 
    '''
    out_dataset = str_to_dict(dataset)
    sens = []
    for i in list(out_dataset['id']):
        subject_ent = out_dataset['subject_entity'][i]['word']
        object_ent = out_dataset['object_entity'][i]['word']
        sub_type = out_dataset['subject_entity'][i]['type']
        obj_type = out_dataset['object_entity'][i]['type']
        sen = out_dataset['sentence'][i]
        sen = sen.replace(subject_ent, f' @ * {sub_type} * {subject_ent} @ ')
        sen = sen.replace(object_ent, f' # ^ {obj_type} ^ {object_ent} # ')
        sens.append(sen)

    dataset['sentence'] = sens
        
    return dataset


def use_typed_entity_mark_punc_8(dataset):
    '''
    entity 전과 후에 @ * {sub_type} * {subject_ent} @ , $ ^ {obj_type} ^ {object_ent} $ 삽입 
    '''
    out_dataset = str_to_dict(dataset)
    sens = []
    for i in list(out_dataset['id']):
        subject_ent = out_dataset['subject_entity'][i]['word']
        object_ent = out_dataset['object_entity'][i]['word']
        sub_type = out_dataset['subject_entity'][i]['type']
        obj_type = out_dataset['object_entity'][i]['type']
        sen = out_dataset['sentence'][i]
        sen = sen.replace(subject_ent, f' @ * {sub_type} * {subject_ent} @ ')
        sen = sen.replace(object_ent, f' $ ^ {obj_type} ^ {object_ent} $ ')
        sens.append(sen)

    dataset['sentence'] = sens
        
    return dataset


def use_typed_entity_mark_punc_9(dataset):
    '''
    7번에 오리지널 문장을 그대로 붙이기 
    '''
    out_dataset = str_to_dict(dataset)
    sens = []
    for i in list(out_dataset['id']):
        subject_ent = out_dataset['subject_entity'][i]['word']
        object_ent = out_dataset['object_entity'][i]['word']
        sub_type = out_dataset['subject_entity'][i]['type']
        obj_type = out_dataset['object_entity'][i]['type']
        sen = out_dataset['sentence'][i]
        sen2 = out_dataset['sentence'][i]
        sen = sen.replace(subject_ent, f' @ * {sub_type} * {subject_ent} @ ')
        sen = sen.replace(object_ent, f' # ^ {obj_type} ^ {object_ent} # ')
        sens.append(sen + sen2)

    dataset['sentence'] = sens
        
    return dataset


    


