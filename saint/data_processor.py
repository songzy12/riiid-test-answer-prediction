import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


def get_question_df(path_questions):
    dtype_questions = {
        "question_id": "int32",
        # 'bundle_id': 'int32',
        # 'correct_answer': 'int8',
        "part": "int8",
        # 'tags': 'object',
    }
    questions = pd.read_csv(
        path_questions,
        dtype=dtype_questions,
        usecols=dtype_questions.keys(),
        index_col="question_id",
    )
    return questions


def get_train_df(path_train):
    dtype = {
        "answered_correctly": "int8",
        # 'row_id': 'int64',
        # 'timestamp': 'int64',
        "user_id": "int32",
        "content_id": "int16",
        # 'content_type_id': 'int8',
        "task_container_id": "int16",
        # 'user_answer': 'int8',
        "prior_question_elapsed_time": "float32",
        # 'prior_question_had_explanation': 'boolean'
    }
    df = pd.read_csv(path_train, usecols=dtype.keys(), dtype=dtype, nrows=10 ** 6)
    df = df[df.answered_correctly != -1]
    df = df.groupby("user_id").head(1500)
    return df


def transform_questions(questions):
    part_ids = questions.part.max() + 1
    return questions, part_ids


def transform_df(df, questions):
    df["prior_question_elapsed_time"] = (
        df["prior_question_elapsed_time"].fillna(0).astype(np.float32) / 300000
    )
    content_ids = questions.index.max() + 2
    df = df.join(questions, on="content_id")
    df["content_id"] += 1
    df["task_container_id"] += 1
    task_container_ids = 10001
    return df, content_ids, task_container_ids


def get_df(path_questions, path_train):

    questions = get_question_df(path_questions)
    df = get_train_df(path_train)

    questions, part_ids = transform_questions(questions)
    df, content_ids, task_container_ids = transform_df(df, questions)

    df = {uid: u.drop(columns="user_id") for uid, u in df.groupby("user_id")}
    return df, part_ids, content_ids, task_container_ids
