import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

# import random


# def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
#     print("Starting Evaluation.....")
#     """
#     Evaluates the submission for a particular challenge phase and returns score
#     Arguments:

#         `test_annotations_file`: Path to test_annotation_file on the server
#         `user_submission_file`: Path to file submitted by the user
#         `phase_codename`: Phase to which submission is made

#         `**kwargs`: keyword arguments that contains additional submission
#         metadata that challenge hosts can use to send slack notification.
#         You can access the submission metadata
#         with kwargs['submission_metadata']

#         Example: A sample submission metadata can be accessed like this:
#         >>> print(kwargs['submission_metadata'])
#         {
#             'status': u'running',
#             'when_made_public': None,
#             'participant_team': 5,
#             'input_file': 'https://abc.xyz/path/to/submission/file.json',
#             'execution_time': u'123',
#             'publication_url': u'ABC',
#             'challenge_phase': 1,
#             'created_by': u'ABC',
#             'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
#             'method_name': u'Test',
#             'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
#             'participant_team_name': u'Test Team',
#             'project_url': u'http://foo.bar',
#             'method_description': u'ABC',
#             'is_public': False,
#             'submission_result_file': 'https://abc.xyz/path/result/file.json',
#             'id': 123,
#             'submitted_at': u'2017-03-20T19:22:03.880652Z'
#         }
#     """
#     output = {}
#     if phase_codename == "dev":
#         print("Evaluating for Dev Phase")
#         output["result"] = [
#             {
#                 "train_split": {
#                     "Metric1": random.randint(0, 99),
#                     "Metric2": random.randint(0, 99),
#                     "Metric3": random.randint(0, 99),
#                     "Total": random.randint(0, 99),
#                 }
#             }
#         ]
#         # To display the results in the result file
#         output["submission_result"] = output["result"][0]["train_split"]
#         print("Completed evaluation for Dev Phase")
#     elif phase_codename == "test":
#         print("Evaluating for Test Phase")
#         output["result"] = [
#             {
#                 "train_split": {
#                     "Metric1": random.randint(0, 99),
#                     "Metric2": random.randint(0, 99),
#                     "Metric3": random.randint(0, 99),
#                     "Total": random.randint(0, 99),
#                 }
#             },
#             {
#                 "test_split": {
#                     "Metric1": random.randint(0, 99),
#                     "Metric2": random.randint(0, 99),
#                     "Metric3": random.randint(0, 99),
#                     "Total": random.randint(0, 99),
#                 }
#             },
#         ]
#         # To display the results in the result file
#         output["submission_result"] = output["result"][0]
#         print("Completed evaluation for Test Phase")
#     return output



def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation ==>  ", phase_codename)
    # print("main.py file... 1")

    print(user_submission_file)
    user_predictions = np.loadtxt(user_submission_file)
    print(user_predictions)
    
    #print(test_annotation_file)
    test_true_labels = np.loadtxt(test_annotation_file)
    #print(test_true_labels)
    
    if phase_codename == "train_split_ttt":
        print("Evaluating for train_split_ttt")
        output = evaluate_train_split_phase(test_true_labels, user_predictions)
        print("Completed evaluation train_split_ttt")
    # elif phase_codename == "det_test_mta":
    #     print("Evaluating for Detection Test (MTA)")
    #     output = evaluate_det_test_phase(test_true_labels, user_predictions)
    #     print("Completed evaluation for Detection Test Phase")
    # elif phase_codename == "class_val_mta":
    #     print("Evaluating for Family Classification Validation Phase (MTA)")
    #     output = evaluate_class_val_phase(test_true_labels, user_predictions)
    #     print("Completed evaluation for Family Classification Validation Phase (MTA)")
    # elif phase_codename == "class_test_mta":
    #     print("Evaluating for Family Classification Test Phase (MTA)")
    #     output = evaluate_class_test_phase(test_true_labels, user_predictions)
    #     print("Completed evaluation for Family Classification Test Phase (MTA)")
    # elif phase_codename == "det_val_ustc":
    #     print("Evaluating for Detection Validation Phase (USTC)")
    #     output = evaluate_det_val_ustc_phase(test_true_labels, user_predictions)
    #     print("Completed evaluation for Detection Validation Phase (USTC)")
    # elif phase_codename == "det_test_ustc":
    #     print("Evaluating for Detection Test (USTC)")
    #     output = evaluate_det_test_ustc_phase(test_true_labels, user_predictions)
    #     print("Completed evaluation for Detection Test Phase (USTC)")
    # elif phase_codename == "class_val_ustc":
    #     print("Evaluating for Family Classification Validation Phase (USTC)")
    #     output = evaluate_class_val_ustc_phase(test_true_labels, user_predictions)
    #     print("Completed evaluation for Family Classification Validation Phase (USTC)")
    # elif phase_codename == "class_test_ustc":
    #     print("Evaluating for Family Classification Test Phase (USTC)")
    #     output = evaluate_class_test_ustc_phase(test_true_labels, user_predictions)
    #     print("Completed evaluation for Family Classification Test Phase (USTC)")
    return output


def evaluate_train_split_phase(true_labels, user_predictions):
    output = {}
    output["result"] = [
        {
            "train_split": {
                "Accuracy":  accuracy_score(true_labels, user_predictions),
                "Precision": precision_score(true_labels, user_predictions, average='macro'),
                "Recall":    recall_score(true_labels, user_predictions, average='macro'),
                "F1":        f1_score(true_labels, user_predictions, average='macro'),
            }
        }
    ]
    # To display the results in the result file
    output["submission_result"] = output["result"][0]["train_split"]
    return output

def evaluate_det_test_phase(true_labels, user_predictions):
    output = {}
    output["result"] = [
        {
            "det_test_split_mta": {
                "Accuracy":  accuracy_score(true_labels, user_predictions),
                "Precision": precision_score(true_labels, user_predictions, average='macro'),
                "Recall":    recall_score(true_labels, user_predictions, average='macro'),
                "F1":        f1_score(true_labels, user_predictions, average='macro'),
            }
        }
    ]
    # To display the results in the result file
    output["submission_result"] = output["result"][0]["det_test_split_mta"]
    return output

def evaluate_class_val_phase(true_labels, user_predictions):
    output = {}
    output["result"] = [
        {
            "class_val_split_mta": {
                "Accuracy":  accuracy_score(true_labels, user_predictions),
                "Precision": precision_score(true_labels, user_predictions, average='macro'),
                "Recall":    recall_score(true_labels, user_predictions, average='macro'),
                "F1":        f1_score(true_labels, user_predictions, average='macro'),
            }
        }
    ]
    # To display the results in the result file
    output["submission_result"] = output["result"][0]["class_val_split_mta"]
    return output

def evaluate_class_test_phase(true_labels, user_predictions):
    output = {}
    output["result"] = [
        {
            "class_test_split_mta": {
                "Accuracy":  accuracy_score(true_labels, user_predictions),
                "Precision": precision_score(true_labels, user_predictions, average='macro'),
                "Recall":    recall_score(true_labels, user_predictions, average='macro'),
                "F1":        f1_score(true_labels, user_predictions, average='macro'),
            }
        }
    ]
    # To display the results in the result file
    output["submission_result"] = output["result"][0]["class_test_split_mta"]
    return output

def evaluate_det_val_ustc_phase(true_labels, user_predictions):
    output = {}
    output["result"] = [
        {
            "det_val_split_ustc": {
                "Accuracy":  accuracy_score(true_labels, user_predictions),
                "Precision": precision_score(true_labels, user_predictions, average='macro'),
                "Recall":    recall_score(true_labels, user_predictions, average='macro'),
                "F1":        f1_score(true_labels, user_predictions, average='macro'),
            }
        }
    ]
    # To display the results in the result file
    output["submission_result"] = output["result"][0]["det_val_split_ustc"]
    return output

def evaluate_det_test_ustc_phase(true_labels, user_predictions):
    output = {}
    output["result"] = [
        {
            "det_test_split_ustc": {
                "Accuracy":  accuracy_score(true_labels, user_predictions),
                "Precision": precision_score(true_labels, user_predictions, average='macro'),
                "Recall":    recall_score(true_labels, user_predictions, average='macro'),
                "F1":        f1_score(true_labels, user_predictions, average='macro'),
            }
        }
    ]
    # To display the results in the result file
    output["submission_result"] = output["result"][0]["det_test_split_ustc"]
    return output

def evaluate_class_val_ustc_phase(true_labels, user_predictions):
    output = {}
    output["result"] = [
        {
            "class_val_split_ustc": {
                "Accuracy":  accuracy_score(true_labels, user_predictions),
                "Precision": precision_score(true_labels, user_predictions, average='macro'),
                "Recall":    recall_score(true_labels, user_predictions, average='macro'),
                "F1":        f1_score(true_labels, user_predictions, average='macro'),
            }
        }
    ]
    # To display the results in the result file
    output["submission_result"] = output["result"][0]["class_val_split_ustc"]
    return output

def evaluate_class_test_ustc_phase(true_labels, user_predictions):
    output = {}
    output["result"] = [
        {
            "class_test_split_ustc": {
                "Accuracy":  accuracy_score(true_labels, user_predictions),
                "Precision": precision_score(true_labels, user_predictions, average='macro'),
                "Recall":    recall_score(true_labels, user_predictions, average='macro'),
                "F1":        f1_score(true_labels, user_predictions, average='macro'),
            }
        }
    ]
    # To display the results in the result file
    output["submission_result"] = output["result"][0]["class_test_split_ustc"]
    return output