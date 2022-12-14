import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score



def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....", phase_codename)
    print("main.py file... 1")

    print(user_submission_file)
    user_predictions = np.loadtxt(user_submission_file)
    print(user_predictions)
    
    #print(test_annotation_file)
    test_true_labels = np.loadtxt(test_annotation_file)
    #print(test_true_labels)
    
    if phase_codename == "det_val_mta":
        print("Evaluating for Detection Validation Phase (MTA)")
        output = evaluate_det_val_phase(test_true_labels, user_predictions)
        print("Completed evaluation for Detection Validation Phase (MTA)")
    elif phase_codename == "det_test_mta":
        print("Evaluating for Detection Test (MTA)")
        output = evaluate_det_test_phase(test_true_labels, user_predictions)
        print("Completed evaluation for Detection Test Phase")
    elif phase_codename == "class_val_mta":
        print("Evaluating for Family Classification Validation Phase (MTA)")
        output = evaluate_class_val_phase(test_true_labels, user_predictions)
        print("Completed evaluation for Family Classification Validation Phase (MTA)")
    elif phase_codename == "class_test_mta":
        print("Evaluating for Family Classification Test Phase (MTA)")
        output = evaluate_class_test_phase(test_true_labels, user_predictions)
        print("Completed evaluation for Family Classification Test Phase (MTA)")
    elif phase_codename == "det_val_ustc":
        print("Evaluating for Detection Validation Phase (USTC)")
        output = evaluate_det_val_ustc_phase(test_true_labels, user_predictions)
        print("Completed evaluation for Detection Validation Phase (USTC)")
    elif phase_codename == "det_test_ustc":
        print("Evaluating for Detection Test (USTC)")
        output = evaluate_det_test_ustc_phase(test_true_labels, user_predictions)
        print("Completed evaluation for Detection Test Phase (USTC)")
    elif phase_codename == "class_val_ustc":
        print("Evaluating for Family Classification Validation Phase (USTC)")
        output = evaluate_class_val_ustc_phase(test_true_labels, user_predictions)
        print("Completed evaluation for Family Classification Validation Phase (USTC)")
    elif phase_codename == "class_test_ustc":
        print("Evaluating for Family Classification Test Phase (USTC)")
        output = evaluate_class_test_ustc_phase(test_true_labels, user_predictions)
        print("Completed evaluation for Family Classification Test Phase (USTC)")
        
    return output


def evaluate_det_val_phase(true_labels, user_predictions):
    output = {}
    output["result"] = [
        {
            "det_val_split_mta": {
                "Accuracy":  accuracy_score(true_labels, user_predictions),
                "Precision": precision_score(true_labels, user_predictions, average='macro'),
                "Recall":    recall_score(true_labels, user_predictions, average='macro'),
                "F1":        f1_score(true_labels, user_predictions, average='macro'),
            }
        }
    ]
    # To display the results in the result file
    output["submission_result"] = output["result"][0]["det_val_split_mta"]
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